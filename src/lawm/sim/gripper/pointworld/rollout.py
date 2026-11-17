"""PointWorld's predicted motion of a candidate gripper trajectory: one forward pass of its 10-step window through the release pipeline, reduced to the mean displacement of the object and of the rest of the scene."""
from __future__ import annotations

import numpy as np
import torch

CONTEXT, H_CHUNK = 1, 10
T_CHUNK = CONTEXT + H_CHUNK
OBJECT_MARKER = np.array([255.0, 0.0, 255.0], dtype=np.float32)  # object rows carry this colour through the pipeline, which reorders points


def marked_colors(colors, object_mask):
    """Colours with the object rows painted with the marker, raw 0-255."""
    out = colors.astype(np.float32).copy()
    out[object_mask] = OBJECT_MARKER
    return out


def recover_object_mask(colors):
    """Which points carry the marker, for colours on either scale."""
    colors = np.asarray(colors, dtype=np.float32)
    unit = colors.max() <= 1.5
    return np.linalg.norm(colors - (OBJECT_MARKER / 255.0 if unit else OBJECT_MARKER)[None, :], axis=-1) < (5.0 / 255.0) * (1.0 if unit else 255.0)


def without_robot(obs):
    """The observation with the scene points on the robot removed, and how many went."""
    keep = ~np.asarray(obs["scene_is_robot"], dtype=bool)
    out = dict(obs)
    for key in ("scene_pts", "scene_colors", "scene_normals", "scene_obj_mask", "scene_is_robot"):
        out[key] = obs[key][keep]
    return out, int((~keep).sum())


class Rollout:
    """The prediction of one replan's observation; pipeline and collate are PointWorld's release functions."""

    def __init__(self, model, model_args, obs, device, pipeline, collate, batch):
        """Keep the scene with its object marked, the camera data and the gripper cloud, and check the object survives the pipeline."""
        self.model, self.args, self.device, self.pipeline, self.collate, self.batch = model, model_args, device, pipeline, collate, batch
        self.points = np.asarray(obs["scene_pts"], dtype=np.float32)
        self.colors = marked_colors(obs["scene_colors"], np.asarray(obs["scene_obj_mask"], bool))
        self.normals = np.asarray(obs["scene_normals"], dtype=np.float32)
        self.cameras = {k: v for k, v in obs.items() if k.startswith("cam")}
        self.gripper_open = float(obs["gripper_open"])
        self.robot_pts = np.asarray(obs["robot_pts"], dtype=np.float32)
        self.robot_normals = np.asarray(obs["robot_normals"], dtype=np.float32)
        first = self.sample(np.tile(self.robot_pts[None], (T_CHUNK, 1, 1)))
        self.n_object_post = int(recover_object_mask(np.asarray(first["scene_colors"])[0]).sum())
        assert self.n_object_post, "no object points survived PointWorld's pipeline: the object cost would score an empty set"

    def sample(self, robot_pts):
        """One release-format sample of this scene with a gripper trajectory (T, Nr, 3), run through the pipeline."""
        ns, nr = len(self.points), robot_pts.shape[1]
        sample = dict(robot_flows=np.array(robot_pts, dtype=np.float32, copy=True), robot_colors=np.zeros((T_CHUNK, nr, 3), dtype=np.float32), robot_normals=np.tile(self.robot_normals[None], (T_CHUNK, 1, 1)),
                      robot_exists=np.ones((T_CHUNK, nr), dtype=bool), scene_flows=np.tile(self.points[None], (T_CHUNK, 1, 1)), scene_colors=np.tile(self.colors[None], (T_CHUNK, 1, 1)),
                      scene_normals=np.tile(self.normals[None], (T_CHUNK, 1, 1)), scene_exists=np.ones((T_CHUNK, ns), dtype=bool), scene_visibility=np.ones((T_CHUNK, ns), dtype=bool),
                      scene_depth_valid_mask=np.ones((T_CHUNK, ns), dtype=bool), right_gripper_open=np.full((T_CHUNK, 1), self.gripper_open, dtype=np.float32), __domain__="droid",
                      joint_positions=np.zeros((T_CHUNK, 7), dtype=np.float32), gripper_positions=np.full((T_CHUNK,), 1.0 - self.gripper_open, dtype=np.float32))
        sample.update({k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.cameras.items()})  # the pipeline changes the extrinsics in place
        return self.pipeline(sample, domain="droid", mode="test", args=self.args, include_scene_data=True)

    def decode(self, trajectories):
        """(object, scene) cumulative predicted displacement, each (K, 10, 3), of trajectories (K, 10, 3) in metres."""
        trajectories = np.asarray(trajectories, dtype=np.float32)
        object_disp, scene_disp = np.zeros((len(trajectories), H_CHUNK, 3), dtype=np.float32), np.zeros((len(trajectories), H_CHUNK, 3), dtype=np.float32)
        for lo in range(0, len(trajectories), self.batch):
            chunk = trajectories[lo:lo + self.batch]
            steps = np.concatenate([np.zeros((len(chunk), 1, 3), np.float32), chunk], axis=1)
            robot = self.robot_pts[None, None] + steps[:, :, None, :]
            processed = [self.sample(r) for r in robot]
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in self.collate(processed, self.args).items()}
            with torch.no_grad():
                pred = self.model(batch, training=False)["scene_flows"].detach().float().cpu().numpy()
            for i, item in enumerate(processed):
                colors = np.asarray(item["scene_colors"])[0].astype(np.float32)
                motion = pred[i, 1:, :len(colors)] - pred[i, 0:1, :len(colors)]
                is_object = recover_object_mask(colors)
                if is_object.any():
                    object_disp[lo + i] = motion[:, is_object].mean(1)
                if (~is_object).any():
                    scene_disp[lo + i] = motion[:, ~is_object].mean(1)
        return object_disp, scene_disp
