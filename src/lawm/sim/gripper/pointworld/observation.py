"""A PointWorld observation of a live robosuite scene: rendered RGB-D from two cameras in the world frame, the object mask from the simulator's segmentation, and the gripper's own surface as the action cloud."""
from __future__ import annotations

import numpy as np

from lawm.sim.clouds import BOX, box_triangles, geom_triangles

H, W = 180, 320  # the resolution PointWorld's release requires
CAMERAS = ("agentview", "frontview")
WORKSPACE_MIN = np.array([0.00, -0.40, -0.30], dtype=np.float32) - 0.5 * np.array([0.70, 0.80, 1.50], dtype=np.float32)  # DROID's relaxed box, in the robot base frame
WORKSPACE_MAX = np.array([0.70, 0.40, 1.20], dtype=np.float32) + 0.5 * np.array([0.70, 0.80, 1.50], dtype=np.float32)
TABLE_CUT_MARGIN = 0.02
N_ROBOT = 2048
OPEN_SEP_MM = 139.1
ROBOT_PREFIXES = ("robot0_", "gripper0_", "fixed_mount0_")
LEFT_PAD, RIGHT_PAD = "gripper0_right_left_fingertip_visual", "gripper0_right_right_fingertip_visual"
OBJECT_MARGIN_M = 0.02
SCENE_KEYS = ("scene_pts", "scene_colors", "scene_normals", "scene_obj_mask", "scene_is_robot")


def render_camera(env, camera):
    """RGB, metric depth, per-pixel geom ids, intrinsics, extrinsics and the world-to-pixel matrix of one camera, upright."""
    from robosuite.utils import camera_utils
    rgb, depth = env.sim.render(height=H, width=W, camera_name=camera, depth=True)
    depth = np.asarray(camera_utils.get_real_depth_map(env.sim, depth[::-1].copy())).reshape(H, W).astype(np.float32)
    geoms = np.asarray(camera_utils.get_camera_segmentation(env.sim, camera, H, W))[..., 1].reshape(H, W).astype(np.int32)
    return (rgb[::-1].copy().astype(np.uint8), depth, geoms, camera_utils.get_camera_intrinsic_matrix(env.sim, camera, H, W).astype(np.float32),
            camera_utils.get_camera_extrinsic_matrix(env.sim, camera).astype(np.float32), camera_utils.get_camera_transform_matrix(env.sim, camera, H, W).astype(np.float32))


def backproject(rgb, depth, geoms, world_to_pixel):
    """World points, colours and geom ids of the pixels with valid depth, and the world-point grid with its validity."""
    rows, cols = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij")
    z = depth[rows, cols].astype(np.float64)
    valid = z > 1e-4
    grid = np.einsum("ij,rcj->rci", np.linalg.inv(world_to_pixel.astype(np.float64)), np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1))[..., :3]
    return grid[valid].astype(np.float32), rgb[rows, cols][valid].astype(np.uint8), geoms[rows, cols][valid].astype(np.int32), grid.astype(np.float32), valid


def grid_normals(grid, valid):
    """Normals of a depth grid from the cross product of neighbouring points; their sign is not turned towards the camera."""
    dx, dy = np.zeros_like(grid), np.zeros_like(grid)
    dx[:, :-1] = grid[:, 1:] - grid[:, :-1]
    dy[:-1, :] = grid[1:, :] - grid[:-1, :]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)[valid].astype(np.float32)


def face_camera(normals, points, camera_pos):
    """Normals with their sign turned towards the camera."""
    flip = ((camera_pos[None] - points) * normals).sum(axis=-1) < 0
    return np.where(flip[:, None], -normals, normals)


def alt_box_triangles(half):
    """Vertices and triangles of a box."""
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    index = lambda sx, sy, sz: (sx > 0) * 4 + (sy > 0) * 2 + (sz > 0)  # noqa: E731
    faces = []
    for a, b, c, d, e, f in (((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, -1), (-1, 1, 1), (-1, -1, 1)), ((1, -1, -1), (1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1), (1, 1, 1)),
                             ((-1, -1, -1), (1, -1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1), (1, -1, 1)), ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, -1), (1, 1, 1), (-1, 1, 1)),
                             ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, -1, -1), (1, 1, -1), (-1, 1, -1)), ((-1, -1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1), (1, 1, 1))):
        faces += [[index(*a), index(*b), index(*c)], [index(*d), index(*e), index(*f)]]
    return corners, np.array(faces, dtype=np.int64)


def local_triangles(model, gid, outward):
    """A geom's local vertices and triangles; boxes have outward normals only when asked."""
    if int(model.geom_type[gid]) == BOX:
        return box_triangles(model.geom_size[gid]) if outward else alt_box_triangles(model.geom_size[gid])
    return geom_triangles(model, gid)


def triangle_areas(verts, faces):
    """(cross products, areas) of a triangle mesh."""
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return cross, 0.5 * np.linalg.norm(cross, axis=1)


def sample_with_normals(model, data, gid, n, rng, outward):
    """n area-weighted surface points of one geom in the world frame, with their face normals."""
    verts, faces = local_triangles(model, gid, outward)
    cross, areas = triangle_areas(verts, faces)
    face = rng.choice(len(faces), size=n, p=areas / areas.sum())
    root, u2 = np.sqrt(rng.uniform(size=n)), rng.uniform(size=n)
    weights = np.stack([1.0 - root, root * (1.0 - u2), root * u2], axis=-1)
    points = np.einsum("nk,nkd->nd", weights, verts[faces][face])
    normals = cross[face]
    size = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, size, out=np.zeros_like(normals), where=size > 1e-12)
    rot, pos = data.geom_xmat[gid].reshape(3, 3), data.geom_xpos[gid]
    return (points @ rot.T + pos).astype(np.float32), (normals @ rot.T).astype(np.float32)


def visual_geoms(sim, body):
    """Visual geoms of one body: names holding "visual", else names ending in "_vis", else all of them."""
    names = [(g, sim.model.geom_id2name(g) or "") for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == body]
    return [g for g, n in names if "visual" in n] or [g for g, n in names if n.endswith("_vis")] or [g for g, _ in names]


class Observer:
    """Captures a PointWorld observation at each replan; made once per trial, after the reset."""

    def __init__(self, env, object_geoms, object_body, gripper_bodies, seed=0, outward_normals=False):
        """Look up the geoms of the object, the robot and the gripper and the table height; outward_normals selects the box face winding."""
        sim = env.sim
        names = {sim.model.geom_id2name(g) for g in range(sim.model.ngeom)}
        self.object_gids = {sim.model.geom_name2id(n) for n in object_geoms if n in names}
        assert self.object_gids, f"none of {object_geoms} exist in this model"
        self.table_z = float(env.model.mujoco_arena.table_top_abs[2]) if hasattr(env.model.mujoco_arena, "table_top_abs") else float(sim.data.geom_xpos[sim.model.geom_name2id("table_collision")][2] + sim.model.geom_size[sim.model.geom_name2id("table_collision")][2])
        self.object_body = sim.model.body_name2id(object_body)
        self.object_radius = max(float(np.linalg.norm(sim.model.geom_size[g])) for g in self.object_gids) + OBJECT_MARGIN_M
        self.robot_gids = {g for g in range(sim.model.ngeom) if (sim.model.body_id2name(sim.model.geom_bodyid[g]) or "").startswith(ROBOT_PREFIXES)}
        self.gripper_bodies = [sim.model.body_name2id(b) for b in gripper_bodies]
        self.base = sim.model.body_name2id("robot0_link0")
        self.pads = [sim.model.geom_name2id(LEFT_PAD), sim.model.geom_name2id(RIGHT_PAD)] if {LEFT_PAD, RIGHT_PAD} <= names else None
        self.seed, self.outward = seed, outward_normals

    def robot_cloud(self, sim):
        """The gripper's surface as the action cloud: points and normals, split over its visual geoms by area."""
        model, data, rng = sim.model._model, sim.data._data, np.random.default_rng(self.seed)
        gids = [g for b in self.gripper_bodies for g in visual_geoms(sim, b)]
        areas = np.array([triangle_areas(*local_triangles(model, g, self.outward))[1].sum() for g in gids])
        counts = np.floor(areas / areas.sum() * N_ROBOT).astype(int)
        counts[np.argmax(counts)] += N_ROBOT - counts.sum()
        parts = [sample_with_normals(model, data, g, int(c), rng, self.outward) for g, c in zip(gids, counts) if c > 0]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def capture(self, env):
        """The observation of the current state: scene points, colours, normals and masks, the camera data, the action cloud and how open the gripper is."""
        sim = env.sim
        parts, payload = [], {}
        for i, camera in enumerate(CAMERAS):
            rgb, depth, geoms, intrinsic, extrinsic, world_to_pixel = render_camera(env, camera)
            points, colours, ids, grid, valid = backproject(rgb, depth, geoms, world_to_pixel)
            normals = grid_normals(grid, valid)
            if self.outward:
                normals = face_camera(normals, points, extrinsic[:3, 3])
            parts.append((points, colours, normals, np.isin(ids, list(self.object_gids)), np.isin(ids, list(self.robot_gids))))
            payload.update({f"cam{i}_initial_rgb": rgb, f"cam{i}_initial_depth": depth, f"cam{i}_intrinsic": intrinsic, f"cam{i}_extrinsic": extrinsic, f"cam{i}_exists": np.array(True)})
        obs = {k: np.concatenate([p[j] for p in parts], 0) for j, k in enumerate(SCENE_KEYS)}
        far = np.linalg.norm(obs["scene_pts"] - sim.data.body_xpos[self.object_body][None], axis=-1) > self.object_radius
        obs["scene_obj_mask"] = obs["scene_obj_mask"] & ~far
        base_pos, base_rot = sim.data.body_xpos[self.base].astype(np.float32), sim.data.body_xmat[self.base].reshape(3, 3).astype(np.float32)
        in_base = (obs["scene_pts"] - base_pos[None]) @ base_rot
        keep = np.all((in_base >= WORKSPACE_MIN[None]) & (in_base <= WORKSPACE_MAX[None]), axis=-1) & (obs["scene_pts"][:, 2] >= self.table_z - TABLE_CUT_MARGIN)
        obs = {k: v[keep] for k, v in obs.items()}
        obs.update(payload)
        obs["robot_pts"], obs["robot_normals"] = self.robot_cloud(sim)
        separation = float(np.linalg.norm(sim.data.geom_xpos[self.pads[0]] - sim.data.geom_xpos[self.pads[1]])) if self.pads else 0.0
        obs["gripper_open"] = np.float32(np.clip(separation * 1000.0 / OPEN_SEP_MM, 0.0, 1.0))
        return obs
