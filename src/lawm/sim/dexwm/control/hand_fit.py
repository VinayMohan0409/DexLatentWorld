"""Geometry to action for the DexWM hand: recover the wrist pose and the finger joint angles from predicted hand points, using the kinematics read from MuJoCo's own model, and turn them into the controller's absolute action."""
from __future__ import annotations

import re

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


MIN_PALM_POINTS = 12  # a rigid fit of fewer points is not trusted


def quat_to_matrix(quat):
    """Rotation matrix of a MuJoCo (w, x, y, z) quaternion."""
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def kabsch(local, world):
    """Rotation R and translation t with world ~= local @ R.T + t, for index-corresponded points (reflection-safe)."""
    lc, wc = local.mean(0), world.mean(0)
    u, _, vt = np.linalg.svd((local - lc).T @ (world - wc))
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, wc - r @ lc


def axis_rotation(axis, angle):
    """Rotation matrix about a unit axis by an angle (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class HandKinematics:
    """Forward kinematics of the hand bodies (name prefix) from the model's own offsets and hinge joints, and the fit of root pose and joint angles to points drawn by a SurfaceSampler."""

    def __init__(self, model, sampler, prefix="gripper0_right_"):
        """Collect the hand bodies in parent-first order, their static offsets and joints, and put the sampler's points into their body frames."""
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
        self.bodies = [b for b in range(1, model.nbody) if names[b].startswith(prefix)]
        index = {b: i for i, b in enumerate(self.bodies)}
        self.parent = np.array([index.get(int(model.body_parentid[b]), -1) for b in self.bodies])
        roots = np.flatnonzero(self.parent < 0)
        assert len(roots) == 1, f"the hand must have one root body, found {len(roots)}"
        self.root = int(roots[0])
        self.pos = np.array([model.body_pos[b] for b in self.bodies])
        self.rot = np.array([quat_to_matrix(model.body_quat[b]) for b in self.bodies])
        joints = {}
        for b in self.bodies:
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
                assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, "only hinge joints are supported"
                joints[int(re.search(r"joint_(\d+)", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)).group(1))] = (index[b], j)
        self.joint_body = np.array([joints[k][0] for k in sorted(joints)])
        ids = [joints[k][1] for k in sorted(joints)]
        self.axis = np.array([model.jnt_axis[j] / np.linalg.norm(model.jnt_axis[j]) for j in ids])
        self.joint_pos = np.array([model.jnt_pos[j] for j in ids])
        self.limits = np.array([model.jnt_range[j] for j in ids])
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in ids])  # where each joint angle sits in qpos
        self.n_joints = len(ids)
        owner_body = np.array([index[int(model.geom_bodyid[g])] for g in sampler.owner])
        geom_rot = np.array([quat_to_matrix(model.geom_quat[g]) for g in sampler.owner])
        self.point_body = owner_body
        group = np.full(len(self.bodies), -1)  # the chain (child of the root and everything below it) each body belongs to; the root is -1
        for i in range(len(self.bodies)):
            if i != self.root:
                a = i
                while self.parent[a] != self.root:
                    a = self.parent[a]
                group[i] = a
        self.chains = []
        for top in sorted(set(group[group >= 0])):
            bodies = [i for i in range(len(self.bodies)) if group[i] == top]
            joints = [k for k, b in enumerate(self.joint_body) if group[b] == top]
            if joints:
                self.chains.append((bodies, joints, np.flatnonzero(group[owner_body] == top)))
        self.local = np.einsum("nij,nj->ni", geom_rot, sampler.local) + model.geom_pos[sampler.owner]  # in the frame of the body owning each point

    def link_poses(self, root_rot, root_pos, q, bodies=None):
        """World rotation (n_bodies, 3, 3) and position (n_bodies, 3) of the hand bodies (all, or only the listed ones, parents first) for a root pose and joint angles."""
        rot, pos = np.zeros_like(self.rot), np.zeros_like(self.pos)
        rot[self.root], pos[self.root] = root_rot, root_pos
        joint_of = {int(b): k for k, b in enumerate(self.joint_body)}
        for i in (range(len(self.bodies)) if bodies is None else bodies):
            if i == self.root:
                rot[i], pos[i] = root_rot, root_pos
                continue
            p = self.parent[i]
            r_static, p_static = rot[p] @ self.rot[i], rot[p] @ self.pos[i] + pos[p]
            if i in joint_of:
                k = joint_of[i]
                r_joint = axis_rotation(self.axis[k], q[k])
                jp = self.joint_pos[k]
                rot[i], pos[i] = r_static @ r_joint, p_static + r_static @ (jp - r_joint @ jp)
            else:
                rot[i], pos[i] = r_static, p_static
        return rot, pos

    def points(self, root_rot, root_pos, q):
        """World positions (n, 3) of the sampled points for a root pose and joint angles."""
        rot, pos = self.link_poses(root_rot, root_pos, q)
        return np.einsum("nij,nj->ni", rot[self.point_body], self.local) + pos[self.point_body]

    def fit(self, predicted, q_init, max_nfev=60):
        """Wrist pose and joint angles that best explain the predicted points: a rigid fit on the palm points, then bounded least squares over the joints of each finger chain on its own points, started from q_init. Returns (rot, pos, q, info)."""
        palm = self.point_body == self.root
        assert palm.sum() >= MIN_PALM_POINTS, f"only {int(palm.sum())} palm points"
        rot, pos = kabsch(self.local[palm], predicted[palm])
        q = np.clip(np.asarray(q_init, dtype=float), self.limits[:, 0] + 1e-6, self.limits[:, 1] - 1e-6)  # the optimiser needs a start strictly inside the bounds
        nfev = 0
        for bodies, joints, rows in self.chains:
            def residual(x, bodies=bodies, joints=joints, rows=rows):
                """Flattened point errors of this chain for joint angles x."""
                full = q.copy()
                full[joints] = x
                r, p = self.link_poses(rot, pos, full, bodies)
                return (np.einsum("nij,nj->ni", r[self.point_body[rows]], self.local[rows]) + p[self.point_body[rows]] - predicted[rows]).ravel()
            sol = least_squares(residual, q[joints], bounds=(self.limits[joints, 0], self.limits[joints, 1]), max_nfev=max_nfev)
            q[joints] = sol.x
            nfev += int(sol.nfev)
        error = np.linalg.norm(self.points(rot, pos, q) - predicted, axis=1)
        return rot, pos, q, dict(nfev=nfev, mean_residual_mm=float(error.mean() * 1000), max_residual_mm=float(error.max() * 1000))


def hand_state_to_action(root_rot, root_pos, q, root_rot_now, root_pos_now, eef_pos_now, eef_rot_now):
    """The controller's absolute target (wrist position, wrist rotation vector, 16 finger angles): the root's motion since now is applied to the controller's current end-effector pose."""
    r_rel = root_rot @ root_rot_now.T
    t_rel = root_pos - r_rel @ root_pos_now
    return r_rel @ eef_pos_now + t_rel, Rotation.from_matrix(r_rel @ eef_rot_now).as_rotvec(), np.asarray(q, dtype=float)
