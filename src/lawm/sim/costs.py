"""Planner costs on predicted point displacements, in millimetres: distance to the goal, the end-effector floor barrier and ManiSkill's push reward, on masks of the segment ids."""
from __future__ import annotations

import numpy as np

SEG_HAND, SEG_OBJECT, SEG_TABLE, SEG_DISTRACTOR = 0, 1, 2, 3  # as lawm.data.sequences; rows are the same physical point in every cloud compared, so the match is per point, not Chamfer


def terminal_weights(n_steps):
    """All weight on the last horizon step."""
    w = np.zeros(n_steps, dtype=np.float32)
    w[-1] = 1.0
    return w


def step_weights(n_steps, step_index):
    """All weight on one horizon step (the executed one); equals terminal_weights for step_index = -1."""
    w = np.zeros(n_steps, dtype=np.float32)
    w[step_index] = 1.0
    return w


def horizon_weights(n_steps, last_weight=0.5):
    """Weights rising towards the last step, which keeps last_weight of the total: horizon_weights(5) = [0.05, 0.10, 0.15, 0.20, 0.50]."""
    if n_steps <= 1:
        return np.array([1.0], dtype=np.float32)
    w = np.arange(1, n_steps, dtype=np.float32)
    return np.concatenate([w / w.sum() * (1.0 - last_weight), [last_weight]]).astype(np.float32)


def goal_cost_mm(pred_deltas, coords0, goal, mask, weights):
    """Horizon-weighted mean per-point distance (mm) from the predicted points to the goal points on the mask rows; pred_deltas is (pop, n_steps, N, 3) cumulative from frame 0."""
    start, target = coords0[None, mask, :], goal[None, mask, :]
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, w in enumerate(weights):
        if w:
            total += w * np.linalg.norm(start + pred_deltas[:, k][:, mask, :] - target, axis=-1).mean(axis=-1) * 1000.0
    return total


def eef_floor_penalty_mm(pred_deltas, hand_mask, cur_eef_z, min_eef_z, ramp_depth_mm=10.0, ramp_scale_mm=800.0, contact_penalty_mm=5000.0):
    """One-sided barrier on the predicted hand height going below min_eef_z (metres): 0 at or above it, a squared ramp over ramp_depth_mm, then flat contact_penalty_mm; the worst step counts."""
    dz = pred_deltas[:, :, hand_mask, 2].mean(axis=2)  # hand-centroid dz per step, as the executor clamps it
    depth_mm = (min_eef_z - (cur_eef_z + dz)) * 1000.0  # positive: below the floor
    ramp = np.clip(depth_mm / max(ramp_depth_mm, 1e-6), 0.0, 1.0) ** 2
    return np.where(depth_mm >= ramp_depth_mm, contact_penalty_mm, ramp * ramp_scale_mm).max(axis=1)


def maniskill_push_cost_mm(pred_deltas, coords0, goal, hand_mask, obj_mask, weights, push_dir, offset_m, leading_face_frac=0.2, tanh_scale=5.0,
                           reached_thresh_m=0.01):
    """Negated ManiSkill3 PushCube-v1 dense reward on the predicted state: reach (1 - tanh(5 d_tcp), d_tcp from the leading hand face to the push pose) + place (1 - tanh(5 d_xy), only once d_tcp < 0.01) + height keeping (1 - tanh(5 |dz|), times place)."""
    obj_pts = coords0[obj_mask]
    obj_now, obj_z0 = obj_pts.mean(0), obj_pts[:, 2].mean()
    proj = obj_pts @ push_dir
    push_pose = obj_now - push_dir * (float(proj.max() - proj.min()) / 2.0 + offset_m)
    hand_pts = coords0[hand_mask]
    face = (hand_pts @ push_dir) >= np.quantile(hand_pts @ push_dir, 1.0 - leading_face_frac)
    start_face, start_obj = hand_pts[face][None], coords0[None, obj_mask, :]
    goal_xy = goal[obj_mask].mean(0)[:2]
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, w in enumerate(weights):
        if not w:
            continue
        d_tcp = np.linalg.norm((start_face + pred_deltas[:, k][:, hand_mask, :][:, face, :]).mean(axis=1) - push_pose[None], axis=-1)
        obj_c = (start_obj + pred_deltas[:, k][:, obj_mask, :]).mean(axis=1)
        place = (1.0 - np.tanh(tanh_scale * np.linalg.norm(obj_c[:, :2] - goal_xy[None], axis=-1))) * (d_tcp < reached_thresh_m)
        height = (1.0 - np.tanh(tanh_scale * np.abs(obj_c[:, 2] - obj_z0))) * place
        total += w * -((1.0 - np.tanh(tanh_scale * d_tcp)) + place + height)
    return total
