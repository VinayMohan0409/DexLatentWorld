"""The DexWM robot's right Allegro hand as a point cloud sampled from MuJoCo's own visual geoms: palm, finger links and DIGIT fingertips, without the wrist-camera parts."""
from __future__ import annotations

import mujoco
import numpy as np

from lawm.sim import clouds

ACCESSORY_MESHES = ("camera_mount", "zed_x")  # wrist camera and its mount: real but not part of the hand; the model never saw a camera on any hand


def hand_geom_ids(model, prefix="gripper0_right_", accessories=ACCESSORY_MESHES):
    """Visual mesh geoms of every body whose name starts with prefix, without the accessory meshes."""
    gids = []
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if not name.startswith(prefix):
            continue
        for g in clouds.body_geom_ids(model, body, "visual"):
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) if model.geom_type[g] == clouds.MESH else None
            if mesh is not None and not mesh.endswith(accessories):
                gids.append(g)
    assert gids, f"no hand geoms under bodies starting with {prefix!r}"
    return gids


def hand_cloud(model, data, n, rng, prefix="gripper0_right_"):
    """n points over the hand's visual surface, split by surface area; returns (points (n, 3) float32, geom id of each point)."""
    return clouds.geoms_cloud(model, data, hand_geom_ids(model, prefix), n, rng)


def hand_finger_mask(model, owner, prefix="gripper0_right_"):
    """True for the hand points owned by a finger body and False for the palm (the hand's root body); owner is the geom id of each point."""
    names = {b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(1, model.nbody)}
    hand = {b for b, n in names.items() if n.startswith(prefix)}
    roots = [b for b in hand if int(model.body_parentid[b]) not in hand]
    assert len(roots) == 1, f"the hand must have one root body, found {len(roots)}"
    return model.geom_bodyid[np.asarray(owner)] != roots[0]
