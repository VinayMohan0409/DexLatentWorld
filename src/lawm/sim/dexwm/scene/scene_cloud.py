"""A DexWM scene as a point cloud in the layout the model is trained on (hand, object, table, distractors; 1024 points each), sampled from the simulator's own meshes: no cameras, no depth, no masks."""
from __future__ import annotations

import re
from contextlib import contextmanager

import mujoco
import numpy as np

from lawm.sim import clouds
from lawm.sim.costs import SEG_DISTRACTOR, SEG_HAND, SEG_OBJECT, SEG_TABLE
from lawm.sim.dexwm.scene import hand_surface

TABLE_RADIUS = 1.0  # metres from the object: the training table reaches 1.06 m (99th percentile) around the object, its median point 0.53 m
DISTRACTOR_RADIUS = 1.1  # the training distractor centres reach 1.07 m
SUPPORT_BAND = 0.02  # metres around the height of the surface the object rests on
FREE_ABOVE = 0.25  # an upward-facing surface has nothing within this height above it (an underside, or a surface under a cabinet, fails)
MOVERS = re.compile(r"^(robot0_base|left_eef_target|right_eef_target|obj_main|distr_.*_main)$")
DISTRACTOR = re.compile(r"^distr_.*_main$")
VISUAL_GROUP = 1
HIDDEN_GROUP = 5


def top_level_body(model, body):
    """The ancestor of a body that is a direct child of the world."""
    while model.body_parentid[body] != 0:
        body = model.body_parentid[body]
    return body


def body_name(model, body):
    """Name of a body ('' if unnamed)."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""


def mover_geom_ids(model, movers=MOVERS):
    """Geoms of every top-level body whose name matches movers (the robot, the manipulated object, the distractors)."""
    hidden = set()
    for body in range(1, model.nbody):
        if movers.match(body_name(model, top_level_body(model, body))):
            first = int(model.body_geomadr[body])
            hidden.update(range(first, first + int(model.body_geomnum[body])))
    return sorted(hidden)


def visual_geom_ids(model, body_id):
    """Visual geoms of one body and everything below it."""
    ids = []
    for b in range(1, model.nbody):
        a = b
        while a != 0 and a != body_id:
            a = model.body_parentid[a]
        if a == body_id:
            ids += clouds.body_geom_ids(model, b, "visual")
    return ids


def distractor_bodies(model, data, centers, radius=DISTRACTOR_RADIUS):
    """Names of the distractor bodies within radius (horizontally) of any of the given object centres, e.g. its start and target position."""
    out = []
    for body in range(1, model.nbody):
        name = body_name(model, body)
        if DISTRACTOR.match(name) and model.body_parentid[body] == 0 and any(np.linalg.norm(data.xpos[body][:2] - c[:2]) < radius for c in centers):
            out.append(name)
    return out


@contextmanager
def movers_hidden(model, movers=MOVERS):
    """The movers' geoms are out of the visual group inside the block and restored after it."""
    hidden = mover_geom_ids(model, movers)
    saved = model.geom_group[hidden].copy()
    model.geom_group[hidden] = HIDDEN_GROUP
    try:
        yield
    finally:
        model.geom_group[hidden] = saved


def support_height(model, data, centers, movers=MOVERS):
    """Height of the highest static surface straight below any of the centers (an object held past the counter's edge sees only the floor below it)."""
    group = np.zeros(6, dtype=np.uint8)
    group[VISUAL_GROUP] = 1
    gid = np.zeros(1, dtype=np.int32)
    with movers_hidden(model, movers):
        hits = [(c[2], mujoco.mj_ray(model, data, np.asarray(c, dtype=np.float64), np.array([0.0, 0.0, -1.0]), group, 1, -1, gid)) for c in centers]
    heights = [z - d for z, d in hits if d > 0]
    assert heights, "no surface below the object"
    return max(heights)


def support_cloud(model, data, center, n, rng, radius=TABLE_RADIUS, band=SUPPORT_BAND, free=FREE_ABOVE, movers=MOVERS, also_under=()):
    """n points uniform over the upward-facing static surfaces within radius of center at the support height of center and the also_under points, with the movers left out."""
    z0 = support_height(model, data, [center, *also_under], movers)
    group = np.zeros(6, dtype=np.uint8)
    group[VISUAL_GROUP] = 1
    gid = np.zeros(1, dtype=np.int32)
    with movers_hidden(model, movers):
        verts, faces, offset = [], [], 0
        for g in np.flatnonzero(model.geom_group == VISUAL_GROUP):
            if np.linalg.norm(data.geom_xpos[g][:2] - center[:2]) > radius + model.geom_rbound[g] or abs(data.geom_xpos[g][2] - z0) > band + model.geom_rbound[g]:
                continue
            if model.geom_type[g] not in (clouds.SPHERE, clouds.BOX, clouds.CYLINDER, clouds.MESH):
                continue
            v, f = clouds.geom_triangles(model, g)
            v = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            tri = v[f]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            flat = np.abs(normal[:, 2]) > 0.9 * np.linalg.norm(normal, axis=1)
            level = (np.abs(tri[:, :, 2] - z0) < band).all(axis=1)
            keep = flat & level
            verts.append(v)
            faces.append(f[keep] + offset)
            offset += len(v)
        verts, faces = np.concatenate(verts), np.concatenate(faces)
        assert len(faces), "no flat surface at the height of the support within the radius"
        kept, draws = [], 0
        while sum(len(k) for k in kept) < n and draws < 8:
            candidates = clouds.sample_triangles(verts, faces, 20 * n, rng)[0]
            candidates = candidates[np.linalg.norm(candidates[:, :2] - center[:2], axis=1) < radius]
            up = np.array([0.0, 0.0, 1.0])
            ok = np.array([mujoco.mj_ray(model, data, p + 0.005 * up, up, group, 1, -1, gid) < 0 or mujoco.mj_ray(model, data, p + 0.005 * up, up, group, 1, -1, gid) > free
                           for p in candidates])
            kept.append(candidates[ok])
            draws += 1
    points = np.concatenate(kept)
    assert len(points) >= n, f"only {len(points)} free support points within {radius} m, need {n}"
    return points[rng.choice(len(points), size=n, replace=False)].astype(np.float32)


class DexWMScene:
    """Point-cloud builder of one episode: samplers drawn once, so row i is the same physical point at every step; the table is static."""

    def __init__(self, model, data, object_centers, n_hand, n_object, n_table, n_distractor, seed=0, prefix="gripper0_right_", object_body="obj_main"):
        """Fit the samplers at the current (start) state; object_centers are the object's positions at the start and the target, which decide the distractors and the table."""
        rng = np.random.default_rng(seed)
        self.hand = clouds.SurfaceSampler(model, hand_surface.hand_geom_ids(model, prefix), n_hand, rng)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        self.object = clouds.SurfaceSampler(model, visual_geom_ids(model, body), n_object, rng)
        self.distractor_names = distractor_bodies(model, data, object_centers)
        self.distractors = [clouds.SurfaceSampler(model, visual_geom_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)), n_distractor, rng) for name in self.distractor_names]
        self.table = support_cloud(model, data, np.asarray(object_centers[0], dtype=np.float64), n_table, rng, also_under=object_centers[1:]) if n_table else np.zeros((0, 3), np.float32)
        self.counts = [n_hand, n_object, len(self.table)] + [n_distractor] * len(self.distractors)
        self.segment = np.concatenate([np.full(c, s, dtype=np.int8) for c, s in zip(self.counts, [SEG_HAND, SEG_OBJECT, SEG_TABLE] + [SEG_DISTRACTOR] * len(self.distractors))])

    def __call__(self, data):
        """(coords float32 (N, 3), segment int8 (N,)) for the current state: hand, object, table, then each distractor."""
        parts = [self.hand(data), self.object(data), self.table] + [d(data) for d in self.distractors]
        return np.concatenate(parts).astype(np.float32), self.segment
