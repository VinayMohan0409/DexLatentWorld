"""The point cloud of a gripper-task scene: hand, moved object and static scene as rows with a fixed meaning, drawn once on the simulator's geoms and re-posed at every call."""
from __future__ import annotations

import mujoco
import numpy as np

from lawm.sim.clouds import BOX, SurfaceSampler, body_geom_ids, box_top_cloud, geom_triangles
from lawm.sim.costs import SEG_HAND, SEG_OBJECT, SEG_TABLE

ROBOTIQ_BODIES = ("gripper0_right_right_gripper", "gripper0_right_left_outer_knuckle", "gripper0_right_left_inner_finger", "gripper0_right_left_inner_knuckle",
                  "gripper0_right_right_outer_knuckle", "gripper0_right_right_inner_finger", "gripper0_right_right_inner_knuckle")
PLATE_BODY = "gripper0_right_wiping_gripper"
N_HAND, N_PLATE, N_OBJECT, N_TABLE = 2048, 400, 400, 2048  # points sampled per entity from the simulator geometry


def geom_id(model, name):
    """Id of the geom called name."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"no geom called {name!r}"
    return gid


def body_id(model, name):
    """Id of the body called name."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"no body called {name!r}"
    return bid


def named_visual_geoms(model, bodies):
    """Geoms of the bodies whose name contains "visual" (how the Robotiq's drawn parts are marked), in body and geom order."""
    ids = [g for b in bodies for g in range(model.ngeom) if model.geom_bodyid[g] == body_id(model, b) and "visual" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    assert ids, f"no visual geoms on {bodies}"
    return ids


def triangle_area(model, gid):
    """Surface area of a geom's tessellation."""
    verts, faces = geom_triangles(model, gid)
    tri = verts[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()


class AreaSplit:
    """n points over several geoms, the count of each geom proportional to its area (rounded down, the rest to the largest count), drawn once; calling it places them from the simulator's current geom poses."""

    def __init__(self, model, gids, n, rng):
        """Draw the points of every geom."""
        areas = np.array([triangle_area(model, g) for g in gids])
        counts = np.floor(areas / areas.sum() * n).astype(int)
        counts[np.argmax(counts)] += n - counts.sum()
        self.parts = [SurfaceSampler(model, [g], int(c), rng) for g, c in zip(gids, counts) if c > 0]

    def __call__(self, data):
        """The points in the world frame, float32 (n, 3)."""
        return np.concatenate([p(data) for p in self.parts], axis=0)

    @property
    def owner(self):
        """Geom id of every point."""
        return np.concatenate([p.owner for p in self.parts])


class SceneCloud:
    """Rows: hand, then object, then static scene; row i is the same physical point at every call. Static parts are drawn once and never move."""

    def __init__(self, model, data, hand, obj, static, rng):
        """hand and obj are (geoms, n); static is a list of (point array, ...) already drawn in the world frame."""
        self.hand, self.obj = AreaSplit(model, *hand, rng), AreaSplit(model, *obj, rng)
        self.static = np.concatenate(static, axis=0).astype(np.float32)
        n_hand, n_obj = hand[1], obj[1]
        self.segment = np.concatenate([np.full(n_hand, SEG_HAND), np.full(n_obj, SEG_OBJECT), np.full(len(self.static), SEG_TABLE)]).astype(np.int8)

    def __call__(self, sim):
        """(points (N, 3) float32, segment (N,) int8) for the simulator's current state."""
        data = sim.data._data
        return np.concatenate([self.hand(data), self.obj(data), self.static], axis=0), self.segment


def table_top(model, data, rng, n=N_TABLE):
    """n points on the top face of the table's visual box."""
    return box_top_cloud(model, data, "table_visual", n, rng)


def bin_collision(model, data, body, n, rng):
    """n points over the collision geoms of a bin (floor and four walls), split by area, in the world frame."""
    return AreaSplit(model, body_geom_ids(model, body_id(model, body), "collision"), n, rng)(data)


def stack_scene(sim, seed=0, n_hand=N_HAND, n_obj=N_OBJECT, n_table=N_TABLE):
    """Stack: the Robotiq hand, cubeA, and the table top."""
    model, data, rng = sim.model._model, sim.data._data, np.random.default_rng(seed)
    return SceneCloud(model, data, (named_visual_geoms(model, ROBOTIQ_BODIES), n_hand), ([geom_id(model, "cubeA_g0")], n_obj), [table_top(model, data, rng, n_table)], rng)


def milk_scene(sim, seed=0, n_hand=N_HAND, n_obj=N_OBJECT, n_table=N_TABLE):
    """PickPlaceMilk: the Robotiq hand, the milk carton, and the two bins (half the static points each)."""
    model, data, rng = sim.model._model, sim.data._data, np.random.default_rng(seed)
    carton = body_geom_ids(model, body_id(model, "Milk_main"), "visual")
    bins = [bin_collision(model, data, "bin1", n_table // 2, rng), bin_collision(model, data, "bin2", n_table - n_table // 2, rng)]
    return SceneCloud(model, data, (named_visual_geoms(model, ROBOTIQ_BODIES), n_hand), (carton, n_obj), bins, rng)


def plate_geoms(model):
    """The wiping gripper's collision boxes: the surface that pushes (its 1 mm edge spheres are left out)."""
    return [g for g in body_geom_ids(model, body_id(model, PLATE_BODY), "collision") if int(model.geom_type[g]) == BOX]


def push_scene(sim, seed=0, n_hand=N_PLATE, n_obj=N_OBJECT, n_table=N_TABLE):
    """Push: the wiping gripper's collision boxes, the pushed object and the table top."""
    model, data, rng = sim.model._model, sim.data._data, np.random.default_rng(seed)
    return SceneCloud(model, data, (plate_geoms(model), n_hand), ([geom_id(model, "cube_g0")], n_obj), [table_top(model, data, rng, n_table)], rng)


def reach_scene(sim, seed=0, n_hand=N_HAND, n_obj=N_OBJECT, n_table=N_TABLE):
    """Reach: the Robotiq hand, Lift's cube and the table top."""
    model, data, rng = sim.model._model, sim.data._data, np.random.default_rng(seed)
    return SceneCloud(model, data, (named_visual_geoms(model, ROBOTIQ_BODIES), n_hand), ([geom_id(model, "cube_g0")], n_obj), [table_top(model, data, rng, n_table)], rng)
