"""Tests of the DexWM scene cloud: layout and counts, support surface (top faces only, free above, movers see-through), distractor selection, stable rows."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from lawm.sim.dexwm.scene import scene_cloud as S  # noqa: E402

CUBE = 'vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1"'
XML = f"""
<mujoco>
  <asset>
    <mesh name="gripper0_right_palm" {CUBE} scale="0.04 0.04 0.02"/>
    <mesh name="piece" {CUBE} scale="0.03 0.03 0.03"/>
  </asset>
  <worldbody>
    <geom name="floor" type="box" size="5 5 0.01" pos="0 0 -0.01" group="1"/>
    <geom name="counter" type="box" size="1.5 1.5 0.015" pos="0 0 0.905" group="1"/>
    <geom name="far_counter" type="box" size="0.4 0.4 0.015" pos="3.0 0 0.905" group="1"/>
    <geom name="low_step" type="box" size="0.3 0.3 0.015" pos="0.2 -0.9 0.5" group="1"/>
    <geom name="cabinet" type="box" size="0.2 0.2 0.05" pos="0.6 0.6 1.2" group="1"/>
    <geom name="counter_collision" type="box" size="1.5 1.5 0.5" pos="0 0 0.4" group="0"/>
    <body name="obj_main" pos="0 0 1.0">
      <freejoint/>
      <geom name="obj" type="mesh" mesh="piece" group="1" contype="0" conaffinity="0"/>
    </body>
    <body name="distr_near_main" pos="0.5 0.2 0.95"><geom type="mesh" mesh="piece" group="1" contype="0" conaffinity="0"/></body>
    <body name="distr_far_main" pos="2.5 0 0.95"><geom type="mesh" mesh="piece" group="1" contype="0" conaffinity="0"/></body>
    <body name="robot0_base" pos="-1 0 0.9"><geom name="arm" type="box" size="0.05 0.05 0.3" pos="0.3 0 0.3" group="1" contype="0" conaffinity="0"/></body>
    <body name="gripper0_right_allegro_hand" pos="0.2 0 1.2">
      <freejoint/>
      <geom name="palm" type="mesh" mesh="gripper0_right_palm" group="1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def setup():
    """Compile the toy kitchen: a counter (top at z = 0.92) with an overhanging cabinet, a far counter, a lower step, an object, two distractors, a robot arm and a hand."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_table_points_are_on_the_top_face_at_the_support_height():
    """Table points have z = 0.92 (the counter top the object rests on), none on the lower step or the far counter, and none beyond the radius."""
    model, data = setup()
    pts = S.support_cloud(model, data, np.array([0.0, 0.0, 1.0]), 500, np.random.default_rng(0), radius=1.0)
    assert len(pts) == 500 and np.allclose(pts[:, 2], 0.92, atol=1e-5)
    assert (np.linalg.norm(pts[:, :2], axis=1) <= 1.0 + 1e-6).all()


def test_nothing_under_the_overhanging_cabinet():
    """Counter points under the cabinet (0.4 m to 0.8 m in x and y, 0.28 m above the counter) are free above only beyond FREE_ABOVE, so with a larger free height they are dropped."""
    model, data = setup()
    pts = S.support_cloud(model, data, np.array([0.0, 0.0, 1.0]), 400, np.random.default_rng(0), free=0.4)
    under = (pts[:, 0] > 0.4) & (pts[:, 0] < 0.8) & (pts[:, 1] > 0.4) & (pts[:, 1] < 0.8)
    assert not under.any()


def test_underside_of_a_thin_slab_is_excluded():
    """With a band wide enough to include the slab's underside (0.03 m below the top), no point is on it."""
    model, data = setup()
    pts = S.support_cloud(model, data, np.array([0.0, 0.0, 1.0]), 400, np.random.default_rng(0), band=0.04)
    assert np.allclose(pts[:, 2], 0.92, atol=1e-5)


def test_movers_are_see_through_and_restored():
    """The counter under the object is sampled although the object hides it, and the geoms' groups are unchanged afterwards."""
    model, data = setup()
    groups = model.geom_group.copy()
    pts = S.support_cloud(model, data, np.array([0.0, 0.0, 1.0]), 1500, np.random.default_rng(0), radius=0.5)
    assert ((np.abs(pts[:, 0]) < 0.03) & (np.abs(pts[:, 1]) < 0.03)).any() and np.array_equal(groups, model.geom_group)


def test_distractors_within_the_radius_only():
    """The near distractor is selected from either object centre; the far one only if an object centre is close to it."""
    model, data = setup()
    assert S.distractor_bodies(model, data, [np.array([0.0, 0.0, 1.0])]) == ["distr_near_main"]
    assert S.distractor_bodies(model, data, [np.array([0.0, 0.0, 1.0]), np.array([2.0, 0.0, 1.0])]) == ["distr_near_main", "distr_far_main"]


def test_scene_layout_counts_and_stable_rows():
    """The cloud is hand, object, table, distractor with 1024 points each; moving the object moves only its rows, and the table rows never move."""
    model, data = setup()
    scene = S.DexWMScene(model, data, [np.array([0.0, 0.0, 1.0])], 1024, 1024, 1024, 1024, seed=1)
    a, seg = scene(data)
    assert len(a) == 4 * 1024 and [int((seg == s).sum()) for s in range(4)] == [1024, 1024, 1024, 1024]
    assert (seg[:1024] == S.SEG_HAND).all() and (seg[1024:2048] == S.SEG_OBJECT).all() and (seg[2048:3072] == S.SEG_TABLE).all()
    obj_joint = model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_main")]
    data.qpos[model.jnt_qposadr[obj_joint]:model.jnt_qposadr[obj_joint] + 3] += [0.1, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    b, _ = scene(data)
    moved = np.linalg.norm(b - a, axis=1) > 1e-6
    assert moved[1024:2048].all() and not moved[:1024].any() and not moved[2048:].any()
    assert np.allclose(b[1024:2048] - a[1024:2048], [0.1, 0, 0], atol=1e-5)


def test_zero_table_points_is_allowed():
    """With n_table = 0 the cloud has no table rows."""
    model, data = setup()
    scene = S.DexWMScene(model, data, [np.array([0.0, 0.0, 1.0])], 1024, 1024, 0, 1024)
    assert (scene(data)[1] == S.SEG_TABLE).sum() == 0


def test_object_past_the_counter_edge_uses_the_support_of_its_target():
    """An object held past the counter's edge sees only the floor below it; with its target over the counter the table is the counter top."""
    model, data = setup()
    past_edge, target = np.array([1.6, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    assert abs(S.support_height(model, data, [past_edge])) < 1e-6
    pts = S.support_cloud(model, data, past_edge, 300, np.random.default_rng(0), also_under=[target])
    assert np.allclose(pts[:, 2], 0.92, atol=1e-5)


def test_entity_point_counts_are_settings():
    """Each entity takes its own number of points, as the original's 1024, 400, 2048 and 400."""
    model, data = setup()
    scene = S.DexWMScene(model, data, [np.array([0.0, 0.0, 1.0])], n_hand=1024, n_object=400, n_table=2048, n_distractor=400)
    assert [int((scene(data)[1] == s).sum()) for s in range(4)] == [1024, 400, 2048, 400]
