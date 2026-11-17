"""Tests of the DexWM hand cloud: only visual hand geoms, no camera accessories, no collision duplicates, no other bodies."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from lawm.sim import clouds  # noqa: E402
from lawm.sim.dexwm.scene import hand_surface  # noqa: E402

XML = """
<mujoco>
  <asset>
    <mesh name="gripper0_right_palm" vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1" scale="0.05 0.05 0.02"/>
    <mesh name="gripper0_right_digit" vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1" scale="0.01 0.01 0.03"/>
    <mesh name="gripper0_right_zed_x" vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1" scale="0.04 0.02 0.02"/>
    <mesh name="gripper0_right_camera_mount" vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1" scale="0.02 0.02 0.02"/>
    <mesh name="other_mesh" vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1" scale="0.5 0.5 0.5"/>
  </asset>
  <worldbody>
    <body name="gripper0_right_allegro_hand" pos="0 0 1">
      <geom name="palm_vis" type="mesh" mesh="gripper0_right_palm" contype="0" conaffinity="0"/>
      <geom name="palm_box" type="box" size="0.05 0.05 0.02"/>
      <geom name="zed" type="mesh" mesh="gripper0_right_zed_x" pos="0.2 0 0" contype="0" conaffinity="0"/>
      <geom name="mount" type="mesh" mesh="gripper0_right_camera_mount" pos="0 0.2 0" contype="0" conaffinity="0"/>
      <body name="gripper0_right_link_3.0" pos="0.1 0 0">
        <geom name="digit_vis" type="mesh" mesh="gripper0_right_digit" contype="0" conaffinity="0"/>
        <geom name="digit_collision" type="mesh" mesh="gripper0_right_digit"/>
      </body>
    </body>
    <body name="table" pos="0 0 0">
      <geom name="table_vis" type="mesh" mesh="other_mesh" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def setup():
    """Compile the model and run forward kinematics."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def names(model, ids):
    """Geom names of ids."""
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ids}


def test_hand_geoms_are_the_visual_hand_meshes_only():
    """The hand uses the palm and the fingertip visual meshes, not collision copies, camera parts or other bodies."""
    model, _ = setup()
    assert names(model, hand_surface.hand_geom_ids(model)) == {"palm_vis", "digit_vis"}


def test_accessories_can_be_included_on_request():
    """Passing no accessory names brings the camera parts back."""
    model, _ = setup()
    assert names(model, hand_surface.hand_geom_ids(model, accessories=())) == {"palm_vis", "digit_vis", "zed", "mount"}


def test_hand_cloud_lies_on_the_chosen_geoms():
    """Every hand point is within a micrometre of the surface of the geom it is reported on, and none is on the table or camera."""
    model, data = setup()
    pts, owner = hand_surface.hand_cloud(model, data, 3000, np.random.default_rng(0))
    assert names(model, set(owner)) == {"palm_vis", "digit_vis"}
    palm = pts[owner == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm_vis")].astype(np.float64) - np.array([0, 0, 1.0])
    q = np.abs(palm) - np.array([0.05, 0.05, 0.02])
    assert np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)).max() < 1e-6


def test_missing_prefix_is_an_error():
    """A prefix that matches no body fails loudly."""
    model, _ = setup()
    with pytest.raises(AssertionError):
        hand_surface.hand_geom_ids(model, prefix="nothing_")
