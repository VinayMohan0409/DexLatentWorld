"""Tests of geometry to action: kinematics against MuJoCo, recovering known hand states from points (clean, noisy, beyond the limits), the action conversion."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from lawm.sim import clouds  # noqa: E402
from lawm.sim.dexwm.control import hand_fit as A  # noqa: E402

XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="gripper0_right_allegro_hand" pos="0.3 -0.2 1.0" quat="0.9238795 0 0 -0.3826834">
      <geom name="palm" type="box" size="0.05 0.04 0.01"/>
      <body name="gripper0_right_link_0.0" pos="0 0.03 0.0" quat="0.999048 -0.0436194 0 0">
        <joint name="gripper0_right_joint_0.0" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="box" size="0.01 0.01 0.02" pos="0 0 0.02"/>
        <body name="gripper0_right_link_1.0" pos="0 0 0.04">
          <joint name="gripper0_right_joint_1.0" axis="0 1 0" range="-0.3 1.7"/>
          <geom type="box" size="0.01 0.012 0.03" pos="0 0 0.03" quat="0.9659258 0.2588190 0 0"/>
        </body>
      </body>
      <body name="gripper0_right_link_2.0" pos="0 -0.03 0.0">
        <joint name="gripper0_right_joint_2.0" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="box" size="0.01 0.01 0.02" pos="0 0 0.02"/>
        <body name="gripper0_right_link_3.0" pos="0 0 0.04">
          <joint name="gripper0_right_joint_3.0" pos="0.005 0 0" axis="-1 0 0" range="0.1 1.5"/>
          <geom type="box" size="0.012 0.01 0.03" pos="0 0 0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def setup(n=600, seed=0):
    """Compile the toy hand (palm, two two-joint fingers with offset geoms and a joint off the body origin), a sampler and its kinematics."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sampler = clouds.SurfaceSampler(model, list(range(model.ngeom)), n, np.random.default_rng(seed))
    return model, data, sampler, A.HandKinematics(model, sampler)


def pose(model, data, q):
    """Set the four joint angles and return the world points MuJoCo places the sampler's rows at."""
    data.qpos[:4] = q
    mujoco.mj_forward(model, data)


def root_pose(model, data):
    """World rotation and position of the root body."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_allegro_hand")
    return data.xmat[b].reshape(3, 3).copy(), data.xpos[b].copy()


def test_kinematics_places_points_exactly_where_mujoco_does():
    """Our forward kinematics of the sampled points equals MuJoCo's own for random joint angles."""
    model, data, sampler, kin = setup()
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(kin.limits[:, 0], kin.limits[:, 1])
        pose(model, data, q)
        r, p = root_pose(model, data)
        assert np.abs(kin.points(r, p, q) - sampler(data)).max() < 1e-5


def test_fit_recovers_a_known_state_from_clean_points():
    """Starting 0.2 rad away, the fit returns the true joint angles and the true wrist pose."""
    model, data, sampler, kin = setup()
    q_true = np.array([0.3, 0.9, -0.2, 0.7])
    pose(model, data, q_true)
    r, p = root_pose(model, data)
    rot, pos, q, info = kin.fit(sampler(data).astype(float), q_true + 0.2)
    assert np.abs(q - q_true).max() < 1e-4 and np.abs(rot - r).max() < 1e-5 and np.abs(pos - p).max() < 1e-5 and info["mean_residual_mm"] < 0.01


def test_fit_with_noise_stays_close():
    """With 1 mm of Gaussian noise on every point the joint angles are within 0.05 rad and the wrist within 1 mm."""
    model, data, sampler, kin = setup(n=900)
    q_true = np.array([-0.2, 1.2, 0.3, 0.5])
    pose(model, data, q_true)
    r, p = root_pose(model, data)
    noisy = sampler(data).astype(float) + np.random.default_rng(3).normal(scale=0.001, size=(900, 3))
    rot, pos, q, _ = kin.fit(noisy, q_true)
    assert np.abs(q - q_true).max() < 0.05 and np.linalg.norm(pos - p) < 0.001


def test_fit_never_leaves_the_joint_limits():
    """Points that need a joint beyond its range give an angle at the limit, not past it."""
    model, data, sampler, kin = setup()
    pose(model, data, np.array([0.3, 1.0, 0.0, 0.8]))
    points = sampler(data).astype(float)
    kin.limits[1] = [-0.3, 0.5]  # the same points now need joint 1 at 1.0 rad
    _, _, q, _ = kin.fit(points, np.zeros(4))
    assert q[1] <= 0.5 + 1e-9 and (q >= kin.limits[:, 0] - 1e-9).all() and (q <= kin.limits[:, 1] + 1e-9).all()


def test_action_applies_the_roots_motion_to_the_controllers_pose():
    """No motion keeps the end-effector pose and the angles; a root translation and a 90 degree turn about z move the end-effector accordingly."""
    eye, zero = np.eye(3), np.zeros(3)
    eef_pos, eef_rot = np.array([0.5, 0.1, 1.0]), eye
    pos, rotvec, q = A.hand_state_to_action(eye, zero, [0.1, 0.2], eye, zero, eef_pos, eef_rot)
    assert np.allclose(pos, eef_pos) and np.allclose(rotvec, 0) and np.allclose(q, [0.1, 0.2])
    pos, _, _ = A.hand_state_to_action(eye, np.array([0.1, 0, 0]), [0], eye, zero, eef_pos, eef_rot)
    assert np.allclose(pos, eef_pos + [0.1, 0, 0])
    turn = A.axis_rotation(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    pos, rotvec, _ = A.hand_state_to_action(turn, zero, [0], eye, zero, eef_pos, eef_rot)
    assert np.allclose(pos, turn @ eef_pos) and np.allclose(rotvec, [0, 0, np.pi / 2])
