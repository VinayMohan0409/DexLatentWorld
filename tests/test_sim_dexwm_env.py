"""Tests of the demo environment on a toy kitchen: reset and goal, observation, state switches that forget the last episode, the three task distances, the success flag, and the controller settings."""
import os

import sim_settings as cfg
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import sim_toy as T  # noqa: E402
from lawm.sim.dexwm.env import kitchen as D  # noqa: E402

DEXWM_ROOT = os.environ.get("DEXWM_ROOT", "")
ALLEGRO_XML = os.path.join(DEXWM_ROOT, "robot_sim_dexwm/robosuite_murp/robosuite/models/assets/grippers/wonik_allegro_2.xml")


def make(flag=False):
    """A demo environment on the toy kitchen with the stand-in wrapper."""
    env = T.Env()
    T.Wrapper.flag = flag
    return D.DexWMEnv(env, T.episode(env), cfg.env(), make_wrapper=T.Wrapper), env


def obj_x(env):
    """The object's x position in the toy."""
    return float(env.mj.xpos[env.obj_body_id["obj"]][0])


def test_reset_gives_start_observation_and_goal_from_the_target_frame():
    """Reach starts at frame 0 and has its goal at frame 1: the object rows of the goal are 5 cm further along x, everything else is the same."""
    sim, env = make()
    obs = sim.reset("reach")
    assert obs.coords.shape == (len(sim.segment), 3) and sim.goal.shape == obs.coords.shape and sim.goal.dtype == np.float32
    e = cfg.env()
    assert [int((sim.segment == s).sum()) for s in range(4)] == [e.n_hand, e.n_object, e.n_table, e.n_distractor]  # the toy has one distractor
    moved = np.linalg.norm(sim.goal - obs.coords, axis=1) > 1e-6
    assert moved[sim.segment == 1].all() and not moved[sim.segment != 1].any()
    assert np.allclose(sim.goal[sim.segment == 1] - obs.coords[sim.segment == 1], [0.05, 0, 0], atol=1e-5)
    assert obj_x(env) == pytest.approx(0.0)


def test_observation_fields():
    """The root pose is the hand root body's, the 16 angles come from qpos in joint order, the reference is the controller's."""
    sim, env = make()
    sim.reset("reach")
    env.mj.qpos[sim.kinematics.qpos_adr] = np.linspace(0.1, 0.9, 16)
    mujoco.mj_forward(env.model, env.mj)
    obs = sim.observe()
    hand = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_allegro_hand")
    assert np.allclose(obs.root_pose[:3, 3], env.mj.xpos[hand]) and np.allclose(obs.root_pose[:3, :3], env.mj.xmat[hand].reshape(3, 3)) and obs.root_pose[3].tolist() == [0, 0, 0, 1]
    assert np.allclose(obs.q, np.linspace(0.1, 0.9, 16)) and np.allclose(obs.ref_pos, env.mj.xpos[hand]) and np.array_equal(obs.ref_rot, np.eye(3))
    assert obs.coords.shape == (len(sim.segment), 3)


def test_targets_given_to_the_success_wrapper():
    """Reach gets the five hand bodies at the target frame, place the object position there, grasp neither."""
    sim, env = make()
    sim.reset("reach")
    tips = np.array([env.mj.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, b)] for b in D.TIP_BODIES])
    assert np.allclose(sim.wrapped.target_hand_pose, tips) and sim.wrapped.target_obj_pos is None
    sim.reset("place")
    assert np.allclose(sim.wrapped.target_obj_pos, [0.25, 0, 1.0]) and sim.wrapped.target_hand_pose is None
    sim.reset("grasp")
    assert sim.wrapped.target_obj_pos is None and sim.wrapped.target_hand_pose is None and sim.wrapped.task_name == "grasp"


def test_reset_forgets_the_last_episode():
    """After the object was moved, the controls and solver warm start were left over and a goal was set, a reset gives the same observation as the first one and clears them."""
    sim, env = make()
    first = sim.reset("grasp")
    env.mj.qpos[:3] += [0.3, 0.2, 0.1]
    env.mj.ctrl[:] = 5.0
    env.mj.qacc_warmstart[:] = 7.0
    second = sim.reset("grasp")
    assert np.array_equal(first.coords, second.coords) and np.array_equal(first.root_pose, second.root_pose)
    assert not env.mj.ctrl.any() and not env.mj.qacc_warmstart.any() and env.site_updates > 0
    assert env.parts["right"].goal_resets > 0


def test_goal_is_independent_of_how_many_times_it_is_reset():
    """Switching start, target, start again, twice, gives the same goal cloud each time."""
    sim, _ = make()
    sim.reset("place")
    a = sim.goal.copy()
    sim.reset("place")
    assert np.array_equal(a, sim.goal)


def test_distances_of_the_three_tasks():
    """Place measures object to target, grasp wrist to object, reach the mean of the five hand bodies to their targets."""
    sim, env = make()
    sim.reset("place")
    assert sim.distance() == pytest.approx(0.05)
    env.mj.qpos[0] += 0.1
    mujoco.mj_forward(env.model, env.mj)
    assert sim.distance() == pytest.approx(0.15)
    sim.reset("grasp")
    hand = env.mj.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_allegro_hand")]
    assert sim.distance() == pytest.approx(np.linalg.norm(hand - env.mj.xpos[env.obj_body_id["obj"]]))
    sim.reset("reach")
    assert sim.distance() == pytest.approx(0.0, abs=1e-9)
    env.mj.qpos[sim.kinematics.qpos_adr[1]] += 0.3
    mujoco.mj_forward(env.model, env.mj)
    assert sim.distance() > 0.0


def test_step_returns_the_success_flag_of_the_wrapper():
    """The action reaches the environment and the flag comes back as a bool."""
    sim, env = make(flag=True)
    sim.reset("reach")
    action = np.arange(47, dtype=np.float32)
    assert sim.step(action) is True and np.array_equal(env.actions[-1], action)
    T.Wrapper.flag = False
    assert sim.step(action) is False


def test_controller_settings_are_applied_once_at_construction():
    """The arm gains are set (z doubled, rotations x10, no interpolation) and the finger controller runs the PD law."""
    sim, env = make()
    right = env.parts["right"]
    assert np.allclose(right.kp, [300, 300, 600, 3000, 3000, 3000]) and right.interpolator_pos is None and right.interpolator_ori is None
    fingers = env.parts["right_gripper"]
    fingers.goal_qpos[:] = 1.0
    assert np.allclose(fingers.run_controller(), 0.35) and fingers.base_calls == 1


@pytest.mark.skipif(not os.path.exists(ALLEGRO_XML), reason="set DEXWM_ROOT to the DexWM clone")
def test_original_root_from_the_grip_site_is_the_hand_root_body():
    """In the simulator's own Allegro model the controller's grip site sits at the origin of the eef body, 0.145 m along z of the hand root, so root = site pose minus that offset = the root body itself."""
    model = mujoco.MjModel.from_xml_path(ALLEGRO_XML)
    data = mujoco.MjData(model)
    data.qpos[:16] = np.linspace(0.0, 0.8, 16)
    mujoco.mj_forward(model, data)
    root = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "allegro_hand")]
    rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "allegro_hand")].reshape(3, 3)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    assert np.allclose(data.site_xpos[site] - data.site_xmat[site].reshape(3, 3)[:, 2] * 0.145, root, atol=1e-12)
    assert np.allclose(data.site_xmat[site].reshape(3, 3), rot, atol=1e-12)


def test_build_env_cameras_switch(monkeypatch):
    """Cameras off: no renderer. Cameras on: the offscreen renderer is built but no camera is drawn at any step (no camera observations); the absolute world-frame arm targets are set either way."""
    import sys
    import types

    made = []
    monkeypatch.setitem(sys.modules, "robosuite", types.SimpleNamespace(make=lambda **kw: made.append(kw) or kw))
    args = lambda: dict(env_name="K", env_kwargs=dict(controller_configs=dict(body_parts=dict(right={}, left={}))))  # noqa: E731
    D.build_env(args(), 10)
    D.build_env(args(), 10, cameras=True)
    off, on = made
    assert (off["has_offscreen_renderer"], on["has_offscreen_renderer"]) == (False, True) and off["use_camera_obs"] is False and on["use_camera_obs"] is False
    for kw in made:
        assert kw["control_freq"] == 10 and kw["controller_configs"]["body_parts"]["right"] == dict(input_type="absolute", input_ref_frame="world")


def test_render_draws_on_demand_and_returns_upright_images(monkeypatch):
    """render asks the simulator for one camera at the given size, flips the image upright, converts depth to metres, and refuses when the cameras are off."""
    import sys
    import types

    calls = []
    image = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    sim, env = make()
    env.sim.render = lambda **kw: calls.append(kw) or ((image, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])) if kw["depth"] else image)
    with pytest.raises(AssertionError):
        sim.render("cam", 2, 3)
    sim.cameras = True
    assert np.array_equal(sim.render("cam", 2, 3), image[::-1]) and calls[-1] == dict(width=3, height=2, camera_name="cam", depth=False)
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", types.SimpleNamespace(get_real_depth_map=lambda sim_, d: d * 10))
    monkeypatch.setitem(sys.modules, "robosuite.utils", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "robosuite", types.SimpleNamespace(utils=sys.modules["robosuite.utils"]))
    rgb, depth = sim.render("cam", 2, 3, depth=True)
    assert np.array_equal(rgb, image[::-1]) and np.allclose(depth, np.array([[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]) * 10) and calls[-1]["depth"] is True
