"""Tests of the executor: the root pose from the end-effector site, the action layout, the translation limit, the rotation smoothing and its memory across replans."""
import sim_settings as cfg
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

from lawm.sim.dexwm.control import executor as E  # noqa: E402

EYE = np.eye(4)
Q = np.linspace(0.0, 1.5, 16)


def command(executor, move=(0.0, 0.0, 0.0), turn=(0.0, 0.0, 0.0), ref_pos=(0.1, 0.2, 0.3), ref_rot=np.eye(3)):
    """One command whose fitted root is the observed root (identity) moved by `move` and turned by `turn` (rotation vector)."""
    state = (Rotation.from_rotvec(turn).as_matrix(), np.array(move, dtype=float), Q)
    return executor.command(state, EYE, np.array(ref_pos, dtype=float), ref_rot)


def test_action_layout_and_type():
    """The action has 47 float32 numbers; position, rotation vector and the 16 finger angles sit where the controller reads them and the rest is zero."""
    action, _ = command(E.Executor(cfg.exec_settings(), np.zeros(3)), move=(0.02, 0, 0))
    assert action.shape == (47,) and action.dtype == np.float32
    assert np.allclose(action[0:3], [0.12, 0.2, 0.3]) and np.array_equal(action[15:31], Q.astype(np.float32))
    assert not action[6:15].any() and not action[31:].any()


def test_small_move_is_not_clipped():
    """A move under the limit is sent as it is and reported in mm."""
    action, mm = command(E.Executor(cfg.exec_settings(), np.zeros(3)), move=(0.03, 0.04, 0.0))
    assert mm == pytest.approx(50.0) and np.allclose(action[0:3] - [0.1, 0.2, 0.3], [0.03, 0.04, 0.0], atol=1e-7)


def test_large_move_is_cut_to_the_limit_in_the_same_direction():
    """A move over the limit keeps its direction, is cut to exactly 60 mm, and the unclipped length is still reported."""
    action, mm = command(E.Executor(cfg.exec_settings(), np.zeros(3)), move=(0.06, 0.08, 0.0))
    shift = action[0:3].astype(float) - [0.1, 0.2, 0.3]
    assert mm == pytest.approx(100.0) and np.linalg.norm(shift) == pytest.approx(0.06, abs=1e-6) and np.allclose(shift / 0.06, [0.6, 0.8, 0.0], atol=1e-5)


def test_rotation_goes_halfway_from_the_start_and_remembers_it():
    """The first target is halfway from the start rotation; the next replan starts from that smoothed target, not from the raw one."""
    executor = E.Executor(cfg.exec_settings(), np.zeros(3))
    first, _ = command(executor, turn=(0.0, 0.0, 0.8))
    assert np.allclose(first[3:6], [0.0, 0.0, 0.4], atol=1e-6)
    second, _ = command(executor, turn=(0.0, 0.0, 0.8))
    assert np.allclose(second[3:6], [0.0, 0.0, 0.6], atol=1e-6)


def test_rotation_share_follows_the_setting():
    """ori_alpha = 1 sends the raw target and 0 keeps the previous one."""
    assert np.allclose(command(E.Executor(cfg.exec_settings(ori_alpha=1.0), np.zeros(3)), turn=(0, 0, 0.8))[0][3:6], [0, 0, 0.8], atol=1e-6)
    assert np.allclose(command(E.Executor(cfg.exec_settings(ori_alpha=0.0), np.zeros(3)), turn=(0, 0, 0.8))[0][3:6], 0.0, atol=1e-6)


def test_executors_do_not_share_state():
    """Two executors keep their own last rotation (no state leaks between episodes)."""
    a, b = E.Executor(cfg.exec_settings(), np.zeros(3)), E.Executor(cfg.exec_settings(), np.zeros(3))
    command(a, turn=(0, 0, 0.8))
    assert np.allclose(command(b, turn=(0, 0, 0.8))[0][3:6], [0, 0, 0.4], atol=1e-6)
