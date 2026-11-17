"""Tests of the vendored DexWM success checks: file integrity, and the reach / grasp / place thresholds on a fake environment."""
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

VENDOR = Path(__file__).resolve().parents[1] / "src" / "lawm" / "sim" / "dexwm" / "success"
SHA256 = "e90f529e1b79e22346aea72175a469675429029cfb55edaf2c831374139a857d"


def test_vendored_file_is_unmodified():
    """The copied sim_wrappers.py has the SHA-256 recorded in NOTICE."""
    assert hashlib.sha256((VENDOR / "sim_wrappers.py").read_bytes()).hexdigest() == SHA256
    assert SHA256 in (VENDOR / "NOTICE").read_text()


class FakeWrapperBase:
    """The part of robosuite's Wrapper the success checks use: attribute lookups fall through to the wrapped environment."""

    def __init__(self, env):
        """Keep the environment."""
        self.env = env

    def __getattr__(self, name):
        """Forward unknown attributes to the environment."""
        return getattr(self.env, name)

    def step(self, action):
        """Return the environment's four-tuple."""
        return self.env.step(action)


@pytest.fixture
def wrapper_module(monkeypatch):
    """Import the vendored module with a minimal stand-in for robosuite.wrappers.Wrapper when robosuite is not installed."""
    try:
        import robosuite.wrappers  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "robosuite", types.ModuleType("robosuite"))
        wrappers = types.ModuleType("robosuite.wrappers")
        wrappers.Wrapper = FakeWrapperBase
        monkeypatch.setitem(sys.modules, "robosuite.wrappers", wrappers)
    monkeypatch.syspath_prepend(str(VENDOR))
    sys.modules.pop("sim_wrappers", None)
    import sim_wrappers
    return sim_wrappers


class FakeEnv:
    """An environment with named bodies at given positions and a contact flag."""

    def __init__(self, positions, contact=False):
        """positions maps body name -> xyz; 'obj' is the manipulated object."""
        names = list(positions)
        self.names = names
        self.sim = types.SimpleNamespace(data=types.SimpleNamespace(body_xpos=np.array([positions[n] for n in names], dtype=float)),
                                         model=types.SimpleNamespace(body_name2id=lambda n: names.index(n)))
        self.objects = {"obj": types.SimpleNamespace(name="obj")}
        self.obj_body_id = {"obj": names.index("obj")}
        gripper = types.SimpleNamespace(root_body="robot0_right_hand")
        self.robots = [types.SimpleNamespace(gripper={"right": gripper})]
        self.contact = contact

    def check_contact(self, gripper, obj):
        """The configured contact flag."""
        return self.contact

    def step(self, action):
        """Observation, reward, done, info as robosuite returns."""
        return {"o": 1}, 0.0, False, {}


TIPS = ["robot0_right_hand", "gripper0_right_link_3.0", "gripper0_right_link_7.0", "gripper0_right_link_11.0", "gripper0_right_link_15.0"]


def hand_at(offset, obj=(0.0, 0.0, 0.0)):
    """Five hand-point positions at an offset from the origin, plus the object."""
    pos = {n: np.array(offset, dtype=float) + 0.01 * i for i, n in enumerate(TIPS)}
    pos["obj"] = np.array(obj, dtype=float)
    return pos


def test_reach_uses_the_mean_distance_of_the_five_points(wrapper_module):
    """Reach succeeds when the mean tip distance to the target is below 0.15 m and fails above it."""
    target = np.array([hand_at([0, 0, 0])[n] for n in TIPS])
    near = wrapper_module.EvaluationWrapper(FakeEnv(hand_at([0.14, 0, 0])), "reach", target_hand_pose=target)
    far = wrapper_module.EvaluationWrapper(FakeEnv(hand_at([0.16, 0, 0])), "reach", target_hand_pose=target)
    assert near.check_success() and not far.check_success()


def test_grasp_needs_both_distance_and_contact(wrapper_module):
    """Grasp needs the hand within 0.20 m of the object and a contact."""
    ok = wrapper_module.EvaluationWrapper(FakeEnv(hand_at([0.1, 0, 0]), contact=True), "grasp")
    no_contact = wrapper_module.EvaluationWrapper(FakeEnv(hand_at([0.1, 0, 0]), contact=False), "grasp")
    far = wrapper_module.EvaluationWrapper(FakeEnv(hand_at([0.3, 0, 0]), contact=True), "grasp")
    assert ok.check_success() and not no_contact.check_success() and not far.check_success()


def test_place_uses_the_object_distance_to_its_target(wrapper_module):
    """Place succeeds when the object is within 0.10 m of the target position."""
    target = np.array([0.0, 0.0, 0.0])
    assert wrapper_module.EvaluationWrapper(FakeEnv(hand_at([1, 1, 1], obj=[0.09, 0, 0])), "place", target_obj_pos=target).check_success()
    assert not wrapper_module.EvaluationWrapper(FakeEnv(hand_at([1, 1, 1], obj=[0.11, 0, 0])), "place", target_obj_pos=target).check_success()


def test_step_returns_the_observation_and_the_success(wrapper_module):
    """step returns (observation, success) as DexWM's evaluation loop expects."""
    w = wrapper_module.EvaluationWrapper(FakeEnv(hand_at([1, 1, 1], obj=[0.0, 0, 0])), "place", target_obj_pos=np.zeros(3))
    obs, success = w.step(np.zeros(3))
    assert obs == {"o": 1} and success
