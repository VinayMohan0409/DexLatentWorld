"""Tests of the gripper environments: the patches are idempotent, and Stack, PickPlaceMilk and PushEnv build, reset and step with the expected action sizes and object names."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

robosuite = pytest.importorskip("robosuite")
from lawm.sim.gripper import patches, push_env  # noqa: E402

VENDOR = Path(__file__).resolve().parents[1] / "src" / "lawm" / "sim" / "gripper" / "vendor"
SHA256 = "673fc8a4e8fd647b0373f9e14c7c35f64e3b247ad5147d3e1d8a3aa87d4fd038"


def make(name, **kwargs):
    """One environment with the patches applied, as every gripper task builds it."""
    patches.apply_all()
    patches.register_vanilla_pickplace()
    return robosuite.make(env_name=name, robots="Panda", controller_configs=patches.fixed_impedance_controller_config(), has_renderer=False,
                          has_offscreen_renderer=False, use_camera_obs=False, control_freq=20, **kwargs)


def test_vendored_pickplace_file_has_the_recorded_hash():
    """The copied upstream PickPlace file has the SHA-256 written in NOTICE."""
    assert hashlib.sha256((VENDOR / "vanilla_pick_place.py").read_bytes()).hexdigest() == SHA256
    assert SHA256 in (VENDOR / "NOTICE").read_text()


def test_patches_can_be_applied_twice():
    """A second apply_all changes nothing (the patched functions keep their marker)."""
    patches.apply_all()
    from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
    run_controller = SimpleGripController.run_controller
    patches.apply_all()
    assert SimpleGripController.run_controller is run_controller and run_controller._mimic_patched


def test_controller_config_is_fixed_impedance():
    """The arm controller has the fixed-impedance mode, so an action has no gain terms."""
    assert patches.fixed_impedance_controller_config()["body_parts"]["right"]["impedance_mode"] == "fixed"


@pytest.mark.parametrize("name,kwargs,action_dim", [("Stack", dict(gripper_types="Robotiq140Gripper"), 7), ("PickPlaceMilk", dict(gripper_types="Robotiq140Gripper"), 7),
                                                    ("PushEnv", dict(gripper_types="WipingGripper", push_object="puck", ignore_done=True), 6)])
def test_environment_builds_resets_and_steps(name, kwargs, action_dim):
    """Each environment builds, resets, takes a step and has the expected action size."""
    env = make(name, **kwargs)
    env.reset()
    env.step(np.zeros(env.action_dim))
    assert env.action_dim == action_dim
    env.close()


@pytest.mark.parametrize("kind", list(push_env.PUSH_OBJECTS))
def test_push_objects_are_always_called_cube(kind):
    """Every pushable shape is named cube, so its body and geom names stay cube_main and cube_g0."""
    env = make("PushEnv", gripper_types="WipingGripper", push_object=kind, ignore_done=True)
    env.reset()
    model = env.sim.model
    assert model.body_name2id("cube_main") >= 0 and model.geom_name2id("cube_g0") >= 0
    assert np.allclose(model.geom_size[model.geom_name2id("cube_g0")][:len(push_env.PUSH_OBJECTS[kind]["size"])], push_env.PUSH_OBJECTS[kind]["size"])
    env.close()


def test_unknown_push_object_is_refused():
    """An unknown object kind stops the construction."""
    with pytest.raises(AssertionError):
        push_env.PushEnv(robots="Panda", gripper_types="WipingGripper", push_object="sphere")
