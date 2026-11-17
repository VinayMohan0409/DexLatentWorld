"""The robosuite environments of the gripper tasks, built the way every gripper task builds them."""
from __future__ import annotations

import robosuite

from lawm.sim.gripper import patches, push_env  # noqa: F401  (importing push_env registers PushEnv)

CONTROL_FREQ = 20
GRIPPERS = {"Stack": "Robotiq140Gripper", "PickPlaceMilk": "Robotiq140Gripper", "PushEnv": "WipingGripper", "Lift": "Robotiq140Gripper"}


def build_env(name, horizon=None, cameras=False, **kwargs):
    """A Panda environment (cameras: with the offscreen renderer PointWorld observes through) with the patches applied and fixed-impedance control; PushEnv takes push_object ("puck", "cube", ...)."""
    patches.apply_all()
    patches.register_vanilla_pickplace()
    extra = dict(horizon=horizon) if horizon is not None else {}
    if name == "PushEnv":
        extra.update(initialization_noise="default", ignore_done=True)
    return robosuite.make(env_name=name, robots="Panda", gripper_types=GRIPPERS[name], controller_configs=patches.fixed_impedance_controller_config(),
                          has_renderer=False, has_offscreen_renderer=cameras, use_camera_obs=False, control_freq=CONTROL_FREQ, **extra, **kwargs)
