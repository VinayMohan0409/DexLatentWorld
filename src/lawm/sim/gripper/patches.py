"""Fixes that make the local robosuite fork run a Panda with a Robotiq or wiping gripper; apply_all() once before robosuite.make, and pass fixed_impedance_controller_config() as controller_configs."""
from __future__ import annotations

import numpy as np

ACTUATED_QPOS_IDX = np.array([0, 3])  # Robotiq140: the two driven joints among the six joints it reports


def patch_observable_validity_lenient():
    """An observable whose validity check fails is disabled with a message instead of raising (the fork's mobile-base observables look up bodies a fixed-base env does not have)."""
    from robosuite.utils.observables import Observable
    if getattr(Observable._check_sensor_validity, "_lenient_patched", False):
        return
    original = Observable._check_sensor_validity

    def lenient(self):
        """Run the original check; on any error disable the observable."""
        try:
            original(self)
        except Exception as error:
            print(f"[patch] disabling broken observable {self.name!r}: {error}", flush=True)
            self._data_shape, self._is_number, self._enabled, self._active = (1,), True, False, False

    lenient._lenient_patched = True
    Observable._check_sensor_validity = lenient


def patch_simple_grip_mimic_joints():
    """SimpleGripController compares its goal with the actuated joints only when the gripper reports more joints than it has actuators (Robotiq140: 2 actuators, 6 joints)."""
    from robosuite.controllers.parts.controller import Controller
    from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
    if getattr(SimpleGripController.run_controller, "_mimic_patched", False):
        return
    original = SimpleGripController.run_controller

    def run_controller(self):
        """The original when joints and actuators match; otherwise the same control law on the actuated joints."""
        if len(self.joint_pos) == self.control_dim:
            return original(self)
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))
        self.update()
        desired = self.interpolator.get_interpolated_goal() if self.interpolator is not None else self.goal_qpos
        actuated = self.joint_pos[ACTUATED_QPOS_IDX]
        if self.use_action_scaling:
            ctrl_range = np.stack([self.actuator_min, self.actuator_max], axis=-1)
            bias, weight = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0]), 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            self.ctrl = bias + weight * np.clip(desired - actuated, -1, 1)
        else:
            self.ctrl = desired
        Controller.run_controller(self)
        return self.ctrl

    def set_goal(self, action, set_qpos=None):
        """Goal joint positions: given directly, or the current actuated positions plus the (scaled) action."""
        self.update()
        if set_qpos is not None:
            self.goal_qpos = np.array(set_qpos)
        else:
            delta = np.array(action)
            assert len(delta) == self.control_dim
            if self.use_action_scaling:
                delta = self.scale_action(delta)
            base = self.joint_pos[ACTUATED_QPOS_IDX] if len(self.joint_pos) != self.control_dim else self.joint_pos
            self.goal_qpos = base + delta
        if self.position_limits is not None:
            self.goal_qpos = np.clip(self.goal_qpos, self.position_limits[0], self.position_limits[1])
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    run_controller._mimic_patched = True
    set_goal._mimic_patched = True
    SimpleGripController.run_controller = run_controller
    SimpleGripController.set_goal = set_goal


def fixed_impedance_controller_config(robot="Panda"):
    """The fork's composite controller config with the arm's OSC in fixed-impedance mode: 7 numbers per action (6 pose, 1 gripper) and no gain terms."""
    from robosuite.controllers import load_composite_controller_config
    config = load_composite_controller_config(robot=robot)
    config["body_parts"]["right"]["impedance_mode"] = "fixed"
    return config


def patch_stack_reward_noop():
    """Stack.reward returns 0 (its own reward reads a Robotiq140 geom key that does not exist; the planner never uses the reward)."""
    from robosuite.environments.manipulation.stack import Stack
    if getattr(Stack.reward, "_noop_patched", False):
        return

    def reward(self, action=None):
        """No reward."""
        return 0.0

    reward._noop_patched = True
    Stack.reward = reward


def register_vanilla_pickplace():
    """Import the vendored upstream PickPlace family, which re-registers it over the fork's broken versions."""
    from lawm.sim.gripper.vendor import vanilla_pick_place  # noqa: F401


def apply_all():
    """The three patches every gripper environment needs."""
    patch_observable_validity_lenient()
    patch_simple_grip_mimic_joints()
    patch_stack_reward_noop()
