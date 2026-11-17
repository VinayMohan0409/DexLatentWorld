"""A toy DexWM kitchen for the environment tests: a real MuJoCo model (counter, object, distractors, a 16-joint hand) behind a stand-in for the parts of robosuite the wrapper touches."""
from types import SimpleNamespace

import mujoco
import numpy as np

CUBE = 'vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1"'


def finger(index):
    """The chain of 4 hinge links (joint numbers 4 * index .. 4 * index + 3) of one finger."""
    xml = ""
    for k in range(4):
        n = 4 * index + k
        xml += f'<body name="gripper0_right_link_{n}.0" pos="{0.03 * (index - 1.5) if k == 0 else 0} 0 {0.04 if k else 0.05}"><joint name="gripper0_right_joint_{n}.0" axis="{"0 0 1" if k == 0 else "0 1 0"}" range="-0.5 1.5"/><geom type="mesh" mesh="link" group="1" contype="0" conaffinity="0"/>'
    return xml + "</body>" * 4


XML = f"""
<mujoco>
  <asset>
    <mesh name="gripper0_right_palm" {CUBE} scale="0.04 0.04 0.02"/>
    <mesh name="link" {CUBE} scale="0.01 0.01 0.02"/>
    <mesh name="piece" {CUBE} scale="0.03 0.03 0.03"/>
  </asset>
  <worldbody>
    <geom name="counter" type="box" size="1.5 1.5 0.015" pos="0 0 0.905" group="1"/>
    <body name="obj_main" pos="0 0 1.0"><freejoint/><geom name="obj" type="mesh" mesh="piece" group="1" contype="0" conaffinity="0"/></body>
    <body name="distr_near_main" pos="0.5 0.2 0.95"><geom type="mesh" mesh="piece" group="1" contype="0" conaffinity="0"/></body>
    <body name="robot0_base" pos="-1 0 0.9"><geom name="arm" type="box" size="0.05 0.05 0.3" pos="0.3 0 0.3" group="1" contype="0" conaffinity="0"/></body>
    <body name="robot0_right_hand" pos="0.2 0.1 1.3"/>
    <body name="gripper0_right_allegro_hand" pos="0.2 0 1.2" quat="0.9238795 0 0 0.3826834">
      <geom name="palm" type="mesh" mesh="gripper0_right_palm" group="1" contype="0" conaffinity="0"/>
      {"".join(finger(i) for i in range(4))}
    </body>
  </worldbody>
</mujoco>
"""

WAYPOINTS = np.array([b"s0", b"s0", b"s1-w1", b"s1-w1", b"s2-w1", b"s2-w1", b"s3-w1"], dtype=object)  # reach (0 -> 1), grasp (2 -> 3), place (6 -> 5)


class Part:
    """A part controller stand-in: refreshes its reference from the hand, counts goal resets."""

    def __init__(self, hand):
        """Keep the hand pose source."""
        self.hand, self.goal_resets, self.ref_ori_mat = hand, 0, np.eye(3)
        self.ref_pos = np.zeros(3)

    def update(self, force=False):
        """The reference position follows the hand."""
        self.ref_pos = np.array(self.hand())

    def reset_goal(self):
        """Count the reset."""
        self.goal_resets += 1


class FingerBase:
    """The class a finger controller derives from: its own torque law runs here."""

    def run_controller(self):
        """Count the call of the original torque law."""
        self.base_calls = getattr(self, "base_calls", 0) + 1


class Fingers(FingerBase):
    """A finger controller stand-in with the attributes the PD law reads."""

    def __init__(self):
        """Zero goal and state, torque limits +-0.35."""
        self.goal_qpos, self.joint_pos, self.joint_vel = np.zeros(16), np.zeros(16), np.zeros(16)
        self.actuator_min, self.actuator_max, self.ctrl = np.full(16, -0.35), np.full(16, 0.35), np.zeros(16)

    def update(self):
        """Nothing to refresh."""


class Env:
    """The robosuite environment as the wrapper sees it, backed by the toy MuJoCo model; the state is the qpos."""

    def __init__(self):
        """Compile the model and build the stand-ins."""
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.mj = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.mj)
        hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_allegro_hand")
        self.parts = {"right": Part(lambda: self.mj.xpos[hand]), "right_gripper": Fingers()}
        self.sim = SimpleNamespace(model=SimpleNamespace(_model=self.model, body_name2id=lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)),
                                   data=SimpleNamespace(_data=self.mj, body_xpos=self.mj.xpos),
                                   set_state_from_flattened=self.set_state, forward=lambda: mujoco.mj_forward(self.model, self.mj))
        self.robots = [SimpleNamespace(composite_controller=SimpleNamespace(part_controllers=self.parts), gripper={"right": SimpleNamespace(root_body="gripper0_right_allegro_hand")})]
        self.objects = {"obj": SimpleNamespace(name="obj")}
        self.obj_body_id = {"obj": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_main")}
        self.actions, self.site_updates = [], 0

    def set_state(self, state):
        """Set the stored qpos."""
        self.mj.qpos[:] = state

    def check_contact(self, gripper, obj):
        """No contact in the toy."""
        return False

    def close(self):
        """Count the close."""
        self.closed = True

    def update_sites(self):
        """Count the bookkeeping call."""
        self.site_updates += 1

    def step(self, action):
        """Record the action."""
        self.actions.append(np.array(action))
        return None, 0.0, False, {}


def episode(env):
    """A 7-frame episode: the object slides 5 cm along x per frame, the fingers stay open."""
    states = np.tile(env.mj.qpos.copy(), (7, 1))
    states[:, 0] = 0.05 * np.arange(7)
    return dict(demo="toy", states=states, waypoints=WAYPOINTS)


class Wrapper:
    """An EvaluationWrapper stand-in: remembers the task and targets, answers every step with a settable flag."""

    flag = False

    def __init__(self, env, task_name, target_hand_pose=None, target_obj_pos=None):
        """Keep what the real wrapper would keep."""
        self.env, self.task_name, self.target_hand_pose, self.target_obj_pos = env, task_name, target_hand_pose, target_obj_pos

    def step(self, action):
        """Pass the action to the environment and return (observation, success flag)."""
        self.env.step(action)
        return None, self.flag
