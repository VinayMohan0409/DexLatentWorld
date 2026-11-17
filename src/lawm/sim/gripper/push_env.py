"""PushEnv: robosuite's Lift task with the lifted object replaced by one of four pushable shapes (always named "cube")."""
from __future__ import annotations

from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

PUSH_OBJECTS = {
    "box": dict(shape="box", size=[0.025, 0.025, 0.025], friction=None, rgba=[0.85, 0.25, 0.2, 1]),
    "cube": dict(shape="box", size=[0.035, 0.035, 0.035], friction=None, rgba=[0.9, 0.7, 0.15, 1]),
    "can": dict(shape="cylinder", size=[0.028, 0.045], friction=None, rgba=[0.2, 0.4, 0.85, 1]),
    "puck": dict(shape="cylinder", size=[0.0675, 0.018], friction=[0.05, 0.05, 0.01], rgba=[0.3, 0.75, 0.3, 1]),
}


def build_object(kind):
    """The robosuite object of a PUSH_OBJECTS entry, named "cube"."""
    spec = PUSH_OBJECTS[kind]
    material = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat", tex_attrib={"type": "cube"},
                              mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"})
    kwargs = dict(name="cube", size=spec["size"], rgba=spec["rgba"], material=material, friction=spec["friction"])
    return BoxObject(**kwargs) if spec["shape"] == "box" else CylinderObject(**kwargs)


class PushEnv(Lift):
    """Lift with the object built from PUSH_OBJECTS[push_object] ("box", "cube", "can" or "puck") and placed in x [-0.10, 0.00], y [-0.10, 0.10] on the table."""

    def __init__(self, *args, push_object="box", **kwargs):
        """Remember the object kind, then build the Lift environment."""
        assert push_object in PUSH_OBJECTS, f"push_object must be one of {list(PUSH_OBJECTS)}"
        self.push_object = push_object
        super().__init__(*args, **kwargs)

    def _load_model(self):
        """Lift._load_model with the object construction replaced."""
        ManipulationEnv._load_model(self)
        self.robots[0].robot_model.set_base_xpos(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.cube = build_object(self.push_object)
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler", mujoco_objects=self.cube, x_range=[-0.10, 0.00], y_range=[-0.10, 0.10], rotation=None,
                ensure_object_boundary_in_range=False, ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.001)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=self.cube)
