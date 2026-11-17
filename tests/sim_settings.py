"""The settings of the tests: the values of the real config files, with single fields replaced; imports stay inside the functions so tests without mujoco can use it."""
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "stage3"
DEXWM = json.loads((CONFIGS / "dexwm.json").read_text())
GRIPPER = json.loads((CONFIGS / "gripper.json").read_text())


def cem(task="reach", **changes):
    """The search settings of a DexWM task."""
    from lawm.sim.dexwm.planning.cem_replan import CemSettings
    return CemSettings(**{**DEXWM["cem"][task], **changes})


def loop(task="reach", **changes):
    """The stops of a DexWM task."""
    from lawm.sim.dexwm.closed_loop import LoopSettings
    return LoopSettings(**{**DEXWM["loop"][task], **changes})


def exec_settings(**changes):
    """The DexWM executor settings."""
    from lawm.sim.dexwm.control.executor import ExecSettings
    return ExecSettings(**{**DEXWM["exec"], **changes})


def env(**changes):
    """The DexWM kitchen settings."""
    from lawm.sim.dexwm.env.kitchen import EnvSettings
    return EnvSettings(**{**DEXWM["env"], **changes})


def weights(task):
    """The cost weights of a DexWM task (place: one per stage)."""
    from lawm.sim.dexwm.planning.plan_costs import CostWeights
    raw = DEXWM["weights"][task]
    return tuple(CostWeights(**w) for w in raw) if isinstance(raw, list) else CostWeights(**raw)


def hover(**changes):
    """The hover gate settings."""
    from lawm.sim.dexwm.planning.phase_gates import HoverSettings
    return HoverSettings(**{**DEXWM["gates"]["hover"], **changes})


def secure(**changes):
    """The secure gate settings."""
    from lawm.sim.dexwm.planning.phase_gates import SecureSettings
    return SecureSettings(**{**DEXWM["gates"]["secure"], **changes})


def gates():
    """Both gates' settings."""
    from lawm.sim.dexwm.planning.phase_gates import GateSettings
    return GateSettings(hover(), secure())


def gripper_planner(**changes):
    """The gripper planner settings."""
    from lawm.sim.gripper.planner import PlannerSettings
    return PlannerSettings(**{**GRIPPER["planner"], **changes})
