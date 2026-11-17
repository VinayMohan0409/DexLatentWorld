"""Runs the gripper tasks from a config in one process: one environment per task, one trial per seed, one result file per trial (finished trials are skipped on a rerun)."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lawm.sim.gripper import envs, milk, push, reach, stack
from lawm.sim.gripper.pointworld.planner import PointWorldPlanner, PointWorldSettings
from lawm.sim.gripper.planner import LawmPlanner, PlannerSettings

TASKS = {"stack": (stack, "Stack"), "milk": (milk, "PickPlaceMilk"), "push_puck": (push, "PushEnv"), "push_cube": (push, "PushEnv"), "reach": (reach, "Lift")}


@dataclass(frozen=True)
class RunConfig:
    """What a run depends on: the tasks and seeds, the search settings, the number of candidates per decoder call, and the settings of each task."""

    tasks: tuple
    seeds: tuple
    planner: PlannerSettings
    copies: int
    stack: stack.StackSettings
    milk: milk.MilkSettings
    push_puck: push.PushSettings
    push_cube: push.PushSettings
    reach: reach.ReachSettings
    pointworld: PointWorldSettings


def load_config(path):
    """The RunConfig of a JSON file; an unknown or misspelt key is an error."""
    raw = json.loads(Path(path).read_text())
    return RunConfig(tuple(raw["tasks"]), tuple(raw["seeds"]), PlannerSettings(**raw["planner"]), raw["copies"], stack.StackSettings(**raw["stack"]), milk.MilkSettings(**raw["milk"]),
                     push.PushSettings(**raw["push_puck"]), push.PushSettings(**raw["push_cube"]), reach.ReachSettings(**{**raw["reach"], "angles_deg": tuple(raw["reach"]["angles_deg"])}), PointWorldSettings(**raw["pointworld"]))


def horizon(task, settings):
    """The environment horizon: Milk's is max(1000, 500 + replans * steps per replan * 2), the other tasks keep robosuite's default."""
    return max(1000, 500 + settings.replans * settings.steps_per_replan * 2) if task == "milk" else None


def result_path(out_dir, task, seed):
    """The result file of a trial."""
    return Path(out_dir) / f"{task}_seed{seed}.json"


def run_task(task, config, planner, out_dir, seeds=None):
    """Run the seeds of a task that have no result file yet; the environment is built only if some seed is left."""
    listed = range(len(config.reach.angles_deg)) if task == "reach" else config.seeds
    todo = [s for s in (seeds if seeds is not None else listed) if not result_path(out_dir, task, s).exists()]
    if not todo:
        return
    module, env_name = TASKS[task]
    settings = getattr(config, task)
    cameras = hasattr(planner, "bind")  # the PointWorld planner renders its observations
    build = lambda: envs.build_env(env_name, horizon=horizon(task, settings), cameras=cameras, **(dict(push_object=settings.object) if module is push else {}))  # noqa: E731
    env = None if module is reach else build()
    try:
        for seed in todo:
            trial_env = env or build()  # every Reach trial starts from the random state robosuite.make leaves
            if cameras:
                planner.bind(trial_env, task)
            result_path(out_dir, task, seed).write_text(json.dumps(module.run_trial(trial_env, planner, seed, settings), indent=1) + "\n")
            if env is None:
                trial_env.close()
    finally:
        if env is not None:
            env.close()


def run_all(config, planner, out_dir, tasks=None, seeds=None):
    """Run the tasks into out_dir, which keeps a copy of the config and refuses a different one; returns the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.dumps(asdict(config), sort_keys=True, indent=1)
    saved = out_dir / "config_used.json"
    if saved.exists():
        assert saved.read_text() == snapshot, f"{saved} differs from this config: use a new output folder"
    else:
        saved.write_text(snapshot)
    for task in tasks or config.tasks:
        run_task(task, config, planner, out_dir, seeds)
    return summarize(out_dir)


def summarize(out_dir):
    """Per task: trials done, trials whose grasp setup worked, real successes and trials that reached the stop distance."""
    table = {}
    for path in sorted(Path(out_dir).glob("*_seed*.json")):
        r = json.loads(path.read_text())
        row = table.setdefault(path.name.split("_seed")[0], dict(trials=0, setup_ok=0, real_success=0, converged=0))
        row["trials"] += 1
        row["setup_ok"] += int(r["setup_ok"])
        row["real_success"] += int(r.get("real_success", False))
        row["converged"] += int(r.get("converged", False))
    return table


def build_planner(config, checkpoint, manifest, prior):
    """The planner of a checkpoint on the GPU (CPU if there is none), with the latent prior that was built from this very checkpoint."""
    import torch

    from lawm.sim import model_adapter
    model = model_adapter.load_model(checkpoint, manifest, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return LawmPlanner(model_adapter.ModelAdapter(model, model_adapter.load_prior(prior, checkpoint), copies=config.copies), config.planner)


def build_pointworld(config, repo, checkpoint, log_dir):
    """The PointWorld planner of a released checkpoint, its repository left as it is."""
    import torch

    from lawm.sim.gripper.pointworld import model
    return PointWorldPlanner(*model.load(repo, checkpoint, log_dir), config.pointworld, seed_torch=torch.manual_seed)


def main():
    """python -m lawm.sim.gripper.runner --config configs/stage3/gripper.json --out DIR --planner lawm --checkpoint best.pt --manifest run_manifest.json --prior prior.npz, or --planner pointworld --pointworld-repo REPO --checkpoint model-best.pt [--tasks stack milk] [--seeds 0 1 ...]."""
    from lawm.sim.dexwm.runner import record_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--planner", choices=("lawm", "pointworld"), default="lawm")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", help="run_manifest.json of the training run (lawm)")
    parser.add_argument("--prior", help="latent prior file built from this checkpoint (lawm)")
    parser.add_argument("--pointworld-repo", help="PointWorld's repository (pointworld)")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.planner == "lawm":
        planner = build_planner(config, args.checkpoint, args.manifest, args.prior)
    else:
        planner = build_pointworld(config, args.pointworld_repo, args.checkpoint, args.out)
    record_model(args.out, args.checkpoint, args.prior)
    print(json.dumps(run_all(config, planner, args.out, args.tasks, args.seeds), indent=1))


if __name__ == "__main__":
    main()
