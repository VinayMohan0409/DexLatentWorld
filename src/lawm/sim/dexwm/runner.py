"""Runs the DexWM benchmark from a config: one kitchen per demo, one episode per task with its own seed, one result file per episode (finished episodes are skipped on a rerun)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lawm.data.config import resolve_data_path
from lawm.sim.dexwm.control.executor import ExecSettings
from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planning.phase_gates import GateSettings, HoverSettings, SecureSettings
from lawm.sim.dexwm.planning.plan_costs import CostWeights
from lawm.sim.dexwm.planning.task_planner import TaskPlanner
from lawm.sim.dexwm.planning.cem_replan import CemSettings
from lawm.sim.dexwm.closed_loop import LoopSettings, run_episode


@dataclass(frozen=True)
class RunConfig:
    """Everything a run depends on: the 50-episode manifest, the tasks, the episode and scene seeds, the settings of the kitchen, the executor and the phase gates, and per task the loop (stops), the search and the cost weights (place: one per secure stage)."""

    manifest: str
    tasks: tuple
    seed: int
    scene_seed: int
    env: kitchen.EnvSettings
    exec: ExecSettings
    loop: dict
    cem: dict
    weights: dict
    gates: GateSettings


def load_config(path):
    """The RunConfig of a JSON file (the manifest path is relative to the file); an unknown or misspelt key is an error."""
    raw = json.loads(Path(path).read_text())
    return RunConfig(str(Path(path).parent / raw["manifest"]), tuple(raw["tasks"]), raw["seed"], raw["scene_seed"], kitchen.EnvSettings(**raw["env"]),
                     ExecSettings(**raw["exec"]), {task: LoopSettings(**c) for task, c in raw["loop"].items()}, {task: CemSettings(**c) for task, c in raw["cem"].items()},
                     {task: tuple(CostWeights(**s) for s in w) if isinstance(w, list) else CostWeights(**w) for task, w in raw["weights"].items()}, GateSettings(HoverSettings(**raw["gates"]["hover"]), SecureSettings(**raw["gates"]["secure"])))


def episode_seed(seed, demo, task):
    """The planner seed of one episode: fixed by (run seed, demo, task), different for every episode."""
    return int(np.random.SeedSequence([seed, int(demo.rsplit("_", 1)[1]), benchmark.TASKS.index(task)]).generate_state(1)[0])


def run_task(sim, task, model, config, demo):
    """One episode of a task on the demo's kitchen."""
    sim.reset(task, config.scene_seed)
    planner = TaskPlanner(task, sim.goal, sim.segment, sim.finger_mask, model, episode_seed(config.seed, demo, task), config.cem[task], config.weights[task], config.gates)
    return run_episode(sim, planner, sim.kinematics, config.exec, config.loop[task])


def result_path(out_dir, demo, task):
    """The result file of an episode."""
    return Path(out_dir) / f"{demo}_{task}.json"


def open_demo(demo, config):
    """The kitchen of a manifest demo, built from its stored episode."""
    episodes_dir = resolve_data_path(json.loads(Path(config.manifest).read_text())["episodes_dir"])
    return kitchen.open_demo(benchmark.load_episode(episodes_dir, demo), config.env)


def run_demo(demo, config, model, out_dir, tasks=None, opener=open_demo):
    """Run the tasks of a demo that have no result file yet; the kitchen is only built if some task is left."""
    todo = [t for t in (tasks or config.tasks) if not result_path(out_dir, demo, t).exists()]
    if not todo:
        return
    sim = opener(demo, config)
    try:
        for task in todo:
            result = run_task(sim, task, model, config, demo)
            summary = dict(demo=demo, task=task, seed=episode_seed(config.seed, demo, task), success=result.success, holds_at_end=result.holds_at_end,
                           stop_reason=result.stop_reason, n_replans=result.n_replans, final_error_mm=result.records[-1]["error_mm"], records=result.records)
            result_path(out_dir, demo, task).write_text(json.dumps(summary, indent=1) + "\n")
    finally:
        sim.env.close()


def run_all(config, model, out_dir, demos=None, tasks=None, opener=open_demo):
    """Run the demos of the manifest (or the listed ones) into out_dir, which keeps a copy of the config and refuses a different one."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.dumps(config, default=lambda o: o.__dict__, sort_keys=True, indent=1)
    saved = out_dir / "config_used.json"
    if saved.exists():
        assert saved.read_text() == snapshot, f"{saved} differs from this config: use a new output folder"
    else:
        saved.write_text(snapshot)
    listed = [e["demo"] for e in json.loads(Path(config.manifest).read_text())["episodes"]]
    for demo in demos or listed:
        assert demo in listed, f"{demo} is not one of the {len(listed)} manifest episodes"
        run_demo(demo, config, model, out_dir, tasks, opener)
    return summarize(out_dir)


def summarize(out_dir):
    """Per task: episodes done, successes (10 steps in a row at some point) and episodes where the check still holds at the end."""
    table = {}
    for path in sorted(Path(out_dir).glob("demo_*.json")):
        r = json.loads(path.read_text())
        row = table.setdefault(r["task"], dict(episodes=0, success=0, holds_at_end=0))
        row["episodes"] += 1
        row["success"] += int(r["success"])
        row["holds_at_end"] += int(r["holds_at_end"])
    return table


def build_model(checkpoint, manifest, prior):
    """The model adapter of a checkpoint on the GPU (CPU if there is none): the Stage 2 model, and the latent prior that was built from this very checkpoint."""
    import torch

    from lawm.sim import model_adapter
    model = model_adapter.load_model(checkpoint, manifest, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model_adapter.ModelAdapter(model, model_adapter.load_prior(prior, checkpoint))


def record_model(out_dir, checkpoint, prior):
    """Write model_used.json (checkpoint path and SHA-256, prior path) into out_dir; a folder that already holds another model's record is refused."""
    from lawm.sim import model_adapter
    record = json.dumps(dict(checkpoint=str(checkpoint), checkpoint_sha256=model_adapter.checkpoint_sha256(checkpoint), prior=str(prior)), indent=1)
    saved = Path(out_dir) / "model_used.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if saved.exists():
        assert saved.read_text() == record, f"{saved} differs from this model: use a new output folder"
    else:
        saved.write_text(record)


def main():
    """python -m lawm.sim.dexwm.runner --config configs/stage3/dexwm.json --out DIR --checkpoint best.pt --manifest run_manifest.json --prior prior.npz [--demos demo_133 ...] [--tasks reach ...]."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="run_manifest.json of the training run (the model's settings)")
    parser.add_argument("--prior", required=True, help="latent prior file built from this checkpoint")
    parser.add_argument("--demos", nargs="*")
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()
    model = build_model(args.checkpoint, args.manifest, args.prior)
    record_model(args.out, args.checkpoint, args.prior)
    print(json.dumps(run_all(load_config(args.config), model, args.out, args.demos, args.tasks), indent=1))


if __name__ == "__main__":
    main()
