"""Tests for lawm.train on a tiny synthetic dataset with a stand-in backbone that, like drop-path, consumes random numbers."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fake_backbone import FakeBackbone
from synthetic_data import make_dataset

from lawm.data.dataset import WindowDataset, collate
from lawm.data.index import build_index
from lawm.model.chunked import ChunkedLAWM
from lawm.train.evaluation import ablation_rows, apply_condition, evaluate, sharded_loader
from lawm.train.reproducibility import evaluation_stream
from lawm.train.trainer import main, truncate_log

GAP, STEPS = 2, 2
TRAIN = {f"subject1/h1/{i}": (30, i, None) for i in range(3)}
VAL = {f"subject3/k2/{i}": (30, 10 + i, None) for i in range(3)}
CPU = torch.device("cpu")


class NoisyBackbone(FakeBackbone):
    """Adds fresh random noise to every feature level on each call, so the random stream matters (as drop-path does)."""
    def forward(self, data):
        """Run the fake backbone, then add random noise to the features of every level"""
        out = super().forward(data)
        level = out
        while level is not None:
            level["feat"] = level["feat"] + 0.01 * torch.randn_like(level["feat"])
            level = level.get("pooling_parent")
        return out


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Module-scoped fixture that builds a tiny project tree with synthetic data, a split file and a config"""
    root = tmp_path_factory.mktemp("proj")
    data = root / "lawm" / "data"
    make_dataset(data / "d", {**TRAIN, **VAL})
    (root / "configs").mkdir()
    (root / "splits").mkdir()
    (root / "splits/s.json").write_text(json.dumps(dict(train=list(TRAIN), val=list(VAL))))
    (root / "configs/c.json").write_text(json.dumps(dict(
        dataset="synthetic", data_dir="d", variant="t", split_file="splits/s.json",
        embodiments=dict(names=[], train_on="all", held_out=[]), window=dict(gap=GAP, n_steps=STEPS, points_per_entity=0),
        normalisation=dict(windows=20, seed=0))))
    return root


def run(project, name, *extra, backbone=NoisyBackbone):
    """Run the trainer main on the tiny project with CPU-friendly arguments and return the results and runs directories"""
    root = project
    argv = ["--config", str(root / "configs/c.json"), "--run-name", name, "--runs-root", str(root / "lawm/runs"),
            "--results-root", str(root / "results"), "--epochs", "2", "--k", "2", "--m", "2", "--workers", "0", "--val-batch-size", "2",
            "--ablation-windows", "6", "--final-ablation-windows", "6", "--print-every", "1000", *extra]
    import os
    os.environ["LAWM_DATA_ROOT"] = str(root / "lawm/data")
    main(argv, backbone_factory=backbone, device=CPU)
    return root / "results" / name, root / "lawm/runs" / name


def read_csv(path):
    """Read a CSV file into a list of row dicts"""
    with open(path) as f:
        return list(csv.DictReader(f))


def test_a_short_run_writes_manifest_logs_metrics_and_checkpoints(project):
    """A two-epoch run writes the manifest, step log, metrics, ablation tables and last and best checkpoints with the expected contents"""
    results, runs = run(project, "short")
    manifest = json.loads((results / "run_manifest.json").read_text())
    assert manifest["determinism"]["deterministic_algorithms"] and manifest["run"]["windows_per_step"] == 4
    assert manifest["data"]["train_windows"] > 0 and manifest["stats_sha256"] and "git_commit" in manifest["environment"]
    steps = read_csv(results / "train_steps.csv")
    assert len(steps) == 2 * manifest["run"]["steps_per_epoch"] and all(np.isfinite(float(r["total_loss"])) for r in steps)
    metrics = read_csv(results / "metrics.csv")
    names = {(r["epoch"], r["split"], r["metric"]) for r in metrics}
    assert {("1", "train", "mover_5mm"), ("2", "val", "overall")} <= names
    assert {r["metric"] for r in metrics} == {"mover_5mm", "mover_2mm", "dynamic", "static", "overall"}
    ablation = read_csv(results / "ablation.csv")
    assert {r["condition"] for r in ablation if r["epoch"] == "1"} == {"correct", "zero"}
    final = [r for r in ablation if r["epoch"] == "best"]
    assert {r["condition"] for r in final} == {"correct", "zero", "shuffled_across", "fixed_donor"}
    assert all(int(r["n_points"]) > 0 for r in final if r["metric"] == "overall")
    assert (runs / "last.pt").exists() and (runs / "best.pt").exists()
    assert set(torch.load(runs / "best.pt", weights_only=False)) == {"model", "epoch", "val", "metric", "config"}


def test_the_same_run_twice_gives_identical_losses_and_the_comparison_script_agrees(project):
    """Two identical runs give identical step logs, compare_runs.py reports them identical, and a different seed differs"""
    a, _ = run(project, "twin_a")
    b, _ = run(project, "twin_b")
    assert (a / "train_steps.csv").read_text() == (b / "train_steps.csv").read_text()
    out = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "scripts/compare_runs.py"), str(a), str(b)], capture_output=True, text=True)
    assert out.returncode == 0 and "identical" in out.stdout
    c, _ = run(project, "twin_c", "--seed", "1")
    assert (a / "train_steps.csv").read_text() != (c / "train_steps.csv").read_text()


@pytest.mark.parametrize("workers", ["0", "2"])  # a DataLoader with workers iterates its sampler twice per epoch
def test_resuming_after_one_epoch_reproduces_an_uninterrupted_run_exactly(project, workers):
    """Resuming after one epoch reproduces the logs, metrics and final weights of an uninterrupted run"""
    full, full_runs = run(project, f"full{workers}", "--workers", workers)
    part, part_runs = run(project, f"part{workers}", "--epochs", "1", "--workers", workers)
    run(project, f"part{workers}", "--resume", "--workers", workers)
    assert (full / "train_steps.csv").read_text() == (part / "train_steps.csv").read_text()
    for x, y in zip(read_csv(full / "metrics.csv"), read_csv(part / "metrics.csv")):
        assert x == y
    a, b = torch.load(full_runs / "last.pt", weights_only=False)["model"], torch.load(part_runs / "last.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def same_state(x, y):
    """Return whether two state dicts have the same keys and equal values"""
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) if torch.is_tensor(x[k]) else x[k] == y[k] for k in x)


@pytest.mark.parametrize("workers", ["0", "2"])
def test_resuming_from_a_pause_inside_an_epoch_reproduces_an_uninterrupted_run_exactly(project, workers):
    """Resuming from a mid-epoch pause reproduces the logs, final model and optimiser state, and best checkpoint of an uninterrupted run"""
    full, full_runs = run(project, f"mfull{workers}", "--workers", workers)
    stop = json.loads((full / "run_manifest.json").read_text())["run"]["steps_per_epoch"] * 3 // 2  # halfway through epoch 2
    part, part_runs = run(project, f"mpart{workers}", "--workers", workers, "--max-steps", str(stop), "--save-every", "4")
    paused = torch.load(part_runs / "last.pt", weights_only=False)
    assert not paused["epoch_complete"] and paused["epoch"] == 2 and paused["step"] == stop
    run(project, f"mpart{workers}", "--resume", "--workers", workers)
    for name in ("train_steps.csv", "metrics.csv", "ablation.csv"):
        assert (full / name).read_text() == (part / name).read_text(), name
    a, b = (torch.load(r / "last.pt", weights_only=False) for r in (full_runs, part_runs))
    assert same_state(a["model"], b["model"]) and same_state(a["opt"]["state"][0], b["opt"]["state"][0]) and a["step"] == b["step"]
    best_a, best_b = (torch.load(r / "best.pt", weights_only=False) for r in (full_runs, part_runs))
    assert same_state(best_a["model"], best_b["model"]) and best_a["epoch"] == best_b["epoch"] and best_a["val"] == best_b["val"]


def test_truncate_log_keeps_the_header_and_the_rows_that_pass(tmp_path):
    """truncate_log keeps the header and the leading rows accepted by the predicate while preserving the line endings"""
    path = tmp_path / "log.csv"
    path.write_bytes(b"epoch,step\r\n1,1\r\n1,2\r\n2,3\r\n")
    truncate_log(path, lambda f: int(f[1]) <= 2)
    assert path.read_bytes() == b"epoch,step\r\n1,1\r\n1,2\r\n"


def test_resume_refuses_other_settings_and_a_finished_run_is_not_overwritten(project):
    """Resuming with changed settings raises, and starting a run whose name already exists raises FileExistsError"""
    run(project, "guard", "--epochs", "1")
    with pytest.raises(ValueError, match="cannot resume"):
        run(project, "guard", "--resume", "--k", "3")
    with pytest.raises(FileExistsError):
        run(project, "guard")


def test_resume_without_a_checkpoint_refuses_to_append_to_the_logs_of_an_interrupted_start(project):
    """Resuming with logs but no checkpoint raises instead of appending to a stale log"""
    results = project / "results" / "stale"
    results.mkdir(parents=True)
    (results / "train_steps.csv").write_text("epoch,step,recon_loss,contrastive_loss,total_loss,grad_norm\n1,1,0,0,0,0\n")
    with pytest.raises(FileExistsError, match="no checkpoint to resume from"):
        run(project, "stale", "--resume")


def test_evaluation_leaves_the_training_random_stream_untouched_and_is_repeatable():
    """evaluation_stream restores the global RNG state on exit and gives the same draws each time for a given seed"""
    torch.manual_seed(5)
    before = torch.get_rng_state()
    with evaluation_stream(7, CPU):
        first = torch.rand(3)
    assert torch.equal(torch.get_rng_state(), before)
    with evaluation_stream(7, CPU):
        assert torch.equal(first, torch.rand(3))


@pytest.fixture(scope="module")
def val_ds(project):
    """Module-scoped fixture returning the validation WindowDataset in evaluation mode"""
    root = project / "lawm/data/d"
    return WindowDataset(root, build_index(root, list(VAL), GAP, STEPS), GAP, STEPS, m=1, train=False)


def test_ablation_batches_come_from_different_sequences_and_conditions_do_what_they_say(val_ds):
    """ablation_rows picks consecutive windows from different sequences and apply_condition zeroes, shuffles or replaces latents as named, raising on invalid input"""
    rows = ablation_rows(val_ds, 9, 2)
    assert len(rows) == 9
    seqs = [str(val_ds.seq[r]) for r in rows]
    assert all(seqs[i] != seqs[i + 1] for i in range(len(seqs) - 1))
    with pytest.raises(ValueError, match="exceeds"):
        ablation_rows(val_ds, 9, 4)
    zs = torch.randn(2, 3, 4, 8)
    assert torch.equal(apply_condition("zero", zs, None), torch.zeros_like(zs))
    assert torch.equal(apply_condition("shuffled_across", zs, ["a", "b", "c"])[:, 1], zs[:, 0])
    assert apply_condition("shuffled_across", zs[:, :1], ["a"]) is None
    with pytest.raises(ValueError, match="different sequences"):
        apply_condition("shuffled_across", zs, ["a", "a", "b"])
    fixed = torch.randn(2, 1, 4, 8)
    assert torch.equal(apply_condition("fixed_donor", zs, None, fixed)[:, 2], fixed[:, 0])


@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_every_validation_window_is_used_exactly_once_across_ranks(val_ds, world_size):
    """sharded_loader covers every validation window exactly once across all ranks"""
    seen = []
    for rank in range(world_size):
        for batch in sharded_loader(val_ds, range(len(val_ds)), rank, world_size, 4, 0, CPU):
            seen += batch["transition_idx"].tolist()
    assert sorted(seen) == list(range(len(val_ds)))


def test_validation_totals_do_not_depend_on_batch_size_or_sharding(val_ds):
    """Summed evaluation totals match whether computed in one batch or sharded over ranks with small batches"""
    torch.manual_seed(0)
    stats = dict(per_step_mean=np.zeros((STEPS, 3)), per_step_var=np.full((STEPS, 3), 1e-4))
    model = ChunkedLAWM(stats, n_steps=STEPS, backbone=FakeBackbone(), seed=0).eval()  # the stand-in without noise
    rows = list(range(0, len(val_ds), 7))
    whole = evaluate(model, sharded_loader(val_ds, rows, 0, 1, len(rows), 0, CPU), CPU, False, ("correct", "zero"))
    pieces = [evaluate(model, sharded_loader(val_ds, rows, r, 3, 2, 0, CPU), CPU, False, ("correct", "zero")) for r in range(3)]
    for c in whole:
        assert np.allclose(whole[c], sum(p[c] for p in pieces), rtol=1e-6)


def test_a_run_with_plain_self_attention_the_profiler_and_a_validation_subset(project):
    """A run with plain decoder self-attention, profiling and a validation subset has the right parameters, profile files, timing log and capped validation point counts"""
    results, runs = run(project, "plain", "--decoder-self-attention", "plain", "--val-windows", "8", "--profile-steps", "1")
    keys = torch.load(runs / "best.pt", weights_only=False)["model"]
    assert not any("self_adaln" in k for k in keys) and any("decoder.self_attn" in k for k in keys)
    assert (results / "profile_table.txt").read_text().strip()
    summary = json.loads((results / "profile_summary.json").read_text())
    assert summary["profiled_steps"] == 1 and summary["wall_ms"] > 0 and summary["gpu_busy_fraction"] == 0  # no GPU on this machine
    timing = read_csv(results / "train_timing.csv")
    assert len(timing) == len(read_csv(results / "train_steps.csv")) and all(float(r["step_s"]) > 0 for r in timing)
    val_n = [int(r["n_points"]) for r in read_csv(results / "metrics.csv") if r["split"] == "val" and r["metric"] == "overall" and r["step"] == "1"]
    assert val_n and all(0 < n < 8 * 80 for n in val_n)  # 8 windows of fewer than 80 points each
