"""Train the chunked latent-action model on windows of the compact point-cloud dataset; every run writes a manifest, per-step losses and metrics."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from lawm.data.config import data_root, dataset_dir, read_json
from lawm.data.dataset import WindowDataset, collate
from lawm.data.index import fingerprint, load_or_build_index
from lawm.data.sampler import LengthGroupedBatchSampler
from lawm.data.stats import compute_norm_stats, load_norm_stats, save_norm_stats
from lawm.model.backbone import GRID_SIZE, PTV3_CONFIG
from lawm.model.chunked import ChunkedLAWM
from lawm.model.losses import embodiment_info_nce_loss, movement_weighted_huber_loss
from lawm.model.metrics import NAMES, step_sums, summarise
from lawm.train.evaluation import (ablation_rows, compute_context, donor_latents, evaluate, sharded_loader, to_device)
from lawm.train.reproducibility import (configure_determinism, describe_environment, evaluation_stream, file_sha256, get_random_state,
                                        seed_training, set_random_state, write_manifest)

REPO = Path(__file__).resolve().parents[3]
UNCHECKED_ON_RESUME = {"epochs", "resume", "workers", "max_steps", "print_every", "save_every"}  # everything else must match the run being continued
CHECKPOINT_METRIC = "mover_5mm"


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser for a training run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/h2o_50pct.json"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-root", default=os.environ.get("LAWM_RUNS_ROOT"), help="checkpoints and caches (default: <LAWM_DATA_ROOT>/../runs)")
    ap.add_argument("--results-root", default=str(REPO / "results"), help="small per-run outputs meant for git")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k", type=int, default=4, help="transition groups per GPU per step; each group holds m embodiments of one window")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=5.0, help="0 = no clipping")
    ap.add_argument("--contrastive-lambda", type=float, default=0.15)
    ap.add_argument("--contrastive-tau", type=float, default=0.1)
    ap.add_argument("--z-dim", type=int, default=8)
    ap.add_argument("--n-queries", type=int, default=4)
    ap.add_argument("--ada-weight-init-std", type=float, default=0.02)
    ap.add_argument("--decoder-self-attention", choices=("adaln", "plain"), default="adaln",
                    help="the self-attention over all points after the cross-attention: adaln = conditioned on the latents (the paper's models), "
                         "plain = ordinary pre-LayerNorm self-attention without conditioning")
    ap.add_argument("--drop-path", type=float, default=PTV3_CONFIG["drop_path"])
    ap.add_argument("--shuffle-orders", action=argparse.BooleanOptionalAction, default=PTV3_CONFIG["shuffle_orders"])
    ap.add_argument("--voxel-dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="give the backbone one point per 5 mm voxel in every frame (two points in one voxel make the sparse convolution depend on "
                         "thread timing, so runs would not repeat); every point keeps its row and receives its voxel's features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--determinism", choices=("strict", "warn", "off"), default="strict",
                    help="strict: deterministic algorithms only (an operation without one raises); warn: warn instead (some kernels stay "
                         "non-deterministic); off")
    ap.add_argument("--val-batch-size", type=int, default=8)
    ap.add_argument("--ablation-windows", type=int, default=1000, help="fixed validation subset for the per-epoch correct-vs-zero check")
    ap.add_argument("--final-ablation-windows", type=int, default=4000, help="subset for the full ablation on the best checkpoint")
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--val-windows", type=int, default=0, help="0 = the whole validation split; otherwise this many evenly spaced windows (smoke tests)")
    ap.add_argument("--profile-steps", type=int, default=0, help="profile this many training steps (after 3 warm-up steps) and write profile_table.txt")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (0 = whole run); for smoke tests and repeat checks")
    ap.add_argument("--save-every", type=int, default=500, help="write last.pt every this many steps inside an epoch too (0 = at epoch ends only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="build the window index and the normalisation statistics, then stop (no GPU needed)")
    return ap


def setup_distributed():
    """Initialise NCCL when WORLD_SIZE > 1 and return (rank, local_rank, world_size), or (0, 0, 1) for a single process."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 0, 1
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))
    return rank, local_rank, world_size


def rank0_first(rank: int, world_size: int, fn):
    """Run fn on rank 0 (it may write a cache), wait for it, then on every other rank, so the cache is never written twice."""
    out = fn() if rank == 0 else None
    if world_size > 1:
        dist.barrier()
    return out if rank == 0 else fn()


def rows_from_totals(epoch, split, condition, totals):
    """CSV rows [epoch, split, condition, step, name, mean error, count] for each step and set, plus a mean row per set, from summed totals."""
    s = summarise(totals)
    out = []
    for i, name in enumerate(NAMES):
        for k in range(s["per_step"].shape[0]):
            out.append([epoch, split, condition, k + 1, name, s["per_step"][k, i], int(s["count"][k, i])])
        out.append([epoch, split, condition, "mean", name, s["mean"][i], int(s["count"][:, i].sum())])
    return out


def append_rows(path: Path, header, rows) -> None:
    """Append rows to the CSV at path, writing the header first if the file is new."""
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def atomic_save(obj, path: Path) -> None:
    """torch.save obj to a temporary file and rename it over path, so a crash never leaves a partial file."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def truncate_log(path: Path, keep) -> None:
    """Drop the rows of a CSV log that a resumed run writes again; keep(fields) decides per row, the header stays."""
    if path.exists():
        lines = path.read_bytes().decode().splitlines(keepends=True)
        path.write_bytes("".join(lines[:1] + [line for line in lines[1:] if keep(line.split(","))]).encode())


def start_profiler(steps: int, device, results_dir: Path, run_dir: Path):
    """torch.profiler over `steps` training steps after 3 warm-up steps: a table of the costliest operations and a summary of GPU busy time."""
    from torch.profiler import ProfilerActivity, profile, schedule
    cuda = device.type == "cuda"

    def done(p):
        """Profiler callback: write the operation table, the GPU-busy summary and the Chrome trace."""
        key = "self_cuda_time_total" if cuda else "self_cpu_time_total"
        (results_dir / "profile_table.txt").write_text(p.key_averages().table(sort_by=key, row_limit=40, max_name_column_width=70))
        summary = {}
        try:  # wall time of the profiled steps and the time during which at least one GPU kernel was running (union of kernel intervals)
            steps_ = [e for e in p.events() if e.name.startswith("ProfilerStep")]
            kernels = sorted((e.time_range.start, e.time_range.end) for e in p.events()
                             if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith("ProfilerStep"))
            start, end = min(e.time_range.start for e in steps_), max(e.time_range.end for e in steps_)
            busy, last = 0.0, start
            for a, b in kernels:
                a, b = max(a, last), min(b, end)
                if b > a:
                    busy, last = busy + b - a, b
            summary = dict(profiled_steps=steps, wall_ms=(end - start) / 1000, gpu_busy_ms=busy / 1000, gpu_busy_fraction=busy / (end - start))
        except Exception as err:  # the profiler's event API differs between torch versions
            summary = dict(error=repr(err))
        (results_dir / "profile_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        p.export_chrome_trace(str(run_dir / "profile_trace.json"))

    prof = profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else []),
                   schedule=schedule(wait=2, warmup=1, active=steps, repeat=1), on_trace_ready=done)
    prof.start()
    return prof


def train_step_loss(model, ddp_model, batch, args):
    """One batch -> (total loss, reconstruction loss, contrastive loss, detached per-step metric sums)."""
    coords, offset, segment = batch["coords"], batch["offset"], batch["segment"]
    outs, zs = ddp_model(coords, offset, return_latents=True)
    losses, sums = [], []
    for k in range(1, model.n_steps + 1):
        delta_gt = (coords[k] - coords[0]).unsqueeze(1)
        losses.append(movement_weighted_huber_loss(outs[k - 1]["delta_norm"], delta_gt, offset,
                                                   model.out_mean_windowed[k - 1:k], model.out_std_windowed[k - 1:k]))
        sums.append(step_sums(outs[k - 1]["delta"].detach().float(), delta_gt, segment))
    recon = torch.stack(losses).mean()
    contrastive = embodiment_info_nce_loss(zs[0].float(), batch["transition_idx"], tau=args.contrastive_tau) if args.contrastive_lambda > 0 else recon * 0
    return recon + args.contrastive_lambda * contrastive, recon, contrastive, torch.stack(sums)


def main(argv=None, backbone_factory=None, device=None):
    """Run training from command-line arguments argv (or only prepare data with --prepare-only); backbone_factory and device override the defaults."""
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = read_json(cfg_path)
    if cfg["embodiments"]["train_on"] != "all" or cfg["embodiments"]["held_out"]:
        raise NotImplementedError("only training on all embodiments with none held out is supported")
    split = read_json(cfg_path.parent.parent / cfg["split_file"])
    window = cfg["window"]
    gap, n_steps, points, voxel = window["gap"], window["n_steps"], window.get("points_per_entity", 0), window.get("voxel_size", 0.0)
    if voxel and abs(voxel - GRID_SIZE) > 1e-12:
        raise ValueError(f"voxel_size {voxel} must equal the backbone's grid size {GRID_SIZE}, or the clouds still contain duplicate voxels")

    rank, local_rank, world_size = setup_distributed()
    if device is None:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    determinism = configure_determinism(args.determinism)
    seed_training(args.seed, rank)

    runs_root = Path(args.runs_root) if args.runs_root else data_root().parent / "runs"
    run_dir, results_dir, cache_dir = runs_root / args.run_name, Path(args.results_root) / args.run_name, runs_root / "_cache"
    resuming = args.resume and (run_dir / "last.pt").exists()
    if rank == 0 and not args.prepare_only:
        if not resuming and any(p.exists() for p in (run_dir / "last.pt", results_dir / "metrics.csv", results_dir / "train_steps.csv")):
            raise FileExistsError(f"{args.run_name} already has results" + (" but no checkpoint to resume from" if args.resume else "; pass --resume to continue it")
                                  + "; choose a new --run-name")
        run_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    ds_dir = dataset_dir(cfg)
    train_idx = rank0_first(rank, world_size, lambda: load_or_build_index(cache_dir, "train", ds_dir, split["train"], gap, n_steps, points))
    val_idx = rank0_first(rank, world_size, lambda: load_or_build_index(cache_dir, "val", ds_dir, split["val"], gap, n_steps, points))
    norm = cfg["normalisation"]
    fp = fingerprint(ds_dir, split["train"])
    stats_path = cache_dir / f"norm_stats_train_{fp[:12]}_g{gap}_n{n_steps}_p{points}_v{round(voxel * 1000)}_w{norm['windows']}_s{norm['seed']}.npz"
    stats_args = dict(fingerprint=fp, split="train", n_steps=n_steps, gap=gap, windows=norm["windows"], seed=norm["seed"], points_per_entity=points, voxel_size=voxel)

    def make_stats():
        """Compute and save the training normalisation statistics unless cached, then load them."""
        if not stats_path.exists():
            stats_ds = WindowDataset(ds_dir, train_idx, gap, n_steps, m=1, train=False, points_per_entity=points, voxel_size=voxel)
            stats = compute_norm_stats(stats_ds, windows=norm["windows"], seed=norm["seed"], workers=max(args.workers, 1))
            save_norm_stats(stats_path, stats, fp, "train")
        return load_norm_stats(stats_path, **stats_args)
    stats = rank0_first(rank, world_size, make_stats)

    train_ds = WindowDataset(ds_dir, train_idx, gap, n_steps, m=args.m, seed=args.seed, train=True, points_per_entity=points, voxel_size=voxel)
    val_ds = WindowDataset(ds_dir, val_idx, gap, n_steps, m=1, train=False, points_per_entity=points, voxel_size=voxel)
    if args.prepare_only:
        print(f"prepared: {len(train_ds)} train and {len(val_ds)} val windows, points per entity {points}, statistics {stats_path}", flush=True)
        return
    sampler = LengthGroupedBatchSampler(train_ds.length, batch_size=args.k, seed=args.seed, rank=rank, world_size=world_size,
                                        group_keys=(train_ds.seq.tolist(), train_ds.f.tolist()), min_start_frame_gap=n_steps * gap)
    loader_seed = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0, generator=loader_seed)
    val_rows = range(len(val_ds)) if not args.val_windows else np.linspace(0, len(val_ds) - 1, min(args.val_windows, len(val_ds))).round().astype(int).tolist()
    val_loader = sharded_loader(val_ds, val_rows, rank, world_size, args.val_batch_size, args.workers, device)
    epoch_rows = ablation_rows(val_ds, args.ablation_windows, args.val_batch_size)
    epoch_loader = sharded_loader(val_ds, epoch_rows, rank, world_size, args.val_batch_size, args.workers, device)
    donor_row = epoch_rows[len(epoch_rows) // 2]

    ptv3_cfg = dict(PTV3_CONFIG, drop_path=args.drop_path, shuffle_orders=args.shuffle_orders)
    model = ChunkedLAWM(stats, n_steps=n_steps, cfg=ptv3_cfg, z_dim=args.z_dim, n_queries=args.n_queries,
                        ada_weight_init_std=args.ada_weight_init_std, backbone=backbone_factory() if backbone_factory else None,
                        seed=args.seed, decoder_self_attention=args.decoder_self_attention, voxel_dedup=args.voxel_dedup).to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank]) if world_size > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]

    start_epoch, skip, resumed_sums, best, step = 1, 0, None, float("inf"), 0
    config = {k: v for k, v in vars(args).items() if k not in UNCHECKED_ON_RESUME}
    if resuming:
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if ck["world_size"] != world_size or ck["config"] != config:
            raise ValueError(f"cannot resume {args.run_name}: world size or settings differ from the run being continued "
                             f"(stored world size {ck['world_size']}, now {world_size}; changed settings: "
                             f"{sorted(k for k in config if ck['config'].get(k) != config[k])})")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        set_random_state(ck["random_state"][rank], device)
        best, step = ck["best"], ck["step"]
        start_epoch, skip, resumed_sums = (ck["epoch"] + 1, 0, None) if ck["epoch_complete"] else (ck["epoch"], ck["batches_done"], ck["sums"][rank])
        if rank == 0:  # a run killed after its last checkpoint may have logged work that is about to be redone
            for name in ("train_steps.csv", "train_timing.csv"):
                truncate_log(results_dir / name, lambda f: int(f[1]) <= step)
            for name in ("metrics.csv", "ablation.csv"):
                truncate_log(results_dir / name, lambda f: f[0].isdigit() and int(f[0]) < start_epoch)
    elif rank == 0:
        write_manifest(results_dir, command=[sys.executable, *sys.argv] if argv is None else list(argv), args=vars(args), config_file=str(cfg_path), config=cfg,
                       config_sha256=file_sha256(cfg_path), split_sha256=file_sha256(cfg_path.parent.parent / cfg["split_file"]),
                       stats_file=str(stats_path), stats_sha256=file_sha256(stats_path),
                       data=dict(dataset_dir=str(ds_dir), train_fingerprint=fp, points_per_entity=points,
                                 train_windows=len(train_ds), val_windows=len(val_rows), ablation_windows=len(epoch_rows)),
                       run=dict(world_size=world_size, windows_per_gpu_step=args.k * args.m, windows_per_step=args.k * args.m * world_size,
                                steps_per_epoch=len(sampler), parameters=sum(p.numel() for p in model.parameters()),
                                ptv3_config=ptv3_cfg, device=str(device)),
                       determinism=determinism, environment=describe_environment(device, REPO))

    eval_seed = args.seed + 10_000
    step_log = None
    if rank == 0:
        fresh = not (results_dir / "train_steps.csv").exists()
        step_log = open(results_dir / "train_steps.csv", "a", newline="")
        if fresh:
            step_log.write("epoch,step,recon_loss,contrastive_loss,total_loss,grad_norm\n")

    def save_checkpoints(epoch, done, sums=None, improved=False, val_value=None):
        """sums = each rank's training sums so far: a state inside `epoch` after `done` steps; None: the epoch is complete."""
        states = [None] * world_size
        state = dict(random=get_random_state(device), sums=None if sums is None else sums.cpu())
        if world_size > 1:
            dist.all_gather_object(states, state)
        else:
            states = [state]
        if rank == 0:
            step_log.flush()
            timing_log.flush()
            if improved:  # before last.pt, which records the new best: a kill in between then only repeats this epoch's improvement
                atomic_save(dict(model=model.state_dict(), epoch=epoch, val=val_value, metric=CHECKPOINT_METRIC, config=config), run_dir / "best.pt")
            atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, batches_done=done, epoch_complete=sums is None, step=step,
                             best=best, config=config, world_size=world_size, random_state=[s["random"] for s in states],
                             sums=[s["sums"] for s in states]), run_dir / "last.pt")

    timing_log = None
    if rank == 0:
        fresh = not (results_dir / "train_timing.csv").exists()
        timing_log = open(results_dir / "train_timing.csv", "a", newline="")
        if fresh:
            timing_log.write("epoch,step,data_wait_s,step_s\n")
    prof = None
    if args.profile_steps and rank == 0:
        prof = start_profiler(args.profile_steps, device, results_dir, run_dir)

    stop = paused = False
    steps_per_epoch = len(sampler)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        sampler.epoch, sampler.skip = epoch - 1, skip
        done, skip = sampler.skip, 0
        t0 = time.time()
        t_ready = time.perf_counter()
        sums = torch.zeros(n_steps, len(NAMES), 2, dtype=torch.float64, device=device) if resumed_sums is None else resumed_sums.to(device)
        resumed_sums = None
        recent = []
        for batch in loader:
            t_batch = time.perf_counter()
            batch = to_device(batch, device)
            with compute_context(device, args.amp):
                total, recon, contrastive, batch_sums = train_step_loss(model, ddp_model, batch, args)
            opt.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip if args.grad_clip > 0 else float("inf"))
            opt.step()
            step += 1
            done += 1
            sums += batch_sums
            row = torch.stack([recon.detach().float(), contrastive.detach().float(), total.detach().float()])
            if world_size > 1:
                dist.all_reduce(row)
                row /= world_size
            row = row.tolist() + [float(grad_norm)]
            if not np.isfinite(row).all():
                raise FloatingPointError(f"non-finite loss or gradient at step {step}: {row}")
            if rank == 0:
                t_done = time.perf_counter()
                timing_log.write(f"{epoch},{step},{t_batch - t_ready:.4f},{t_done - t_batch:.4f}\n")  # waiting for data, then compute (ends at the loss read-back)
                step_log.write(f"{epoch},{step},{row[0]!r},{row[1]!r},{row[2]!r},{row[3]!r}\n")
                recent.append(row)
                if step % args.print_every == 0:
                    print(f"[epoch {epoch} step {step}] recon={np.mean([r[0] for r in recent]):.4f} contrastive={np.mean([r[1] for r in recent]):.4f} "
                          f"grad_norm={np.mean([r[3] for r in recent]):.3f}", flush=True)
                    recent = []
                    step_log.flush()
                    timing_log.flush()
            if prof is not None:
                prof.step()  # may write the profile (tens of seconds), so the data-wait clock starts after it
            t_ready = time.perf_counter()
            if args.save_every and step % args.save_every == 0 and done < steps_per_epoch:
                save_checkpoints(epoch, done, sums)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop and done < steps_per_epoch:  # paused inside the epoch: --resume continues from this exact step
            save_checkpoints(epoch, done, sums)
            model.eval()
            with evaluation_stream(eval_seed, device):
                val = evaluate(model, val_loader, device, args.amp, ("correct",), distributed=world_size > 1)["correct"]
            if rank == 0:
                print(f"[paused at epoch {epoch} step {step}] val {CHECKPOINT_METRIC} (mean over steps) = "
                      f"{float(summarise(val)['mean'][NAMES.index(CHECKPOINT_METRIC)]):.3f} mm (not logged: the epoch is not finished)", flush=True)
            paused = True
            break
        if world_size > 1:
            dist.all_reduce(sums)
        model.eval()
        with evaluation_stream(eval_seed, device):
            val = evaluate(model, val_loader, device, args.amp, ("correct",), distributed=world_size > 1)["correct"]
        with evaluation_stream(eval_seed, device):
            fixed = donor_latents(model, val_ds, donor_row, device, args.amp)
            abl = evaluate(model, epoch_loader, device, args.amp, ("correct", "zero"), fixed, distributed=world_size > 1)
        model.train()
        val_value = float(summarise(val)["mean"][NAMES.index(CHECKPOINT_METRIC)])
        improved = bool(np.isfinite(val_value) and val_value < best)
        best = val_value if improved else best
        if rank == 0:
            header = ["epoch", "split", "condition", "step", "metric", "mm", "n_points"]
            append_rows(results_dir / "metrics.csv", header, rows_from_totals(epoch, "train", "correct", sums.cpu().numpy())
                        + rows_from_totals(epoch, "val", "correct", val))
            append_rows(results_dir / "ablation.csv", header, [r for c, t in abl.items() for r in rows_from_totals(epoch, "val_subset", c, t)])
            step_log.flush()
            print(f"[epoch {epoch}] val {CHECKPOINT_METRIC} (mean over steps) = {val_value:.3f} mm"
                  f"{'  -> best.pt' if improved else ''}  ({time.time() - t0:.0f} s)", flush=True)
        save_checkpoints(epoch, done, None, improved, val_value)
        if stop:
            break

    if world_size > 1:
        dist.barrier()
    if not paused and (run_dir / "best.pt").exists():  # no best checkpoint exists if the validation metric was never finite
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
        model.eval()
        final_rows = ablation_rows(val_ds, args.final_ablation_windows, args.val_batch_size)
        with evaluation_stream(eval_seed, device):
            fixed = donor_latents(model, val_ds, final_rows[len(final_rows) // 2], device, args.amp)
            final = evaluate(model, sharded_loader(val_ds, final_rows, rank, world_size, args.val_batch_size, args.workers, device),
                             device, args.amp, ("correct", "zero", "shuffled_across", "fixed_donor"), fixed, distributed=world_size > 1)
        if rank == 0:
            append_rows(results_dir / "ablation.csv", ["epoch", "split", "condition", "step", "metric", "mm", "n_points"],
                        [r for c, t in final.items() for r in rows_from_totals("best", "val_subset_final", c, t)])
    if prof is not None:
        prof.stop()
    if step_log:
        step_log.close()
        timing_log.close()
    if device.type == "cuda" and rank == 0:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GiB on rank 0", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
