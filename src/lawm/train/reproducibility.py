"""Seeding, deterministic mode, random-state capture and the run manifest that records everything a rerun needs."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

NUMERICS_ENV = ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS", "SPCONV_ALGO_TIME_LIMIT", "SPCONV_DEBUG_SAVE_PATH",
                "TORCH_CUDNN_V8_API_ENABLED", "NCCL_ALGO", "NCCL_PROTO", "LAWM_BUILD_LOCK", "LAWM_DATA_ROOT")


def configure_determinism(mode: str) -> dict:
    """Set deterministic algorithms by mode (strict raises, warn only warns and some kernels stay non-deterministic, off); cuDNN autotuning is always off."""
    deterministic = mode != "off"
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before the first cuBLAS call
    torch.use_deterministic_algorithms(deterministic, warn_only=mode == "warn")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    return dict(mode=mode, deterministic_algorithms=deterministic, warn_only=mode == "warn", cudnn_deterministic=deterministic, cudnn_benchmark=False,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)


def seed_training(seed: int, rank: int) -> None:
    """Every rank gets its own stream (seed + rank), so the backbone's random masks differ between ranks but repeat between runs."""
    random.seed(seed + rank)
    np.random.seed((seed + rank) % 2 ** 32)
    torch.manual_seed(seed + rank)


@contextlib.contextmanager
def evaluation_stream(seed: int, device):
    """Run an evaluation pass on a fixed random stream and leave the training stream exactly as it was."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def get_random_state(device) -> dict:
    """Capture the Python, NumPy and torch CPU random states, plus torch CUDA's on a cuda device."""
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state())
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def set_random_state(state: dict, device) -> None:
    """Restore the random states captured by get_random_state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())  # a checkpoint loaded with map_location=cuda holds these on the GPU; the setters need the CPU
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def file_sha256(path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(cmd, cwd=None) -> str:
    """Stripped stdout of a command, or an empty string if it cannot be run."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # git or nvidia-smi missing on this node
        return ""


def describe_environment(device, repo: Path) -> dict:
    """Versions, hardware, scheduler and numerics settings of this process."""
    packages = {}
    for name in ("torch", "numpy", "spconv-cu120", "spconv-cu118", "spconv-cu124", "cumm-cu120", "cumm-cu118", "cumm-cu124", "timm", "torch-scatter"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    diff = _run(["git", "diff", "HEAD"], repo)
    env = dict(python=sys.version, platform=platform.platform(), node=platform.node(), packages=packages,
               git_commit=_run(["git", "rev-parse", "HEAD"], repo), git_dirty=bool(_run(["git", "status", "--porcelain"], repo)),
               git_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
               slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in (
                   "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")},
               numerics_env={k: os.environ.get(k) for k in NUMERICS_ENV})
    if device.type == "cuda":
        env.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(device),
                   gpus_visible=torch.cuda.device_count(),
                   driver=_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0])
    return env


def write_manifest(run_dir: Path, **sections) -> None:
    """Write the given sections as run_manifest.json in run_dir."""
    (Path(run_dir) / "run_manifest.json").write_text(json.dumps(sections, indent=1, sort_keys=True, default=str) + "\n")
