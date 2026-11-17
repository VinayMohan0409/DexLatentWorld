"""Per-module initialisation streams: with an init seed, each named module draws its starting weights from its own generator state."""
from __future__ import annotations

import contextlib
import hashlib

import torch

_SEED = None


@contextlib.contextmanager
def init_seed(seed):
    """Within this block `own_stream` is active; seed None (the default) leaves initialisation on the global random stream."""
    global _SEED
    previous, _SEED = _SEED, seed
    try:
        yield
    finally:
        _SEED = previous


@contextlib.contextmanager
def own_stream(name: str):
    """Build the modules inside this block from a stream fixed by (init seed, name), so changing the shape of one module cannot shift the weights of another."""
    if _SEED is None:
        yield
        return
    key = int.from_bytes(hashlib.blake2b(f"{_SEED}|{name}".encode(), digest_size=8).digest(), "little") % 2 ** 63
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(key)
        yield
