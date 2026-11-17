"""Point Transformer V3 encoder (pinned copy of Utonia's, random weights) and helpers to run it on batches of ragged clouds."""
from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path

import torch

PTV3_CONFIG = dict(
    in_channels=3, order=("z", "z-trans"), stride=(2, 2, 2), enc_depths=(2, 2, 3, 2), enc_channels=(48, 96, 144, 192),
    enc_num_head=(2, 4, 6, 8), enc_patch_size=(1024, 1024, 1024, 1024), enc_mode=True, enable_flash=False,
    drop_path=0.3,        # stochastic depth: the chance of skipping a block rises linearly from 0 to this over the 9 blocks (training only)
    shuffle_orders=True)  # a random choice among the serialisation orders at each forward pass (training only); both are the backbone's own defaults
GRID_SIZE = 0.005  # metres: the finest voxel size of the first pooling stage


def import_spconv_once() -> None:
    """Import spconv under a file lock (LAWM_BUILD_LOCK or a temp-folder file) so processes starting together do not race on its first-import compilation."""
    lock = Path(os.environ.get("LAWM_BUILD_LOCK", Path(tempfile.gettempdir()) / "lawm_spconv_build.lock"))
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            import spconv.pytorch  # noqa: F401
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_backbone(cfg=PTV3_CONFIG):
    """Encoder-only Point Transformer V3 with random weights."""
    import_spconv_once()
    from lawm.model.vendor.utonia.model import PointTransformerV3
    return PointTransformerV3(**cfg)


def make_data_dict(coord, offset, grid_size=GRID_SIZE, grid_coord=None):
    """The backbone's input dict: the centred coordinates are also the only point feature, and grid_coord is computed by the backbone unless given."""
    data = dict(coord=coord, feat=coord, offset=offset, grid_size=grid_size)
    if grid_coord is not None:
        data["grid_coord"] = grid_coord
    return data


def voxel_representatives(coord, offset, grid_size=GRID_SIZE):
    """Keep the first point of each occupied voxel; returns (kept rows, each input row's position in kept, cloud ends within kept, kept rows' voxels)."""
    n, device = coord.shape[0], coord.device
    batch = torch.repeat_interleave(torch.arange(len(offset), device=device), torch.diff(offset, prepend=offset.new_zeros(1)).long())
    grid = torch.div(coord - coord.min(0)[0], grid_size, rounding_mode="trunc").int()
    _, voxel = torch.unique(torch.cat([batch[:, None].int(), grid], dim=1), dim=0, return_inverse=True)
    first = torch.full((int(voxel.max()) + 1,), n, dtype=torch.long, device=device).scatter_reduce_(0, voxel, torch.arange(n, device=device), "amin")
    kept = first.sort().values
    rank = torch.empty(n, dtype=torch.long, device=device)
    rank[kept] = torch.arange(len(kept), device=device)
    return kept, rank[first[voxel]], torch.bincount(batch[kept], minlength=len(offset)).cumsum(0).int(), grid[kept]


def collect_pooling_chain(point):
    """From the backbone's coarsest output, walk the stored parents down to the finest level: [coarsest, ..., finest]."""
    chain = [point]
    while "pooling_parent" in chain[-1].keys():
        chain.append(chain[-1]["pooling_parent"])
    return chain
