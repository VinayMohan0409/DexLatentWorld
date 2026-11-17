"""The DexWM planner's view of a Stage 2 model: the latent prior, and a decoder that encodes an observed cloud once (one copy) and decodes candidate latents in chunks of a fixed size."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CHUNK = 16  # candidates per decoder call (32 does not fit an 11 GB GPU at 6144 points; results do not depend on it)


def load_model(checkpoint, manifest, device, backbone=None):
    """The Stage 2 model of a checkpoint in eval mode, built from the settings in the run's manifest (the checkpoint's own statistics replace the placeholder ones); backbone replaces the sparse encoder (tests only)."""
    from lawm.model.chunked import ChunkedLAWM
    run = json.loads(Path(manifest).read_text())
    args, ptv3 = run["args"], {k: tuple(v) if isinstance(v, list) else v for k, v in run["run"]["ptv3_config"].items()}
    n_steps = run["config"]["window"]["n_steps"]
    placeholder = dict(per_step_mean=np.zeros((n_steps, 3)), per_step_var=np.ones((n_steps, 3)))
    model = ChunkedLAWM(placeholder, n_steps=n_steps, cfg=ptv3, z_dim=args["z_dim"], n_queries=args["n_queries"], ada_weight_init_std=args["ada_weight_init_std"],
                        backbone=backbone, decoder_self_attention=args["decoder_self_attention"], voxel_dedup=args["voxel_dedup"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.eval()


def encode_cloud(model, coords):
    """Frame-0 feature levels of one cloud centred on its mean, built as ChunkedLAWM.encode_window builds them for frame 0 (with the one-point-per-voxel wrapper when the model has it)."""
    from lawm.model.backbone import collect_pooling_chain, make_data_dict, voxel_representatives
    device = next(model.parameters()).device
    cloud = torch.from_numpy((coords - coords.mean(axis=0, keepdims=True)).astype(np.float32)).to(device)
    offset = torch.tensor([len(cloud)], device=device).int()
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        if model.voxel_dedup:
            kept, rows, kept_offset, grid = voxel_representatives(cloud, offset)
            chain = collect_pooling_chain(model.ptv3(make_data_dict(cloud[kept], kept_offset, grid_coord=grid)))
            chain[-1]["dedup_rows"], chain[-1]["dedup_offset"] = rows, offset
            expected = cloud[kept]
        else:
            chain = collect_pooling_chain(model.ptv3(make_data_dict(cloud, offset)))
            expected = cloud
    assert torch.equal(chain[-1].coord, expected), "the backbone reordered the points of a cloud"
    return chain


def repeated_queries(model, chain, copies=CHUNK):
    """The decoder's per-point queries of one encoded cloud, repeated for `copies` candidates, and their offsets."""
    with torch.no_grad():
        queries, offset = model.decoder.point_queries(chain)
    n = int(offset[-1])
    return queries.repeat(copies, 1), torch.arange(1, copies + 1, device=queries.device, dtype=offset.dtype) * n


def decode_chunked(model, queries, z, n_points, copies=CHUNK):
    """Cumulative displacements (pop, n_steps, n_points, 3) float32 of candidate latents z (pop, n_steps, n_queries, z_dim): decoded in chunks of exactly `copies` against repeated_queries, the last chunk padded with its last candidate and the padding dropped."""
    device = queries[0].device
    pop = len(z)
    pad = (-pop) % copies
    if pad:
        z = np.concatenate([z, np.repeat(z[-1:], pad, axis=0)], axis=0)
    out = []
    for start in range(0, len(z), copies):
        zs = torch.from_numpy(np.ascontiguousarray(z[start:start + copies].transpose(1, 0, 2, 3))).float().to(device)
        with torch.no_grad():
            steps = model.decode_all_steps(None, zs, queries)
        out.append(np.stack([s["delta"].squeeze(1).detach().cpu().numpy().reshape(copies, n_points, 3) for s in steps], axis=1))
    return np.concatenate(out, axis=0)[:pop].astype(np.float32)


class ModelAdapter:
    """What the planner needs of a model: prior = (z_mean, z_std) and decoder(coords, segment) -> decode(z) for an observed cloud (the model reads coordinates only, so segment is not used)."""

    def __init__(self, model, prior, copies=CHUNK):
        """Keep the model, its latent prior (mean and spread, each (n_steps, n_queries, z_dim)) and the chunk size."""
        self.model, self.prior, self.copies = model, prior, copies

    def decoder(self, coords, segment):
        """decode(z) for the cloud observed now: the encoder runs once on one copy here, each decode call only runs the decoder."""
        queries = repeated_queries(self.model, encode_cloud(self.model, coords), self.copies)
        return lambda z: decode_chunked(self.model, queries, z, len(coords), self.copies)


def checkpoint_sha256(checkpoint):
    """Hex SHA-256 of a checkpoint file."""
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_prior(model, windows):
    """(mean, std) over the windows of the latents the encoder gives them, each (n_steps, n_queries, z_dim); every window is centred coordinates (n_steps + 1, N, 3)."""
    device = next(model.parameters()).device
    zs = []
    for coords in windows:
        cloud = torch.as_tensor(coords, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, z = model.encode_window(cloud, torch.tensor([cloud.shape[1]], device=device).int())
        zs.append(z.squeeze(1).cpu().numpy())
    zs = np.stack(zs)
    return zs.mean(axis=0), zs.std(axis=0)


def save_prior(path, prior, checkpoint, n_windows):
    """Store a prior with the SHA-256 of the checkpoint whose encoder made it and the number of windows."""
    np.savez(path, z_mean=prior[0], z_std=prior[1], checkpoint_sha256=checkpoint_sha256(checkpoint), n_windows=n_windows)


def load_prior(path, checkpoint):
    """The (mean, std) stored in path; an error if it was made by another checkpoint's encoder."""
    stored = np.load(path)
    assert str(stored["checkpoint_sha256"]) == checkpoint_sha256(checkpoint), f"{path} was built from a different checkpoint than {checkpoint}"
    return stored["z_mean"], stored["z_std"]
