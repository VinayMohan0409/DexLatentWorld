"""PointWorld's released checkpoint through its own code: the repository is passed in, never edited."""
from __future__ import annotations

import os
import sys


def load(repo, checkpoint, log_dir):
    """(model, model args, device, release pipeline, collate) of a checkpoint; needs the GPU environment PointWorld runs in."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    from arguments import parse_args
    from dataset_components.collate import custom_collate_fn
    from dataset_components.pipeline import apply_release_pipeline_to_sample
    from pointworld.checkpoint_contract import read_checkpoint_contract, train_domains_from_data_contract
    from training.trainer import Trainer

    os.chdir(repo)  # its norm statistics path is relative to the repository
    sys.argv = ["pointworld", "--model_path", checkpoint, "--distributed", "false", "--log_dir", log_dir, "--exp_name", "lawm_gripper"]
    args = parse_args()
    _, contract = read_checkpoint_contract(torch.load(checkpoint, map_location="cpu", weights_only=False), context=f"checkpoint '{checkpoint}'")
    args.domains = train_domains_from_data_contract(contract, context=f"checkpoint '{checkpoint}'")
    model = Trainer(args, inference_only=True, data_info_dict=None).model.eval()
    return model, args, next(model.parameters()).device, apply_release_pipeline_to_sample, custom_collate_fn
