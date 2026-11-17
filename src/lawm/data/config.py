"""Locate the dataset through LAWM_DATA_ROOT and read the JSON config and split files."""
from __future__ import annotations

import json
import os
from pathlib import Path


def data_root() -> Path:
    """The folder that holds the dataset, from the LAWM_DATA_ROOT environment variable."""
    root = os.environ.get("LAWM_DATA_ROOT")
    if not root:
        raise RuntimeError("Set LAWM_DATA_ROOT to the folder that contains the dataset (see README.md).")
    return Path(root)


def resolve_data_path(path) -> Path:
    """A path from a config: used as written if absolute, otherwise relative to LAWM_DATA_ROOT."""
    path = Path(path)
    return path if path.is_absolute() else data_root() / path


def read_json(path) -> dict:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())


def dataset_dir(config: dict) -> Path:
    """The folder with the <subject>/<session>/<take> tree of compact point-cloud files."""
    return data_root() / config["data_dir"]
