"""Open the external UniDex package (robot-hand retargeting and the H2O dataset index) and restore the process state afterwards."""
from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

_SINGLE_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def open_unidex(unidex_root: Path, dataset_name: str = "H2o") -> Iterator[Tuple[object, object]]:
    """Yield (visualizer module, dataset) for UniDex at unidex_root, running with its root as working directory and restoring the directory and sys.path afterwards."""
    for var in _SINGLE_THREAD_VARS:
        os.environ.setdefault(var, "1")
    root = Path(unidex_root).resolve()
    previous_cwd, previous_path = os.getcwd(), list(sys.path)
    sys.path[:0] = [str(root / "HandAdapter"), str(root)]
    os.chdir(root)
    try:
        import visualizer as uv  # UniDex's module; also builds its (unused here) web viewer definitions

        if dataset_name not in uv.MANO_TIP_INDEX_MAP:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        uv.MANO_TIP_INDEX = uv.MANO_TIP_INDEX_MAP[dataset_name]
        cfg = copy.deepcopy(uv.DATASET_CONFIG[dataset_name])
        cfg["load_pcd"] = False  # hands only: no RGB-D point clouds are read
        yield uv, uv.hydra.utils.instantiate(cfg)
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
