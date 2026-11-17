"""Compare the per-step losses of two runs exactly: `python scripts/compare_runs.py RUN_A RUN_B` (folders with train_steps.csv)."""
import csv
import sys
from pathlib import Path


def read(folder):
    """Rows of one run's train_steps.csv as dicts."""
    with open(Path(folder) / "train_steps.csv") as f:
        return list(csv.DictReader(f))


a, b = read(sys.argv[1]), read(sys.argv[2])
n = min(len(a), len(b))
for i in range(n):
    for col in ("recon_loss", "contrastive_loss", "total_loss", "grad_norm"):
        if a[i][col] != b[i][col]:
            print(f"first difference at step {a[i]['step']} ({col}): {a[i][col]} vs {b[i][col]}")
            sys.exit(1)
print(f"identical over {n} steps" + ("" if len(a) == len(b) else f" (the runs have {len(a)} and {len(b)} steps)"))
