"""Cross-decode on a tiny synthetic dataset with the stand-in backbone: window choice, batching, and agreement with evaluate()."""
import numpy as np
import pytest
import torch

from fake_backbone import FakeBackbone
from synthetic_data import make_dataset

from lawm.data.index import build_index
from lawm.data.sequences import Sequence
from lawm.eval import cross_decode as cd
from lawm.model.chunked import ChunkedLAWM
from lawm.train.evaluation import evaluate

CPU = torch.device("cpu")
GAP, STEPS = 2, 2
SEQS = {"subject1/h1/0": (20, 1, {3: [5]}), "subject1/h1/1": (16, 2, None), "subject1/h2/0": (18, 3, None)}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Synthetic dataset root and its window index."""
    root = tmp_path_factory.mktemp("cd")
    make_dataset(root, SEQS)
    return root, build_index(root, list(SEQS), GAP, STEPS)


@pytest.fixture(scope="module")
def model():
    """Stand-in model in eval mode."""
    torch.manual_seed(0)
    stats = dict(per_step_mean=np.zeros((STEPS, 3)), per_step_var=np.full((STEPS, 3), 1e-4))
    return ChunkedLAWM(stats, n_steps=STEPS, backbone=FakeBackbone(), seed=0).eval()


def test_window_choice_is_seeded_capped_and_only_uses_windows_valid_for_every_embodiment(data):
    """the same seed gives the same windows, at most per_sequence per sequence, none invalid for a listed embodiment"""
    _, index = data
    a = cd.select_windows(index, [0, 3, 5], per_sequence=4, seed=0)
    assert a == cd.select_windows(index, [0, 3, 5], per_sequence=4, seed=0)
    assert a != cd.select_windows(index, [0, 3, 5], per_sequence=4, seed=1)
    for s in SEQS:
        assert len([w for w in a if w[0] == s]) == 4
    bad = {(str(s), int(f)) for s, f, v in zip(index["seq"], index["f"], index["valid_window"][:, 3]) if not v}
    assert bad and not bad & set(a)  # embodiment 3 is invalid at frame 5 of the first sequence
    assert bad & set(cd.select_windows(index, [0, 1], per_sequence=100, seed=0))  # not filtered when 3 is not listed


def test_batches_never_span_two_sequences():
    """groups are consecutive, at most batch_size long, and each holds one sequence"""
    w = [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("c", 0), ("c", 1)]
    assert cd.batches(w, 2) == [[("a", 0), ("a", 1)], [("a", 2)], [("b", 0)], [("c", 0), ("c", 1)]]


def test_the_diagonal_equals_evaluate_and_columns_share_one_target_set(data, model):
    """cell (e, e) is exactly evaluate()'s 'correct' condition on the same windows; every cell of a column scores the same points"""
    root, index = data
    emb = [0, 2, 5]
    windows = cd.select_windows(index, emb, per_sequence=3, seed=0)
    totals = cd.cross_decode(model, root, windows, emb, GAP, 0, 4, CPU, False, log=lambda *_: None)
    seqs = {s: Sequence(root / s) for s in SEQS}
    for col, e in enumerate(emb):
        loader = [cd.window_batch(seqs, g, e, GAP, STEPS) for g in cd.batches(windows, 4)]
        ref = evaluate(model, loader, CPU, False)["correct"]
        assert np.allclose(totals[col, col], ref, rtol=1e-6)
        assert np.array_equal(totals[:, col, ..., 1], np.broadcast_to(totals[col, col, ..., 1], totals[:, col, ..., 1].shape))
    assert not np.allclose(totals[0, 1], totals[1, 1])  # another embodiment's latents change the prediction


def test_results_do_not_depend_on_batch_size_with_the_stand_in(data, model):
    """with a backbone that has no batch-dependent grid, batch size only regroups the sums"""
    root, index = data
    emb = [1, 4]
    windows = cd.select_windows(index, emb, per_sequence=3, seed=2)
    a = cd.cross_decode(model, root, windows, emb, GAP, 0, 1, CPU, False, log=lambda *_: None)
    b = cd.cross_decode(model, root, windows, emb, GAP, 0, 8, CPU, False, log=lambda *_: None)
    assert np.allclose(a, b, rtol=1e-6)


def test_matrices_and_regimes():
    """per-step and mean-over-steps matrices come from sums / counts; regimes split by trained flags"""
    totals = np.zeros((2, 2, 2, 5, 2))
    totals[..., 0, 0] = [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]
    totals[..., 0, 1] = 10
    m = cd.matrices(totals)
    assert m["step1"][0, 1] == 3.0 and m["mean_over_steps"][1, 0] == 5.5
    r = cd.regime_summary(m["mean_over_steps"], [True, False])
    assert r["diag_seen"] == 1.5 and r["seen_to_unseen"] == 3.5 and r["unseen_to_seen"] == 5.5 and r["diag_unseen"] == 7.5


def test_a_partial_run_matches_the_full_run_on_its_cells(data, model):
    """decoding only some pairs gives exactly the full run's values there and zeros elsewhere; per-encode seeding is accepted"""
    root, index = data
    emb = [0, 2, 5]
    windows = cd.select_windows(index, emb, per_sequence=2, seed=0)
    full = cd.cross_decode(model, root, windows, emb, GAP, 0, 4, CPU, False, log=lambda *_: None, names=["a", "b", "c"], seed=7)
    cells = np.array([[False, False, True], [False, False, True], [True, True, True]])
    part = cd.cross_decode(model, root, windows, emb, GAP, 0, 4, CPU, False, log=lambda *_: None, cells=cells, names=["a", "b", "c"], seed=7)
    assert np.allclose(part[cells], full[cells], rtol=1e-6) and not part[~cells].any()
