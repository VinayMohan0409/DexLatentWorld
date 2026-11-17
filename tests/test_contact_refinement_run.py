"""Unit tests for the pure helpers in lawm.preprocessing.contact_refinement.run."""
import pytest

pytest.importorskip("torch")

from lawm.preprocessing.contact_refinement.optimizer import RefineConfig  # noqa: E402
from lawm.preprocessing.contact_refinement.run import config_hash  # noqa: E402


def test_config_hash_is_stable_and_sensitive_to_hyperparameters_and_inputs():
    """config_hash repeats for the same inputs, changes with a hyperparameter or source hash, and exposes the payload"""
    base, payload = config_hash(RefineConfig(), {"000000.txt": "aa"})
    assert base == config_hash(RefineConfig(), {"000000.txt": "aa"})[0]
    assert base != config_hash(RefineConfig(n_iter=10), {"000000.txt": "aa"})[0]
    assert base != config_hash(RefineConfig(), {"000000.txt": "bb"})[0]
    assert payload["lam_temporal"] == 10.0 and payload["source_hashes"] == {"000000.txt": "aa"}
