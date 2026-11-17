"""Registry consistency, appended-embodiment scene rules and the export's stored-scene guard."""
import numpy as np
import pytest

from lawm.embodiments.registry import load_registry  # noqa: E402
from lawm.preprocessing.export_extra_embodiments import keep_stored_scene, scene_difference  # noqa: E402
from lawm.preprocessing.hand_samplers import HAND_TYPES  # noqa: E402
from lawm.preprocessing.scene_geometry import EMBODIMENT_NAMES  # noqa: E402


def test_registry_starts_with_the_stored_order():
    """the first nine registry entries are the stored embodiments, in order, all trained, robots mapped to their UniDex folders"""
    reg = load_registry()
    names = list(reg)[: len(EMBODIMENT_NAMES)]
    assert names == [n.lower() for n in EMBODIMENT_NAMES]
    assert [reg[n].unidex_hand for n in names[1:]] == list(HAND_TYPES)
    assert all(reg[n].trained for n in names)


def test_seed_names_reproduce_stored_hands_and_separate_variants():
    """stored hands seed with their UniDex name (as the stored clouds); a variant seeds with its own name"""
    reg = load_registry()
    assert reg["shadow"].seed_name == "Shadow"
    assert reg["shadow_noforearm"].seed_name == "shadow_noforearm"
    assert reg["shadow_noforearm"].exclude_links == ("forearm",) and reg["shadow_noforearm"].display == "shadow"


def test_registry_rejects_bad_entries(tmp_path):
    """unknown keys and robots without a UniDex hand are errors"""
    p = tmp_path / "r.json"
    p.write_text('{"embodiments": {"x": {"kind": "robot", "trained": false}}}')
    with pytest.raises(ValueError):
        load_registry(p)
    p.write_text('{"embodiments": {"x": {"kind": "robot", "unidex_hand": "X", "trained": false, "typo": 1}}}')
    with pytest.raises(ValueError):
        load_registry(p)


def _scene(rng, counts):
    """A small stored-format scene with the given distractor counts."""
    return {"table": rng.random((8, 3), dtype=np.float32), "object_canonical": rng.random((8, 3), dtype=np.float32),
            "plane_n": np.array([0, 0, 1], np.float32), "plane_d": np.float32(0), "table_uvc": np.eye(3, dtype=np.float32),
            "cam_poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)), "object_id": 1, "n_frames": np.int32(2),
            "distractor_points": [rng.random((k, 8, 3), dtype=np.float32) for k in counts],
            "distractor_counts": np.array(counts, np.int32)}


def test_stored_scene_guard_tolerates_rounding_and_keeps_stored_arrays():
    """rounding-level differences pass and are replaced by the stored arrays; added distractors come from the rebuild"""
    rng = np.random.default_rng(0)
    base = _scene(rng, [1, 2])
    rebuilt = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    rebuilt["table"] = base["table"] + np.float32(1e-7)
    extra = rng.random((3, 8, 3), dtype=np.float32)
    rebuilt["distractor_points"] = [d + np.float32(1e-7) for d in base["distractor_points"]] + [extra]
    rebuilt["distractor_counts"] = np.array([1, 2, 3], np.int32)
    assert scene_difference(base, rebuilt, 2) < 1e-6
    out = keep_stored_scene(base, rebuilt, 2)
    assert out["table"] is base["table"] and out["distractor_points"][1] is base["distractor_points"][1]
    assert out["distractor_points"][2] is extra and len(out["distractor_points"]) == 3


def test_stored_scene_guard_rejects_discrete_changes():
    """a different distractor count for a stored embodiment is an error, not a tolerance question"""
    rng = np.random.default_rng(1)
    base = _scene(rng, [1, 2])
    other = _scene(np.random.default_rng(1), [1, 3])
    with pytest.raises(RuntimeError):
        scene_difference(base, other, 2)
