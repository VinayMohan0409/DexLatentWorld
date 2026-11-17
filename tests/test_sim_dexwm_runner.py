"""Tests of the runner: the config file, the episode seeds, and whole runs on the toy kitchen (every piece from the planner to the environment, with a stand-in model): result files, resume, config guard."""
import sim_settings as cfg
import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import sim_toy as T  # noqa: E402
from lawm.sim.dexwm import runner as R  # noqa: E402
from lawm.sim.dexwm.env import kitchen as D  # noqa: E402
from lawm.sim.dexwm.planning import task_planner as P  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "stage3" / "dexwm.json"


class Model:
    """A stand-in model: the prior, and a decoder that moves every hand point up to 1 cm along x."""

    prior = (np.zeros((5, 4, 8), dtype=np.float32), np.full((5, 4, 8), 0.5, dtype=np.float32))

    def decoder(self, coords, segment):
        """The decode function for one observed cloud."""
        hand = segment == 0

        def decode(z):
            """Latents (n, 5, 4, 8) to displacements (n, 5, N, 3)."""
            out = np.zeros((len(z), 5, len(coords), 3), dtype=np.float32)
            out[:, :, hand, 0] = (1.0 / (1.0 + np.exp(-z.reshape(len(z), -1).sum(axis=1))))[:, None, None] * 0.01
            return out
        return decode


def small_config(tmp_path, max_replans=2):
    """The real config with a manifest of one toy demo, a tiny search and a short loop."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(episodes_dir=str(tmp_path), episodes=[dict(demo="demo_7")])))
    config = R.load_config(CONFIG)
    cem = {t: cfg.cem(pop=8, iters=1) for t in config.tasks}
    loop = {t: cfg.loop(max_replans=max_replans) for t in config.tasks}
    return R.RunConfig(str(manifest), config.tasks, config.seed, config.scene_seed, config.env, config.exec, loop, cem, config.weights, config.gates)


def toy_opener(opened):
    """An opener that builds the toy kitchen and counts the demos opened."""
    def opener(demo, config):
        """The toy kitchen with the stand-in success wrapper."""
        opened.append(demo)
        env = T.Env()
        T.Wrapper.flag = False
        return D.DexWMEnv(env, T.episode(env), config.env, make_wrapper=T.Wrapper)
    return opener


def test_config_file_matches_the_defaults_in_code():
    """The runner reads the config file into the settings classes."""
    config = R.load_config(CONFIG)
    assert config.env == cfg.env() and config.exec == cfg.exec_settings() and config.loop == {t: cfg.loop(t) for t in config.tasks} and config.gates == cfg.gates() and config.loop["grasp"].stall_patience is None
    assert config.weights == {t: cfg.weights(t) for t in config.tasks} and config.cem == {t: cfg.cem(t) for t in config.tasks} and config.tasks == ("reach", "grasp", "place") and Path(config.manifest).name == "dexwm_episodes.json"
    assert Path(config.manifest).exists() and config.env.cameras is False


def test_unknown_config_key_is_an_error(tmp_path):
    """A misspelt setting does not pass silently."""
    raw = json.loads(CONFIG.read_text())
    raw["loop"]["grasp"]["max_replan"] = 5
    path = tmp_path / "c.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(TypeError):
        R.load_config(path)


def test_episode_seeds_are_repeatable_and_all_different():
    """The same (seed, demo, task) always gives the same seed; other demos, tasks and run seeds give other ones."""
    seeds = {R.episode_seed(s, d, t) for s in (0, 1) for d in ("demo_1", "demo_2") for t in ("reach", "grasp", "place")}
    assert len(seeds) == 12 and R.episode_seed(0, "demo_1", "reach") == R.episode_seed(0, "demo_1", "reach")


def test_whole_run_on_the_toy_kitchen(tmp_path):
    """Three tasks of one demo run end to end: a result file each with the records of the replans, a summary, the kitchen closed."""
    opened = []
    out = tmp_path / "out"
    summary = R.run_all(small_config(tmp_path), Model(), out, opener=toy_opener(opened))
    assert opened == ["demo_7"] and sorted(summary) == ["grasp", "place", "reach"]
    expected = dict(reach=("converged", 1), grasp=("max_replans", 2), place=("max_replans", 2))  # the toy hand never moves: reach starts at distance 0
    for task, (reason, n) in expected.items():
        r = json.loads(R.result_path(out, "demo_7", task).read_text())
        assert (r["stop_reason"], r["n_replans"], len(r["records"]), r["success"]) == (reason, n, n, False) and r["seed"] == R.episode_seed(0, "demo_7", task)
        assert r["final_error_mm"] == r["records"][-1]["error_mm"] and r["records"][0]["executed"] in ("best_sample", "hold")
    assert (out / "config_used.json").exists()


def test_rerun_skips_finished_episodes_and_does_not_open_the_kitchen(tmp_path):
    """With every result present nothing is opened; with one task missing the kitchen is opened and only that task runs."""
    opened, out = [], tmp_path / "out"
    config = small_config(tmp_path)
    R.run_all(config, Model(), out, opener=toy_opener(opened))
    R.run_all(config, Model(), out, opener=toy_opener(opened))
    assert opened == ["demo_7"]
    before = R.result_path(out, "demo_7", "reach").read_text()
    R.result_path(out, "demo_7", "place").unlink()
    R.run_all(config, Model(), out, opener=toy_opener(opened))
    assert opened == ["demo_7", "demo_7"] and R.result_path(out, "demo_7", "place").exists() and R.result_path(out, "demo_7", "reach").read_text() == before


def test_different_config_in_the_same_folder_is_refused(tmp_path):
    """A folder keeps the config of its first run; another config needs another folder."""
    out = tmp_path / "out"
    R.run_all(small_config(tmp_path), Model(), out, tasks=["reach"], opener=toy_opener([]))
    with pytest.raises(AssertionError):
        R.run_all(small_config(tmp_path, max_replans=3), Model(), out, tasks=["reach"], opener=toy_opener([]))


def test_unknown_demo_is_refused(tmp_path):
    """Only manifest demos can be run."""
    with pytest.raises(AssertionError):
        R.run_all(small_config(tmp_path), Model(), tmp_path / "out", demos=["demo_999"], opener=toy_opener([]))


def test_weights_and_stops_of_each_task_reach_the_episode(monkeypatch, tmp_path):
    """The planner gets its task's weights and search budget and the loop its task's stops: grasp without a stall stop, the others with one."""
    import dataclasses
    config = small_config(tmp_path)
    config = dataclasses.replace(config, loop={**config.loop, "grasp": cfg.loop(max_replans=2, stall_patience=None)}, weights={**config.weights, "reach": R.CostWeights(fingertip=3.0)})
    seen = []
    real = R.run_episode
    monkeypatch.setattr(R, "run_episode", lambda sim, planner, kin, ex, loop: seen.append((planner.weights, planner.settings, loop)) or real(sim, planner, kin, ex, loop))
    R.run_all(config, Model(), tmp_path / "out", opener=toy_opener([]))
    by_task = dict(zip(("reach", "grasp", "place"), seen))
    assert by_task["reach"][0].fingertip == 3.0 and by_task["grasp"][0] == cfg.weights("grasp") and by_task["place"][0] == cfg.weights("place")
    assert by_task["grasp"][2].stall_patience is None and by_task["reach"][2].stall_patience == 15 and by_task["place"][1].pop == 8


def test_model_record_is_written_once_and_a_different_model_is_refused(tmp_path):
    """The output folder remembers which checkpoint made it (path and SHA-256); the same model passes again, another is refused."""
    pytest.importorskip("torch")
    ck, other = tmp_path / "a.pt", tmp_path / "b.pt"
    ck.write_bytes(b"one"), other.write_bytes(b"two")
    R.record_model(tmp_path / "out", ck, "prior.npz")
    R.record_model(tmp_path / "out", ck, "prior.npz")
    assert json.loads((tmp_path / "out" / "model_used.json").read_text())["checkpoint_sha256"].startswith("7692c3ad")
    with pytest.raises(AssertionError):
        R.record_model(tmp_path / "out", other, "prior.npz")


def test_config_paths_are_relative_to_the_data_root(tmp_path, monkeypatch):
    """A relative path in a config is read under LAWM_DATA_ROOT, an absolute one as written; the shipped configs hold no absolute path."""
    from lawm.data.config import resolve_data_path
    monkeypatch.setenv("LAWM_DATA_ROOT", str(tmp_path))
    assert resolve_data_path("stage3/dexwm/eval") == tmp_path / "stage3" / "dexwm" / "eval"
    assert resolve_data_path("/elsewhere/x") == Path("/elsewhere/x")
    for name in ("dexwm_episodes", "dexwm_tuning_episodes"):
        assert "/lus/" not in (CONFIG.parent / f"{name}.json").read_text()
