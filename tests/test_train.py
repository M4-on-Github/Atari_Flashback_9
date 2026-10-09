"""Trainer tests (plain script, see docs/contracts.md §0). Run: container/run.sh python tests/test_train.py

All runs are CPU, tiny, and write to a temporary root (never the repo's runs/ or checkpoints/).
"""
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from export import export  # noqa: E402
from fake_env import FakeVecGames  # noqa: E402
from fb9.model import Agent  # noqa: E402
from fb9.selfplay import SelfPlay  # noqa: E402
from train import Args, train  # noqa: E402

torch.set_num_threads(1)


def fake_factory(num_actions: int = 3):
    def make(args: Args, num_workers: int):
        return FakeVecGames(args.num_games, num_actions=num_actions, seed=args.seed)
    return make


def tiny_args(**overrides) -> Args:
    base = dict(game="surround", num_games=4, num_steps=16, total_samples=6720, lr=1e-3, pool_fraction=0.25,
                snapshot_every=300, pool_size=5, checkpoint_every=10**9, cuda=False, seed=1,
                num_workers=1, run_name="run", minibatch_size=32)
    base.update(overrides)
    return Args(**base)


def test_model_shapes() -> None:
    agent = Agent(num_actions=5)
    obs = torch.randint(0, 256, (3, 6, 84, 84), dtype=torch.uint8)
    logits, value = agent(obs)
    assert logits.shape == (3, 5) and logits.dtype == torch.float32, logits.shape
    assert value.shape == (3,), value.shape
    action, logp, ent, val = agent.get_action_and_value(obs)
    assert action.shape == (3,) and logp.shape == (3,) and ent.shape == (3,) and val.shape == (3,)
    # uint8 input is scaled by 1/255: all-255 and all-0 frames must give different outputs
    white = torch.full((1, 6, 84, 84), 255, dtype=torch.uint8)
    black = torch.zeros((1, 6, 84, 84), dtype=torch.uint8)
    assert not torch.allclose(agent(white)[0], agent(black)[0])
    # uint8 path equals the explicit float path on the same values
    assert torch.allclose(agent(obs)[0], agent(obs.float())[0], atol=1e-5)


def test_selfplay_slots_and_pfsp() -> None:
    sp = SelfPlay(num_games=8, pool_fraction=0.25, pool_size=3, seed=0)
    assert sp.num_mirror_games == 6 and sp.num_pool_games == 2
    # mirror games 0..5 -> slots 0..11; pool game 6 (even) learner seat 0 -> slot 12; game 7 (odd) seat 1 -> 15
    assert sp.learner_slots.tolist() == list(range(12)) + [12, 15], sp.learner_slots
    assert sp.opponent_slots.tolist() == [13, 14]
    assert sp.num_learner_slots == 2 * 6 + 2
    assert sp.opponent_slot_of(6) == 13 and sp.opponent_slot_of(7) == 14

    # empty pool: pool games use the current learner (None)
    sp.resample_opponents()
    groups = sp.opponent_slot_groups()
    assert len(groups) == 1 and groups[0][0] is None and sorted(groups[0][1].tolist()) == [13, 14]

    # PFSP-hard weights
    a = sp.add_snapshot(100)
    b = sp.add_snapshot(200)
    a.winrate, b.winrate = 0.5, 1.0
    assert np.allclose(sp.pfsp_weights(), [0.30, 0.05])
    b.winrate = 0.0
    assert np.allclose(sp.pfsp_weights(), [0.30, 1.05])

    # sampling frequency follows the weights (1.05 vs 0.30)
    counts = {100: 0, 200: 0}
    for _ in range(4000):
        counts[sp.sample_opponent().samples] += 1
    frac = counts[200] / 4000
    assert abs(frac - 1.05 / 1.35) < 0.03, frac

    # EMA winrate (alpha 0.1): win -> 0.5*0.9+0.1 = 0.55; draw -> 0.5; loss -> 0.45
    sp2 = SelfPlay(num_games=4, pool_fraction=0.5, pool_size=2, seed=0)  # mirror 0,1; pool 2 (seat 0), 3 (seat 1)
    s = sp2.add_snapshot(10)
    sp2.opponents[2] = s
    sp2.opponents[3] = s
    sp2.on_episode_end(2, np.array([1.0, -1.0], dtype=np.float32))   # game 2 learner = seat 0 -> win
    assert abs(s.winrate - 0.55) < 1e-9, s.winrate
    sp2.opponents[3] = s
    sp2.on_episode_end(3, np.array([1.0, 0.0], dtype=np.float32))    # game 3 learner = seat 1 -> return 0 = draw
    assert abs(s.winrate - (0.9 * 0.55 + 0.05)) < 1e-9, s.winrate  # draw counts as 0.5
    sp2.opponents[2] = s
    sp2.on_episode_end(2, np.array([-1.0, 1.0], dtype=np.float32))   # loss
    assert abs(s.winrate - (0.9 * (0.9 * 0.55 + 0.05))) < 1e-9, s.winrate
    # mirror games never touch the pool
    sp2.on_episode_end(0, np.array([1.0, -1.0], dtype=np.float32))
    assert sp2.opponents[0] is None

    # eviction keeps the newest pool_size snapshots
    sp3 = SelfPlay(num_games=4, pool_fraction=0.5, pool_size=2, seed=0)
    for k in (1, 2, 3):
        sp3.add_snapshot(k)
    assert [x.samples for x in sp3.pool] == [2, 3]


def test_learns_on_fake_env() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_learn_"))
    try:
        args = tiny_args(run_name="learn", update_epochs=2)
        res = train(args, env_factory=fake_factory(), root=root)
        rets = np.array(res["episode_returns"])
        assert len(rets) >= 20, f"too few episodes: {len(rets)}"
        q = max(1, len(rets) // 4)
        early, late = float(rets[:q].mean()), float(rets[-q:].mean())
        print(f"    episodes={len(rets)} early_mean={early:.2f} late_mean={late:.2f} updates={res['updates']}")
        assert late > early + 10.0, (early, late)
        assert (root / "runs" / "learn").exists()
        assert (root / "checkpoints" / "learn" / "pool").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_resume() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_resume_"))
    try:
        args = tiny_args(run_name="resume", checkpoint_every=500, snapshot_every=300, total_samples=1500)
        res1 = train(args, env_factory=fake_factory(), root=root)
        batch = res1["batch_size"]
        ckpt_dir = root / "checkpoints" / "resume"
        assert (ckpt_dir / "latest.pt").exists() and list(ckpt_dir.glob("ckpt_*.pt"))
        s1 = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        assert s1["samples"] == res1["samples"] and s1["updates"] == res1["updates"]
        assert len(s1["pool"]) >= 1, "expected at least one snapshot in the pool"
        for entry in s1["pool"]:
            assert (ckpt_dir / "pool" / f"snap_{entry['samples']}.pt").exists()

        # resume with the same total: zero updates, the re-saved weights must be bit-identical
        res2 = train(tiny_args(run_name="resume", checkpoint_every=500, snapshot_every=300,
                               total_samples=s1["samples"], resume=True), env_factory=fake_factory(), root=root)
        s2 = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        assert res2["updates"] == s1["updates"] and s2["samples"] == s1["samples"]
        assert s2["pool"] == s1["pool"]
        for k, v in s1["model"].items():
            assert torch.equal(v, s2["model"][k]), k

        # resume and train further: counters continue from the checkpoint
        res3 = train(tiny_args(run_name="resume", checkpoint_every=500, snapshot_every=300,
                               total_samples=s1["samples"] + 2 * batch, resume=True),
                     env_factory=fake_factory(), root=root)
        assert res3["updates"] == s1["updates"] + 2, (res3["updates"], s1["updates"])
        assert res3["samples"] == s1["samples"] + 2 * batch
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_export_roundtrip() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_export_"))
    try:
        args = tiny_args(run_name="export", total_samples=16 * 7 * 2)
        train(args, env_factory=fake_factory(num_actions=5), root=root)
        ckpt = root / "checkpoints" / "export" / "latest.pt"
        out = root / "models" / "surround"
        export(ckpt, out)
        assert (out / "model.ts").exists() and (out / "config.json").exists()

        import json
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["game"] == "surround" and cfg["ale_mode"] == 1 and cfg["frameskip"] == 15 and cfg["stack"] == 4
        assert cfg["obs_shape"] == [6, 84, 84] and cfg["action_ids"] == [0, 2, 3, 4, 5]
        assert len(cfg["action_names"]) == 5 and cfg["samples"] == 16 * 7 * 2

        model = torch.jit.load(str(out / "model.ts"), map_location="cpu")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        agent = Agent(num_actions=5)
        agent.load_state_dict(state["model"])
        agent.eval()
        x = torch.randint(0, 256, (2, 6, 84, 84), dtype=torch.uint8)
        with torch.no_grad():
            got = model(x)
            ref, _ = agent(x)
        assert got.shape == (2, 5) and got.dtype == torch.float32, got.shape
        assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    finally:
        shutil.rmtree(root, ignore_errors=True)


TESTS = [test_model_shapes, test_selfplay_slots_and_pfsp, test_learns_on_fake_env,
         test_checkpoint_resume, test_export_roundtrip]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}", flush=True)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
