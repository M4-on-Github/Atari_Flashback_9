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
from fb9.grid import GRID_OBS_SHAPE  # noqa: E402
from fb9.model import Agent, GridAgent  # noqa: E402
from fb9.selfplay import BOT_ACTION, SelfPlay  # noqa: E402
from train import Args, train  # noqa: E402

torch.set_num_threads(1)


class BotFakeVecGames(FakeVecGames):
    """Fake env that counts BOT_ACTION slots and plays them as action 0 (the real VecGames plays SurroundBot)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot_actions_seen = 0

    def step(self, actions: np.ndarray):
        a = np.asarray(actions).reshape(self.num_slots).copy()
        self.bot_actions_seen += int(np.sum(a == BOT_ACTION))
        a[a == BOT_ACTION] = 0
        return super().step(a)


def fake_factory(num_actions: int = 3, made: list | None = None):
    def make(args: Args, num_workers: int):
        env = BotFakeVecGames(args.num_games, num_actions=num_actions, seed=args.seed, obs=args.obs)
        if made is not None:
            made.append(env)
        return env
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


def test_selfplay_bot_games() -> None:
    # 8 games: mirror 0..3, pool 4 (seat 0), 5 (seat 1), bot 6 (seat 0), 7 (seat 1)
    sp = SelfPlay(num_games=8, pool_fraction=0.25, pool_size=3, seed=0, bot_fraction=0.25)
    assert (sp.num_mirror_games, sp.num_pool_games, sp.num_bot_games) == (4, 2, 2)
    assert sp.bot_slots.tolist() == [13, 14], sp.bot_slots
    assert sp.opponent_slots.tolist() == [9, 10, 13, 14]
    assert sp.learner_slots.tolist() == list(range(8)) + [8, 11, 12, 15], sp.learner_slots
    assert [sp.is_bot(g) for g in range(8)] == [False] * 6 + [True] * 2
    assert sp.is_mirror(3) is True and sp.is_mirror(4) is False and sp.is_mirror(6) is False

    # bot games never get a snapshot and are not in the opponent groups
    snap = sp.add_snapshot(10)
    sp.resample_opponents()
    assert sp.opponents[6] is None and sp.opponents[7] is None
    groups = sp.opponent_slot_groups()
    assert len(groups) == 1 and groups[0][0] is snap and sorted(groups[0][1].tolist()) == [9, 10]
    assert "bot_winrate" in sp.pool_stats() and sp.pool_stats()["bot_winrate"] == 0.5

    # bot EMA: game 6 learner seat 0 wins -> 0.55; game 7 learner seat 1 loses -> 0.9 * 0.55
    sp.on_episode_end(6, np.array([1.0, -1.0], dtype=np.float32))
    assert abs(sp.bot_winrate - 0.55) < 1e-9, sp.bot_winrate
    sp.on_episode_end(7, np.array([0.0, -1.0], dtype=np.float32))
    assert abs(sp.bot_winrate - 0.9 * 0.55) < 1e-9, sp.bot_winrate
    assert snap.winrate == 0.5 and sp.opponents[6] is None   # pool winrates untouched by bot games

    # without bot games: no bot_winrate key, and the bot slots are empty
    plain = SelfPlay(num_games=4, pool_fraction=0.5, pool_size=2, seed=0)
    assert "bot_winrate" not in plain.pool_stats() and plain.bot_slots.size == 0

    for bad in (dict(pool_fraction=0.5, bot_fraction=0.75), dict(pool_fraction=0.0, bot_fraction=1.5)):
        try:
            SelfPlay(num_games=4, pool_size=2, seed=0, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_train_bot_fraction_smoke() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_bot_"))
    made: list = []
    try:
        args = tiny_args(run_name="bots", bot_fraction=0.25, pool_fraction=0.25)   # 4 games: 2 mirror, 1 pool, 1 bot
        res = train(args, env_factory=fake_factory(made=made), root=root)
        env = made[0]
        assert res["updates"] >= 1 and env.bot_actions_seen > 0, env.bot_actions_seen
        s = torch.load(root / "checkpoints" / "bots" / "latest.pt", map_location="cpu", weights_only=False)
        assert s["args"]["bot_fraction"] == 0.25
        assert (root / "runs" / "bots").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    try:
        train(tiny_args(run_name="bad", bot_fraction=0.25, game="combat"), env_factory=fake_factory(), root=root)
    except ValueError:
        return
    raise AssertionError("bot_fraction with game=combat should raise ValueError")


def test_learns_on_fake_env() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_learn_"))
    try:
        args = tiny_args(run_name="learn", update_epochs=2, obs="pixels")
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
        args = tiny_args(run_name="export", total_samples=16 * 7 * 2, obs="pixels")
        train(args, env_factory=fake_factory(num_actions=5), root=root)
        ckpt = root / "checkpoints" / "export" / "latest.pt"
        out = root / "models" / "surround"
        export(ckpt, out)
        assert (out / "model.ts").exists() and (out / "config.json").exists()

        import json
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["game"] == "surround" and cfg["ale_mode"] == 1 and cfg["frameskip"] == 15 and cfg["stack"] == 4
        assert cfg["obs"] == "pixels" and cfg["obs_shape"] == [6, 84, 84] and cfg["action_ids"] == [0, 2, 3, 4, 5]
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


def test_grid_obs_checkpoint_resume_and_export() -> None:
    """Grid is the Surround default: checkpoint args record it, resume refuses another obs, export writes grid config."""
    root = Path(tempfile.mkdtemp(prefix="fb9_test_grid_"))
    try:
        args = tiny_args(run_name="grid", total_samples=16 * 7 * 2, checkpoint_every=500)
        assert args.obs == ""
        train(args, env_factory=fake_factory(num_actions=5), root=root)
        ckpt = root / "checkpoints" / "grid" / "latest.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert state["args"]["obs"] == "grid", state["args"]["obs"]

        try:
            train(tiny_args(run_name="grid", total_samples=16 * 7 * 2 + 1, resume=True, obs="pixels",
                            checkpoint_every=500), env_factory=fake_factory(num_actions=5), root=root)
        except ValueError as e:
            assert "obs" in str(e), e
        else:
            raise AssertionError("resuming a grid checkpoint with obs=pixels should raise ValueError")

        out = root / "models" / "surround_grid"
        export(ckpt, out)
        import json
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["obs"] == "grid" and cfg["obs_shape"] == [6, 18, 38] and cfg["stack"] == 2, cfg
        assert cfg["frameskip"] == 15 and cfg["samples"] == 16 * 7 * 2
        assert list(GRID_OBS_SHAPE) == cfg["obs_shape"]

        model = torch.jit.load(str(out / "model.ts"), map_location="cpu")
        agent = GridAgent(num_actions=5)
        agent.load_state_dict(state["model"])
        agent.eval()
        x = torch.randint(0, 256, (2,) + GRID_OBS_SHAPE, dtype=torch.uint8)
        with torch.no_grad():
            got = model(x)
            ref, _ = agent(x)
        assert got.shape == (2, 5) and got.dtype == torch.float32, got.shape
        assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_obs_arg_validation() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_obsargs_"))
    try:
        for bad in (dict(obs="rgb"), dict(obs="grid", game="combat")):
            try:
                train(tiny_args(run_name="badobs", **bad), env_factory=fake_factory(), root=root)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {bad}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


TESTS = [test_model_shapes, test_selfplay_slots_and_pfsp, test_selfplay_bot_games, test_train_bot_fraction_smoke,
         test_learns_on_fake_env, test_checkpoint_resume, test_export_roundtrip, test_grid_obs_checkpoint_resume_and_export,
         test_obs_arg_validation]

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
