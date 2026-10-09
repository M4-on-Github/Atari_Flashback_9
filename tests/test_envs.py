"""Tests for fb9/envs.py (docs/contracts.md §3.3). Run: container/run.sh python tests/test_envs.py"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fb9.envs import EnvConfig, TwoPlayerGame, VecGames  # noqa: E402
from fb9.games import GAMES, LEGACY_FRAMESKIP, frameskip_from_args  # noqa: E402
from fb9.preprocess import OBS_SHAPE  # noqa: E402


class _RecordingALE:
    """Proxy around the ALE that records the action array of every emulator frame."""

    def __init__(self, ale):
        self._ale = ale
        self.acts: list[np.ndarray] = []

    def act(self, a):
        self.acts.append(np.array(a, copy=True))
        return self._ale.act(a)

    def __getattr__(self, name):
        return getattr(self._ale, name)


def _same_info(a: dict, b: dict) -> bool:
    return (a["game"] == b["game"] and a["episode_frames"] == b["episode_frames"]
            and np.array_equal(a["episode_return"], b["episode_return"]))


def test_shapes_dtypes_seats():
    for train in (False, True):
        g = TwoPlayerGame(EnvConfig(game="surround", train=train, obs="pixels"), seed=0)
        obs = g.reset()
        assert obs.shape == (2,) + OBS_SHAPE and obs.dtype == np.uint8, (obs.shape, obs.dtype)
        assert np.all(obs[0, 4] == 255) and np.all(obs[0, 5] == 0)
        assert np.all(obs[1, 4] == 0) and np.all(obs[1, 5] == 255)
        obs, rew, done, info = g.step(np.array([1, 2]))
        assert obs.shape == (2,) + OBS_SHAPE and obs.dtype == np.uint8
        assert rew.shape == (2,) and rew.dtype == np.float32
        assert isinstance(done, (bool, np.bool_)) and not done and info == {}
        assert np.all(obs[0, 4] == 255) and np.all(obs[0, 5] == 0)
        assert np.all(obs[1, 4] == 0) and np.all(obs[1, 5] == 255)
    cfg = EnvConfig(game="combat", train=True)
    obs = TwoPlayerGame(cfg, seed=0).reset()
    assert obs.shape == (2,) + OBS_SHAPE
    assert np.all(obs[0, 4] == 255) and np.all(obs[1, 5] == 255)
    vec = VecGames(EnvConfig(game="surround", train=True, obs="pixels"), num_games=2, num_workers=0, seed=0)
    vobs = vec.reset()
    assert vobs.shape == (4,) + OBS_SHAPE and vobs.dtype == np.uint8
    assert vec.num_slots == 4 and vec.num_games == 2 and vec.num_actions == 5
    vec.close()


def test_eval_stacks_identical_and_deterministic():
    cfg = EnvConfig(game="surround", train=False, obs="pixels")
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 5, size=(60, 2))
    runs = []
    for _ in range(2):
        g = TwoPlayerGame(cfg, seed=42)
        obs = [g.reset()]
        rews = []
        for a in seq:
            o, r, done, _ = g.step(a)
            obs.append(o)
            rews.append(r)
            assert np.array_equal(o[0, :4], o[1, :4]), "eval stacks of the two seats must match"
        runs.append((np.stack(obs), np.stack(rews)))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def test_train_mode_seeded_reproducible():
    cfg = EnvConfig(game="surround", train=True)
    outs = []
    for _ in range(2):
        g = TwoPlayerGame(cfg, seed=5)
        obs = [g.reset()]
        for k in range(30):
            obs.append(g.step(np.array([k % 5, (k * 3) % 5]))[0])
        outs.append(np.stack(obs))
    assert np.array_equal(outs[0], outs[1])


def _pettingzoo_env(game: str):
    from pettingzoo.atari import combat_tank_v2, surround_v2

    max_frames = GAMES[game].max_frames
    if game == "surround":
        # surround's raw_env fixes mode_num (ALE mode 1 = first available mode) and uses the minimal action set
        return surround_v2.parallel_env(max_cycles=max_frames)
    # combat's mode is derived from these flags: (False, False) -> 1, plus has_maze -> 2
    return combat_tank_v2.parallel_env(has_maze=True, is_invisible=False, billiard_hit=False,
                                       full_action_space=True, max_cycles=max_frames)


def _check_pettingzoo_equivalence(game: str, ale_seed: int, max_steps: int):
    """Our env (train=False) vs PettingZoo stepped one frame at a time with the same action per decision."""
    spec = GAMES[game]
    pz = _pettingzoo_env(game)
    assert list(pz.unwrapped.action_mapping) == list(spec.action_ids), "action sets differ"
    assert pz.unwrapped.mode == spec.mode, (pz.unwrapped.mode, spec.mode)

    g = TwoPlayerGame(EnvConfig(game=game, train=False), seed=0)
    g.reset(ale_seed=ale_seed)
    pz_obs, _ = pz.reset(seed=ale_seed)
    assert np.array_equal(pz_obs["first_0"], g.render_rgb()), "first screens differ"

    rng = np.random.default_rng(123)
    frames = 0
    pz_total = np.zeros(2, dtype=np.float32)
    for step in range(max_steps):
        acts = rng.integers(0, spec.num_actions, size=2)
        _, our_rew, our_done, info = g.step(acts)
        pz_rew = np.zeros(2, dtype=np.float32)
        pz_over, nfr, last_obs = False, 0, None
        for _ in range(g.frameskip):
            last_obs, rews, _, _, _ = pz.step({"first_0": int(acts[0]), "second_0": int(acts[1])})
            nfr += 1
            pz_rew += np.array([rews["first_0"], rews["second_0"]], dtype=np.float32)
            if pz.unwrapped.ale.game_over():
                pz_over = True
                break
        frames += nfr
        pz_total += pz_rew
        assert np.array_equal(our_rew, pz_rew), f"step {step}: rewards {our_rew} vs PettingZoo {pz_rew}"
        assert our_done == pz_over, f"step {step}: done {our_done} vs PettingZoo game over {pz_over}"
        if pz_over:
            assert info["episode_frames"] == frames, (info["episode_frames"], frames)
            assert np.array_equal(info["episode_return"], pz_total), "episode return differs"
            return step + 1, frames
        assert np.array_equal(last_obs["first_0"], g.render_rgb()), f"step {step}: screens differ"
    raise AssertionError(f"{game}: episode did not end within {max_steps} steps")


def test_pettingzoo_equivalence_surround():
    steps, frames = _check_pettingzoo_equivalence("surround", ale_seed=1234, max_steps=2500)
    print(f"  surround: {steps} steps / {frames} frames to game over, rewards and screens identical")


def test_pettingzoo_equivalence_combat():
    steps, frames = _check_pettingzoo_equivalence("combat", ale_seed=1234, max_steps=2200)
    print(f"  combat: {steps} steps / {frames} frames to game over, rewards and screens identical")


def test_delay_exact():
    """With known per-player delays and no sticky, the action executed at frame t is the one chosen d frames ago."""
    ids = np.array(GAMES["surround"].action_ids)
    n_steps = 30
    rng = np.random.default_rng(5)
    intended = rng.integers(1, 5, size=(n_steps, 2))   # UP/RIGHT/LEFT/DOWN, so every step is a real move
    for d in ([0, 0], [3, 7], [10, 1]):
        g = TwoPlayerGame(EnvConfig(game="surround", train=True, sticky_p=0.0, max_delay=10, augment=False), seed=3)
        g.reset(ale_seed=11)
        assert all(0 <= x <= 10 for x in g._delay), "drawn delays out of range"
        g._delay = list(d)   # force the per-episode draw
        rec = _RecordingALE(g.ale)
        g.ale = rec
        for k in range(n_steps):
            g.step(intended[k])
        assert len(rec.acts) == g.frameskip * n_steps
        for t, a in enumerate(rec.acts):
            for p in (0, 1):
                idx = 0 if t < d[p] else intended[(t - d[p]) // g.frameskip, p]
                assert a[p] == ids[idx], f"d={d} frame {t} player {p}: executed {a[p]}, expected {ids[idx]}"


def test_sticky_and_eval_flag():
    ids = GAMES["surround"].action_ids
    # sticky_p=1 in train mode: the previously executed action (NOOP at the start) is repeated forever
    g = TwoPlayerGame(EnvConfig(game="surround", train=True, sticky_p=1.0, max_delay=0, augment=False), seed=1)
    g.reset(ale_seed=2)
    rec = _RecordingALE(g.ale)
    g.ale = rec
    for _ in range(10):
        g.step(np.array([2, 3]))
    assert all(np.all(a == ids[0]) for a in rec.acts)
    # train=False ignores sticky and delay entirely: executed == intended, frame for frame
    g = TwoPlayerGame(EnvConfig(game="surround", train=False, sticky_p=1.0, max_delay=10), seed=1)
    g.reset(ale_seed=2)
    rec = _RecordingALE(g.ale)
    g.ale = rec
    seq = [(1, 4), (2, 0), (3, 3)]
    for a in seq:
        g.step(np.array(a))
    for t, a in enumerate(rec.acts):
        exp = np.array([ids[seq[t // g.frameskip][0]], ids[seq[t // g.frameskip][1]]])
        assert np.array_equal(a, exp), (t, a, exp)


def test_frameskip_per_game():
    """Surround decides every 15 frames (one cell move), Combat every 4; EnvConfig.frameskip overrides the game."""
    for game, want in (("surround", 15), ("combat", 4)):
        assert GAMES[game].frameskip == want, game
        g = TwoPlayerGame(EnvConfig(game=game, train=False), seed=0)
        g.reset(ale_seed=3)
        assert g.frameskip == want
        g.step(np.zeros(2, dtype=np.int64))
        assert g.t == want, (game, g.t)
    g = TwoPlayerGame(EnvConfig(game="surround", train=False, frameskip=4), seed=0)
    g.reset(ale_seed=3)
    g.step(np.zeros(2, dtype=np.int64))
    assert g.frameskip == 4 and g.t == 4, (g.frameskip, g.t)


def test_frameskip_from_args():
    assert frameskip_from_args({}) == LEGACY_FRAMESKIP == 4
    assert frameskip_from_args({"frameskip": 15}) == 15
    assert frameskip_from_args({"frameskip": 0}) == 4


def test_vec_workers_match_inprocess():
    cfg = EnvConfig(game="surround", train=False)
    a = VecGames(cfg, num_games=4, num_workers=0, seed=7)
    b = VecGames(cfg, num_games=4, num_workers=2, seed=7)
    try:
        oa, ob = a.reset(), b.reset()
        assert np.array_equal(oa, ob)
        rng = np.random.default_rng(9)
        episodes_done = 0
        for _ in range(2500):
            acts = rng.integers(0, a.num_actions, size=a.num_slots)
            oa, ra, da, ia = a.step(acts)
            ob, rb, db, ib = b.step(acts)
            assert np.array_equal(oa, ob), "observations differ"
            assert np.array_equal(ra, rb) and np.array_equal(da, db), "rewards/dones differ"
            assert len(ia) == len(ib) and all(_same_info(x, y) for x, y in zip(ia, ib)), "infos differ"
            episodes_done += len(ia)
            if episodes_done >= 2:   # two auto-resets exercised
                break
        assert episodes_done >= 2, "auto-reset not exercised"
    finally:
        a.close()
        b.close()


def test_vec_auto_reset_semantics():
    cfg = EnvConfig(game="surround", train=False)
    frameskip = GAMES["surround"].frameskip
    vec = VecGames(cfg, num_games=2, num_workers=0, seed=21)
    try:
        vec.reset()
        rng = np.random.default_rng(3)
        steps_since_reset = [0, 0]
        acc = np.zeros(vec.num_slots, dtype=np.float32)
        finished = 0
        for _ in range(2000):
            acts = rng.integers(0, vec.num_actions, size=vec.num_slots)
            obs, rew, dones, infos = vec.step(acts)
            acc += rew
            for i in range(2):
                steps_since_reset[i] += 1
            for info in infos:
                i = info["game"]
                assert dones[2 * i] and dones[2 * i + 1], "both slots of a finished game must be done"
                assert np.array_equal(info["episode_return"], acc[2 * i:2 * i + 2]), "episode return mismatch"
                n = steps_since_reset[i]
                assert frameskip * (n - 1) < info["episode_frames"] <= frameskip * n, (n, info["episode_frames"])
                # returned obs is the first obs of the new episode: the reset filled all 4 stack slots
                for s in (2 * i, 2 * i + 1):
                    assert np.array_equal(obs[s, 0], obs[s, 3]), "obs after auto-reset is not a fresh stack"
                acc[2 * i:2 * i + 2] = 0
                steps_since_reset[i] = 0
                finished += 1
            assert dones.sum() == 2 * len(infos)
            if finished >= 2:
                break
        assert finished >= 2, "expected at least two finished episodes"
    finally:
        vec.close()


def test_speed():
    frameskip = GAMES["surround"].frameskip
    for workers in (1, 2):
        vec = VecGames(EnvConfig(game="surround", train=True), num_games=4, num_workers=workers, seed=0)
        try:
            vec.reset()
            rng = np.random.default_rng(0)
            steps = 200
            acts_all = rng.integers(0, vec.num_actions, size=(steps, vec.num_slots))
            t0 = time.perf_counter()
            for k in range(steps):
                vec.step(acts_all[k])
            dt = time.perf_counter() - t0
            game_steps = steps * vec.num_games
            print(f"  {workers} worker(s), 4 games: {steps / dt:.1f} vec-steps/s, "
                  f"{game_steps / dt:.0f} game-steps/s, {game_steps * frameskip / dt:.0f} frames/s")
        finally:
            vec.close()


def test_bot_seat_in_process_and_workers():
    """Seat 1 = BOT_ACTION (SurroundBot), seat 0 random: both paths agree exactly, and the bot wins most rounds."""
    from fb9.selfplay import BOT_ACTION
    n, steps = 4, 1200
    cfg = EnvConfig(game="surround", train=True)
    runs = {}
    for workers in (0, 2):
        vec = VecGames(cfg, num_games=n, num_workers=workers, seed=5)
        try:
            obs = [vec.reset()]
            rng = np.random.default_rng(1)
            infos_all: list[dict] = []
            for _ in range(steps):
                acts = np.zeros(vec.num_slots, dtype=np.int64)
                acts[0::2] = rng.integers(0, vec.num_actions, size=n)
                acts[1::2] = BOT_ACTION
                o, _, _, infos = vec.step(acts)
                obs.append(o)
                infos_all.extend(infos)
            runs[workers] = (np.stack(obs), infos_all)
        finally:
            vec.close()
    assert np.array_equal(runs[0][0], runs[2][0]), "bot games differ between in-process and worker paths"
    infos = runs[0][1]
    rets = np.array([i["episode_return"] for i in infos])
    bot_wins = int(np.sum(rets[:, 1] > rets[:, 0]))
    print(f"    {len(infos)} finished episodes, bot (seat 1) better in {bot_wins}")
    assert len(infos) >= 8, f"too few episodes finished: {len(infos)}"
    assert bot_wins > 0.7 * len(infos), f"bot better in only {bot_wins}/{len(infos)} episodes"
    assert rets[:, 1].sum() > rets[:, 0].sum(), "bot seat return sum should beat the random seat"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
