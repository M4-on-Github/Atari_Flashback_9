"""Tests for the Surround grid observation (fb9/grid.py, grid mode in fb9/envs.py, GridAgent in fb9/model.py).
Run: container/run.sh python tests/test_grid.py"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fb9.bots import EMPTY, OPP, OWN, WALL_CODE, SurroundBot, parse_grid  # noqa: E402
from fb9.envs import EnvConfig, TwoPlayerGame, VecGames  # noqa: E402
from fb9.grid import (C_BLUE, C_EMPTY, C_GREEN, C_WALL, GRID_OBS_SHAPE, GridStack,  # noqa: E402
                      classify_cells, grid_planes, obs_shape)
from fb9.model import GridAgent, Agent, make_agent  # noqa: E402


def _bot_frames(seed: int, n_steps: int) -> tuple[list[np.ndarray], int]:
    """RGB screens of a SurroundBot vs SurroundBot game, plus the number of round changes seen on the way."""
    g = TwoPlayerGame(EnvConfig(game="surround", train=False, obs="grid"), seed=seed)
    bots = [SurroundBot(seed=seed), SurroundBot(seed=seed + 1)]
    g.reset()
    for b in bots:
        b.reset()
    frames = [g.render_rgb()]
    for _ in range(n_steps):
        rgb = g.render_rgb()
        acts = np.array([bots[p].act(rgb, p) for p in (0, 1)])
        _, _, done, _ = g.step(acts)
        if done:
            g.reset()
            for b in bots:
                b.reset()
        frames.append(g.render_rgb())
    occupied = [int(np.count_nonzero(parse_grid(f, 0) != EMPTY)) for f in frames]
    round_changes = sum(1 for a, b in zip(occupied, occupied[1:]) if b < a)
    return frames, round_changes


def _expected_codes(rgb: np.ndarray, seat: int) -> np.ndarray:
    """classify_cells must give the same cells as parse_grid, with colours mapped to the blue/green codes."""
    ref = parse_grid(rgb, seat)
    own_code, opp_code = (C_BLUE, C_GREEN) if seat == 0 else (C_GREEN, C_BLUE)
    return np.select([ref == EMPTY, ref == WALL_CODE, ref == OWN, ref == OPP],
                     [C_EMPTY, C_WALL, own_code, opp_code]).astype(np.int8)


def test_classify_matches_parse_grid():
    frames, round_changes = _bot_frames(seed=0, n_steps=400)
    assert len(frames) == 401
    assert round_changes >= 1, "no round change in 400 steps"
    mismatches = 0
    for rgb in frames:
        for seat in (0, 1):
            codes = classify_cells(rgb)
            mismatches += int(np.count_nonzero(codes != _expected_codes(rgb, seat)))
    assert mismatches == 0, f"classify_cells disagrees with parse_grid on {mismatches} cells"
    print(f"    {len(frames)} frames, {round_changes} round changes, classify == parse_grid on every cell")


def test_classify_robust_to_noise():
    frames, _ = _bot_frames(seed=0, n_steps=400)
    rng = np.random.default_rng(0)
    for rgb in frames:
        noise = rng.normal(0.0, 8.0, size=rgb.shape)
        offset = rng.integers(-12, 13, size=3)   # constant per channel for this frame
        noisy = np.clip(np.round(rgb.astype(np.float64) + noise + offset), 0, 255).astype(np.uint8)
        assert np.array_equal(classify_cells(noisy), classify_cells(rgb)), "noise changed a cell class"


def test_grid_stack_shape_order_symmetry():
    frames, _ = _bot_frames(seed=1, n_steps=20)
    a, b, c = frames[0], frames[1], frames[2]
    s0, s1 = GridStack(seat=0), GridStack(seat=1)
    o0 = s0.reset(a)
    assert o0.shape == GRID_OBS_SHAPE == (6, 18, 38) and o0.dtype == np.uint8, (o0.shape, o0.dtype)
    pa0, pa1 = grid_planes(classify_cells(a), 0), grid_planes(classify_cells(a), 1)
    assert np.array_equal(o0, np.concatenate([pa0, pa0]))
    o0 = s0.push(b)
    pb0 = grid_planes(classify_cells(b), 0)
    assert np.array_equal(o0[:3], pa0), "oldest frame must come first"
    assert np.array_equal(o0[3:], pb0), "newest frame must come last"
    o0_copy = o0.copy()
    o0 = s0.push(c)
    assert np.array_equal(o0_copy[3:], o0[:3]), "push must drop the oldest frame"
    assert o0 is not s0.buf, "push must return a copy"
    # seat symmetry on the same frame: occupied equal, own and opponent planes swapped
    p0 = GridStack(seat=0).reset(b)
    p1 = GridStack(seat=1).reset(b)
    assert np.array_equal(p0[0], p1[0]) and np.array_equal(p0[3], p1[3])
    assert np.array_equal(p0[1], p1[2]) and np.array_equal(p0[2], p1[1])
    assert np.array_equal(pa1[1], pa0[2]) and np.array_equal(pa1[2], pa0[1])


def test_two_player_grid_obs():
    g = TwoPlayerGame(EnvConfig(game="surround", train=False, obs="grid"), seed=4)
    assert g.obs_shape == (6, 18, 38) and g.obs_kind == "grid"
    obs = g.reset()
    assert obs.shape == (2, 6, 18, 38) and obs.dtype == np.uint8, obs.shape
    assert np.count_nonzero(obs[0, 4]) == 1 and np.count_nonzero(obs[1, 4]) == 1   # both heads alive at reset
    bots = [SurroundBot(seed=4), SurroundBot(seed=5)]
    for _ in range(40):
        rgb = g.render_rgb()
        acts = np.array([bots[p].act(rgb, p) for p in (0, 1)])
        obs, _, done, _ = g.step(acts)
        assert obs.shape == (2, 6, 18, 38) and obs.dtype == np.uint8
        rgb = g.render_rgb()
        for seat in (0, 1):
            assert np.array_equal(obs[seat, 3:], grid_planes(classify_cells(rgb), seat)), f"seat {seat} newest planes"
            # while both snakes are alive, exactly one own-head pixel in the newest frame
            if len(np.argwhere(parse_grid(rgb, seat) == OWN)) == 1:
                assert np.count_nonzero(obs[seat, 4]) == 1, f"seat {seat}: own-head count"
        if done:
            obs = g.reset()
            bots = [SurroundBot(seed=4), SurroundBot(seed=5)]


def test_invalid_combinations():
    try:
        TwoPlayerGame(EnvConfig(game="combat", obs="grid"), seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("combat + grid must raise ValueError")
    g = TwoPlayerGame(EnvConfig(game="surround", train=False, obs="pixels"), seed=0)
    obs = g.reset()
    assert obs.shape == (2, 6, 84, 84), obs.shape
    obs, _, _, _ = g.step(np.array([1, 2]))
    assert obs.shape == (2, 6, 84, 84)
    assert obs_shape("pixels") == (6, 84, 84) and obs_shape("grid") == (6, 18, 38)
    try:
        obs_shape("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown obs kind must raise ValueError")


def test_vec_grid_two_workers():
    vec = VecGames(EnvConfig(game="surround", train=True, obs="grid"), num_games=4, num_workers=2, seed=3)
    try:
        assert vec.obs_shape == (6, 18, 38)
        obs = vec.reset()
        assert obs.shape == (8, 6, 18, 38) and obs.dtype == np.uint8, obs.shape
        rng = np.random.default_rng(0)
        for _ in range(5):
            acts = rng.integers(0, vec.num_actions, size=vec.num_slots)
            obs, rew, done, infos = vec.step(acts)
            assert obs.shape == (8, 6, 18, 38) and obs.dtype == np.uint8
            assert rew.shape == (8,) and done.shape == (8,)
            assert set(np.unique(obs)) <= {0, 255}
    finally:
        vec.close()


def test_grid_agent_shapes():
    net = GridAgent(num_actions=5)
    x = np.random.default_rng(0).integers(0, 256, size=(3, 6, 18, 38)).astype(np.uint8)
    xt = torch.from_numpy(x)
    logits, value = net(xt)
    assert logits.shape == (3, 5) and value.shape == (3,), (logits.shape, value.shape)
    action, logp, ent, val = net.get_action_and_value(xt)
    assert action.shape == (3,) and logp.shape == (3,) and ent.shape == (3,) and val.shape == (3,)
    assert net.network[-2].out_features == 512 and net.num_actions == 5
    assert isinstance(make_agent("grid", 5), GridAgent)
    assert isinstance(make_agent("pixels", 5), Agent)
    try:
        make_agent("bogus", 5)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown obs kind must raise ValueError")


def test_speed_grid_vs_pixels():
    steps = 300
    for kind in ("grid", "pixels"):
        rng = np.random.default_rng(0)
        g = TwoPlayerGame(EnvConfig(game="surround", train=False, obs=kind), seed=0)
        g.reset()
        t0 = time.perf_counter()
        for _ in range(steps):
            _, _, done, _ = g.step(rng.integers(1, 5, size=2))
            if done:
                g.reset()
        dt = time.perf_counter() - t0
        print(f"    {kind}: {steps / dt:.0f} steps/s ({steps} steps)")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
