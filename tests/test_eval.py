"""Evaluation tests (plain script, see docs/contracts.md §0). Run: container/run.sh python tests/test_eval.py

Covers the scripted Surround bot (real games and synthetic screens) and an end-to-end evaluate() run on a temporary
root (never the repo's runs/ or checkpoints/).
"""
import glob
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fb9.bots import (BG, SEAT_COLORS, WALL, X0, Y0, CELL_H, CELL_W, NCOLS, NROWS, RandomBot,  # noqa: E402
                      SurroundBot, parse_grid, S_DOWN, S_LEFT, S_RIGHT, S_UP)
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402

from fb9.envs import EnvConfig, TwoPlayerGame  # noqa: E402
from fb9.model import Agent, GridAgent  # noqa: E402
from fb9.evaluate import Args, _load_agent, evaluate  # noqa: E402

torch.set_num_threads(1)
OPPOSITE = {S_UP: S_DOWN, S_DOWN: S_UP, S_LEFT: S_RIGHT, S_RIGHT: S_LEFT}
DELTA = {S_UP: (-1, 0), S_RIGHT: (0, 1), S_LEFT: (0, -1), S_DOWN: (1, 0)}


def paint_screen(own_head, opp_head, trail, seat: int) -> np.ndarray:
    """Synthetic 210x160x3 screen: background, trail cells (wall colour), and 4x9 heads (seat colours)."""
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    rgb[:] = BG.astype(np.uint8)

    def cell(rc, color):
        r, c = rc
        rgb[Y0 + CELL_H * r:Y0 + CELL_H * (r + 1), X0 + CELL_W * c:X0 + CELL_W * (c + 1)] = color.astype(np.uint8)

    for rc in trail:
        cell(rc, WALL)
    if opp_head is not None:
        cell(opp_head, SEAT_COLORS[1 - seat])
    cell(own_head, SEAT_COLORS[seat])
    return rgb


def test_parse_grid_finds_heads():
    for seat in (0, 1):
        rgb = paint_screen((4, 7), (10, 20), [(4, 6), (3, 6)], seat)
        grid = parse_grid(rgb, seat)
        assert grid.shape == (NROWS, NCOLS), grid.shape
        assert grid[4, 7] == 2 and grid[10, 20] == 3, (grid[4, 7], grid[10, 20])
        assert grid[4, 6] == 1 and grid[3, 6] == 1 and grid[0, 0] == 0


def test_surround_bot_beats_random_rounds():
    """Real games: SurroundBot (both seats) vs RandomBot. Rounds are scored by the sign of the per-round reward."""
    won = lost = 0
    for seed in (11, 12):
        for seat in (0, 1):
            game = TwoPlayerGame(EnvConfig("surround", train=False), seed=seed)
            game.reset()
            bot, rnd = SurroundBot(seed), RandomBot(5, seed + 100)
            bot.reset()
            done = False
            while not done:
                rgb = game.render_rgb()
                acts = np.zeros(2, dtype=np.int64)
                acts[seat] = bot.act(rgb, seat)
                acts[1 - seat] = rnd.act(rgb, 1 - seat)
                _, r, done, _ = game.step(acts)
                if r[seat] > 0:
                    won += 1
                elif r[seat] < 0:
                    lost += 1
    rate = won / max(1, won + lost)
    print(f"    SurroundBot vs RandomBot: rounds won {won}, lost {lost}, win rate {rate:.2f}")
    assert won + lost >= 8, f"too few rounds decided: {won + lost}"
    assert rate >= 0.8, f"win rate {rate:.2f} < 0.80"


def test_surround_bot_never_reverses():
    """Synthetic screens: the bot never reverses into its neck, never enters a trail or leaves the grid."""
    for seat in (0, 1):
        bot = SurroundBot(seed=seat)
        head = (9, 10)
        trail: list[tuple[int, int]] = []
        heading = None
        for step in range(40):
            rgb = paint_screen(head, (1, 1) if seat == 0 else (16, 30), trail, seat)
            action = bot.act(rgb, seat)
            assert action in (S_UP, S_RIGHT, S_LEFT, S_DOWN), f"not a direction: {action}"
            if heading is not None:
                assert action != OPPOSITE[heading], f"seat {seat} step {step}: reversed into neck"
            dr, dc = DELTA[action]
            nxt = (head[0] + dr, head[1] + dc)
            assert 0 <= nxt[0] < NROWS and 0 <= nxt[1] < NCOLS, f"seat {seat} step {step}: left the grid"
            assert nxt not in trail, f"seat {seat} step {step}: entered a trail"
            trail.append(head)
            head, heading = nxt, action
        # round restart: cleared board, head back at its start cell -> state is reset and a move is chosen
        action = bot.act(paint_screen((9, 10), None, [], seat), seat)
        assert action in (S_UP, S_RIGHT, S_LEFT, S_DOWN)


def test_evaluate_end_to_end():
    """Two random checkpoints in the train.py format -> state.json, elo.json, TensorBoard file and mp4 appear."""
    root = Path(tempfile.mkdtemp(prefix="fb9_eval_test_"))
    try:
        run = "fake_run"
        ckpt_dir = root / "checkpoints" / run
        ckpt_dir.mkdir(parents=True)
        for samples in (1000, 2000):
            agent = Agent(5)
            state = {"model": agent.state_dict(), "optimizer": {}, "samples": samples, "updates": 1, "pool": [],
                     "args": {}}
            torch.save(state, ckpt_dir / f"ckpt_{samples}.pt")
        state = evaluate(Args(run=run, game="surround", episodes=2, concurrency=4), root=root, max_steps=60)
        eval_dir = root / "runs" / run / "eval"
        assert [e["samples"] for e in state["evaluated"]] == [1000, 2000], state["evaluated"]
        assert (eval_dir / "state.json").exists()
        elo = __import__("json").loads((eval_dir / "elo.json").read_text())
        assert set(elo["ratings"]) == {"ckpt_1000.pt", "ckpt_2000.pt"}, elo
        assert elo["ratings"]["ckpt_1000.pt"] == 1000.0
        assert glob.glob(str(eval_dir / "events.out.tfevents*")), "no TensorBoard event file"
        for samples in (1000, 2000):
            mp4 = root / "runs" / run / "videos" / f"{samples}.mp4"
            assert mp4.exists() and mp4.stat().st_size > 0, mp4
        # a second run evaluates nothing new
        again = evaluate(Args(run=run, game="surround", episodes=2, concurrency=4), root=root, max_steps=60)
        assert len(again["evaluated"]) == 2
        print(f"    end-to-end: elo {elo['ratings']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_evaluate_legacy_pixel_and_grid_checkpoints():
    """A pixel checkpoint without 'obs' in its args loads as pixel Agent; a grid checkpoint evaluates, and it skips
    the pixel checkpoint as an opponent (printed note, no vs_prev games)."""
    root = Path(tempfile.mkdtemp(prefix="fb9_eval_obs_"))
    try:
        run = "obs_run"
        ckpt_dir = root / "checkpoints" / run
        ckpt_dir.mkdir(parents=True)
        pixel = Agent(5)
        torch.save({"model": pixel.state_dict(), "optimizer": {}, "samples": 1000, "updates": 1, "pool": [],
                    "args": {"game": "surround", "frameskip": 15}}, ckpt_dir / "ckpt_1000.pt")
        grid = GridAgent(5)
        torch.save({"model": grid.state_dict(), "optimizer": {}, "samples": 2000, "updates": 1, "pool": [],
                    "args": {"game": "surround", "obs": "grid", "frameskip": 15}}, ckpt_dir / "ckpt_2000.pt")

        agent, frameskip, obs_kind = _load_agent(ckpt_dir / "ckpt_1000.pt", 5)
        assert isinstance(agent, Agent) and frameskip == 15 and obs_kind == "pixels", (type(agent), frameskip, obs_kind)
        agent, frameskip, obs_kind = _load_agent(ckpt_dir / "ckpt_2000.pt", 5)
        assert isinstance(agent, GridAgent) and frameskip == 15 and obs_kind == "grid", (type(agent), obs_kind)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            state = evaluate(Args(run=run, game="surround", episodes=2, concurrency=4), root=root, max_steps=60)
        assert [e["samples"] for e in state["evaluated"]] == [1000, 2000], state["evaluated"]
        first, second = state["evaluated"]
        assert first["vs_random"]["games"] == 2 and first["vs_prev"] == {}, first
        assert second["vs_random"]["games"] == 2 and second["vs_prev"] == {}, second
        assert "skip opponent ckpt_1000.pt: obs pixels != grid of ckpt_2000.pt" in out.getvalue(), out.getvalue()
        assert (root / "runs" / run / "videos" / "2000.mp4").exists()
        elo = json.loads((root / "runs" / run / "eval" / "elo.json").read_text())
        assert set(elo["ratings"]) == {"ckpt_1000.pt", "ckpt_2000.pt"}, elo
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
