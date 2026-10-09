"""SearchBot tests (plain script, see docs/contracts.md §0). Run: container/run.sh python tests/test_search_bot.py

Real games only: SearchBot vs RandomBot (both seats), SearchBot vs SurroundBot (both seats, step-capped), the
never-reverse rule checked against headings inferred from the screen, and the act() timing.
"""
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from fb9.bots import (OWN, RandomBot, S_DOWN, S_LEFT, S_RIGHT, S_UP,  # noqa: E402
                      SurroundBot, parse_grid, _DIR_DELTA, _OPPOSITE)
from fb9.envs import EnvConfig, TwoPlayerGame  # noqa: E402
from fb9.search_bot import SearchBot  # noqa: E402

ACTIONS = (S_UP, S_RIGHT, S_LEFT, S_DOWN)


def _game(seed: int) -> TwoPlayerGame:
    game = TwoPlayerGame(EnvConfig("surround", train=True, sticky_p=0.25, max_delay=0, augment=False), seed)
    game.reset()
    return game


def _play(seed: int, seat: int, opponent: str, max_steps: int, on_step=None):
    """One game: SearchBot at seat, opponent ('random' or 'flood') at the other seat. Returns (won, lost, steps).
    on_step(rgb, action, seat) is called with every SearchBot decision."""
    game = _game(seed)
    me = SearchBot(seed)
    opp = RandomBot(5, seed + 100) if opponent == "random" else SurroundBot(seed)
    won = lost = steps = 0
    done = False
    while not done and steps < max_steps:
        rgb = game.render_rgb()
        acts = np.zeros(2, dtype=np.int64)
        acts[seat] = me.act(rgb, seat)
        acts[1 - seat] = opp.act(rgb, 1 - seat)
        if on_step is not None:
            on_step(rgb, int(acts[seat]), seat)
        _, r, done, _ = game.step(acts)
        steps += 1
        if r[seat] > 0:
            won += 1
        elif r[seat] < 0:
            lost += 1
    return won, lost, steps


def _head(rgb, seat):
    """Own head cell (row, col), or None if the head is not visible."""
    pos = np.argwhere(parse_grid(rgb, seat) == OWN)
    return (int(pos[0][0]), int(pos[0][1])) if len(pos) == 1 else None


def test_never_reverses():
    """Across real game steps the returned action is a direction and never the reverse of the heading, where the
    heading is inferred independently here from the own head's last one-cell move on screen."""
    checked = 0
    cadence = []   # agent steps per own-head cell move (measured on the screen, not from the bot)
    for seat in (0, 1):
        state = {"prev": None, "heading": None, "since_move": 0}

        def on_step(rgb, action, seat=seat, state=state):
            nonlocal checked
            assert action in ACTIONS, f"not a direction: {action}"
            head = _head(rgb, seat)
            if head is not None:
                if state["prev"] is not None and head != state["prev"]:
                    dr, dc = head[0] - state["prev"][0], head[1] - state["prev"][1]
                    if max(abs(dr), abs(dc)) == 1 and (dr == 0 or dc == 0):   # one-cell move
                        state["heading"] = next(a for a, d in _DIR_DELTA.items() if d == (dr, dc))
                        cadence.append(state["since_move"] + 1)
                    else:                                                       # new round: head jumped
                        state["heading"] = None
                    state["since_move"] = 0
                else:
                    state["since_move"] += 1
                state["prev"] = head
            if state["heading"] is not None:
                assert action != _OPPOSITE[state["heading"]], \
                    f"seat {seat}: reversed {state['heading']} -> {action}"
                checked += 1

        _play(seed=21 + seat, seat=seat, opponent="random", max_steps=600, on_step=on_step)
    assert checked > 500, f"too few checked steps: {checked}"
    print(f"    {checked} decisions checked; agent steps per own-head cell move: "
          f"mean {np.mean(cadence):.2f}, min {min(cadence)}, max {max(cadence)}")


def test_beats_random():
    won = lost = 0
    for seed in (31, 32):
        for seat in (0, 1):
            w, l, _ = _play(seed, seat, "random", max_steps=3000)
            won, lost = won + w, lost + l
    rate = won / max(1, won + lost)
    print(f"    SearchBot vs RandomBot: rounds won {won}, lost {lost}, win rate {rate:.2f}")
    assert won + lost >= 8, f"too few rounds decided: {won + lost}"
    assert rate >= 0.95, f"win rate {rate:.2f} < 0.95"


def test_beats_flood_fill_bot():
    """Full games are long (about 5k agent steps), so each game is capped; rounds are counted as they finish."""
    won = lost = 0
    for seed in (41, 42):
        for seat in (0, 1):
            w, l, steps = _play(seed, seat, "flood", max_steps=2000)
            print(f"    seed {seed} seat {seat}: rounds won {w}, lost {l} in {steps} agent steps")
            won, lost = won + w, lost + l
    rate = won / max(1, won + lost)
    print(f"    SearchBot vs SurroundBot: rounds won {won}, lost {lost}, round win rate {rate:.2f}")
    assert won + lost >= 8, f"too few rounds decided: {won + lost}"
    assert rate >= 0.70, f"round win rate {rate:.2f} < 0.70"


def test_act_speed():
    """Mean act() time over real frames (cache hits included), against the flood-fill bot."""
    game = _game(51)
    me, opp = SearchBot(51), SurroundBot(51)
    times = []
    done = False
    while not done and len(times) < 600:
        rgb = game.render_rgb()
        acts = np.zeros(2, dtype=np.int64)
        t0 = time.perf_counter()
        acts[0] = me.act(rgb, 0)
        times.append(time.perf_counter() - t0)
        acts[1] = opp.act(rgb, 1)
        _, _, done, _ = game.step(acts)
    mean_ms = 1000 * float(np.mean(times))
    print(f"    act() over {len(times)} frames: mean {mean_ms:.2f} ms, max {1000 * max(times):.2f} ms")
    assert mean_ms < 10.0, f"mean act() {mean_ms:.2f} ms >= 10 ms"


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
