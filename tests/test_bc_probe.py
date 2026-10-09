"""Tests for move_scores / HeadingTracker / choose_move (fb9/bots.py) and the behaviour-cloning probe (tools/bc_probe.py).
Run: container/run.sh python tests/test_bc_probe.py
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from fb9.bots import (EMPTY, NCOLS, NROWS, OPP, OWN, S_DOWN, S_LEFT, S_RIGHT, S_UP,  # noqa: E402
                      WALL_CODE, SurroundBot, choose_move, move_scores, parse_grid, _DIR_DELTA, _OPPOSITE,
                      RISK_PENALTY, _reachable_area)
from fb9.envs import EnvConfig, TwoPlayerGame  # noqa: E402
import bc_probe  # noqa: E402

def _corridor_grid(opp: tuple[int, int] | None = None) -> np.ndarray:
    """Own head at (9,10) moving right. UP leads into a closed 3-cell pocket, RIGHT into an open 8x20 block, DOWN and
    LEFT are walls. Optional opponent head."""
    g = np.full((NROWS, NCOLS), WALL_CODE, dtype=np.int8)
    g[9:17, 11:31] = EMPTY
    g[8, 10] = g[7, 10] = g[7, 11] = EMPTY   # pocket
    g[9, 10] = OWN
    if opp is not None:
        g[opp] = OPP
    return g

def test_move_scores_hand_grid():
    """Pocket vs open board, walls, the reverse of the heading, and the risk penalty next to the opponent head."""
    scores = move_scores(_corridor_grid(), S_RIGHT)
    assert scores == {S_UP: 3.0, S_RIGHT: 160.0, S_LEFT: -np.inf, S_DOWN: -np.inf}, scores
    assert choose_move(scores, S_RIGHT) == S_RIGHT

    # opponent head at (10,12): RIGHT's first cell (9,11) is next to it (penalty 5, and one cell less of area).
    # UP's cell (8,10) is two rows away, so it has no penalty.
    scores = move_scores(_corridor_grid(opp=(10, 12)), S_RIGHT)
    assert scores == {S_UP: 3.0, S_RIGHT: 154.0, S_LEFT: -np.inf, S_DOWN: -np.inf}, scores

    # heading None: nothing is reversed, but walls still are
    scores = move_scores(_corridor_grid(), None)
    assert scores[S_LEFT] == -np.inf and scores[S_DOWN] == -np.inf and scores[S_RIGHT] == 160.0, scores

    # a reversed heading is excluded: heading UP makes DOWN illegal even where it is open
    g = _corridor_grid()
    g[10, 10] = EMPTY   # open the cell below the head
    assert move_scores(g, S_UP)[S_DOWN] == -np.inf
    assert move_scores(g, S_RIGHT)[S_DOWN] > -np.inf

    # no own head visible: empty dict
    g = _corridor_grid()
    g[9, 10] = EMPTY
    assert move_scores(g, S_RIGHT) == {}

def test_heading_tracker_round_reset():
    """Heading follows one-cell moves, a jump clears it, and a cleared board starts a new round."""
    from fb9.bots import HeadingTracker
    t = HeadingTracker()
    g = _corridor_grid()
    assert t.update(g) == (9, 10) and t.heading is None
    g2 = _corridor_grid()
    g2[9, 10] = EMPTY
    g2[9, 11] = OWN
    g2[9, 10] = WALL_CODE   # the trail left behind
    assert t.update(g2) == (9, 11) and t.heading == S_RIGHT
    g3 = np.full((NROWS, NCOLS), EMPTY, dtype=np.int8)   # new round: board cleared, head elsewhere
    g3[2, 2] = OWN
    assert t.update(g3) == (2, 2) and t.heading is None

def _legacy_act(bot_state: dict, rgb: np.ndarray, seat: int) -> int:
    """Verbatim copy of SurroundBot.act before the move_scores refactor (the reference for the equivalence test)."""
    grid = parse_grid(rgb, seat)
    occupied = int(np.count_nonzero(grid != EMPTY))
    own = np.argwhere(grid == OWN)
    opp = np.argwhere(grid == OPP)
    if occupied < bot_state["prev_occupied"]:
        bot_state.update(prev_head=None, heading=None, prev_occupied=0)
    bot_state["prev_occupied"] = occupied
    heading = bot_state["heading"]
    if len(own) != 1:
        return heading if heading is not None else S_UP
    head = (int(own[0][0]), int(own[0][1]))
    if bot_state["prev_head"] is not None and head != bot_state["prev_head"]:
        dr = head[0] - bot_state["prev_head"][0]
        dc = head[1] - bot_state["prev_head"][1]
        if max(abs(dr), abs(dc)) == 1 and (dr == 0 or dc == 0):
            bot_state["heading"] = next(a for a, d in _DIR_DELTA.items() if d == (dr, dc))
        else:
            bot_state["heading"] = None
    bot_state["prev_head"] = head
    heading = bot_state["heading"]
    opp_head = (int(opp[0][0]), int(opp[0][1])) if len(opp) == 1 else None
    blocked = grid != EMPTY
    blocked[head] = True
    best_action, best_score = None, None
    for action, (dr, dc) in _DIR_DELTA.items():
        if heading is not None and action == _OPPOSITE[heading]:
            continue
        nxt = (head[0] + dr, head[1] + dc)
        if not (0 <= nxt[0] < NROWS and 0 <= nxt[1] < NCOLS) or blocked[nxt]:
            continue
        area = _reachable_area(blocked, nxt)
        risky = opp_head is not None and max(abs(nxt[0] - opp_head[0]), abs(nxt[1] - opp_head[1])) <= 1
        score = (area - (RISK_PENALTY if risky else 0), action == heading)
        if best_score is None or score > best_score:
            best_action, best_score = action, score
    if best_action is None:
        return heading if heading is not None else S_UP
    return best_action

def test_surround_bot_unchanged():
    """SurroundBot's actions match the pre-refactor logic over 300 steps of a fixed-seed game (both seats)."""
    game = TwoPlayerGame(EnvConfig("surround", train=False), seed=5)
    game.reset()
    bots = [SurroundBot(seed=5), SurroundBot(seed=6)]
    legacy = [{"prev_head": None, "heading": None, "prev_occupied": 0} for _ in (0, 1)]
    n = 0
    rounds = 0
    for _ in range(300):
        rgb = game.render_rgb()
        acts = np.zeros(2, dtype=np.int64)
        for p in (0, 1):
            acts[p] = bots[p].act(rgb, p)
            ref = _legacy_act(legacy[p], rgb, p)
            assert acts[p] == ref, f"seat {p} step {n}: new {acts[p]} != legacy {ref}"
        _, _, done, _ = game.step(acts)
        n += 1
        if done:
            rounds += 1
            game.reset()
            for b in bots:
                b.reset()
            legacy = [{"prev_head": None, "heading": None, "prev_occupied": 0} for _ in (0, 1)]
    print(f"    {n} steps x 2 seats identical to the legacy logic ({rounds} episode ends)")

def test_collect_grid_samples():
    """Worker output: grid obs are (6,18,38) in {0,255}, own head visible in the newest frame, pixel obs (6,84,84)."""
    out = bc_probe._collect(("train", 0, 60, 3, 0.15, 0.1, 10.0))
    assert out["grid"].shape == (60, 6, NROWS, NCOLS), out["grid"].shape
    assert int(out["recorded"]) >= 60
    assert out["pix"].shape == (60, 6, 84, 84) and out["pix"].dtype == np.uint8
    assert set(np.unique(out["grid"]).tolist()) <= {0, 255}
    own_now = out["grid"][:, 4].reshape(60, -1).sum(axis=1) // 255          # newest frame, own-head plane
    occ_now = out["grid"][:, 3].reshape(60, -1)
    assert (own_now == 1).all(), "own head not exactly one cell in the newest frame"
    assert ((out["grid"][:, 4] == 255) <= (out["grid"][:, 3] == 255)).all(), "own head outside occupied cells"
    assert (occ_now == 255).sum() > 0
    legal = out["scores"] > -np.inf
    assert legal.any(axis=1).all(), "sample without a legal move"
    assert np.isfinite(out["scores"][legal]).all()
    print(f"    {len(out['scores'])} samples from {int(out['games'])} games; grid obs OK")

def test_keep_trivial_filter():
    """keep_trivial=1.0 keeps every candidate (nothing filtered); 0.1 gives a clearly more decisive kept set."""
    def decisive_frac(keep: float) -> tuple[float, dict]:
        out = bc_probe._collect(("train", 0, 300, 4, 0.15, keep, 10.0))
        assert len(out["scores"]) == 300
        return float(bc_probe._decisive_mask(out["scores"], 10.0).mean()), out

    d_all, unfiltered = decisive_frac(1.0)
    assert int(unfiltered["recorded"]) == 300, "keep_trivial=1.0 must keep every candidate"
    d_kept, filtered = decisive_frac(0.1)
    print(f"    decisive fraction: unfiltered {d_all:.3f}, keep_trivial=0.1 {d_kept:.3f} "
          f"({int(filtered['recorded'])} candidates for 300 kept)")
    assert d_kept > d_all + 0.2, f"filter did not raise the decisive fraction: {d_all:.3f} -> {d_kept:.3f}"

def test_bc_probe_end_to_end():
    """Tiny run: both models, one epoch, CPU, JSON written with sane metrics."""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results.json"
        args = bc_probe.Args(train_samples=600, test_samples=200, workers=2, epochs=1, cuda=False, play_games=0,
                             out=str(out))
        res = bc_probe.run(args)
        data = json.loads(out.read_text())
    assert set(data["models"]) == {"agent_pixels", "grid_agent"}, data["models"].keys()
    for name, m in data["models"].items():
        assert len(m["epochs"]) == 1, name
        t = m["epochs"][0]["test"]
        assert 0.0 <= t["acc_all"] <= 1.0 and 0.0 <= t["illegal_rate"] <= 1.0, (name, t)
        assert m["train"]["n"] == 600, m["train"]["n"]
    base = data["baselines"]["test"]
    rates = [base["decisive_fraction"], base["random_legal"]["acc_all"], base["random_legal"]["illegal_rate"],
             base["always_straight"]["acc_all"], base["always_straight"]["illegal_rate"]]
    for key in ("acc_decisive", "acc_trivial", "area_regret_decisive"):
        v = base["random_legal"][key]
        if key != "area_regret_decisive" and v is not None:
            rates.append(v)
            rates.append(base["always_straight"][key])
    assert all(0.0 <= r <= 1.0 for r in rates), rates
    assert res["data"]["train"]["kept"] == 600 and res["data"]["test"]["kept"] == 200
    assert res["data"]["train"]["candidates"] >= 600
    print(f"    pixels acc_all {data['models']['agent_pixels']['epochs'][0]['test']['acc_all']:.3f}, "
          f"grid acc_all {data['models']['grid_agent']['epochs'][0]['test']['acc_all']:.3f}, "
          f"random legal acc_all {base['random_legal']['acc_all']:.3f}")

def test_search_teacher_collection():
    """teacher="search": -inf exactly on off-grid/occupied directions, row[0] == row[heading], flags and metadata sane."""
    quota = 300
    out = bc_probe._collect(("train", 0, quota, 7, 0.15, 1.0, 10.0, "search"))
    assert len(out["scores"]) == quota
    scores, grid, heading = out["scores"], out["grid"], out["heading"]
    assert set(np.unique(grid).tolist()) <= {0, 255}
    occupied = grid[:, 3] == 255            # newest frame, occupied plane (heads included)
    own = grid[:, 4] == 255
    for i in range(quota):
        r, c = np.argwhere(own[i])[0]
        h = int(heading[i])
        assert scores[i, 0] == scores[i, h], (i, scores[i], h)
        for a in bc_probe.DIRECTIONS:
            nr, nc = r + bc_probe._DIR_DELTA[a][0], c + bc_probe._DIR_DELTA[a][1]
            blocked = not (0 <= nr < NROWS and 0 <= nc < NCOLS) or occupied[i, nr, nc]
            assert np.isneginf(scores[i, a]) == blocked, (i, a, scores[i], blocked)
    assert out["disagree"].dtype == bool
    n_dis = int(out["disagree"].sum())
    assert n_dis > 0, "no disagreeing sample among 300 kept"
    assert (out["head_dist"] >= 1).all(), out["head_dist"].min()
    assert ((out["flood_choice"] >= 1) & (out["flood_choice"] <= 4)).all()
    assert int(out["mismatch"]) >= 0 and int(out["searched"]) >= quota
    print(f"    {quota} search-labelled samples: disagree {n_dis}, near {(out['head_dist'] <= 6).sum()}, "
          f"heading mismatches {int(out['mismatch'])} of {int(out['searched'])} searched states")

def test_bc_probe_search_end_to_end():
    """teacher="search", grid only, one epoch, one play episode per seat and opponent: JSON has the new fields."""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results.json"
        args = bc_probe.Args(train_samples=300, test_samples=150, workers=2, epochs=1, cuda=False, teacher="search",
                             models=("grid",), play_games=1, out=str(out))
        bc_probe.run(args)
        data = json.loads(out.read_text())
        assert set(data["models"]) == {"grid_agent"}, data["models"].keys()
        assert (Path(tmp) / "results_grid.pt").exists()
    t = data["models"]["grid_agent"]["epochs"][0]["test"]
    assert 0.0 <= t["acc_disagree"] <= 1.0 and t["n_disagree"] >= 0 and t["acc_near"] is not None, t
    assert "flood_bot" in data["baselines"]["test"]
    assert data["baselines"]["test"]["flood_bot"]["acc_disagree"] == 0.0
    assert set(data["play"]) == {"vs_search", "vs_flood"}
    for opp in data["play"].values():
        for seat in ("seat0", "seat1"):
            assert opp[seat]["episodes"] == 1
    print(f"    flood_bot acc_all {data['baselines']['test']['flood_bot']['acc_all']:.3f}, "
          f"grid acc_disagree {t['acc_disagree']}, play {data['play']}")

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
