"""Behaviour-cloning probe: can each architecture learn a bot's space-keeping decision from its observation?

Data: Surround games where both seats follow a teacher (epsilon uniform random legal moves). A seat-step with a visible
own head, a known heading and a legal move is a candidate sample, labelled with the per-move teacher scores. Trivial
candidates (straight is in the correct set and the legal scores are close) are kept only with probability keep_trivial,
so most kept samples are decisive. Agent(5) learns from pixels, GridAgent(5) from the grid observation, on the same
samples, with policy logits only.

teacher="flood": SurroundBot (flood-fill) decides and labels (fb9.bots.move_scores).
teacher="search": SearchBot decides; its root move values (SearchBot.last_values) label the samples. Samples where the
flood-fill bot's chosen move is not in the search correct set are flagged `disagree`, and the flood bot's own
choice is stored as `flood_choice`, so the two teachers can be compared on contested states.

Run: container/run.sh python tools/bc_probe.py --help
"""
import json
import multiprocessing as mp
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

from fb9.bots import (EMPTY, NCOLS, NROWS, OPP, S_DOWN, S_LEFT, S_RIGHT, S_UP, SurroundBot, _DIR_DELTA,  # noqa: E402
                      HeadingTracker, choose_move, move_scores, parse_grid)
from fb9.envs import EnvConfig, TwoPlayerGame  # noqa: E402
from fb9.evaluate import EVAL_ENV  # noqa: E402
from fb9.games import GAMES  # noqa: E402
from fb9.grid import GRID_OBS_SHAPE, GridStack  # noqa: E402
from fb9.model import Agent, GridAgent  # noqa: E402
from fb9.preprocess import OBS_SHAPE  # noqa: E402
from fb9.search_bot import SearchBot  # noqa: E402

N_ACTIONS = 5                       # 0 NOOP (keep heading), then S_UP, S_RIGHT, S_LEFT, S_DOWN
DIRECTIONS = (S_UP, S_RIGHT, S_LEFT, S_DOWN)
TIE_TOL = 1e-6
EVAL_TRAIN_SUBSAMPLE = 20_000
NEAR_DIST = 6                       # "near" states: Manhattan distance between the two heads <= NEAR_DIST
NO_OPP_DIST = 99                    # head_dist when the opponent head is not visible (never "near")
PLAY_WORKER = 900                   # game-seed block of the play evaluation (disjoint from data generation)
SIDE_KEYS = ("disagree", "head_dist", "flood_choice")
# Workers are single-threaded (same as fb9/envs.py).
_SINGLE_THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
_SPLIT_ID = {"train": 0, "test": 1}


@dataclass
class Args:
    train_samples: int = 100_000   # kept samples
    test_samples: int = 20_000     # kept samples
    workers: int = 12
    epsilon: float = 0.15          # probability of a uniform random legal move per seat-step during data generation
    keep_trivial: float = 0.1      # probability of keeping a trivial candidate (decisive or disagreeing ones are kept)
    teacher: Literal["flood", "search"] = "flood"   # who plays and labels the data
    epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    critical_spread: float = 10.0  # a state is decisive if NOOP is not in the correct set, or spread >= this
    seed: int = 0
    cuda: bool = True
    models: tuple[str, ...] = ("pixels", "grid")   # which imitators to train: "pixels" (Agent), "grid" (GridAgent)
    play_games: int = 3            # trained GridAgent episodes per seat vs SearchBot and vs SurroundBot (0 = skip)
    out: str = "runs/bc_probe/results.json"


def _game_seed(seed: int, split: str, worker: int, k: int) -> int:
    """Disjoint game seeds: train and test live in different ranges, each worker has its own block."""
    base = seed * 10**10 + (0 if split == "train" else 5 * 10**9)
    return base + worker * 10**5 + k


def _seat_action(scores: dict[int, float], heading: int | None, rng: np.random.Generator, epsilon: float) -> int:
    """Epsilon-random legal direction, else the bot's choice. Falls back to the heading (S_UP if None)."""
    fallback = heading if heading is not None else S_UP
    legal = [a for a in DIRECTIONS if scores.get(a, -np.inf) > -np.inf]
    if not legal:
        return fallback
    if rng.random() < epsilon:
        return int(legal[rng.integers(len(legal))])
    best = choose_move(scores, heading)
    return fallback if best is None else best


def _flood_row(sc: dict[int, float], heading: int) -> np.ndarray:
    """5-entry label row from the flood scores: row[0] = score of the heading, then the four directions."""
    row = np.full(N_ACTIONS, -np.inf, dtype=np.float32)
    for a in DIRECTIONS:
        row[a] = sc.get(a, -np.inf)
    row[0] = row[heading]
    return row


def _search_row(values: dict[int, float], cells: np.ndarray, head: tuple[int, int], heading: int) -> np.ndarray:
    """5-entry label row from SearchBot root values. Directions that are off the grid or occupied are -inf."""
    row = np.full(N_ACTIONS, -np.inf, dtype=np.float32)
    for a in DIRECTIONS:
        r, c = head[0] + _DIR_DELTA[a][0], head[1] + _DIR_DELTA[a][1]
        if values.get(a) is None or not (0 <= r < NROWS and 0 <= c < NCOLS) or cells[r, c] != EMPTY:
            continue
        row[a] = values[a]
    row[0] = row[heading]
    return row


def _disagrees(search_row: np.ndarray, flood_choice: int) -> bool:
    """True if the flood bot's chosen move is not in the search correct set (over the 5 entries)."""
    cs = search_row >= search_row.max() - TIE_TOL
    return not bool(cs[flood_choice])


def _head_dist(cells: np.ndarray, head: tuple[int, int]) -> int:
    """Manhattan distance between the own head and the opponent head (NO_OPP_DIST if it is not visible)."""
    opp = np.argwhere(cells == OPP)
    if len(opp) != 1:
        return NO_OPP_DIST
    return abs(int(opp[0][0]) - head[0]) + abs(int(opp[0][1]) - head[1])


def _collect(job: tuple) -> dict[str, np.ndarray]:
    """Worker: play games until `quota` kept samples are recorded. Returns arrays of exactly `quota` samples.

    job = (split, worker, quota, seed, epsilon, keep_trivial, critical_spread[, teacher]); teacher defaults to "flood".
    A candidate is kept if it is decisive (on the label row), or with probability keep_trivial (no random draw when
    keep_trivial >= 1). In teacher="search" mode a disagreeing candidate is always kept as well.
    """
    split, worker, quota, seed, epsilon, keep_trivial, critical_spread = job[:7]
    teacher = job[7] if len(job) > 7 else "flood"
    search = teacher == "search"
    rng = np.random.default_rng([seed, _SPLIT_ID[split], worker])
    pix = np.zeros((quota,) + OBS_SHAPE, dtype=np.uint8)
    grid = np.zeros((quota,) + GRID_OBS_SHAPE, dtype=np.uint8)
    scores = np.zeros((quota, N_ACTIONS), dtype=np.float32)
    heading = np.zeros(quota, dtype=np.int8)
    side = {"disagree": np.zeros(quota, dtype=bool), "head_dist": np.zeros(quota, dtype=np.int16),
            "flood_choice": np.zeros(quota, dtype=np.int8)} if search else {}
    n = recorded = games = steps = k = mismatch = searched = 0
    while n < quota:
        gs = _game_seed(seed, split, worker, k)
        game = TwoPlayerGame(EnvConfig("surround", train=False, obs="pixels"), gs)
        k += 1
        games += 1
        obs = game.reset()
        trackers = [HeadingTracker(), HeadingTracker()]
        bots = [SearchBot(seed=2 * gs + p) for p in (0, 1)] if search else None
        stacks = [GridStack(seat=0), GridStack(seat=1)]
        gobs = [stacks[p].reset(game.render_rgb()) for p in (0, 1)]
        done = False
        while not done and n < quota:
            rgb = game.render_rgb()
            acts = np.zeros(2, dtype=np.int64)
            for p in (0, 1):
                cells = parse_grid(rgb, p)
                head = trackers[p].update(cells)
                h = trackers[p].heading
                sc = move_scores(cells, h)
                if not search:
                    acts[p] = _seat_action(sc, h, rng, epsilon)
                    legal = any(v > -np.inf for v in sc.values())
                    if head is not None and h is not None and legal and n < quota:
                        row = _flood_row(sc, h)
                        recorded += 1
                        if (_decisive_mask(row[None], critical_spread)[0] or keep_trivial >= 1.0
                                or rng.random() < keep_trivial):
                            pix[n], grid[n], scores[n], heading[n] = obs[p], gobs[p], row, h
                            n += 1
                else:
                    sb = bots[p]
                    sb_act = sb.act(rgb, p)   # every step, so the bot's heading tracking stays correct
                    legal_dirs = [a for a in DIRECTIONS if sc.get(a, -np.inf) > -np.inf]
                    if legal_dirs and rng.random() < epsilon:
                        acts[p] = int(legal_dirs[rng.integers(len(legal_dirs))])
                    else:
                        acts[p] = sb_act
                    values = sb.last_values
                    if values is not None:
                        searched += 1
                        if sb.heading[0] != h:
                            mismatch += 1
                        elif h is not None and head is not None and sc and n < quota:
                            row = _search_row(values, cells, head, h)
                            if (row[1:] > -np.inf).any():
                                fc = choose_move(sc, h) or 0
                                dis = _disagrees(row, fc)
                                recorded += 1
                                if (_decisive_mask(row[None], critical_spread)[0] or dis
                                        or keep_trivial >= 1.0 or rng.random() < keep_trivial):
                                    pix[n], grid[n], scores[n], heading[n] = obs[p], gobs[p], row, h
                                    side["disagree"][n] = dis
                                    side["head_dist"][n] = _head_dist(cells, head)
                                    side["flood_choice"][n] = fc
                                    n += 1
            obs, _, done, _ = game.step(acts)
            steps += 1
            if not done:
                gobs = [stacks[p].push(game.render_rgb()) for p in (0, 1)]
    out = {"pix": pix, "grid": grid, "scores": scores, "heading": heading,
           "games": np.int64(games), "steps": np.int64(steps), "recorded": np.int64(recorded)}
    out.update(side)
    out["mismatch"], out["searched"] = np.int64(mismatch), np.int64(searched)
    return out


@contextmanager
def _single_threaded_env():
    """Workers inherit these at spawn time; restored afterwards."""
    saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
    os.environ.update(_SINGLE_THREAD_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _generate(split: str, n_samples: int, args: Args) -> dict[str, np.ndarray]:
    """Parallel data generation of n_samples kept samples. Arrays are preallocated and filled as results arrive."""
    quotas = [len(c) for c in np.array_split(np.arange(n_samples), args.workers)]
    jobs = [(split, w, q, args.seed, args.epsilon, args.keep_trivial, args.critical_spread, args.teacher)
            for w, q in enumerate(quotas) if q > 0]
    out = {"pix": np.zeros((n_samples,) + OBS_SHAPE, dtype=np.uint8),
           "grid": np.zeros((n_samples,) + GRID_OBS_SHAPE, dtype=np.uint8),
           "scores": np.zeros((n_samples, N_ACTIONS), dtype=np.float32),
           "heading": np.zeros(n_samples, dtype=np.int8)}
    if args.teacher == "search":
        out |= {"disagree": np.zeros(n_samples, dtype=bool), "head_dist": np.zeros(n_samples, dtype=np.int16),
                "flood_choice": np.zeros(n_samples, dtype=np.int8)}
    games = steps = recorded = mismatch = searched = off = 0
    ctx = mp.get_context("spawn")
    with _single_threaded_env():
        pool = ctx.Pool(len(jobs))
    with pool:
        for res in pool.imap(_collect, jobs):
            m = len(res["scores"])
            for key in out:
                out[key][off:off + m] = res[key]
            off += m
            games += int(res["games"])
            steps += int(res["steps"])
            recorded += int(res["recorded"])
            mismatch += int(res["mismatch"])
            searched += int(res["searched"])
            del res
    assert off == n_samples, (off, n_samples)
    out["games"], out["steps"], out["recorded"] = games, steps, recorded
    out["mismatch"], out["searched"] = mismatch, searched
    return out


def _decisive_mask(scores: np.ndarray, critical_spread: float) -> np.ndarray:
    """Decisive: NOOP (keep heading) is not in the correct set, or best - worst over legal moves >= critical_spread."""
    legal = scores > -np.inf
    smax = scores.max(axis=1)
    noop_correct = scores[:, 0] >= smax - TIE_TOL
    smin_legal = np.where(legal, scores, np.inf).min(axis=1)
    return ~noop_correct | ((smax - smin_legal) >= critical_spread)


def _legal_stats(scores: np.ndarray, critical_spread: float) -> dict[str, np.ndarray]:
    """Per-sample: legal mask, correct-set mask, max score, decisive flag."""
    legal = scores > -np.inf
    smax = scores.max(axis=1)
    correct = scores >= smax[:, None] - TIE_TOL
    return {"legal": legal, "correct": correct, "smax": smax.astype(np.float64),
            "decisive": _decisive_mask(scores, critical_spread),
            "fork": (smax - np.where(legal, scores, np.inf).min(axis=1)) >= critical_spread}


def _opt_mean(x: np.ndarray) -> float | None:
    return float(x.mean()) if x.size else None


def _side(d: dict, idx=None) -> dict:
    """The teacher-comparison arrays of a split (or of a subset idx); empty for the flood teacher."""
    return {k: (d[k] if idx is None else d[k][idx]) for k in SIDE_KEYS if k in d}


def _side_metrics(ok: np.ndarray, side: dict | None) -> dict:
    """Accuracy on disagreeing samples and on near states. ok: per-sample accuracy (bool or float)."""
    if not side or "disagree" not in side:
        return {"acc_disagree": None, "acc_near": None, "n_disagree": None, "n_near": None}
    dis = side["disagree"]
    near = side["head_dist"] <= NEAR_DIST
    return {"acc_disagree": _opt_mean(ok[dis]), "acc_near": _opt_mean(ok[near]),
            "n_disagree": int(dis.sum()), "n_near": int(near.sum())}


def _metrics(scores: np.ndarray, pred: np.ndarray, critical_spread: float, side: dict | None = None) -> dict:
    """Metrics of predicted actions (one per sample) against the scores. side: see _side (optional)."""
    st = _legal_stats(scores, critical_spread)
    idx = np.arange(len(pred))
    ok = st["correct"][idx, pred]
    illegal = ~st["legal"][idx, pred]
    chosen = np.where(illegal, -np.inf, scores[idx, pred].astype(np.float64))
    regret = np.where(illegal, st["smax"] + 1.0, st["smax"] - chosen)
    dec = st["decisive"]
    return {
        "acc_all": float(ok.mean()),
        "acc_decisive": _opt_mean(ok[dec]),
        "acc_trivial": _opt_mean(ok[~dec]),
        "area_regret_decisive": _opt_mean(regret[dec]),
        "acc_fork": _opt_mean(ok[st["fork"]]),
        "area_regret_fork": _opt_mean(regret[st["fork"]]),
        "n_fork": int(st["fork"].sum()),
        "illegal_rate": float(illegal.mean()),
        "n": int(len(pred)),
        "n_decisive": int(dec.sum()),
        "pred_hist_decisive": np.bincount(pred[dec], minlength=N_ACTIONS).tolist(),
    } | _side_metrics(ok, side)


def _baselines(scores: np.ndarray, critical_spread: float, side: dict | None = None) -> dict:
    """Always-straight (predict NOOP) and uniform random legal move (exact expectation), on the given samples."""
    st = _legal_stats(scores, critical_spread)
    dec, triv = st["decisive"], ~st["decisive"]
    straight = _metrics(scores, np.zeros(len(scores), dtype=np.int64), critical_spread, side)
    n_legal = st["legal"].sum(axis=1)
    p_correct = st["correct"].sum(axis=1) / n_legal
    smax = st["smax"][:, None]
    mean_legal_regret = np.where(st["legal"], smax - scores.astype(np.float64), 0.0).sum(axis=1) / n_legal
    random_legal = {
        "acc_all": float(p_correct.mean()),
        "acc_decisive": _opt_mean(p_correct[dec]),
        "acc_trivial": _opt_mean(p_correct[triv]),
        "area_regret_decisive": _opt_mean(mean_legal_regret[dec]),
        "acc_fork": _opt_mean(p_correct[st["fork"]]),
        "area_regret_fork": _opt_mean(mean_legal_regret[st["fork"]]),
        "illegal_rate": 0.0,
    } | _side_metrics(p_correct, side)
    return {"always_straight": {k: straight[k] for k in random_legal},
            "random_legal": random_legal,
            "decisive_fraction": float(dec.mean())}


def _bc_loss(logits, scores):
    """-log(sum of softmax probability over the correct set) per sample."""
    correct = scores >= scores.max(dim=1, keepdim=True).values - TIE_TOL
    masked = logits.masked_fill(~correct, float("-inf"))
    return torch.logsumexp(logits, dim=1) - torch.logsumexp(masked, dim=1)


@torch.no_grad()
def _predict(model, obs: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(obs), batch):
        logits = model(torch.from_numpy(obs[i:i + batch]).to(device))[0]
        out.append(logits.argmax(dim=1).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _line(m: dict) -> str:
    return (f"acc_all {m['acc_all']:.3f} acc_decisive {_fmt(m['acc_decisive'])} "
            f"acc_trivial {_fmt(m['acc_trivial'])} regret_decisive {_fmt(m['area_regret_decisive'])} "
            f"acc_fork {_fmt(m['acc_fork'])} regret_fork {_fmt(m['area_regret_fork'])} "
            f"illegal {m['illegal_rate']:.4f} pred_decisive {m['pred_hist_decisive']} "
            f"acc_disagree {_fmt(m.get('acc_disagree'))} (n {_cnt(m.get('n_disagree'))}) "
            f"acc_near {_fmt(m.get('acc_near'))} (n {_cnt(m.get('n_near'))})")


def _train_model(name: str, model, train: dict, test: dict, obs_key: str, eval_idx: np.ndarray,
                 args: Args, device, rng: np.random.Generator) -> tuple[dict, torch.nn.Module]:
    """Supervised training on policy logits; evaluates on the test set after every epoch, train at the end."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    obs, scores = train[obs_key], train["scores"]
    n = len(scores)
    epochs = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        loss_sum = torch.zeros((), device=device)
        for i in range(0, n, args.batch_size):
            idx = np.sort(perm[i:i + args.batch_size])
            x = torch.from_numpy(obs[idx]).to(device)
            s = torch.from_numpy(scores[idx]).to(device)
            loss = _bc_loss(model(x)[0], s).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
        loss = float(loss_sum) / n
        test_m = _metrics(test["scores"], _predict(model, test[obs_key], device), args.critical_spread, _side(test))
        epochs.append({"epoch": epoch, "train_loss": loss, "test": test_m})
        print(f"[{name}] epoch {epoch}/{args.epochs} loss {loss:.4f} | test {_line(test_m)} | "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    train_m = _metrics(scores[eval_idx], _predict(model, obs[eval_idx], device), args.critical_spread,
                       _side(train, eval_idx))
    print(f"[{name}] train subsample ({len(eval_idx)}): {_line(train_m)}", flush=True)
    res = {"params": int(sum(p.numel() for p in model.parameters())), "epochs": epochs, "train": train_m}
    return res, model


@torch.no_grad()
@torch.no_grad()
def _play_eval(model, device, args: Args) -> dict:
    """Trained imitator vs SearchBot and vs SurroundBot, both seats, sampled actions (softmax, not argmax).

    An episode is one round; a round is won by the seat whose reward is positive. Test-range seeds.
    """
    frameskip = GAMES["surround"].frameskip
    model.eval()
    gen = torch.Generator().manual_seed(args.seed)
    opponents = (("vs_search", lambda s: SearchBot(seed=s)), ("vs_flood", lambda s: SurroundBot(seed=s)))
    out = {}
    for oi, (name, make_bot) in enumerate(opponents):
        out[name] = {}
        for seat in (0, 1):
            won = lost = 0
            for e in range(args.play_games):
                k = 1000 * oi + 100 * seat + e
                gs = _game_seed(args.seed, "test", PLAY_WORKER, k)
                game = TwoPlayerGame(EnvConfig(game="surround", frameskip=frameskip, obs="grid", **EVAL_ENV), gs)
                bot = make_bot(gs)
                obs = game.reset()
                done = False
                while not done:
                    logits = model(torch.from_numpy(obs[seat][None]).to(device))[0]
                    probs = torch.softmax(logits.float(), dim=-1).cpu()
                    a = int(torch.multinomial(probs, 1, generator=gen))
                    opp = bot.act(game.render_rgb(), 1 - seat)
                    acts = np.zeros(2, dtype=np.int64)
                    acts[seat], acts[1 - seat] = a, opp
                    obs, r, done, _ = game.step(acts)
                    won += int(r[seat] > 0)
                    lost += int(r[1 - seat] > 0)
            out[name][f"seat{seat}"] = {"rounds_won": won, "rounds_lost": lost, "episodes": args.play_games}
            print(f"play {name} seat {seat}: rounds won {won} lost {lost} over {args.play_games} games", flush=True)
    model.train()
    return out


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _cnt(v: int | None) -> str:
    return "n/a" if v is None else str(v)


def _split_summary(name: str, d: dict, seconds: float, critical_spread: float) -> dict:
    """Print and return one split's generation summary."""
    kept = len(d["scores"])
    dec = float(_decisive_mask(d["scores"], critical_spread).mean())
    print(f"data {name}: {d['steps']} steps ({d['games']} games), kept {kept} of {int(d['recorded'])} candidate "
          f"samples, decisive fraction of kept {dec:.3f}, {seconds:.1f}s", flush=True)
    res = {"kept": kept, "candidates": int(d["recorded"]), "games": int(d["games"]), "steps": int(d["steps"]),
           "decisive_fraction": dec, "seconds": seconds}
    if "disagree" in d:
        res["disagree_fraction"] = float(d["disagree"].mean())
        res["n_near"] = int((d["head_dist"] <= NEAR_DIST).sum())
        res["heading_mismatches"] = int(d["mismatch"])
        res["searched_states"] = int(d["searched"])
        print(f"data {name}: disagree fraction of kept {res['disagree_fraction']:.3f} "
              f"(n {int(d['disagree'].sum())}), near states {res['n_near']}, heading mismatches "
              f"{res['heading_mismatches']} of {res['searched_states']} searched states", flush=True)
    return res


def run(args: Args) -> dict:
    """Generate data, train the selected models, play the grid imitator (optional), write the JSON and return it."""
    assert 0.0 <= args.keep_trivial <= 1.0, args.keep_trivial
    assert set(args.models) <= {"pixels", "grid"} and args.models, args.models
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"device {device}, workers {args.workers}, teacher {args.teacher}, epsilon {args.epsilon}, "
          f"keep_trivial {args.keep_trivial}", flush=True)

    t0 = time.perf_counter()
    train = _generate("train", args.train_samples, args)
    t_train_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = _generate("test", args.test_samples, args)
    t_test_gen = time.perf_counter() - t0
    data = {"train": _split_summary("train", train, t_train_gen, args.critical_spread),
            "test": _split_summary("test", test, t_test_gen, args.critical_spread)}
    total_kept = args.train_samples + args.test_samples
    gen_rate = total_kept / (t_train_gen + t_test_gen)
    print(f"data generation: {gen_rate:.0f} kept samples/s overall", flush=True)

    baselines = {"test": _baselines(test["scores"], args.critical_spread, _side(test))}
    b = baselines["test"]
    print(f"baselines (test): decisive fraction {b['decisive_fraction']:.3f}; "
          f"straight acc_all {b['always_straight']['acc_all']:.3f} acc_decisive "
          f"{_fmt(b['always_straight']['acc_decisive'])}; random legal acc_all {b['random_legal']['acc_all']:.3f} "
          f"acc_decisive {_fmt(b['random_legal']['acc_decisive'])} acc_fork {_fmt(b['random_legal']['acc_fork'])} "
          f"regret_fork {_fmt(b['random_legal']['area_regret_fork'])}", flush=True)
    if "flood_choice" in test:
        fm = _metrics(test["scores"], test["flood_choice"].astype(np.int64), args.critical_spread, _side(test))
        b["flood_bot"] = fm
        print(f"baselines (test) flood_bot: acc_all {fm['acc_all']:.3f} acc_decisive {_fmt(fm['acc_decisive'])} "
              f"acc_near {_fmt(fm['acc_near'])} (n {fm['n_near']}) acc_disagree {_fmt(fm['acc_disagree'])} "
              f"(n {fm['n_disagree']})", flush=True)

    rng = np.random.default_rng(args.seed)
    eval_idx = np.sort(rng.choice(len(train["scores"]), min(EVAL_TRAIN_SUBSAMPLE, len(train["scores"])),
                                  replace=False))
    specs = [("pixels", "agent_pixels", "pix", Agent), ("grid", "grid_agent", "grid", GridAgent)]
    models, trained = {}, {}
    for key, name, obs_key, cls in specs:
        if key not in args.models:
            continue
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        model = cls(N_ACTIONS)
        res, model = _train_model(name, model, train, test, obs_key, eval_idx, args, device, rng)
        res["seconds"] = time.perf_counter() - t0
        models[name] = res
        trained[key] = model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"args": asdict(args), "data": data | {"samples_per_s": gen_rate},
               "baselines": baselines, "models": models}
    if "grid" in trained:
        ckpt = out.with_name(out.stem + "_grid.pt")
        torch.save(trained["grid"].state_dict(), ckpt)
        results["grid_checkpoint"] = str(ckpt)
        print(f"saved {ckpt}", flush=True)
    if "grid" in trained and args.play_games > 0:
        results["play"] = _play_eval(trained["grid"], device, args)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)
    return results


if __name__ == "__main__":
    run(tyro.cli(Args))
