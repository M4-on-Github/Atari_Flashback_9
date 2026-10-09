"""Behaviour-cloning probe: can each architecture learn SurroundBot's space-keeping decision from its observation?

Data: Surround games where both seats follow SurroundBot's choice (epsilon uniform random legal moves). A seat-step
with a visible own head, a known heading and a legal move is a candidate sample, labelled with the per-move bot scores.
Trivial candidates (straight is in the correct set and the legal scores are close) are kept only with probability
keep_trivial, so most kept samples are decisive. Agent(5) learns from pixels, GridAgent(5) from the grid observation,
on the same samples, with policy logits only.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

from fb9.bots import (HeadingTracker, S_DOWN, S_LEFT, S_RIGHT, S_UP, choose_move,  # noqa: E402
                      move_scores, parse_grid)
from fb9.envs import EnvConfig, TwoPlayerGame  # noqa: E402
from fb9.grid import GRID_OBS_SHAPE, GridStack  # noqa: E402
from fb9.model import Agent, GridAgent  # noqa: E402
from fb9.preprocess import OBS_SHAPE  # noqa: E402

N_ACTIONS = 5                       # 0 NOOP (keep heading), then S_UP, S_RIGHT, S_LEFT, S_DOWN
DIRECTIONS = (S_UP, S_RIGHT, S_LEFT, S_DOWN)
TIE_TOL = 1e-6
EVAL_TRAIN_SUBSAMPLE = 20_000
# Workers are single-threaded (same as fb9/envs.py).
_SINGLE_THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
_SPLIT_ID = {"train": 0, "test": 1}


@dataclass
class Args:
    train_samples: int = 100_000   # kept samples
    test_samples: int = 20_000     # kept samples
    workers: int = 12
    epsilon: float = 0.15          # probability of a uniform random legal move per seat-step during data generation
    keep_trivial: float = 0.1      # probability of keeping a trivial candidate (decisive ones are always kept)
    epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    critical_spread: float = 10.0  # a state is decisive if NOOP is not in the correct set, or spread >= this
    seed: int = 0
    cuda: bool = True
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


def _collect(job: tuple[str, int, int, int, float, float, float]) -> dict[str, np.ndarray]:
    """Worker: play games until `quota` kept samples are recorded. Returns arrays of exactly `quota` samples.

    A candidate is kept if it is decisive, or with probability keep_trivial (no random draw when keep_trivial >= 1).
    """
    split, worker, quota, seed, epsilon, keep_trivial, critical_spread = job
    rng = np.random.default_rng([seed, _SPLIT_ID[split], worker])
    pix = np.zeros((quota,) + OBS_SHAPE, dtype=np.uint8)
    grid = np.zeros((quota,) + GRID_OBS_SHAPE, dtype=np.uint8)
    scores = np.zeros((quota, N_ACTIONS), dtype=np.float32)
    heading = np.zeros(quota, dtype=np.int8)
    n = recorded = games = steps = k = 0
    while n < quota:
        game = TwoPlayerGame(EnvConfig("surround", train=False, obs="pixels"), _game_seed(seed, split, worker, k))
        k += 1
        games += 1
        obs = game.reset()
        trackers = [HeadingTracker(), HeadingTracker()]
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
                acts[p] = _seat_action(sc, h, rng, epsilon)
                legal = any(v > -np.inf for v in sc.values())
                if head is not None and h is not None and legal and n < quota:
                    row = np.full(N_ACTIONS, -np.inf, dtype=np.float32)
                    row[0] = sc[h]
                    for a in DIRECTIONS:
                        row[a] = sc[a]
                    recorded += 1
                    if (_decisive_mask(row[None], critical_spread)[0] or keep_trivial >= 1.0
                            or rng.random() < keep_trivial):
                        pix[n], grid[n], scores[n], heading[n] = obs[p], gobs[p], row, h
                        n += 1
            obs, _, done, _ = game.step(acts)
            steps += 1
            if not done:
                gobs = [stacks[p].push(game.render_rgb()) for p in (0, 1)]
    return {"pix": pix, "grid": grid, "scores": scores, "heading": heading,
            "games": np.int64(games), "steps": np.int64(steps), "recorded": np.int64(recorded)}


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
    jobs = [(split, w, q, args.seed, args.epsilon, args.keep_trivial, args.critical_spread)
            for w, q in enumerate(quotas) if q > 0]
    out = {"pix": np.zeros((n_samples,) + OBS_SHAPE, dtype=np.uint8),
           "grid": np.zeros((n_samples,) + GRID_OBS_SHAPE, dtype=np.uint8),
           "scores": np.zeros((n_samples, N_ACTIONS), dtype=np.float32),
           "heading": np.zeros(n_samples, dtype=np.int8)}
    games = steps = recorded = off = 0
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
            del res
    assert off == n_samples, (off, n_samples)
    out["games"], out["steps"], out["recorded"] = games, steps, recorded
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


def _metrics(scores: np.ndarray, pred: np.ndarray, critical_spread: float) -> dict:
    """Metrics of predicted actions (one per sample) against the scores."""
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
    }


def _baselines(scores: np.ndarray, critical_spread: float) -> dict:
    """Always-straight (predict NOOP) and uniform random legal move (exact expectation), on the given samples."""
    st = _legal_stats(scores, critical_spread)
    dec, triv = st["decisive"], ~st["decisive"]
    straight = _metrics(scores, np.zeros(len(scores), dtype=np.int64), critical_spread)
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
    }
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
            f"illegal {m['illegal_rate']:.4f} pred_decisive {m['pred_hist_decisive']}")


def _train_model(name: str, model, train: dict, test: dict, obs_key: str, eval_idx: np.ndarray,
                 args: Args, device, rng: np.random.Generator) -> dict:
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
        test_m = _metrics(test["scores"], _predict(model, test[obs_key], device), args.critical_spread)
        epochs.append({"epoch": epoch, "train_loss": loss, "test": test_m})
        print(f"[{name}] epoch {epoch}/{args.epochs} loss {loss:.4f} | test {_line(test_m)} | "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    train_m = _metrics(scores[eval_idx], _predict(model, obs[eval_idx], device), args.critical_spread)
    print(f"[{name}] train subsample ({len(eval_idx)}): {_line(train_m)}", flush=True)
    return {"params": int(sum(p.numel() for p in model.parameters())), "epochs": epochs, "train": train_m}


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _split_summary(name: str, d: dict, seconds: float, critical_spread: float) -> dict:
    """Print and return one split's generation summary."""
    kept = len(d["scores"])
    dec = float(_decisive_mask(d["scores"], critical_spread).mean())
    print(f"data {name}: {d['steps']} steps ({d['games']} games), kept {kept} of {int(d['recorded'])} candidate "
          f"samples, decisive fraction of kept {dec:.3f}, {seconds:.1f}s", flush=True)
    return {"kept": kept, "candidates": int(d["recorded"]), "games": int(d["games"]), "steps": int(d["steps"]),
            "decisive_fraction": dec, "seconds": seconds}


def run(args: Args) -> dict:
    """Generate data, train both models, write the JSON results and return them."""
    assert 0.0 <= args.keep_trivial <= 1.0, args.keep_trivial
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"device {device}, workers {args.workers}, epsilon {args.epsilon}, keep_trivial {args.keep_trivial}",
          flush=True)

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

    baselines = {"test": _baselines(test["scores"], args.critical_spread)}
    b = baselines["test"]
    print(f"baselines (test): decisive fraction {b['decisive_fraction']:.3f}; "
          f"straight acc_all {b['always_straight']['acc_all']:.3f} acc_decisive "
          f"{_fmt(b['always_straight']['acc_decisive'])}; random legal acc_all {b['random_legal']['acc_all']:.3f} "
          f"acc_decisive {_fmt(b['random_legal']['acc_decisive'])} acc_fork {_fmt(b['random_legal']['acc_fork'])} "
          f"regret_fork {_fmt(b['random_legal']['area_regret_fork'])}", flush=True)

    rng = np.random.default_rng(args.seed)
    eval_idx = np.sort(rng.choice(len(train["scores"]), min(EVAL_TRAIN_SUBSAMPLE, len(train["scores"])),
                                  replace=False))
    models = {}
    for name, obs_key, cls in (("agent_pixels", "pix", Agent), ("grid_agent", "grid", GridAgent)):
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        models[name] = _train_model(name, cls(N_ACTIONS), train, test, obs_key, eval_idx, args, device, rng)
        models[name]["seconds"] = time.perf_counter() - t0

    results = {"args": asdict(args), "data": data | {"samples_per_s": gen_rate},
               "baselines": baselines, "models": models}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)
    return results


if __name__ == "__main__":
    run(tyro.cli(Args))
