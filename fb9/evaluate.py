"""Checkpoint evaluator (plan.md §4.8, docs/contracts.md §4.2). Runs in its own process, on CPU.

CLI: container/run.sh python fb9/evaluate.py --run <run> --game surround [--watch] [--episodes 10]

Per checkpoint (oldest first): vs RandomBot, vs SurroundBot and SearchBot (surround only), vs the previous <=3 evaluated checkpoints
(Elo). Every matchup plays `episodes` games, half with the agent in seat 0 and half in seat 1. Outputs:
  runs/<run>/eval/state.json  evaluated checkpoints and their results
  runs/<run>/eval/elo.json    Elo ratings per checkpoint file
  runs/<run>/eval/            TensorBoard scalars at x = checkpoint samples
  runs/<run>/videos/<samples>.mp4
"""
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from fb9.bots import RandomBot, SurroundBot  # noqa: E402
from fb9.envs import EnvConfig, TwoPlayerGame  # noqa: E402
from fb9.games import GAMES, frameskip_from_args, obs_from_args  # noqa: E402
from fb9.model import make_agent  # noqa: E402
from fb9.search_bot import SearchBot  # noqa: E402

ELO_START = 1000.0
ELO_K = 32.0
ELO_OPPONENTS = 3
POLL_SECONDS = 60
VIDEO_SCALE = 3
EVAL_ENV = dict(train=True, sticky_p=0.25, max_delay=0, augment=False)
CHECKPOINT_RE = re.compile(r"^ckpt_(\d+)\.pt$")


@dataclass
class Args:
    run: str
    game: str = "surround"
    watch: bool = False
    episodes: int = 10
    concurrency: int = 8   # games stepped in lock-step, sharing batched network forwards
    seed: int = 1


@dataclass
class Job:
    tag: str                  # "random" | "bot" | "search" | previous checkpoint file name
    agent: torch.nn.Module    # Agent (pixels) or GridAgent (grid), see fb9/model.py
    opp_kind: str             # "random" | "bot" | "search" | "net"
    seat: int                 # the agent's seat (0 = port 1)
    opp_net: torch.nn.Module | None = None


@dataclass
class Episode:
    tag: str
    agent_ret: float
    opp_ret: float


_GEN = torch.Generator().manual_seed(0)


@torch.no_grad()
def _act(requests: list[tuple[torch.nn.Module, np.ndarray]]) -> list[int]:
    """Sampled action (T=1, as at play time, see fb9/policy.py) for each (net, obs uint8) where obs is (6,84,84) for
    a pixel net or (6,18,38) for a grid net; requests sharing a net run as one batch. Not argmax: argmax is weak when
    several actions do the same thing."""
    out = [0] * len(requests)
    groups: dict[int, list[int]] = {}
    nets: dict[int, torch.nn.Module] = {}
    for k, (net, _) in enumerate(requests):
        groups.setdefault(id(net), []).append(k)
        nets[id(net)] = net
    for key, idxs in groups.items():
        obs = torch.from_numpy(np.stack([requests[k][1] for k in idxs]))
        logits, _ = nets[key](obs)
        acts = torch.multinomial(torch.softmax(logits.float(), dim=-1), 1, generator=_GEN)[:, 0]
        for k, a in zip(idxs, acts.tolist()):
            out[k] = int(a)
    return out


def _make_bot(kind: str, num_actions: int, seed: int):
    if kind == "bot":
        return SurroundBot(seed)
    if kind == "search":
        return SearchBot(seed)
    if kind == "random":
        return RandomBot(num_actions, seed)
    return None


@dataclass
class _Slot:
    job_idx: int
    obs: np.ndarray
    bot: object
    steps: int
    ret: np.ndarray


def _run_jobs(jobs: list[Job], game_name: str, frameskip: int, obs_kind: str, concurrency: int,
              max_steps: int | None, seed: int) -> list[Episode]:
    """Play every job to the end of its episode (or max_steps agent steps), up to `concurrency` games at once.
    obs_kind is the agent's observation kind; every job's agent and opponent net must take it."""
    if not jobs:
        return []
    cfg = EnvConfig(game=game_name, frameskip=frameskip, obs=obs_kind, **EVAL_ENV)
    num_actions = GAMES[game_name].num_actions
    games = [TwoPlayerGame(cfg, seed + s) for s in range(min(concurrency, len(jobs)))]
    results: list[Episode | None] = [None] * len(jobs)
    queue: deque[int] = deque(range(len(jobs)))
    active: dict[int, _Slot] = {}

    def start(s: int, j: int) -> None:
        job = jobs[j]
        obs = games[s].reset()
        active[s] = _Slot(j, obs, _make_bot(job.opp_kind, num_actions, seed + j), 0, np.zeros(2, np.float32))

    for s in range(len(games)):
        start(s, queue.popleft())

    while active:
        actions = {s: np.zeros(2, dtype=np.int64) for s in active}
        agent_req, agent_slots, opp_req, opp_slots = [], [], [], []
        for s, st in active.items():
            job = jobs[st.job_idx]
            agent_req.append((job.agent, st.obs[job.seat]))
            agent_slots.append(s)
            if job.opp_kind == "net":
                opp_req.append((job.opp_net, st.obs[1 - job.seat]))
                opp_slots.append(s)
            else:
                actions[s][1 - job.seat] = st.bot.act(games[s].render_rgb(), 1 - job.seat)
        for s, a in zip(agent_slots, _act(agent_req)):
            actions[s][jobs[active[s].job_idx].seat] = a
        for s, a in zip(opp_slots, _act(opp_req)):
            actions[s][1 - jobs[active[s].job_idx].seat] = a

        for s in list(active):
            st = active[s]
            job = jobs[st.job_idx]
            obs, rewards, done, _ = games[s].step(actions[s])
            st.obs = obs
            st.steps += 1
            st.ret += rewards
            if done or (max_steps is not None and st.steps >= max_steps):
                results[st.job_idx] = Episode(job.tag, float(st.ret[job.seat]), float(st.ret[1 - job.seat]))
                del active[s]
                if queue:
                    start(s, queue.popleft())
    return [r for r in results if r is not None]


def _summary(eps: list[Episode]) -> dict:
    """Wins/draws/losses by the sign of the agent's episode return. winrate = (wins + 0.5 draws) / games."""
    diffs = np.array([e.agent_ret - e.opp_ret for e in eps], dtype=np.float64)
    n = max(len(eps), 1)
    wins, draws, losses = int((diffs > 0).sum()), int((diffs == 0).sum()), int((diffs < 0).sum())
    return {"games": len(eps), "wins": wins, "draws": draws, "losses": losses,
            "winrate": float((wins + 0.5 * draws) / n), "scorediff": float(diffs.mean()) if len(eps) else 0.0}


def _elo_update(rating: float, opp_rating: float, score: float) -> float:
    expected = 1.0 / (1.0 + 10 ** ((opp_rating - rating) / 400.0))
    return rating + ELO_K * (score - expected)


def _load_agent(path: Path, num_actions: int) -> tuple[torch.nn.Module, int, str]:
    """Network, the emulator frameskip and the obs kind it was trained with. Checkpoints without an obs are pixels."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_kind = obs_from_args(ckpt["args"])
    agent = make_agent(obs_kind, num_actions)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    return agent, frameskip_from_args(ckpt["args"]), obs_kind


def _write_video(path: Path, game_name: str, agent: torch.nn.Module, opp_kind: str, opp_net: torch.nn.Module | None,
                 max_steps: int | None, seed: int, frameskip: int, obs_kind: str) -> None:
    """One episode, agent in seat 0, one frame per agent step, upscaled nearest-neighbour. The fps makes playback
    run at about real time (60 emulator frames per second)."""
    cfg = EnvConfig(game=game_name, frameskip=frameskip, obs=obs_kind, **EVAL_ENV)
    num_actions = GAMES[game_name].num_actions
    game = TwoPlayerGame(cfg, seed)
    obs = game.reset()
    bot = _make_bot(opp_kind, num_actions, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = game.render_rgb().shape[:2]
    fps = max(1, round(60 / frameskip))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * VIDEO_SCALE, h * VIDEO_SCALE))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    steps, done = 0, False
    while not done:
        rgb = game.render_rgb()
        frame = np.ascontiguousarray(rgb[:, :, ::-1])
        writer.write(cv2.resize(frame, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE, interpolation=cv2.INTER_NEAREST))
        actions = np.zeros(2, dtype=np.int64)
        actions[0] = _act([(agent, obs[0])])[0]
        if opp_kind == "net":
            actions[1] = _act([(opp_net, obs[1])])[0]
        else:
            actions[1] = bot.act(rgb, 1)
        obs, _, done, _ = game.step(actions)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    rgb = game.render_rgb()
    writer.write(cv2.resize(np.ascontiguousarray(rgb[:, :, ::-1]), None, fx=VIDEO_SCALE, fy=VIDEO_SCALE,
                            interpolation=cv2.INTER_NEAREST))
    writer.release()


def _pending(ckpt_dir: Path, state: dict) -> list[Path]:
    """Checkpoint files not yet evaluated, oldest (lowest samples) first."""
    done = {e["file"] for e in state["evaluated"]}
    found = []
    if ckpt_dir.exists():
        for p in ckpt_dir.iterdir():
            m = CHECKPOINT_RE.match(p.name)
            if m and p.name not in done:
                found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def _read_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _evaluate_checkpoint(args: Args, ckpt_dir: Path, videos_dir: Path, path: Path, state: dict, elo: dict,
                         writer: SummaryWriter, max_steps: int | None) -> dict:
    t0 = time.time()
    num_actions = GAMES[args.game].num_actions
    samples = int(CHECKPOINT_RE.match(path.name).group(1))
    agent, frameskip, obs_kind = _load_agent(path, num_actions)

    prev = [e for e in state["evaluated"] if (ckpt_dir / e["file"]).exists()][-ELO_OPPONENTS:]
    loaded = {e["file"]: _load_agent(ckpt_dir / e["file"], num_actions) for e in prev}
    for e in list(prev):
        if loaded[e["file"]][1] != frameskip:
            print(f"skip opponent {e['file']}: frameskip {loaded[e['file']][1]} != {frameskip} of {path.name}",
                  flush=True)
            prev.remove(e)
        elif loaded[e["file"]][2] != obs_kind:
            print(f"skip opponent {e['file']}: obs {loaded[e['file']][2]} != {obs_kind} of {path.name}", flush=True)
            prev.remove(e)
    prev_nets = {e["file"]: loaded[e["file"]][0] for e in prev}

    jobs: list[Job] = []
    for i in range(args.episodes):
        jobs.append(Job("random", agent, "random", i % 2))
    if args.game == "surround":
        for kind in ("bot", "search"):
            for i in range(args.episodes):
                jobs.append(Job(kind, agent, kind, i % 2))
    for e in prev:
        for i in range(args.episodes):
            jobs.append(Job(e["file"], agent, "net", i % 2, prev_nets[e["file"]]))
    episodes = _run_jobs(jobs, args.game, frameskip, obs_kind, args.concurrency, max_steps, args.seed)
    groups: dict[str, list[Episode]] = {}
    for ep in episodes:
        groups.setdefault(ep.tag, []).append(ep)

    # Elo: start from the mean rating of the opponents, then play each opponent's games with its rating fixed.
    ratings = elo["ratings"]
    rating = float(np.mean([ratings.get(e["file"], ELO_START) for e in prev])) if prev else ELO_START
    vs_prev = {}
    for e in prev:
        eps = groups.get(e["file"], [])
        opp_rating = ratings.get(e["file"], ELO_START)
        for ep in eps:
            score = 1.0 if ep.agent_ret > ep.opp_ret else 0.0 if ep.agent_ret < ep.opp_ret else 0.5
            rating = _elo_update(rating, opp_rating, score)
        vs_prev[e["file"]] = _summary(eps)
    ratings[path.name] = rating

    random_s = _summary(groups.get("random", []))
    bot_s = _summary(groups["bot"]) if args.game == "surround" else None
    search_s = _summary(groups["search"]) if args.game == "surround" else None
    entry = {"file": path.name, "samples": samples, "vs_random": random_s, "vs_bot": bot_s, "vs_search": search_s,
             "vs_prev": vs_prev, "elo": rating}

    writer.add_scalar("eval/winrate_vs_random", random_s["winrate"], samples)
    writer.add_scalar("eval/scorediff_vs_random", random_s["scorediff"], samples)
    if bot_s is not None:
        writer.add_scalar("eval/winrate_vs_bot", bot_s["winrate"], samples)
        writer.add_scalar("eval/scorediff_vs_bot", bot_s["scorediff"], samples)
        writer.add_scalar("eval/winrate_vs_search", search_s["winrate"], samples)
        writer.add_scalar("eval/scorediff_vs_search", search_s["scorediff"], samples)
    writer.add_scalar("eval/elo", rating, samples)
    writer.flush()

    # one video: vs the bot (surround) or vs the previous checkpoint (combat; vs random if there is none yet)
    if args.game == "surround":
        video_opp = ("bot", None)
    elif prev:
        video_opp = ("net", prev_nets[prev[-1]["file"]])
    else:
        video_opp = ("random", None)
    _write_video(videos_dir / f"{samples}.mp4", args.game, agent, video_opp[0], video_opp[1], max_steps, args.seed,
                 frameskip, obs_kind)

    prev_txt = " ".join(f"{k}:{v['wins']}/{v['draws']}/{v['losses']}" for k, v in vs_prev.items()) or "-"
    bot_txt = (f"bot W/D/L {bot_s['wins']}/{bot_s['draws']}/{bot_s['losses']} wr {bot_s['winrate']:.2f} "
               f"sd {bot_s['scorediff']:+.2f} | search W/D/L {search_s['wins']}/{search_s['draws']}/"
               f"{search_s['losses']} wr {search_s['winrate']:.2f} sd {search_s['scorediff']:+.2f} | ") if bot_s else ""
    print(f"eval {path.name} samples {samples} | random W/D/L {random_s['wins']}/{random_s['draws']}/"
          f"{random_s['losses']} wr {random_s['winrate']:.2f} sd {random_s['scorediff']:+.2f} | {bot_txt}"
          f"elo {rating:.0f} | prev {prev_txt} | {time.time() - t0:.0f}s", flush=True)
    return entry


def evaluate(args: Args, root: Path | None = None, max_steps: int | None = None) -> dict:
    """Evaluate every not-yet-evaluated checkpoint of args.run (repeatedly if args.watch).

    root: base directory holding checkpoints/ and runs/ (default: repo root). max_steps: optional cap on agent steps
    per episode (tests and quick checks; a capped episode is scored by the sign of its return so far).
    """
    if args.game not in GAMES:
        raise ValueError(f"unknown game {args.game!r}; choose from {sorted(GAMES)}")
    root = Path(root) if root is not None else REPO
    ckpt_dir = root / "checkpoints" / args.run
    run_dir = root / "runs" / args.run
    eval_dir = run_dir / "eval"
    state_path, elo_path = eval_dir / "state.json", eval_dir / "elo.json"
    state = _read_json(state_path, {"evaluated": []})
    elo = _read_json(elo_path, {"K": ELO_K, "start": ELO_START, "ratings": {}})
    writer = SummaryWriter(str(eval_dir))
    try:
        while True:
            for path in _pending(ckpt_dir, state):
                entry = _evaluate_checkpoint(args, ckpt_dir, run_dir / "videos", path, state, elo, writer, max_steps)
                state["evaluated"].append(entry)
                _write_json(state_path, state)
                _write_json(elo_path, elo)
            if not args.watch:
                break
            time.sleep(POLL_SECONDS)
    finally:
        writer.close()
    return state


def main() -> None:
    torch.set_num_threads(2)
    evaluate(tyro.cli(Args))


if __name__ == "__main__":
    main()
