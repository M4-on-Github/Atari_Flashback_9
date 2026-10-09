"""PPO self-play trainer (docs/contracts.md §4.2). CLI: container/run.sh python train.py --game surround [...]"""
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from fb9.games import GAMES
from fb9.model import Agent
from fb9.preprocess import OBS_SHAPE
from fb9.selfplay import SelfPlay

REPO = Path(__file__).resolve().parent


@dataclass
class Args:
    game: str = "surround"
    run_name: str | None = None
    total_samples: int = 50_000_000
    num_games: int = 64
    num_workers: int = 0  # 0 = auto (usable CPUs - 2, min 1)
    num_steps: int = 128
    lr: float = 2.5e-4
    update_epochs: int = 4
    minibatch_size: int = 2048  # fixed size, so gradient steps scale with the batch (~1 step per 512 samples)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    pool_fraction: float = 0.25
    snapshot_every: int = 2_000_000
    pool_size: int = 20
    checkpoint_every: int = 5_000_000
    resume: bool = False
    seed: int = 1
    cuda: bool = True


def auto_num_workers() -> int:
    """Usable CPUs (affinity-aware, SLURM-safe) minus 2, at least 1."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, n - 2)


def pick_device(cuda: bool) -> torch.device:
    return torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")


def atomic_save(obj: Any, path: Path) -> None:
    """torch.save to a temp file in the same directory, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_snapshot_module(state_dict: dict, num_actions: int, device: torch.device) -> Agent:
    """Frozen copy of a network for use as a pool opponent."""
    net = Agent(num_actions).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def make_game_env(args: Args, num_workers: int):
    from fb9.envs import EnvConfig, VecGames  # imported lazily: the real env needs the ALE

    return VecGames(EnvConfig(game=args.game, train=True), args.num_games, num_workers, args.seed)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor,
                gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over (T, L) tensors. dones[t] = the step-t transition ended its episode (no bootstrap across it)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nxt = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def save_checkpoint(ckpt_dir: Path, agent: Agent, optimizer: torch.optim.Optimizer, samples: int, updates: int,
                    selfplay: SelfPlay, args: Args) -> None:
    state = {
        "model": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "samples": samples,
        "updates": updates,
        "pool": [{"samples": s.samples, "winrate": s.winrate} for s in selfplay.pool],
        "args": asdict(args),
    }
    atomic_save(state, ckpt_dir / f"ckpt_{samples}.pt")
    atomic_save(state, ckpt_dir / "latest.pt")


def train(args: Args, env_factory: Callable[[Args, int], Any] | None = None, root: Path | None = None) -> dict:
    """Run PPO self-play. env_factory(args, num_workers) -> VecGames-like env (tests pass a fake).

    Outputs go under `root` (default: repo root): runs/<run_name>/ and checkpoints/<run_name>/.
    Returns a summary dict including the learner's mirror-game seat-0 episode returns.
    """
    if args.game not in GAMES:
        raise ValueError(f"unknown game {args.game!r}; choose from {sorted(GAMES)}")
    root = Path(root) if root is not None else REPO
    run_name = args.run_name or f"{args.game}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = root / "runs" / run_name
    ckpt_dir = root / "checkpoints" / run_name
    pool_dir = ckpt_dir / "pool"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.cuda)

    num_workers = args.num_workers if args.num_workers > 0 else auto_num_workers()
    env = env_factory(args, num_workers) if env_factory is not None else make_game_env(args, num_workers)
    num_actions = env.num_actions

    agent = Agent(num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    selfplay = SelfPlay(args.num_games, args.pool_fraction, args.pool_size, seed=args.seed)

    samples, updates = 0, 0
    if args.resume:
        latest = ckpt_dir / "latest.pt"
        if not latest.exists():
            raise FileNotFoundError(f"--resume set but {latest} does not exist")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        samples, updates = int(ckpt["samples"]), int(ckpt["updates"])
        for entry in ckpt["pool"]:
            snap_state = torch.load(pool_dir / f"snap_{entry['samples']}.pt", map_location=device,
                                    weights_only=False)
            snap = selfplay.add_snapshot(int(entry["samples"]),
                                         make_snapshot_module(snap_state["model"], num_actions, device))
            snap.winrate = float(entry["winrate"])
        print(f"resumed {run_name} at samples={samples} updates={updates} pool={len(selfplay.pool)}")
    # Opponents are not checkpointed (envs restart on resume): pool games draw fresh opponents.
    selfplay.resample_opponents()

    T = args.num_steps
    N = args.num_games
    L = selfplay.learner_slots
    Lc = selfplay.num_learner_slots
    batch_size = Lc * T
    num_updates_total = max(1, math.ceil(args.total_samples / batch_size))
    num_minibatches = max(1, round(batch_size / args.minibatch_size))
    snap_bucket = samples // args.snapshot_every
    ckpt_bucket = samples // args.checkpoint_every

    obs_buf = torch.zeros((T, Lc) + OBS_SHAPE, dtype=torch.uint8, device=device)
    act_buf = torch.zeros((T, Lc), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, Lc), device=device)
    val_buf = torch.zeros((T, Lc), device=device)
    rew_buf = torch.zeros((T, Lc), device=device)
    done_buf = torch.zeros((T, Lc), device=device)

    writer = SummaryWriter(str(run_dir))
    episode_returns: list[float] = []
    recent = deque(maxlen=50)
    obs = env.reset()
    start = time.time()

    while samples < args.total_samples:
        t0 = time.time()
        frac = max(0.0, 1.0 - updates / num_updates_total)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * frac

        # ---- rollout ----
        ep_returns: list[float] = []
        for step in range(T):
            obs_l = torch.from_numpy(obs[L]).to(device)
            obs_buf[step] = obs_l
            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(obs_l)
                actions = np.zeros(2 * N, dtype=np.int64)
                actions[L] = act.cpu().numpy()
                for opp, slots in selfplay.opponent_slot_groups():
                    net = agent if opp is None else opp.module
                    logits, _ = net(torch.from_numpy(obs[slots]).to(device))
                    actions[slots] = Categorical(logits=logits).sample().cpu().numpy()
            act_buf[step], logp_buf[step], val_buf[step] = act, logp, val

            next_obs, rewards, dones, infos = env.step(actions)
            rew_buf[step] = torch.from_numpy(rewards[L]).to(device)
            done_buf[step] = torch.from_numpy(dones[L].astype(np.float32)).to(device)
            for info in infos:
                g = int(info["game"])
                ret = np.asarray(info["episode_return"], dtype=np.float32)
                if selfplay.is_mirror(g):
                    ep_returns.append(float(ret[0]))
                selfplay.on_episode_end(g, ret)
            obs = next_obs

        with torch.no_grad():
            _, next_value = agent(torch.from_numpy(obs[L]).to(device))
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value, args.gamma, args.gae_lambda)

        # ---- PPO update ----
        t_update = time.time()
        b_obs = obs_buf.reshape((-1,) + OBS_SHAPE)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        stats: dict[str, list[float]] = {k: [] for k in ("pg", "v", "ent", "kl", "clipfrac")}
        for _ in range(args.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for mb in torch.tensor_split(perm, num_minibatches):
                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    stats["kl"].append(((ratio - 1) - logratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > args.clip_coef).float().mean().item())
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_unclipped = (newval - b_ret[mb]) ** 2
                v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                stats["pg"].append(pg_loss.item())
                stats["v"].append(v_loss.item())
                stats["ent"].append(ent.item())

        samples += batch_size
        updates += 1
        t_rollout, t_update = t_update - t0, time.time() - t_update
        elapsed = time.time() - t0
        sps = 2 * N * T / max(elapsed, 1e-9)
        mean = {k: float(np.mean(v)) for k, v in stats.items()}
        var_y = torch.var(b_ret)
        ev = float(1 - torch.var(b_ret - b_val) / var_y) if var_y > 0 else 0.0

        if ep_returns:
            episode_returns.extend(ep_returns)
            recent.extend(ep_returns)
            writer.add_scalar("charts/episode_return", float(np.mean(ep_returns)), samples)
        writer.add_scalar("charts/sps", sps, samples)
        writer.add_scalar("charts/rollout_seconds", t_rollout, samples)
        writer.add_scalar("charts/update_seconds", t_update, samples)
        writer.add_scalar("losses/policy_loss", mean["pg"], samples)
        writer.add_scalar("losses/value_loss", mean["v"], samples)
        writer.add_scalar("losses/entropy", mean["ent"], samples)
        writer.add_scalar("losses/approx_kl", mean["kl"], samples)
        writer.add_scalar("losses/clipfrac", mean["clipfrac"], samples)
        writer.add_scalar("losses/explained_variance", ev, samples)
        writer.add_scalar("pool/size", len(selfplay.pool), samples)
        ps = selfplay.pool_stats()
        if len(selfplay.pool) > 0:
            writer.add_scalar("pool/winrate_mean", ps["winrate_mean"], samples)
            writer.add_scalar("pool/winrate_min", ps["winrate_min"], samples)
        recent_mean = float(np.mean(recent)) if recent else float("nan")
        print(f"upd {updates} samples {samples} sps {sps:.0f} ret50 {recent_mean:.2f} pool {len(selfplay.pool)} "
              f"wr {ps.get('winrate_mean', float('nan')):.2f} pg {mean['pg']:.3f} v {mean['v']:.3f} "
              f"ent {mean['ent']:.3f} kl {mean['kl']:.4f} ev {ev:.2f} t {t_rollout:.1f}+{t_update:.1f}s elapsed {time.time() - start:.0f}s",
              flush=True)

        # ---- snapshots and checkpoints (snapshot first so the saved pool list includes it) ----
        if samples // args.snapshot_every > snap_bucket:
            snap_bucket = samples // args.snapshot_every
            cpu_state = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
            atomic_save({"model": cpu_state, "samples": samples}, pool_dir / f"snap_{samples}.pt")
            selfplay.add_snapshot(samples, make_snapshot_module(agent.state_dict(), num_actions, device))
        if samples // args.checkpoint_every > ckpt_bucket:
            ckpt_bucket = samples // args.checkpoint_every
            save_checkpoint(ckpt_dir, agent, optimizer, samples, updates, selfplay, args)

    save_checkpoint(ckpt_dir, agent, optimizer, samples, updates, selfplay, args)
    writer.close()
    env.close()
    return {"run_name": run_name, "run_dir": str(run_dir), "ckpt_dir": str(ckpt_dir), "samples": samples,
            "updates": updates, "batch_size": batch_size, "episode_returns": episode_returns}


if __name__ == "__main__":
    train(tyro.cli(Args))
