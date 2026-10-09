"""Exploiter league (PPO self-play): alternate frozen exploiters against the main agent and the main agent against them.

Each round r: (1) freeze the main agent as target_r{r}; (2) train an exploiter against that target (league games only)
and freeze it into league/exp_r{r}.pt; (3) train the main agent until (r+1) * main_samples_per_round against all
exploiters so far, plus its pool, bot and mirror games. Restart-safe: finished phases are skipped.
CLI: container/run.sh python league.py --init-ckpt checkpoints/surround_v4/latest.pt --run-name surround_league_v1
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from fb9.games import frameskip_from_args, obs_from_args
from train import REPO, Args, atomic_save, train


@dataclass
class LeagueArgs:
    run_name: str = "surround_league_v1"
    init_ckpt: str = ""              # required: main agent's starting weights (e.g. surround_v4 final)
    rounds: int = 5
    main_samples_per_round: int = 4_000_000
    exploiter_samples: int = 2_000_000
    main_lr: float = 1e-4
    exploiter_lr: float = 2.5e-4
    pool_fraction: float = 0.2
    bot_fraction: float = 0.2
    league_fraction: float = 0.25
    num_games: int = 64
    num_steps: int = 128
    minibatch_size: int = 2048
    num_workers: int = 0
    seed: int = 1
    cuda: bool = True


def saved_samples(path: Path) -> int:
    """Samples recorded in a checkpoint file; 0 if the file does not exist."""
    if not path.exists():
        return 0
    return int(torch.load(path, map_location="cpu", weights_only=False)["samples"])


def run_league(la: LeagueArgs, env_factory=None, root: Path | None = None) -> dict:
    if not la.init_ckpt:
        raise ValueError("init_ckpt is required: the main agent's starting weights")
    root = Path(root) if root is not None else REPO
    run_dir = root / "checkpoints" / la.run_name
    targets_dir = run_dir / "targets"
    league_dir = run_dir / "league"
    latest = run_dir / "latest.pt"
    init_args = torch.load(la.init_ckpt, map_location="cpu", weights_only=False)["args"]
    frameskip, obs = frameskip_from_args(init_args), obs_from_args(init_args)
    exploiter_winrates: list[float] = []

    for r in range(la.rounds):
        # ---- 1. frozen target: the main agent at the start of this round ----
        target = targets_dir / f"target_r{r}.pt"
        if not target.exists():
            src = Path(la.init_ckpt) if (r == 0 and not latest.exists()) else latest
            state = torch.load(src, map_location="cpu", weights_only=False)
            atomic_save({"model": state["model"], "args": state["args"], "samples": state["samples"]}, target)
        target_args = torch.load(target, map_location="cpu", weights_only=False)["args"]

        # ---- 2. exploiter: trained against the target, league games only ----
        exp_path = league_dir / f"exp_r{r}.pt"
        if not exp_path.exists():
            summary = train(Args(game="surround", run_name=f"{la.run_name}_exp{r}", init_ckpt=str(target), resume=True,
                                 opponent_ckpts=(str(target),), league_fraction=1.0, pool_fraction=0.0,
                                 bot_fraction=0.0, total_samples=la.exploiter_samples, lr=la.exploiter_lr,
                                 snapshot_every=10**12, checkpoint_every=1_000_000, num_games=la.num_games,
                                 num_steps=la.num_steps, minibatch_size=la.minibatch_size,
                                 num_workers=la.num_workers, seed=la.seed + 1000 * (r + 1), cuda=la.cuda,
                                 frameskip=frameskip_from_args(target_args), obs=obs_from_args(target_args)),
                            env_factory=env_factory, root=root)
            winrate = summary["league"][0]["winrate"]   # the only league entry: the target; EMA of exploiter's results
            exp = torch.load(Path(summary["ckpt_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            atomic_save({"model": exp["model"], "args": exp["args"], "samples": exp["samples"],
                         "winrate_vs_target": winrate}, exp_path)
        winrate = float(torch.load(exp_path, map_location="cpu", weights_only=False)["winrate_vs_target"])
        exploiter_winrates.append(winrate)
        print(f"league round {r}: exploiter winrate vs main {winrate:.2f}", flush=True)

        # ---- 3. main agent: against all exploiters so far, pool, bot and mirror games ----
        stop = (r + 1) * la.main_samples_per_round
        if saved_samples(latest) < stop:
            train(Args(game="surround", run_name=la.run_name, init_ckpt=la.init_ckpt, resume=True,
                       league_dir=str(league_dir) if la.league_fraction > 0 else "",
                       league_fraction=la.league_fraction, pool_fraction=la.pool_fraction,
                       bot_fraction=la.bot_fraction, total_samples=la.rounds * la.main_samples_per_round,
                       stop_samples=stop, lr=la.main_lr, num_games=la.num_games, num_steps=la.num_steps,
                       minibatch_size=la.minibatch_size, num_workers=la.num_workers, seed=la.seed, cuda=la.cuda,
                       frameskip=frameskip, obs=obs, checkpoint_every=2_000_000),
                  env_factory=env_factory, root=root)

    return {"run_name": la.run_name, "exploiter_winrates": exploiter_winrates,
            "main_samples": saved_samples(latest), "main_ckpt_dir": str(run_dir)}


if __name__ == "__main__":
    print(run_league(tyro.cli(LeagueArgs)))
