"""A4: VecGames throughput (random actions, train=True) at several worker counts."""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fb9.envs import EnvConfig, VecGames  # noqa: E402


@dataclass
class Args:
    game: str = "surround"
    workers: list[int] = field(default_factory=lambda: [1, 2])
    games_per_worker: int = 2
    seconds: float = 20.0


def main(args: Args) -> None:
    for w in args.workers:
        env = VecGames(EnvConfig(game=args.game), num_games=w * args.games_per_worker, num_workers=w, seed=0)
        rng = np.random.default_rng(0)
        env.reset()
        for _ in range(10):
            env.step(rng.integers(env.num_actions, size=env.num_slots))
        steps, t0 = 0, time.time()
        while time.time() - t0 < args.seconds:
            env.step(rng.integers(env.num_actions, size=env.num_slots))
            steps += 1
        dt = time.time() - t0
        env.close()
        print(f"{args.game} workers={w} games={env.num_games}: {steps * env.num_slots / dt:,.0f} samples/s, "
              f"{steps * env.num_games * 4 / dt:,.0f} emulator frames/s", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
