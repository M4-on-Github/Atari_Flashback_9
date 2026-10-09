"""Self-play bookkeeping (docs/contracts.md §4.2): learner slots, opponent pool, PFSP sampling, winrate EMA.

Torch-free on purpose: the trainer attaches the frozen network to each Snapshot as an opaque `module`.
Games 0..M-1 are mirror games; games M..N-1 are pool games with the learner in seat (game index % 2).
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

WINRATE_ALPHA = 0.1
PFSP_FLOOR = 0.05


@dataclass(eq=False)
class Snapshot:
    samples: int             # learner samples at snapshot time (unique id)
    winrate: float = 0.5     # EMA of the learner's results against this snapshot
    module: Any = None       # frozen network, attached by the trainer


class SelfPlay:
    def __init__(self, num_games: int, pool_fraction: float, pool_size: int, seed: int,
                 winrate_alpha: float = WINRATE_ALPHA):
        if not 0.0 <= pool_fraction <= 1.0:
            raise ValueError(f"pool_fraction must be in [0,1], got {pool_fraction}")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.num_games = num_games
        self.num_pool_games = int(round(pool_fraction * num_games))
        self.num_mirror_games = num_games - self.num_pool_games
        self.pool_size = pool_size
        self.alpha = winrate_alpha
        self.rng = np.random.default_rng(seed)
        self.pool: list[Snapshot] = []          # sampling candidates, oldest first
        self.opponents: list[Snapshot | None] = [None] * num_games  # per pool game; None = current learner

        # Per game: -1 = mirror (both seats learner), otherwise the learner's seat.
        self.learner_seat = np.full(num_games, -1, dtype=np.int64)
        pool_ids = np.arange(self.num_mirror_games, num_games)
        self.learner_seat[pool_ids] = pool_ids % 2

        slots = np.arange(2 * num_games)
        game_of_slot, seat_of_slot = slots // 2, slots % 2
        seat = self.learner_seat[game_of_slot]
        self.learner_slot_mask = (seat == -1) | (seat == seat_of_slot)
        self.learner_slots = np.nonzero(self.learner_slot_mask)[0]
        self.opponent_slots = np.nonzero(~self.learner_slot_mask)[0]

    @property
    def num_learner_slots(self) -> int:
        return len(self.learner_slots)

    def is_mirror(self, game: int) -> bool:
        return bool(self.learner_seat[game] == -1)

    def opponent_slot_of(self, game: int) -> int:
        """Slot controlled by the snapshot in a pool game."""
        assert not self.is_mirror(game)
        return 2 * game + (1 - int(self.learner_seat[game]))

    def add_snapshot(self, samples: int, module: Any = None) -> Snapshot:
        """Append a snapshot (winrate 0.5); evict the oldest beyond pool_size. Games keep their current opponent."""
        snap = Snapshot(samples=samples, module=module)
        self.pool.append(snap)
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)
        return snap

    def pfsp_weights(self) -> np.ndarray:
        """PFSP-hard weights (1 - winrate)^2 + 0.05 for each pool candidate."""
        wr = np.array([s.winrate for s in self.pool], dtype=np.float64)
        return (1.0 - wr) ** 2 + PFSP_FLOOR

    def sample_opponent(self) -> Snapshot | None:
        """Draw a snapshot by PFSP weight; None (= current learner) while the pool is empty."""
        if not self.pool:
            return None
        w = self.pfsp_weights()
        idx = self.rng.choice(len(self.pool), p=w / w.sum())
        return self.pool[idx]

    def resample_opponents(self) -> None:
        """Assign a fresh opponent to every pool game."""
        for g in range(self.num_mirror_games, self.num_games):
            self.opponents[g] = self.sample_opponent()

    def opponent_slot_groups(self) -> list[tuple[Snapshot | None, np.ndarray]]:
        """Group opponent slots by the network that controls them (None = current learner)."""
        groups: dict[int, tuple[Snapshot | None, list[int]]] = {}
        for g in range(self.num_mirror_games, self.num_games):
            opp = self.opponents[g]
            entry = groups.setdefault(id(opp), (opp, []))
            entry[1].append(self.opponent_slot_of(g))
        return [(opp, np.asarray(slots, dtype=np.int64)) for opp, slots in groups.values()]

    def on_episode_end(self, game: int, episode_return: np.ndarray) -> None:
        """Update the winrate of the finished pool game's opponent, then re-sample that game's opponent."""
        if self.is_mirror(game):
            return
        opp = self.opponents[game]
        if opp is not None:
            ret = float(episode_return[self.learner_seat[game]])
            result = 1.0 if ret > 0 else (0.0 if ret < 0 else 0.5)
            opp.winrate = (1.0 - self.alpha) * opp.winrate + self.alpha * result
        self.opponents[game] = self.sample_opponent()

    def pool_stats(self) -> dict[str, float]:
        if not self.pool:
            return {"size": 0.0}
        wr = np.array([s.winrate for s in self.pool])
        return {"size": float(len(self.pool)), "winrate_mean": float(wr.mean()), "winrate_min": float(wr.min())}
