"""Self-play bookkeeping (docs/contracts.md §4.2): learner slots, opponent pool, PFSP sampling, winrate EMA.

Torch-free on purpose: the trainer attaches the frozen network to each Snapshot as an opaque `module`.
Games 0..M-1 are mirror games; the next P are pool games with the learner in seat (game index % 2); the last B are
bot games whose opponent is the scripted SurroundBot (played inside the env workers, see BOT_ACTION).
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

WINRATE_ALPHA = 0.1
PFSP_FLOOR = 0.05
BOT_ACTION = -1   # action value for a slot whose seat is played by the game's SurroundBot (VecGames substitutes it)


@dataclass(eq=False)
class Snapshot:
    samples: int             # learner samples at snapshot time (unique id)
    winrate: float = 0.5     # EMA of the learner's results against this snapshot
    module: Any = None       # frozen network, attached by the trainer


class SelfPlay:
    def __init__(self, num_games: int, pool_fraction: float, pool_size: int, seed: int,
                 winrate_alpha: float = WINRATE_ALPHA, bot_fraction: float = 0.0):
        if not 0.0 <= pool_fraction <= 1.0:
            raise ValueError(f"pool_fraction must be in [0,1], got {pool_fraction}")
        if not 0.0 <= bot_fraction <= 1.0:
            raise ValueError(f"bot_fraction must be in [0,1], got {bot_fraction}")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.num_games = num_games
        self.num_pool_games = int(round(pool_fraction * num_games))
        self.num_bot_games = int(round(bot_fraction * num_games))
        if self.num_pool_games + self.num_bot_games > num_games:
            raise ValueError(f"pool_fraction + bot_fraction too large: {self.num_pool_games} pool + "
                             f"{self.num_bot_games} bot games > {num_games} games")
        self.num_mirror_games = num_games - self.num_pool_games - self.num_bot_games
        self.pool_size = pool_size
        self.alpha = winrate_alpha
        self.rng = np.random.default_rng(seed)
        self.pool: list[Snapshot] = []          # sampling candidates, oldest first
        self.opponents: list[Snapshot | None] = [None] * num_games  # per pool game; None = current learner
        self.bot_winrate = 0.5                  # EMA of the learner's results vs SurroundBot (bot games)

        # Per game: -1 = mirror (both seats learner), otherwise the learner's seat.
        self.learner_seat = np.full(num_games, -1, dtype=np.int64)
        pool_ids = np.arange(self.num_mirror_games, self.num_mirror_games + self.num_pool_games)
        self.learner_seat[pool_ids] = pool_ids % 2
        bot_ids = np.arange(num_games - self.num_bot_games, num_games)
        self.learner_seat[bot_ids] = bot_ids % 2

        slots = np.arange(2 * num_games)
        game_of_slot, seat_of_slot = slots // 2, slots % 2
        seat = self.learner_seat[game_of_slot]
        self.learner_slot_mask = (seat == -1) | (seat == seat_of_slot)
        self.learner_slots = np.nonzero(self.learner_slot_mask)[0]
        self.opponent_slots = np.nonzero(~self.learner_slot_mask)[0]
        self.bot_slots = np.asarray([self.opponent_slot_of(g) for g in bot_ids], dtype=np.int64)

    @property
    def num_learner_slots(self) -> int:
        return len(self.learner_slots)

    def is_mirror(self, game: int) -> bool:
        return bool(self.learner_seat[game] == -1)

    def is_bot(self, game: int) -> bool:
        return game >= self.num_games - self.num_bot_games

    def opponent_slot_of(self, game: int) -> int:
        """Slot controlled by the snapshot in a pool game, or by SurroundBot in a bot game."""
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
        for g in range(self.num_mirror_games, self.num_mirror_games + self.num_pool_games):
            self.opponents[g] = self.sample_opponent()

    def opponent_slot_groups(self) -> list[tuple[Snapshot | None, np.ndarray]]:
        """Group opponent slots by the network that controls them (None = current learner)."""
        groups: dict[int, tuple[Snapshot | None, list[int]]] = {}
        for g in range(self.num_mirror_games, self.num_mirror_games + self.num_pool_games):
            opp = self.opponents[g]
            entry = groups.setdefault(id(opp), (opp, []))
            entry[1].append(self.opponent_slot_of(g))
        return [(opp, np.asarray(slots, dtype=np.int64)) for opp, slots in groups.values()]

    def on_episode_end(self, game: int, episode_return: np.ndarray) -> None:
        """Update the winrate of the finished game's opponent, then re-sample a pool game's opponent."""
        if self.is_mirror(game):
            return
        if self.is_bot(game):
            self.bot_winrate = self._ema(self.bot_winrate, game, episode_return)
            return
        opp = self.opponents[game]
        if opp is not None:
            opp.winrate = self._ema(opp.winrate, game, episode_return)
        self.opponents[game] = self.sample_opponent()

    def _ema(self, wr: float, game: int, episode_return: np.ndarray) -> float:
        ret = float(episode_return[self.learner_seat[game]])
        result = 1.0 if ret > 0 else (0.0 if ret < 0 else 0.5)
        return (1.0 - self.alpha) * wr + self.alpha * result

    def pool_stats(self) -> dict[str, float]:
        stats = {"size": 0.0}
        if self.pool:
            wr = np.array([s.winrate for s in self.pool])
            stats = {"size": float(len(self.pool)), "winrate_mean": float(wr.mean()), "winrate_min": float(wr.min())}
        if self.num_bot_games:
            stats["bot_winrate"] = self.bot_winrate
        return stats
