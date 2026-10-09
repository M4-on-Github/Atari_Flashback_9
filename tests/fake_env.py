"""FakeVecGames: stand-in for fb9.envs.VecGames with the same contract (docs/contracts.md §3.2), for trainer tests.

Dynamic: each game draws a target v ~ U{0..A-1} per episode and shows it as the brightness of its four stacked
frames. Seat s is rewarded +1 when its action equals (v + s) mod A and -1 otherwise. This is deliberately NOT
zero-sum (a per-seat learnable task so a shared policy's return can rise even in mirror self-play).
Episodes last U{min_len..max_len} agent steps; episode_frames reports that step count.
"""
import numpy as np

from fb9.preprocess import OBS_SHAPE


class FakeVecGames:
    def __init__(self, num_games: int, num_actions: int = 3, seed: int = 0, min_len: int = 20, max_len: int = 60):
        self.num_games = num_games
        self.num_slots = 2 * num_games
        self.num_actions = num_actions
        self.min_len = min_len
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.target = np.zeros(num_games, dtype=np.int64)
        self.length = np.zeros(num_games, dtype=np.int64)
        self.t = np.zeros(num_games, dtype=np.int64)
        self.ret = np.zeros((num_games, 2), dtype=np.float32)
        self.obs = np.zeros((self.num_slots,) + OBS_SHAPE, dtype=np.uint8)

    def _new_episode(self, g: int) -> None:
        self.target[g] = self.rng.integers(self.num_actions)
        self.length[g] = self.rng.integers(self.min_len, self.max_len + 1)
        self.t[g] = 0
        self.ret[g] = 0.0
        brightness = int(255 * (self.target[g] + 1) / (self.num_actions + 1))
        for seat in (0, 1):
            o = self.obs[2 * g + seat]
            o[:4] = brightness
            o[4:] = 0
            o[4 + seat] = 255

    def reset(self) -> np.ndarray:
        for g in range(self.num_games):
            self._new_episode(g)
        return self.obs.copy()

    def step(self, actions: np.ndarray):
        a = np.asarray(actions).reshape(self.num_slots)
        rewards = np.zeros(self.num_slots, dtype=np.float32)
        dones = np.zeros(self.num_slots, dtype=bool)
        infos: list[dict] = []
        for g in range(self.num_games):
            for seat in (0, 1):
                want = (self.target[g] + seat) % self.num_actions
                rewards[2 * g + seat] = 1.0 if a[2 * g + seat] == want else -1.0
            self.ret[g] += rewards[2 * g:2 * g + 2]
            self.t[g] += 1
            if self.t[g] >= self.length[g]:
                dones[2 * g:2 * g + 2] = True
                infos.append({"game": g, "episode_return": self.ret[g].copy(), "episode_frames": int(self.length[g])})
                self._new_episode(g)
        return self.obs.copy(), rewards, dones, infos

    def close(self) -> None:
        pass
