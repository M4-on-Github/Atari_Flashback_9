"""Fast 2-player Atari environments (docs/contracts.md §3).

TwoPlayerGame: one emulator, one episode at a time, one agent decision per `frameskip` emulator frames.
VecGames: many TwoPlayerGames stepped in worker processes; observations travel through shared memory.
"""
import ctypes
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass

import numpy as np

from fb9.games import GAMES
from fb9.preprocess import OBS_SHAPE, OBS_SIZE, FrameStack, process_frame

MAX_SEED = 2**31 - 1
# Workers are single-threaded: without this each one starts thread pools sized to the whole node (oversubscription).
_SINGLE_THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
NOISE_BANK = 32   # per-episode, per-player noise bank size (frames draw one entry each)


@dataclass
class EnvConfig:
    game: str = "surround"
    train: bool = True          # False => no sticky, no delay, no augmentation (eval / play)
    sticky_p: float = 0.25
    max_delay: int = 10         # frames; per-player delay d ~ U{0..max_delay}, redrawn each episode
    augment: bool = True        # only applies when train=True
    frameskip: int | None = None  # emulator frames per decision; None = GameSpec.frameskip


class TwoPlayerGame:
    """One Surround or Combat emulator. Does not auto-reset; call reset() before step()."""

    def __init__(self, cfg: EnvConfig, seed: int):
        import multi_agent_ale_py as maap

        self.cfg = cfg
        self.spec = GAMES[cfg.game]
        self.num_actions = self.spec.num_actions
        self.frameskip = cfg.frameskip if cfg.frameskip is not None else self.spec.frameskip
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._ids = self.spec.action_ids
        self._rom = self.spec.rom_path()

        maap.ALEInterface.setLoggerMode("error")
        self.ale = maap.ALEInterface()
        self.ale.setFloat(b"repeat_action_probability", 0.0)  # sticky actions are done here, not by ALE

        self._delay_on = cfg.train and cfg.max_delay > 0
        self._sticky_on = cfg.train and cfg.sticky_p > 0
        self._aug_on = cfg.train and cfg.augment
        self._L = cfg.max_delay + 1 if self._delay_on else 1   # length of the intended-action history
        self._stacks = [FrameStack(seat=0), FrameStack(seat=1)]
        self._ale_acts = np.zeros(2, dtype=np.int32)
        self._over = True

    def reset(self, ale_seed: int | None = None) -> np.ndarray:
        """Start a new episode and return obs (2,6,84,84).

        ale_seed: optional explicit ALE random_seed (used by the PettingZoo equivalence test). By default it is
        drawn from this game's rng. Each episode re-seeds with setInt(random_seed) + loadROM + setMode + reset_game,
        the same sequence PettingZoo uses in reset(seed=...); a fresh ROM load per episode is cheap compared to an
        episode (~5.7k-8k frames).
        """
        if ale_seed is None:
            ale_seed = int(self.rng.integers(0, MAX_SEED))
        self.ale.setInt(b"random_seed", int(ale_seed))
        self.ale.loadROM(self._rom)
        self.ale.setMode(self.spec.mode)
        self.ale.reset_game()

        if self._delay_on:
            self._delay = [int(self.rng.integers(0, self.cfg.max_delay + 1)) for _ in range(2)]
        else:
            self._delay = [0, 0]
        if self._aug_on:
            self._aug = [self._draw_augment() for _ in range(2)]

        self._hist = [[0] * self._L, [0] * self._L]   # intended action indices, ring buffer, NOOP-filled
        self._last_exec = [0, 0]                       # previously executed action index (for sticky)
        self._ep_return = np.zeros(2, dtype=np.float32)
        self.t = 0
        self._over = False

        g = self.ale.getScreenGrayscale()
        frame = process_frame(g, g)
        return np.stack([self._stacks[p].reset(self._augment(p, frame)) for p in (0, 1)])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """Advance self.frameskip emulator frames with the given action indices (2,).

        The screen is read only for the last two frames of the step. If the game ends before those, the screen of the
        last executed frame is used as both prev and last.

        Returns obs (2,6,84,84), rewards (2,) float32 summed over the executed frames, done, info.
        When done, info = {"episode_return": (2,) float32, "episode_frames": int}.
        """
        if self._over:
            raise RuntimeError("episode is over; call reset() first")
        acts = (int(actions[0]), int(actions[1]))
        ids, L, ale_acts = self._ids, self._L, self._ale_acts
        max_frames = self.spec.max_frames
        u = self.rng.random((self.frameskip, 2)).tolist() if self._sticky_on else None
        sticky_p = self.cfg.sticky_p

        rewards = np.zeros(2, dtype=np.float32)
        last = prev = None
        done = False
        for k in range(self.frameskip):
            t = self.t
            for p in (0, 1):
                hist = self._hist[p]
                hist[t % L] = acts[p]
                e = hist[(t - self._delay[p]) % L]          # intended d frames ago (NOOP before the start)
                if u is not None and u[k][p] < sticky_p:
                    e = self._last_exec[p]
                self._last_exec[p] = e
                ale_acts[p] = ids[e]
            rewards += self.ale.act(ale_acts)
            self.t = t + 1
            ended = self.ale.game_over() or self.t >= max_frames
            # screens are only needed for the last two frames of a step (or the current one if the game ends
            # earlier in the step, in which case prev = last = the current screen)
            if k >= self.frameskip - 2 or ended:
                prev, last = last, self.ale.getScreenGrayscale()
            if ended:
                done = True
                break
        if prev is None:   # game ended on the first frame of the step
            prev = last

        frame = process_frame(prev, last)
        obs = np.stack([self._stacks[p].push(self._augment(p, frame)) for p in (0, 1)])
        self._ep_return += rewards
        info: dict = {}
        if done:
            self._over = True
            info = {"episode_return": self._ep_return.copy(), "episode_frames": self.t}
        return obs, rewards, done, info

    def render_rgb(self) -> np.ndarray:
        """Current full-screen RGB frame (H,W,3) uint8."""
        return self.ale.getScreenRGB()

    def _draw_augment(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-episode augmentation for one player: pixel LUT (contrast/brightness) and shifted row/col indices.

        Shift with edge padding equals clamped indexing, and the pointwise contrast/brightness commutes with the
        shift, so the LUT is applied after the gather.
        """
        c = float(self.rng.uniform(0.85, 1.15))
        b = float(self.rng.uniform(-20.0, 20.0))
        dy, dx = int(self.rng.integers(-2, 3)), int(self.rng.integers(-2, 3))
        lut = np.clip((np.arange(256, dtype=np.float32) - 128.0) * c + 128.0 + b, 0.0, 255.0).astype(np.uint8)
        rows = np.clip(np.arange(OBS_SIZE) + dy, 0, OBS_SIZE - 1)[:, None]
        cols = np.clip(np.arange(OBS_SIZE) + dx, 0, OBS_SIZE - 1)
        # per-episode bank of Gaussian noise (sigma=3); each frame adds a randomly chosen bank entry
        bank = np.clip(self.rng.normal(0.0, 3.0, (NOISE_BANK, OBS_SIZE, OBS_SIZE)), -127, 127).astype(np.int16)
        return lut, rows, cols, bank

    def _augment(self, p: int, frame: np.ndarray) -> np.ndarray:
        """Train-time augmentation of one 84x84 frame for player p (identity when not training)."""
        if not self._aug_on:
            return frame
        lut, rows, cols, bank = self._aug[p]
        x = lut[frame[rows, cols]].astype(np.int16)
        x += bank[self.rng.integers(NOISE_BANK)]
        np.clip(x, 0, 255, out=x)
        return x.astype(np.uint8)


class _GameBank:
    """A group of games with global ids. Writes observations into the full (2N,6,84,84) buffer `obs`."""

    def __init__(self, cfg: EnvConfig, game_ids: list[int], seed: int, obs: np.ndarray):
        self.ids = game_ids
        self.games = [TwoPlayerGame(cfg, seed + i) for i in game_ids]
        self.obs = obs

    def reset(self) -> None:
        for i, g in zip(self.ids, self.games):
            self.obs[2 * i:2 * i + 2] = g.reset()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """actions (2*len(ids),) -> rewards, dones (both per slot, in bank order), infos. Auto-resets finished games."""
        rewards = np.zeros(2 * len(self.ids), dtype=np.float32)
        dones = np.zeros(2 * len(self.ids), dtype=bool)
        infos: list[dict] = []
        for j, (i, g) in enumerate(zip(self.ids, self.games)):
            obs, r, done, info = g.step(actions[2 * j:2 * j + 2])
            rewards[2 * j:2 * j + 2] = r
            if done:
                dones[2 * j:2 * j + 2] = True
                infos.append({"game": i, **info})
                obs = g.reset()   # the returned obs is the first obs of the next episode
            self.obs[2 * i:2 * i + 2] = obs
        return rewards, dones, infos


def _worker_main(conn, raw, shape: tuple, cfg: EnvConfig, game_ids: list[int], seed: int) -> None:
    try:
        import cv2
        cv2.setNumThreads(1)
        obs = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        bank = _GameBank(cfg, game_ids, seed, obs)
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                bank.reset()
                conn.send(("ok", None))
            elif cmd == "step":
                conn.send(("ok", bank.step(payload)))
            else:
                break
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class VecGames:
    """num_games independent games, 2 slots each (slot 2i = seat 0 of game i, slot 2i+1 = seat 1).

    num_workers=0 steps everything in-process. Otherwise games are split over spawn-context worker processes that
    write observations into a shared RawArray. Obs returned by reset/step are copies (safe to keep).
    """

    def __init__(self, cfg: EnvConfig, num_games: int, num_workers: int, seed: int):
        assert 1 <= num_games and 0 <= num_workers <= num_games, "need 0 <= num_workers <= num_games"
        self.cfg = cfg
        self.num_games = num_games
        self.num_slots = 2 * num_games
        self.num_actions = GAMES[cfg.game].num_actions
        shape = (self.num_slots,) + OBS_SHAPE
        self._closed = False
        self._workers: list[tuple] = []   # (process, connection, slot indices)
        if num_workers == 0:
            self._obs = np.zeros(shape, dtype=np.uint8)
            self._bank = _GameBank(cfg, list(range(num_games)), seed, self._obs)
        else:
            self._bank = None
            ctx = mp.get_context("spawn")
            self._raw = ctx.RawArray(ctypes.c_uint8, int(np.prod(shape)))
            self._obs = np.frombuffer(self._raw, dtype=np.uint8).reshape(shape)
            saved_env = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
            os.environ.update(_SINGLE_THREAD_ENV)   # inherited by the spawned workers
            for ids in np.array_split(np.arange(num_games), num_workers):
                ids = [int(i) for i in ids]
                slots = np.array([s for i in ids for s in (2 * i, 2 * i + 1)], dtype=np.int64)
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(child_conn, self._raw, shape, cfg, ids, seed),
                                   daemon=True)
                proc.start()
                child_conn.close()
                self._workers.append((proc, parent_conn, slots))
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _recv(self, conn):
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"VecGames worker failed:\n{payload}")
        return payload

    def reset(self) -> np.ndarray:
        if not self._workers:
            self._bank.reset()
        else:
            for _, conn, _ in self._workers:
                conn.send(("reset", None))
            for _, conn, _ in self._workers:
                self._recv(conn)
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        assert actions.shape == (self.num_slots,), actions.shape
        rewards = np.zeros(self.num_slots, dtype=np.float32)
        dones = np.zeros(self.num_slots, dtype=bool)
        infos: list[dict] = []
        if not self._workers:
            rewards, dones, infos = self._bank.step(actions)
        else:
            for _, conn, slots in self._workers:
                conn.send(("step", actions[slots]))
            for _, conn, slots in self._workers:
                r, d, inf = self._recv(conn)
                rewards[slots] = r
                dones[slots] = d
                infos.extend(inf)
        return self._obs.copy(), rewards, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for proc, conn, _ in self._workers:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
            conn.close()
        self._workers = []
