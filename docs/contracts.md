# Interface contracts

These are fixed. Code must match them exactly so that separately written modules fit together.
If something here is ambiguous or seems wrong, **stop and ask** — do not guess or change it.

## 0. Ground rules

- Python 3.11, runs **only inside the container**: `container/run.sh python ...` from the repo root
  (`/home/mmyatmau/Atari_Flashback_9`). Never pip/uv/conda install anything; never submit SLURM jobs.
- Available in the container: torch 2.5.1 (CUDA), numpy, opencv-python-headless (`cv2`), gymnasium 0.29.1,
  pettingzoo 1.24.3, supersuit, multi_agent_ale_py, pygame, tensorboard, pyserial, tyro. **No pytest.**
- Tests are plain scripts: `tests/test_<thing>.py` with `def test_*()` functions and a
  `if __name__ == "__main__":` block that runs every `test_*` and prints `PASS`/`FAIL` per test, exiting non-zero on
  failure. Run with `container/run.sh python tests/test_<thing>.py`.
- The dev machine has **2 CPUs and 4 GB RAM**, shared with other agents. Tests use ≤ 2 worker processes, ≤ 4 games,
  and finish in < 2 minutes.
- CLIs use `tyro` with a dataclass of args.
- Style: type hints, short docstrings, no unnecessary abstraction.

## 1. Game facts (measured)

| | Surround | Combat Tank |
|---|---|---|
| ROM | `Path(multi_agent_ale_py.__file__).parent / "roms" / "surround.bin"` | `.../roms/combat.bin` (PAL version) |
| ALE mode | 1 | **2** (classic Tank, maze, straight shots) |
| Screen (H×W) | 210×160 | 256×160 |
| Actions (index → ALE action id) | `[0, 2, 3, 4, 5]` = NOOP, UP, RIGHT, LEFT, DOWN | `list(range(18))` (full set) |
| Episode | ends at `ale.game_over()`; ~5.7k frames random play | fixed 8181 frames (timer) |
| Reward | ±1 per round, zero-sum | ±1 per hit, zero-sum |

- Player 0 = PettingZoo `first_0` = ALE player A = **joystick port 1**. Player 1 = `second_0` = **port 2**.
- `ale.getScreenGrayscale()` is bit-identical to `cv2.cvtColor(ale.getScreenRGB(), cv2.COLOR_RGB2GRAY)`.

ALE setup (copy exactly):

```python
import multi_agent_ale_py as maap
maap.ALEInterface.setLoggerMode("error")
ale = maap.ALEInterface()
ale.setFloat(b"repeat_action_probability", 0.0)   # we do sticky actions ourselves
ale.setInt(b"random_seed", seed)                   # must be set BEFORE loadROM; int32 range
ale.loadROM(str(rom_path))
ale.setMode(mode)
ale.reset_game()
rewards = ale.act(np.array([ale_id_p0, ale_id_p1], dtype=np.int32))  # one emulator frame, returns per-player rewards
ale.game_over(); ale.getScreenGrayscale(); ale.getScreenRGB()
```

`fb9/games.py` holds these facts:

```python
@dataclass(frozen=True)
class GameSpec:
    name: str              # "surround" | "combat"
    rom: str               # "surround" | "combat"
    mode: int
    action_ids: tuple[int, ...]   # index -> ALE action id
    action_names: tuple[str, ...] # index -> e.g. "UP", "UPFIRE"
    max_frames: int        # truncation safety: 108_000 for both

GAMES: dict[str, GameSpec]       # keys "surround", "combat"
```

ALE action ids and names (full set): 0 NOOP, 1 FIRE, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN, 6 UPRIGHT, 7 UPLEFT,
8 DOWNRIGHT, 9 DOWNLEFT, 10 UPFIRE, 11 RIGHTFIRE, 12 LEFTFIRE, 13 DOWNFIRE, 14 UPRIGHTFIRE, 15 UPLEFTFIRE,
16 DOWNRIGHTFIRE, 17 DOWNLEFTFIRE.

`fb9/games.py` also provides `action_to_bitmask(name: str) -> int` for the console bridge:
bit0 UP, bit1 DOWN, bit2 LEFT, bit3 RIGHT, bit4 FIRE (e.g. "UPLEFTFIRE" → 0b10101, "NOOP" → 0).

## 2. Preprocessing — `fb9/preprocess.py` (shared by training, PC play, console play)

One agent decision every **`GameSpec.frameskip` emulator frames** ("a step"): **15** for Surround (one cell move), **4** for Combat. `EnvConfig.frameskip` overrides it. Old checkpoints without a recorded frameskip use 4.

```python
OBS_SIZE = 84
STACK = 4
OBS_SHAPE = (6, 84, 84)        # 4 stacked frames + 2 seat-indicator planes, uint8

def to_gray(rgb: np.ndarray) -> np.ndarray            # HxWx3 uint8 RGB -> HxW uint8 (cv2 RGB2GRAY)
def process_frame(prev_gray, last_gray) -> np.ndarray  # max(prev,last) -> cv2.resize to 84x84 INTER_AREA -> uint8
class FrameStack:
    def __init__(self, seat: int)                      # seat 0 = port 1 / first_0, seat 1 = port 2 / second_0
    def reset(self, frame84: np.ndarray) -> np.ndarray # fill all 4 slots with frame84, return obs (6,84,84)
    def push(self, frame84: np.ndarray) -> np.ndarray  # drop oldest, append, return obs (6,84,84)
```

- `prev_gray`/`last_gray` are the full-screen grayscale frames of the **last two emulator frames of the step**
  (the last two frames of the step, i.e. frames `frameskip-1` and `frameskip`). No cropping; the whole emulator screen is resized to 84×84.
- Obs channel order: `[oldest, ..., newest, seat0_plane, seat1_plane]`. Seat planes are all-255 for the matching seat
  and all-0 for the other (seat 0 → ch4=255, ch5=0; seat 1 → ch4=0, ch5=255).
- The obs returned is a **new array** (callers may keep it).

## 3. Environment — `fb9/envs.py`

### 3.1 Single game: `TwoPlayerGame`

```python
@dataclass
class EnvConfig:
    game: str = "surround"
    train: bool = True          # False => no sticky, no delay, no augmentation (eval / play)
    sticky_p: float = 0.25
    max_delay: int = 10         # frames; per-player delay d ~ U{0..max_delay}, redrawn each episode
    augment: bool = True        # only applies when train=True

class TwoPlayerGame:
    def __init__(self, cfg: EnvConfig, seed: int)
    num_actions: int
    def reset(self) -> np.ndarray                      # obs (2,6,84,84) uint8: [seat0_obs, seat1_obs]
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]
        # actions: (2,) int action *indices*; returns obs (2,6,84,84), rewards (2,) float32 summed over the step's frames,
        # done (game over or max_frames reached), info. Does NOT auto-reset.
        # When done, info = {"episode_return": (2,) float32, "episode_frames": int}
    def render_rgb(self) -> np.ndarray                 # current full-screen RGB (for videos)
```

Per emulator frame, for each player independently (train mode only):

1. **Delay:** the action executed this frame is the action the agent chose `d` frames ago (a FIFO of per-frame
   intended actions, pre-filled with NOOP at reset). `d=0` means no delay.
2. **Sticky:** with probability `sticky_p`, execute the previously *executed* action instead (ALE v5 semantics).

**Augmentation** (train only, parameters drawn per player per episode, applied to each 84×84 frame before it enters
the stack): `x' = clip(contrast*(x-128)+128 + brightness)` with contrast ∈ U[0.85,1.15], brightness ∈ U[-20,20];
integer shift (dy,dx) ∈ U{-2..2}² with edge padding; Gaussian noise σ=3 (redrawn per frame). Seat planes are never
augmented.

**Seeding:** each episode uses a fresh ALE `random_seed` drawn from the game's own `np.random.Generator(seed)`; reload
is not needed — call `ale.setInt(b"random_seed", s)` then `ale.loadROM`+`setMode`+`reset_game` (or keep one ALE
and only reset if re-seeding isn't required; document which you chose and why).

### 3.2 Vectorized: `VecGames`

```python
class VecGames:
    def __init__(self, cfg: EnvConfig, num_games: int, num_workers: int, seed: int)
    num_games: int; num_slots: int  # = 2*num_games
    num_actions: int
    def reset(self) -> np.ndarray                                   # (2N,6,84,84) uint8
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]
        # actions (2N,) int; returns obs (2N,6,84,84) uint8, rewards (2N,) float32, dones (2N,) bool, infos
        # Slot 2i = seat 0 of game i, slot 2i+1 = seat 1 of game i.
        # AUTO-RESET: if game i finished, dones[2i]=dones[2i+1]=True, rewards are the final step's rewards,
        # obs[2i:2i+2] is the FIRST obs of the new episode, and infos contains
        # {"game": i, "episode_return": (2,), "episode_frames": int} for each finished game.
    def close(self)
```

- Worker processes (`multiprocessing` context **"spawn"**, so it is safe after CUDA init), each owning
  `num_games/num_workers` games, stepping them serially. Obs travel through **shared memory** (one
  `multiprocessing.shared_memory` / `RawArray` buffer viewed as a numpy array); only actions/rewards/dones/infos go
  through pipes. Game `i` in worker `w` uses seed `seed + i`.
- `num_workers=0` runs everything in-process (for tests and debugging).

### 3.3 Tests — `tests/test_envs.py`

- Shapes/dtypes; seat planes correct; obs[0] vs obs[1] stacks identical when `train=False`.
- `train=False` with fixed actions is deterministic for a fixed seed.
- **Equivalence vs PettingZoo** (`train=False`): for the same ALE seed, mode and action sequence, our rewards and
  episode termination match `pettingzoo.atari.surround_v2.parallel_env()` stepped one frame at a time
  (PettingZoo has frameskip 1 and no sticky actions by default; set `mode_num`/`full_action_space` to match). For
  Combat use `combat_tank_v2.parallel_env(has_maze=True, is_invisible=False, billiard_hit=False)`.
- Delay: with `max_delay` forced to a known d and sticky 0, an action's effect appears exactly d frames later.
- `VecGames` with 2 workers matches `num_workers=0` exactly for `train=False`; auto-reset semantics.
- A speed print (steps/s, frames/s) for 1 and 2 workers.

## 4. Model, trainer, export

### 4.1 `fb9/model.py`

```python
class Agent(nn.Module):              # Nature CNN, orthogonal init (CleanRL ppo_atari)
    def __init__(self, num_actions: int, in_channels: int = 6)
    def forward(self, obs_uint8: Tensor) -> tuple[Tensor, Tensor]   # (B,6,84,84) uint8 -> logits (B,A), value (B,)
                                                                     # divides by 255 internally
    def get_action_and_value(self, obs, action=None) -> (action, logprob, entropy, value)
```

### 4.2 `train.py` (CLI via tyro)

Args (defaults): `game="surround"`, `run_name=None` (default `f"{game}_{timestamp}"`), `total_samples=50_000_000`,
`num_games=64`, `num_workers=0→auto (os.cpu_count()-2)`, `num_steps=128`, `lr=2.5e-4` (linear anneal),
`update_epochs=4`, `minibatch_size=2048` (minibatch count = round(batch/2048)), `gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.1`, `ent_coef=0.01`,
`vf_coef=0.5`, `max_grad_norm=0.5`, `pool_fraction=0.25`, `snapshot_every=2_000_000`, `pool_size=20`,
`checkpoint_every=5_000_000`, `resume=False`, `seed=1`, `cuda=True`.

Self-play (`fb9/selfplay.py`):

- Games `0 .. M-1` are **mirror** games (both slots controlled by the learner, both slots' samples are trained on),
  `M = num_games - round(pool_fraction*num_games)`. The rest are **pool** games: the learner controls one seat
  (seat 0 for even game index, seat 1 for odd), a frozen snapshot controls the other; only the learner's slot is
  trained on. Learner batch per update = (2M + (num_games-M)) × num_steps.
- A pool game's opponent is re-sampled at each of its episode ends. Until the pool has a snapshot, pool games use the
  current learner as opponent (no training on that slot).
- Pool: snapshot the learner's weights every `snapshot_every` samples, keep the newest `pool_size`. Sampling weight
  for snapshot k = `(1 - winrate_k)**2 + 0.05` (PFSP "hard"), where `winrate_k` is an EMA (α=0.1, init 0.5) of
  the learner's result vs k from pool games (win=1, draw=0.5, loss=0; win/loss = sign of learner's episode return).
- Opponent inference: group pool slots by snapshot and run each snapshot net once per step, `torch.no_grad()`.
- GAE must respect `dones` per slot; the learner's slots in pool games are ordinary trajectories.

Outputs:

- TensorBoard `runs/<run_name>/`: `charts/episode_return` (seat 0 of mirror games), `charts/episode_frames`,
  `charts/sps`, `losses/*`, `pool/size`, `pool/winrate_mean`, `pool/winrate_min`.
- `checkpoints/<run_name>/ckpt_<samples>.pt` and `latest.pt` = `{"model", "optimizer", "samples", "updates",
  "pool": [{"samples": int, "winrate": float}], "args": dict}`; pool weights in
  `checkpoints/<run_name>/pool/snap_<samples>.pt` (`{"model": state_dict, "samples": int}`).
  Write to a temp file and `os.replace` (atomic). `--resume` restores everything from `latest.pt`.
- Prints a one-line progress summary every update.

### 4.3 `export.py`

`export.py --ckpt checkpoints/<run>/latest.pt --out models/<game>/` writes `model.ts` (TorchScript of a wrapper:
uint8 `(B,6,84,84)` → logits `(B,A)` float32, CPU) and `config.json`:
`{"game", "ale_mode", "action_ids", "action_names", "frameskip": 15 (Surround) / 4 (Combat, legacy), "stack": 4, "obs_shape": [6,84,84],
"seat_planes": "ch4=255 for seat0/port1, ch5=255 for seat1/port2", "samples", "source_ckpt"}`.
The exported model must load with only `torch.jit.load` (no repo code).

## 5. Play and console bridge (laptop)

### 5.1 `fb9/policy.py`

```python
class Policy:                        # wraps an exported model.ts + config.json
    def __init__(self, model_dir: str, level: str = "hard")
    def act(self, obs: np.ndarray) -> int   # obs (6,84,84) uint8 -> action index
```

Levels (all sample from the softmax; argmax is weak when several actions do the same thing): `hard` = temperature 1.0;
`medium` = temperature 1.25; `easy` = temperature 1.5 and 15% uniformly random actions. The evaluator samples at T=1 too. (Also `--level` on the CLIs.)

### 5.2 `play_pc.py`

`play_pc.py --game surround --model models/surround --level hard --scale 4` — pygame window at 60 fps, human is
**seat 0 / port 1** (keyboard arrows + space = fire; gamepad if present), model is seat 1. Uses `TwoPlayerGame`
internals or a minimal ALE loop with `fb9/preprocess.py` exactly (`train=False`). Human input is read every
emulator frame; the model decides every `frameskip` frames (from config.json: 15 for Surround) and its action is held for them. Shows score; R restarts,
Esc quits. Must also work headless with `SDL_VIDEODRIVER=dummy` for a smoke test.

### 5.3 `bridge/`

- `arduino/joystick.ino`: 115200 baud. Each received byte is a bitmask (bit0 UP, bit1 DOWN, bit2 LEFT, bit3 RIGHT,
  bit4 FIRE) → pins D2 (UP), D3 (DOWN), D4 (LEFT), D5 (RIGHT), D6 (FIRE). `#define ACTIVE_LOW 1` flips the output
  level. Pressed + released both handled; contradictory UP+DOWN or LEFT+RIGHT → release both. Failsafe: release
  all if no byte for 200 ms. Replies nothing (fire-and-forget).
- `serial_link.py`: `class JoystickLink(port, baud=115200)` with `send(mask: int)`, `release()`, `close()`;
  `FakeLink` that just records masks (for tests).
- `keyboard_to_console.py`: drives port 2 from the laptop keyboard (arrows + space) through `JoystickLink`.
- `capture.py`: `class Capture(device: int|str, width=1280, height=720, fps=60)` using cv2 (`CAP_PROP_FOURCC`
  MJPG), `read() -> (rgb, timestamp)`; `FileCapture(path)` replays a video for tests.
- `calibrate.py`: interactive (cv2 window): user marks the console's game rectangle (corresponding to the full
  emulator screen) → `bridge/crop.json` `{"x","y","w","h"}`; also shows the preprocessed 84×84 next to an emulator
  reference image for comparison.
- `measure_lag.py`: toggles FIRE/directions via `JoystickLink` and measures frames until the screen changes →
  prints p50/p95 lag in frames.
- `console_play.py --game surround --model models/surround --level hard --port /dev/ttyACM0 --device 0`: capture
  → crop (crop.json) → gray → every `frameskip`-th frame `process_frame(prev,last)` → `FrameStack(seat=1)` → `Policy` →
  `action_to_bitmask` → `JoystickLink.send`. Resets the stack when the frame changes drastically (new game).
  `--dry-run` uses `FakeLink` and `FileCapture`.
- Must not import `multi_agent_ale_py` (laptop may not have it) — only `fb9/preprocess.py`, `fb9/policy.py`,
  `fb9/games.py` (keep these three free of ALE imports).
