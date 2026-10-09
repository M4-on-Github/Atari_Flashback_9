# Atari Flashback 9 — RL agents you can play against

**Goal:** train RL agents for **Surround** and **Combat (Tank)** that are genuinely hard to beat, then
play against them (1) on a PC emulator and (2) on the real Atari Flashback 9 console.

Status legend: ✅ done · 🔄 in progress · ⏳ pending · ❓ needs your input

---

## 1. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Games | **Surround** first, **Combat Tank** second | Both are on the FB9, 2-player in the emulator, and joystick-controlled (paddle games like Pong are unreliable on the FB9) |
| Training environment | PettingZoo multi-agent ALE (`surround_v2`, `combat_tank_v2`) | Runs the real 2600 ROMs in true 2-player mode |
| Algorithm | **PPO + self-play with an opponent pool** | Standard for Atari and self-play; handles a moving opponent; scales with CPUs |
| Console interface | Laptop "plays" the console: HDMI capture → policy → Arduino → joystick port | The FB9 is closed hardware; the model can't run on it |
| Compute | pleiades SLURM cluster, everything inside an apptainer container | Cluster rules; reproducible |
| Combat variant | **Classic Tank with maze, straight shots = ALE mode 2** (PettingZoo `billiard_hit=False, has_maze=True`). The PettingZoo default is mode 9 (*Tank-Pong*, bouncing shots), so it must be set explicitly | Most iconic, easiest to match on the console. Verified by screenshots (`docs/screenshots/combat_mode{1,2,9}.png`) |
| Seats on the console | **Human = port 1, agent = port 2** | The FB9 menus are driven from port 1, so the human can navigate normally. The agent indicator lets one model play either seat |
| JAX / GPU emulators | **Not used** (see §4.10) | No JAX version of these 2-player games exists; reimplementations wouldn't match the console's pixels |

## 2. System overview

```
            ┌──────────── CLUSTER (training) ─────────────┐
            │ apptainer: fb9.sif                           │
            │ PettingZoo ALE ─► wrappers ─► PPO self-play  │──► checkpoints/ ─► export ─► model.ts
            └──────────────────────────────────────────────┘                              │
                                                                                           ▼
 ┌──────────────────────── LAPTOP next to the TV (play) ───────────────────────────────────────┐
 │ (a) play_pc.py:      emulator window, you on keyboard/gamepad vs model.ts                    │
 │ (b) console_play.py: capture card ─► crop/preprocess ─► model.ts ─► USB serial ─► Arduino    │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
 FB9 HDMI ─► capture card (loop-out) ─► TV       Arduino ─► optocouplers ─► FB9 port 2 (agent)
                                                      You  ─► joystick     ─► FB9 port 1 (human)
```

The cluster can't reach a display or USB devices, so playing always happens on a local machine.
Inference is a small CNN (~1 ms on CPU), so any laptop is fine.

## 3. Infrastructure rules

- **Everything lives in `~/Atari_Flashback_9`** — nothing in `/data/mmyatmau`.
- **No host Python environments on the cluster.** All code runs via `container/run.sh <cmd>`, which is
  `apptainer exec --nv --no-mount /opt container/fb9.sif <cmd>`. The `--no-mount /opt` is required: the cluster's
  apptainer.conf binds the host `/opt` over the container's `/opt/conda`.
- Image: `container/fb9.def` → `container/fb9.sif` (base `pytorch/pytorch:2.5.1-cuda12.4`, which still
  supports the GTX 1080 Ti / sm_61). Build cache/tmp in `.apptainer/` (gitignored). Rebuild with `container/build.sh` (mksquashfs limited to 1 GB RAM so it fits an interactive session).
- Interactive sessions (2 CPUs, 4 GB) are only for building, smoke tests and debugging. **Training runs via
  `sbatch`.**
- Target node: **pleiades-1-3** (1× GTX 1080 Ti + ~30 CPUs per job; 6 GPUs / ~60 CPUs were free when checked).
  Ada nodes were fully GPU-allocated. The network is tiny, so CPU count (emulator throughput) is the real limit.
- Max job length is 3 days; training checkpoints often and can resume.
- **Watching training:** compute nodes aren't directly reachable. Tunnel TensorBoard through the login node:
  `ssh -J <user>@head1.condo.cs.cmu.edu -L 6006:localhost:6006 <user>@pleiades-1-3`, then open `localhost:6006`.
  Eval mp4s land in `runs/<run>/videos/` for watching without a display.
- Code is committed to git at each phase boundary; `runs/` and `checkpoints/` stay out of git, `models/` (exported) goes in.

## 4. Training specification

### 4.1 Observation & action pipeline (same preprocessing in training, PC play and console play; `[train only]` steps are dropped at play time)

```
raw 210×160 RGB frame
 → sticky actions (p = 0.25)                         [train only]
 → action delay d frames, d ~ U{0..10} per episode   [train only; ≈0–170 ms, covers typical capture + console lag]
 → frame skip 4, max-pool over the last 2 frames (removes flicker)
 → grayscale → resize 84×84
 → augmentation, fixed per episode: brightness/contrast ±15%, shift ±2 px, light noise  [train only]
 → stack the last 4 frames
 → + 2 agent-indicator channels (which player am I)  → input 6×84×84
action: the game's minimal joystick action set (discrete), repeated for 4 frames
```

The delay and augmentation are what let the same model work on the real console. They cost little in emulator play.

- **Why a fixed 0–10 frame range up front:** the hardware won't exist until parts arrive, so training can't wait for
  the lag measurement. A typical end-to-end chain is MJPEG capture (2–4 frames) + emulator input lag (unknown,
  often 2–5) + serial/Arduino (<1) + agent decision cadence (avg ~1.5), which is well inside 10. If B3 measures more,
  we fine-tune the existing checkpoint with a shifted range (about an hour), not retrain.
- **Grayscale caveat:** the agent must tell *its own* trail/tank from the opponent's. If the two player colors have
  similar gray levels, grayscale erases that. The A1 probe checks the luma of both colors. If they're too close we
  use a custom color→gray mapping or keep RGB (12-channel stack). The cost is small.

### 4.2 Network (policy + value)

Nature-DQN CNN shared by actor and critic, orthogonal init:

```
conv 32@8×8 s4 → ReLU → conv 64@4×4 s2 → ReLU → conv 64@3×3 s1 → ReLU → FC 512 → ReLU
  ├─ actor:  linear → logits over actions → Categorical π(a|s)
  └─ critic: linear → V(s)
```

### 4.3 Loss (PPO-clip)

$$L = -\mathbb{E}\big[\min(r_t A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\big] + c_v\,\mathbb{E}\big[(V_\theta(s_t)-R_t)^2\big]_{\text{clipped}} - c_e\,\mathbb{E}\big[H(\pi_\theta(\cdot|s_t))\big]$$

- $r_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$
- Advantage $A_t$ from GAE (γ = 0.99, λ = 0.95), normalized per minibatch
- Return target $R_t = A_t + V_{old}(s_t)$

### 4.4 Hyperparameters (proven Atari defaults, adjusted for many envs)

| Param | Value | Param | Value |
|---|---|---|---|
| parallel games | 64 (= 128 agent slots) | rollout length | 128 steps |
| batch | 16,384 samples | minibatches | 4 |
| epochs per update | 4 | learning rate | 2.5e-4, linear decay to 0 |
| clip ε | 0.1 | value coef $c_v$ | 0.5 |
| entropy coef $c_e$ | 0.01 | max grad norm | 0.5 |
| γ / λ | 0.99 / 0.95 | Adam ε | 1e-5 |

### 4.5 Rewards

The games' own zero-sum scores, with no shaping to start:

- **Surround:** +1 when you win a round, −1 when you lose one.
- **Combat Tank:** +1 when you hit the enemy, −1 when you get hit.
- **Fallback if Combat stalls:** a small, annealed bonus for facing or approaching the enemy.

### 4.6 Self-play scheme

- **75% of games:** learner vs itself, with both seats trained (shared parameters, doubles data).
- **25% of games:** learner vs a **frozen past snapshot**, with only the learner's seat trained. This prevents
  strategy cycling and forgetting.
- **Snapshot pool:** a snapshot every 2M samples; keep the last 20; sample opponents biased toward the ones the
  learner beats least often.
- **Seats:** mirror games train both seats; pool games alternate the learner's seat by game index (half seat 0, half
  seat 1). One model therefore plays either port, with the seat-indicator planes telling it which one.

### 4.7 Budget and stopping

| Game | Target samples | Measured wall time (30 CPUs, 1× 1080 Ti) | Status |
|---|---|---|---|
| Surround | ~50M | ~2.2 h (6.2k learner samples/s) | ⏳ running: `surround_v1` |
| Combat Tank | ~150M | ~7–9 h (Combat steps are a bit slower) | ⏳ |

A4 benchmark (job 50394): the env alone reaches 20k samples/s with 46 workers × 4 games, and scales poorly
(lock-step), but training is GPU-bound: 16, 30 and 46 workers all train at ~5k samples/s. More CPUs do not help.
Each update spends ~2.5 s in the rollout and ~2.1 s in the PPO update (128 games × 128 steps).
PPO uses a fixed minibatch of 2048 (not a fixed count of 4): with 4 minibatches of ~7k, KL stayed ~0.0000 and
the policy barely moved per sample.

**Stop when all four hold:**
1. ≥ 98% wins vs random.
2. Clearly beats the scripted bot.
3. Elo vs past snapshots has flattened.
4. You find it hard to beat.

Otherwise extend the run from its checkpoint.

### 4.8 Evaluation (every 2M samples, logged to TensorBoard)

- 50 games vs **random**
- 50 games vs a **scripted bot**:
  - Surround: parse the grid from pixels and move toward the largest reachable open area (flood fill).
  - Combat: aim-and-shoot heuristic (optional).
- Round-robin vs the last 10 snapshots → **Elo**
- One **mp4** of the current agent vs the best snapshot, so you can watch progress without a display

### 4.9 Difficulty levels (play time)

| Level | Checkpoint | Action selection |
|---|---|---|
| Easy | early checkpoint | sampling, temperature 1.5, plus 15% random actions |
| Medium | mid checkpoint | sampling, T = 1.25 |
| Hard | best checkpoint | sampling, T = 1 (argmax and low T are weak: NOOP / current direction alias "keep going", see §8) |

### 4.10 Where training time goes, and how we speed it up

The bottleneck is the **Atari emulator on CPU** (plus Python overhead per frame), not the neural network. So:

| Option | Verdict |
|---|---|
| **More CPU cores** (sbatch with ~30–48 CPUs) | ✅ Biggest lever; scales almost linearly |
| **Frame skip inside the ALE loop**, grabbing the screen only on the last 2 of every 4 frames, reading grayscale directly from the ALE | ✅ ~1.5–2× fewer copies and less Python per frame; worth doing in A2 |
| Shared-memory vector env (SuperSuit multiprocessing; PufferLib if that's too slow) | ✅ Decided by the A4 benchmark |
| GPU learner (1080 Ti) | ✅ Already planned. A4 also checks whether the PPO update, not rollout, dominates |
| **JAX** (e.g. JAXAtari) | ❌ Its games are hand-written JAX reimplementations, mostly single-player, with no 2-player Surround/Combat. Even if added, their pixels wouldn't match the real ROM the FB9 runs, which breaks console transfer. JAX would only speed up the network, which isn't the bottleneck |
| GPU emulators (CuLE) | ❌ Unmaintained, single-player only |

## 5. Phases

### Track A — software and training (cluster)

| # | Phase | Deliverable | Done when | Status |
|---|---|---|---|---|
| A1 | Container + smoke test | `container/fb9.sif`, probe report | Both envs step and render in 2P; torch sees the 1080 Ti | ✅ |
| A2 | Env harness | `fb9/envs.py` (pipeline §4.1), `fb9/wrappers.py` | Obs shapes correct; delay and augmentation visually checked | ⏳ |
| A3 | Trainer + eval | `train.py`, `fb9/selfplay.py`, `fb9/evaluate.py`, `fb9/bots.py` | 10-min run on the interactive node shows learning vs random; a resume-from-checkpoint test passes | ⏳ |
| A4 | Speed benchmark | `slurm/bench.sbatch` → samples/s at 16/32/48 CPUs | Real hour estimates written into §4.7 | ✅ |
| A5 | Train Surround | `slurm/train.sbatch surround` | Stop criteria in §4.7 | ⏳ |
| A6 | Train Combat Tank | `slurm/train.sbatch combat_tank` | Stop criteria in §4.7 | ⏳ |
| A7 | Export | `export.py` → `models/<game>/{model.ts, config.json}` | Loads and runs on CPU without the repo's training code | ⏳ |

### Track B — play and hardware (laptop + bench)

| # | Phase | Deliverable | Done when | Status |
|---|---|---|---|---|
| B1 | PC play | `play_pc.py`, `requirements-local.txt` | You play both games vs a model on the laptop | ⏳ |
| B2 | Fake joystick | `bridge/arduino/joystick.ino`, `bridge/keyboard_to_console.py` | You drive the FB9 menu and play Surround from the laptop keyboard | ⏳ |
| B3 | Lag measurement | `bridge/measure_lag.py` | p50/p95 end-to-end lag in frames. If p95 > 10 frames → short fine-tune with a shifted delay range | ⏳ |
| B4 | Game-mode match | `docs/screenshots/` (Surround mode 1; Combat mode 2 = Tank with maze) | FB9 game numbers recorded here | ❓ you |
| B5 | Video calibration | `bridge/calibrate.py` → `bridge/crop.json` | Preprocessed console frames match emulator frames | ⏳ |
| B6 | Console play | `bridge/console_play.py --game surround --level hard` | Agent plays port 2, you play port 1 | ⏳ |

**Parts checklist (B2).** The parts belong to the professor's lab. When you pick them up, check each item off (or note
a substitute). No soldering needed. Anything missing costs ~$5–20.

| Item | Purpose |
|---|---|
| USB HDMI capture card **with HDMI loop-out**, 720p60 | Laptop sees the game; TV still gets the picture |
| Arduino Uno (or Nano) | Receives the agent's moves from the laptop over USB serial |
| PC817 optocoupler module, **8 channels** (we need 5: four directions + fire), screw terminals | Isolated electronic "switch presses" |
| DB9 **female** breakout with screw terminals | Plugs into console joystick port 2 |
| Multimeter (if you don't have one, ~$15) | Checks each optocoupler actually shorts its pin to ground before plugging into the console |
| Dupont jumper wires | Connections |

**Hardware wiring (B2).** A 2600 joystick is just switches that short pins to ground:

| DB9 pin | Function | Driven by |
|---|---|---|
| 1 | Up | PC817 #1 |
| 2 | Down | PC817 #2 |
| 3 | Left | PC817 #3 |
| 4 | Right | PC817 #4 |
| 6 | Fire | PC817 #5 |
| 8 | Ground | all optocoupler emitters |

- Module output side: each **OUTx** (the transistor collector) → its DB9 pin; output **GND/G2** → DB9 pin 8. Leave the
  output-side **V2/VCC unconnected**, because these modules have pull-ups meant for level shifting and we must not feed voltage into the port.
- Module input side: Arduino pins D2–D6 → IN1–IN5; **V1/VCC** → Arduino 5V; **G1** → Arduino GND.
- Many of these modules are **active-LOW** (the input pulls the LED's cathode). The firmware has an `ACTIVE_LOW` flag; verify it
  with the multimeter (continuity between OUTx and G2 only while "pressed") before connecting to the console.
- The optocouplers keep the Arduino and the console electrically isolated, and nothing feeds voltage into the port.
- **Firmware failsafe:** release all buttons if no command arrives for 200 ms.

**Console loop (B6).**
0. Navigating menus and starting the game: the human does this with the port-1 joystick, as normal. The agent's frame
   stack resets when a new game is detected (score display returns to 0–0).
1. Capture at 720p60 (cheap MS2109-style cards do 720p60 MJPEG; we pick that mode explicitly).
2. Every 4th frame, max-pool the last 2 frames.
3. Crop, then grayscale, then resize to 84×84.
4. Stack the last 4 frames and run the policy.
5. Send a 1-byte button bitmask to the Arduino.

The end-to-end lag measured in B3 is exactly the delay the agent experiences. Training covers it via D.

## 6. Timeline

Training and PC play don't depend on any hardware. The console work starts whenever you have the professor's parts,
so the two are decoupled.

| When | Track A (cluster) | Track B (you / laptop) |
|---|---|---|
| Day 1, hours 0–3 | A1 container + probe, A2 harness, A3 trainer | Check the console's sockets and the prof's parts against the checklist. B4: match game modes on the FB9 from our screenshots |
| Day 1, hours 3–4 | A4 benchmark → real hour estimates | B1: set up the laptop for PC play |
| Day 1, hours 4–9 | A5 Surround training (sbatch) | Watch eval videos / TensorBoard |
| Day 1 evening | A7 export Surround | **Play Surround on the laptop** 🎮 |
| Overnight → Day 2 | A6 Combat Tank training | — |
| Day 2 | A7 export Combat | **Play Combat on the laptop** 🎮 |
| When you have the prof's parts | (fine-tune if B3 lag > 10 frames) | B2 fake joystick → B3 lag → B5 calibration → **B6 on the console** 🎮 |

Fallback at every stage: `play_pc.py` works even if the console bridge doesn't.

## 7. How the code gets written

- **Haiku coding agents** implement each component from a written brief: interface contract, facts from the probe,
  acceptance tests, and "stop and ask if unsure".
- **Claude reviews every change**: reads the diff, runs the tests in the container, and answers the agents' questions.
- Agents work on non-overlapping files. Shared interfaces are fixed in the briefs before work starts.
- Agents never submit SLURM jobs or install anything on the host; Claude launches the jobs.

## 7b. Repo layout

```
plan.md
container/fb9.def            # apptainer recipe (fb9.sif is gitignored)
slurm/                       # bench.sbatch, train.sbatch
docs/contracts.md            # interface contracts the coding agents build against
docs/screenshots/            # emulator game modes, for matching on the FB9 (B4)
fb9/                         # games.py, preprocess.py, envs.py, model.py, selfplay.py, policy.py, evaluate.py, bots.py
scripts/                     # bench_env.py
tests/                       # plain-script tests: container/run.sh python tests/test_<x>.py
train.py  export.py  play_pc.py  requirements-local.txt
bridge/                      # arduino/joystick.ino, keyboard_to_console.py, measure_lag.py, calibrate.py, console_play.py
runs/  checkpoints/  models/ # outputs (runs/ and checkpoints/ gitignored)
```

## 8. Open gaps

| Gap | How it gets closed | Owner | Status |
|---|---|---|---|
| Container builds and the multi-agent ALE works inside it | `multi-agent-ale-py` built from source (wheels only exist for Python ≤ 3.9). `fb9.sif` 3.4 GB | Claude | ✅ |
| PyTorch 2.5.1 runs on the 1080 Ti (sm_61) | Probe: CUDA available, conv test passes | Claude | ✅ |
| Agent names, action sets, modes, episode lengths, which agent is P1 | Probe (facts below the table) | Claude | ✅ |
| Real samples/s → final training hours | A4 benchmark | Claude | ✅ |
| Scripted Surround bot: grid parsing from pixels | `fb9/bots.py`: 38×18 grid of 4×9-px cells; flood-fill bot wins 40/40 rounds vs random | Claude | ✅ |
| Held-out stronger Surround bot (never a training opponent, so beating it can't be gamed) | `fb9/search_bot.py`: alpha-beta over both snakes' moves, Voronoi territory eval; beats the flood-fill bot 12–0 in rounds, ~4.5 ms/move | Claude | ✅ |
| Player colors still distinguishable in grayscale (§4.1) | Yes: Surround players gray 64 vs 147 (bg 90, walls 167); Combat mode 2 tanks 124 vs 146 (bg 102, maze 208). Grayscale kept | Claude | ✅ |
| `combat_tank_v2` flag defaults (maze / billiard / invisible) and which mode they map to | mode = {1,8,10,13}[invisible,billiard] + has_maze. Default = 9 (Tank-Pong). We use **mode 2** | Claude | ✅ |
| Console has two 9-pin trapezoid joystick sockets on the front ("FB9" = Atari Flashback 9) | Look at the console | **You** | ❓ |
| Hardware: the prof's parts vs the checklist in §5 (B2) | Inventory when you pick them up | **You** | ❓ |
| Play machine | Your laptop next to the TV runs all play/bridge code (inference is ~1 ms on CPU); training stays on the cluster | — | ✅ decided |
| Laptop can run `play_pc.py` (needs `multi-agent-ale-py`) | Python 3.9 env (wheels exist for Linux and Intel Mac), or a C++ compiler + cmake (Apple Silicon/Windows). `console_play.py` doesn't need it, only torch + opencv + pyserial | Claude | ⏳ |
| FB9 game numbers matching the trained modes | B4, using `docs/screenshots/` | You + Claude | ❓ |
| End-to-end console lag (training already covers 0–10 frames) | B3 | You + Claude | ⏳ after parts arrive |
| FB9 emulation close enough to Stella's | B5 calibration + playtest | You + Claude | ⏳ |
| Combat colors differ (emulator PAL ROM vs console NTSC) | B5 calibration maps each console color (background, maze, tank 1, tank 2) to the emulator's gray level; brightness/contrast augmentation covers the rest | Claude | ⏳ |

**A1 probe facts** (used by the code):

- Agents `first_0` = ALE player A = **joystick port 1**, `second_0` = port 2. Surround: `first_0` is the **blue, right-hand**
  snake (score shown top right); `second_0` is green/left. Combat: `first_0` is the **left** tank.
- Surround: 5 actions, ALE ids `[0 NOOP, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN]`, 2P modes `[1,5..12]`, we use mode 1. Random-play
  games last ~5,700 frames (~1.6 min) with ~12 rounds; rewards ±1 per round, zero-sum. Screen 210×160.
- Combat: the ALE ships the **PAL** cartridge (`Combat (32-in-1) (Atari) (PAL)`), while the FB9 runs NTSC. Same game
  logic, but different colors and a taller screen. Full 18-action set, 27 modes, screen **256×160**. Fixed 8,181-frame games (2:16 timer);
  ±1 per hit, zero-sum. Colors differ per mode (mode 2: olive background, blue/pink tanks).
- Speed: ~3,000–3,300 raw frames/s per core through PettingZoo.

**Training facts** (`surround_v1`):

- Never act by argmax. In Surround NOOP, "press the current direction" (and probably the reverse direction) all mean
  "keep going"; the entropy bonus spreads probability over them, so argmax picks a lone turn and the snake steers into
  walls. At 15M samples the argmax agent lost 1–9 to the 5M checkpoint, while sampled play won 6–2. Play levels and the
  evaluator sample (§4.9).
- Surround control (measured): the snake moves exactly one cell every **15 frames** (a ~235-frame pause between
  rounds); a **1-frame** press of a new direction latches the heading at any phase of the cycle; reverse presses are
  ignored. So NOOP, the current direction and the reverse all mean "straight" and only the 2 perpendicular presses
  turn.
- `surround_v1` (frameskip 4) climbed in self-play Elo (1000 → 1345 from 5M to 25M) but won **0 rounds** vs the
  flood-fill bot and the search bot. The 30M checkpoint lost 0–30 rounds vs the flood bot at T=1, 0.5, 0.25, argmax,
  and with each action held for 4 steps. At 25M the policy put ~0.4 probability on turns every step; with ~4 decisions
  per cell, almost every cell turned, and the snake zigzagged into its own trail after ~30 cells.
- Fix for `surround_v2`: per-game frameskip, Surround = 15 (one decision per cell move), Combat stays 4. Legacy
  checkpoints and models without a recorded frameskip are frameskip 4.
- `surround_v2` (frameskip 15, ent_coef 0.005, 128 games × 64 steps, 20M samples, ~2 h at ~3.3k samples/s, entropy
  1.6 → 0.46): the zigzag is gone (long straight runs), self-play Elo 1000 → 2259, the learner beat its pool snapshots
  99% of the time, but it still lost every game to both bots: ~1 round per game vs the flood bot, ~0 vs the search
  bot. Its losses come from driving into regions it closed off itself (dead-end corridors, its own boxes). Self-play
  converged on a narrow style.
- `surround_v3` = v2 + `--bot-fraction 0.25` (a quarter of the games vs the flood-fill SurroundBot, played inside the
  env workers), 30M samples. SearchBot stays held out for evaluation only.
  Result: self-play Elo 2782 but **0/10 games vs both bots at every checkpoint**; training win rate vs the flood bot
  stayed 0.04–0.28 with no trend. More data was not the fix.
- **Supervised probe** (`tools/bc_probe.py`, job 50432, `runs/bc_probe/probe_v1.json`): imitate the flood bot's move
  scores on 100k bot-vs-bot states (decisive ones kept, 10% of trivial ones), test on 20k unseen states. On forks
  (legal moves differ by ≥10 cells of reachable area): pixel `Agent` 43% correct (random legal 50%), 35% illegal
  moves, 86% on its own train states (memorises, doesn't generalise); grid `GridAgent` **86% correct, 0.1% illegal**
  after 1–5 epochs. Conclusion: the 84×84 pixel observation was why v1–v3 couldn't see dead ends.
- `surround_v4` = v3 with `obs="grid"` (18×38 cell grid, `GridAgent`), 30M samples, ~1.8k samples/s. Training win
  rate vs the flood bot rose 0.2 → 0.99 between 2.2M and 2.9M. Held-out eval: **10/0 games vs the flood bot from 4M
  on** (60–5 rounds at 6M), but 0/10 vs the search bot (score diff ≈ −19, flat from 4M through 14M).
  Round-end analysis vs the search bot at 12M (62 rounds, both seats): ~half the losses are crashes near the bot's
  head while the board is still open (many at the round's first head-to-head meeting, ~32 moves in); the rest come
  after the bot cuts v4 off: in every separation v4 had the smaller region (by 6–450 cells). Greedy play doesn't
  help (3/63 rounds, the same game repeated). v4 never learned to contest cutting points, because neither training
  opponent (itself, the flood bot) plays them.
- **Search-teacher probe** (`tools/bc_probe.py --teacher search`, job 50436, `runs/bc_probe/probe_search_v1.json`):
  the grid net imitating SearchBot reaches 85% test accuracy overall, 73% on states where the flood bot's move is not
  SearchBot-best (the flood bot gets 0% there), and 76% with heads ≤6 cells apart (flood bot 63%). Train accuracy 99%,
  so it overfits: there were only 28 training games. Played as a policy, it loses: 0–30 and 10–30 rounds vs SearchBot,
  3–30 and 4–30 vs the flood bot. The net *can* represent the contesting moves, so v4's gap is the training signal
  (no opponent contests space), not network capacity. Next: an exploiter league (§4.6). SearchBot stays held out.
- ⭐ **Candidate: `surround_v4` 12M** (`checkpoints/surround_v4/ckpt_12005376.pt`, copy kept at
  `checkpoints/candidates/surround_v4_12M.pt`). The first agent that beats a scripted bot; not exported yet.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Multi-agent ALE packages are old or unmaintained | Versions pinned in the container; the image is the source of truth |
| Self-play cycles or overfits to itself | Opponent pool + Elo tracking + scripted-bot eval |
| Combat Tank learns slowly | Longer run (3-day limit), shaping fallback (§4.5) |
| Agent strong in the emulator, weak on the console | Delay + augmentation in training, calibration, lag measured before the long runs |
| Damaging the console port | Optocoupler isolation, no voltage into the port, failsafe release |
| GPUs busy | Use 1080 Ti nodes; the model is small enough |
| Job hits the time limit | Regular checkpoints + `--resume` |
| Laptop can't install the 2-player ALE (`play_pc.py`) | Python 3.9 env with prebuilt wheels; or watch eval mp4s; `console_play.py` doesn't need the ALE at all |
| Optocoupler module wired/polarity wrong | Multimeter check before connecting; `ACTIVE_LOW` flag; failsafe release |
| "Wins vs random" is a trivially easy bar | It's only a sanity check. The scripted bot, Elo and your own play are the real criteria |
