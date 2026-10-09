# Atari Flashback 9 — RL agents you can play against

Reinforcement-learning agents for the 2-player games of the **Atari Flashback 9** console: **Surround** first,
**Combat (Tank)** next. They are trained with PPO self-play on a SLURM cluster. You can play them on a PC emulator, or on
the real console: a laptop watches the TV picture through a capture card and presses the console's joystick port 2
through an Arduino.

The full design, the training history and every measured result are in [plan.md](plan.md). The interfaces between the
modules are in [docs/contracts.md](docs/contracts.md).

## Status

| | |
|---|---|
| Surround agent | ✅ playable. Beats the scripted flood-fill bot 10/0, but still loses to the held-out search bot |
| Exploiter league (`surround_league_v1`) | ✅ done (20M). Much stronger in head-to-head games, still 0/10 vs the search bot |
| Combat agent | ⏳ not trained yet |
| Console play | ⏳ hardware being set up (capture card, Arduino, optocouplers) |

Exported models (in `models/`, committed so a `git pull` on the laptop is enough):

| Folder | Checkpoint | Notes |
|---|---|---|
| `models/surround` | `surround_league_v1` 8M | **default**; fewest crashes against strong play |
| `models/surround_league_20M` | `surround_league_v1` 20M (final) | **strongest**: beats 16M 10–0, 18M 8–2 |
| `models/surround_league_16M` | `surround_league_v1` 16M | beats 10M and 12M 10–0 |
| `models/surround_league_10M` | `surround_league_v1` 10M | beats the 8M default 10–0 in head-to-head games |
| `models/surround_v4_30M` | `surround_v4` final | backup |
| `models/surround_v4_12M` | `surround_v4` 12M | backup, the first agent that beat a scripted bot |

## Play on the laptop

The laptop needs plain Python (3.9 or 3.11) without the container. Install the CPU build of torch, then the rest:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-local.txt
```

`multi-agent-ale-py` (the emulator) is only needed for `play_pc.py`. It has wheels for Python ≤ 3.9 only; elsewhere it
builds from source and needs cmake and a C++ compiler. The console bridge doesn't need it.

### 1. Against the emulator

```bash
python play_pc.py --game surround --model models/surround --level hard
```

You are player 1 (arrow keys, space = fire, or a gamepad); the model is player 2. **R** restarts, **Esc** quits.
`--level` is `hard`, `medium` or `easy` (higher sampling temperature; `easy` also makes 15% random moves).

### 2. Against the real console

You play with a normal joystick on **port 1**; the model plays **port 2** through the Arduino.

**Hardware** (parts list, wiring table and safety notes in plan.md §5 Track B):

- an HDMI capture card with loop-out, so the TV still gets the picture;
- an Arduino Uno or Nano running `bridge/arduino/joystick.ino` (115200 baud; it releases every button after 200 ms
  without a command);
- a PC817 optocoupler module driving DB9 pins 1–4 (directions), 6 (fire) and 8 (ground) on port 2. Leave the
  output-side VCC unconnected;
- `ACTIVE_LOW` in the firmware must match your module. Check with a multimeter that each output shorts to ground only
  while "pressed" **before** plugging into the console.

**Steps:**

```bash
# a) Test the joystick link: drive port 2 from the keyboard (arrows + space)
python bridge/keyboard_to_console.py --port /dev/ttyACM0

# b) Calibrate the crop: drag a rectangle over the game area; writes bridge/crop.json
python bridge/calibrate.py --device 0 --reference docs/screenshots/surround_emulator.png

# c) Optional: measure end-to-end lag (training covers 0–10 frames)
python bridge/measure_lag.py --port /dev/ttyACM0 --device 0 --trials 20

# d) Play: start 2-player Surround on the console with your joystick, then
python bridge/console_play.py --game surround --model models/surround --level hard --port /dev/ttyACM0 --device 0
```

- On Windows the port is `COM3` or similar; `--device` is the capture card's camera index.
- Swap `--model` for another folder in `models/` to try a backup agent.
- No hardware at hand? `python bridge/console_play.py --dry-run --video recording.avi` runs the loop on a recording.

**Troubleshooting:**
- **The snake drives straight into a wall at once.** The Surround models read the board by matching each cell to the
  emulator's colours. If the console's colours or the crop are off, the agent sees the wrong board. Redo the
  calibration and compare the two previews.
- **Wrong game mode.** The models are trained on Surround mode 1 (2 players). Find the matching game number in the FB9
  menu (plan.md B4).

## Train on the cluster (pleiades)

Everything Python runs inside the apptainer image. There are no host virtualenvs.

```bash
container/build.sh                                   # builds container/fb9.sif from container/fb9.def (once)
container/run.sh python <script> ...                 # run anything inside the image, from the repo root
```

Training jobs go through SLURM (partition `pleiades`, node `pleiades-1-3`, logs in `slurm/logs/`):

```bash
# Self-play PPO (Surround defaults: grid observation, frameskip 15)
sbatch slurm/train.sbatch surround --run-name surround_v4 --total-samples 30000000 --bot-fraction 0.25

# Exploiter league, warm-started from a finished run (restart-safe: finished phases are skipped)
sbatch slurm/league.sbatch --init-ckpt checkpoints/surround_v4/latest.pt --run-name surround_league_v1

# Evaluation watcher: plays every new checkpoint vs random, the flood-fill bot and the search bot; tracks Elo
sbatch slurm/eval.sbatch surround surround_league_v1

# Export a checkpoint for play (TorchScript + config.json)
container/run.sh python export.py --ckpt checkpoints/<run>/ckpt_<samples>.pt --out models/surround
```

Outputs: TensorBoard logs and eval results in `runs/<run>/`, checkpoints in `checkpoints/<run>/`. Both are gitignored.

**Opponents.**
- Training mixes several opponents:
  - **mirror** games against itself;
  - the **pool** of past snapshots (PFSP);
  - the scripted flood-fill **SurroundBot** (`--bot-fraction`);
  - in the league, frozen **exploiters** trained to beat the current agent (`--league-fraction`).
- The alpha-beta **SearchBot** (`fb9/search_bot.py`) is held out for evaluation only and is never a training opponent.
  So a win against it can't be gamed.

## Tests

Plain scripts, run inside the container:

```bash
for t in grid envs train eval play search_bot bc_probe league; do container/run.sh python tests/test_$t.py; done
```

## Repo layout

```
train.py  league.py  export.py  play_pc.py   # training, exploiter league, export, PC play
fb9/          # games, envs (multi-agent ALE workers), grid obs, model, self-play, policy, evaluate, bots, search bot
bridge/       # console bridge: arduino/joystick.ino, capture, serial link, calibrate, measure_lag, console_play
container/    # apptainer recipe and run.sh (fb9.sif is gitignored)
slurm/        # train / league / eval / bench / probe job scripts
tools/        # bc_probe.py (supervised probe: can a network learn the bots' decisions?)
tests/        # plain-script tests
docs/         # contracts.md, screenshots/
models/       # exported models for play
plan.md       # design, results, open gaps
```

## Notes

- **Agents always sample, never argmax.** In Surround several actions mean "keep going" (NOOP, the current
  direction, the reverse). Argmax then picks a lone turn and steers into walls.
- **Surround decides once per cell move** (frameskip 15). Its observation is an 18×38 grid of cells (two frames)
  rather than pixels: from 84×84 pixels, the network could not see dead ends.
- Combat uses 84×84 grayscale pixels, frameskip 4, and mode 2 (Tank with maze). The ALE ships the PAL cartridge,
  while the FB9 is NTSC, so its colours differ.
