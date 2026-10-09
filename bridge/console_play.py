"""Play the FB9 console on port 2 with the trained model (plan B6, docs/contracts.md §5.3).

Loop: capture frame -> crop (bridge/crop.json) -> gray -> every `frameskip`-th frame max-pool the last two frames and resize to
84x84 -> FrameStack(seat=1) -> Policy -> action_to_bitmask -> JoystickLink.send (1 byte to the Arduino).
A new game is detected when the 84x84 frame jumps (mean abs diff > reset_threshold); the stack is then reset.

Grid models (config "obs": "grid", Surround): the cropped RGB of the last frame of each step is resized to the emulator
screen size (160x210) and classified into the 18x38 cell grid (fb9/grid.py) -> GridStack(seat=1). The first decision
resets the stack; later ones push. Grid mode classifies cells by the nearest emulator palette colour (fb9/bots.py), so
the console colours must be close to the emulator's. Calibration with a reference screenshot checks this.

    container/run.sh python bridge/console_play.py --game surround --model models/surround --level hard \
        --port /dev/ttyACM0 --device 0
    container/run.sh python bridge/console_play.py --dry-run --video recording.avi   # no hardware: FakeLink
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

from bridge.capture import Capture, FileCapture, crop_rgb, load_crop  # noqa: E402
from bridge.serial_link import FakeLink, JoystickLink  # noqa: E402
from fb9.games import GAMES, action_to_bitmask  # noqa: E402
from fb9.grid import GridStack  # noqa: E402
from fb9.policy import Policy  # noqa: E402
from fb9.preprocess import FrameStack, process_frame, to_gray  # noqa: E402

SCREEN_W, SCREEN_H = 160, 210   # emulator screen size; grid mode resizes the cropped console frame to it


@dataclass
class Args:
    game: str = "surround"
    model: str = "models/surround"
    level: str = "hard"
    port: str = "/dev/ttyACM0"
    device: str = "0"
    crop: str = "bridge/crop.json"
    width: int = 1280
    height: int = 720
    fps: int = 60
    dry_run: bool = False           # FakeLink + FileCapture(video): no hardware
    video: str = ""                 # input video for --dry-run
    max_frames: int = 0             # stop after this many captured frames (0 = until Ctrl-C / end of video)
    reset_threshold: float = 40.0   # mean abs diff of consecutive 84x84 frames that means "new game"


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def run(args: Args, link=None, source=None) -> dict:
    """Run the bridge. link/source can be injected (tests). Returns counts for checking."""
    spec = GAMES[args.game]
    policy = Policy(args.model, args.level)
    if policy.action_names != spec.action_names:
        raise ValueError(f"model actions {policy.action_names} != game actions {spec.action_names}")
    crop = load_crop(args.crop)
    if crop is None:
        print(f"note: {args.crop} not found, using the whole capture frame (run bridge/calibrate.py first)")

    if link is None:
        link = FakeLink() if args.dry_run else JoystickLink(args.port)
    if source is None:
        if args.dry_run:
            source = FileCapture(args.video)
        else:
            device = int(args.device) if args.device.isdigit() else args.device
            source = Capture(device, args.width, args.height, args.fps)

    grid = policy.obs_kind == "grid"
    stack = GridStack(seat=1) if grid else FrameStack(seat=1)
    prev_gray = None
    last84 = None
    masks: list[int] = []
    resets = 0
    i = 0
    try:
        while not (args.max_frames and i >= args.max_frames):
            try:
                rgb, _ = source.read()
            except EOFError:
                break
            pos = i % policy.frameskip
            obs = None
            if grid:
                if pos == policy.frameskip - 1:
                    screen = cv2.resize(crop_rgb(rgb, crop), (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
                    if not masks:   # first decision: new game, so start the stack afresh
                        obs = stack.reset(screen)
                        resets += 1
                    else:
                        obs = stack.push(screen)
            else:
                gray = to_gray(crop_rgb(rgb, crop))
                if pos == policy.frameskip - 2:
                    prev_gray = gray
                elif pos == policy.frameskip - 1:
                    frame84 = process_frame(prev_gray, gray)
                    if last84 is None or frame_diff(frame84, last84) > args.reset_threshold:
                        obs = stack.reset(frame84)
                        resets += 1
                    else:
                        obs = stack.push(frame84)
                    last84 = frame84
            if obs is not None:
                action = policy.act(obs)
                mask = action_to_bitmask(policy.action_names[action])
                link.send(mask)
                masks.append(mask)
                if len(masks) % 15 == 0:
                    print(f"decisions={len(masks)} resets={resets} action={policy.action_names[action]} mask={mask:05b}")
            i += 1
    finally:
        link.release()
        link.close()
        source.close()
    return {"frames": i, "decisions": len(masks), "resets": resets, "masks": masks}


def main() -> None:
    args = tyro.cli(Args)
    if args.dry_run and not args.video:
        raise SystemExit("--dry-run needs --video")
    summary = run(args)
    print({k: v for k, v in summary.items() if k != "masks"})


if __name__ == "__main__":
    main()
