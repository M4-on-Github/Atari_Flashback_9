"""Measure end-to-end button lag: laptop send -> Arduino -> optocoupler -> console -> HDMI -> capture (plan B3).

Run it with the FB9 on a static menu screen whose cursor moves with the joystick. Each trial:
  1. wait for a quiet screen (frame-to-frame change below the threshold for 5 frames),
  2. send UP or DOWN (alternating) and re-send it every 50 ms (keeps the failsafe from releasing it),
  3. count captured frames until the crop differs from the quiet frame by more than the threshold.
Prints p50/p95 lag in frames and milliseconds. p95 > 10 frames: fine-tune with a shifted delay range (plan B3).

    container/run.sh python bridge/measure_lag.py --port /dev/ttyACM0 --device 0 --trials 20
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import tyro  # noqa: E402

from bridge.capture import Capture, crop_rgb, load_crop  # noqa: E402
from bridge.serial_link import JoystickLink  # noqa: E402
from fb9.preprocess import to_gray  # noqa: E402

UP_MASK, DOWN_MASK = 1, 2


@dataclass
class Args:
    port: str = "/dev/ttyACM0"
    device: str = "0"               # capture device index or path
    trials: int = 20
    crop: str = "bridge/crop.json"  # measure only the game area if calibrated
    width: int = 1280
    height: int = 720
    fps: int = 60
    threshold: float = 8.0          # mean abs gray difference (0..255) that counts as "the screen changed"
    timeout_frames: int = 60        # give up on a trial after this many frames
    resend_s: float = 0.05


def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main() -> None:
    args = tyro.cli(Args)
    device = int(args.device) if args.device.isdigit() else args.device
    crop = load_crop(args.crop)
    cap = Capture(device, args.width, args.height, args.fps)
    link = JoystickLink(args.port)

    def gray() -> np.ndarray:
        rgb, ts = cap.read()
        return to_gray(crop_rgb(rgb, crop))

    def wait_quiet(prev: np.ndarray) -> np.ndarray:
        quiet, deadline = 0, time.monotonic() + 5.0
        while quiet < 5:
            if time.monotonic() > deadline:
                print("warning: screen never went quiet; measuring anyway")
                break
            cur = gray()
            quiet = quiet + 1 if mean_diff(cur, prev) < args.threshold else 0
            prev = cur
        return prev

    lags_frames, lags_ms = [], []
    try:
        base = wait_quiet(gray())
        for trial in range(args.trials):
            mask = UP_MASK if trial % 2 == 0 else DOWN_MASK
            t0 = time.monotonic()
            last_send = t0
            link.send(mask)
            lag = None
            for frame in range(1, args.timeout_frames + 1):
                cur = gray()
                ts = time.monotonic()
                if mean_diff(cur, base) > args.threshold:
                    lag = frame
                    break
                if ts - last_send >= args.resend_s:
                    link.send(mask)
                    last_send = ts
            link.release()
            if lag is None:
                print(f"trial {trial}: no change within {args.timeout_frames} frames (skipped)")
            else:
                lags_frames.append(lag)
                lags_ms.append((time.monotonic() - t0) * 1000.0)
                print(f"trial {trial}: lag {lag} frames ({lags_ms[-1]:.0f} ms)")
            base = wait_quiet(gray())
    finally:
        link.close()
        cap.close()

    if not lags_frames:
        print("no successful trials: check the crop (bridge/crop.json), the wiring and the menu screen")
        return
    f = np.asarray(lags_frames)
    m = np.asarray(lags_ms)
    print(f"n={len(f)}  frames p50={np.percentile(f, 50):.1f} p95={np.percentile(f, 95):.1f} max={f.max()}  "
          f"ms p50={np.percentile(m, 50):.0f} p95={np.percentile(m, 95):.0f}")
    if np.percentile(f, 95) > 10:
        print("p95 > 10 frames: plan B3 says fine-tune with a shifted delay range before console play")


if __name__ == "__main__":
    main()
