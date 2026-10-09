"""Calibrate the capture crop: drag a rectangle over the game screen, save bridge/crop.json (plan B5).

Shows the live console frame preprocessed exactly as the agent sees it (crop -> gray -> max-pool -> 84x84) next to an
emulator reference image (`--reference`, a PNG of the full emulator screen, e.g. from docs/screenshots/), run through
the same preprocessing. Press q to quit.

    container/run.sh python bridge/calibrate.py --device 0 --reference docs/screenshots/surround_emulator.png
    container/run.sh python bridge/calibrate.py --video recording.avi        # replay a recording instead
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

from bridge.capture import Capture, FileCapture, crop_rgb, save_crop  # noqa: E402
from fb9.preprocess import process_frame, to_gray  # noqa: E402

SHOW = 336  # display size of each 84x84 preview


@dataclass
class Args:
    device: str = "0"
    video: str = ""                 # replay a video file instead of the live device
    width: int = 1280
    height: int = 720
    fps: int = 60
    out: str = "bridge/crop.json"
    reference: str = ""             # optional PNG of the full emulator screen (RGB or BGR image file)


def preview(gray84: np.ndarray, title: str) -> np.ndarray:
    img = cv2.resize(gray84, (SHOW, SHOW), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main() -> None:
    args = tyro.cli(Args)
    if args.video:
        source = FileCapture(args.video)
    else:
        source = Capture(int(args.device) if args.device.isdigit() else args.device, args.width, args.height, args.fps)

    for _ in range(10):  # let the capture settle
        rgb, _ = source.read()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = cv2.selectROI("drag the game screen, then press Enter", bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("no rectangle selected; crop.json not written")
        source.close()
        return
    save_crop(args.out, x, y, w, h)
    print(f"wrote {args.out}: x={x} y={y} w={w} h={h}")

    ref84 = None
    if args.reference:
        ref_bgr = cv2.imread(args.reference, cv2.IMREAD_COLOR)
        if ref_bgr is None:
            raise SystemExit(f"cannot read reference image {args.reference!r}")
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        ref84 = process_frame(ref_gray, ref_gray)

    crop = {"x": x, "y": y, "w": w, "h": h}
    try:
        while True:
            rgb, _ = source.read()
            g = to_gray(crop_rgb(rgb, crop))
            live84 = process_frame(g, g)
            left = preview(live84, "console (live)")
            right = preview(ref84, "emulator reference") if ref84 is not None else \
                np.zeros((SHOW, SHOW, 3), np.uint8)
            if ref84 is None:
                cv2.putText(right, "no --reference given", (6, SHOW // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
            cv2.imshow("calibration: 84x84 agent view", np.hstack([left, right]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except EOFError:
        pass  # end of a replayed video
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
