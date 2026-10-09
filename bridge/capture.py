"""Video sources for the console bridge: a live HDMI capture card, or a video file (tests, dry runs), and crop.json."""
import json
import os
import time

import cv2
import numpy as np


class Capture:
    """Live capture card. Asks for MJPG at width x height x fps (720p60 is the cheap-card mode we use)."""

    def __init__(self, device: int | str, width: int = 1280, height: int = 720, fps: int = 60):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low: no stale frames queued
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open capture device {device!r}")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self) -> tuple[np.ndarray, float]:
        """Blocking read -> (HxWx3 uint8 RGB, monotonic timestamp in seconds)."""
        ok, bgr = self.cap.read()
        ts = time.monotonic()
        if not ok:
            raise RuntimeError("capture read failed (device unplugged or wrong mode?)")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts

    def close(self) -> None:
        self.cap.release()


class FileCapture:
    """Replays a video file with the same read() API. Timestamps come from the frame index and the file's fps."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.index = 0

    def read(self) -> tuple[np.ndarray, float]:
        """-> (HxWx3 uint8 RGB, timestamp). Raises EOFError at the end of the file."""
        ok, bgr = self.cap.read()
        if not ok:
            raise EOFError
        ts = self.index / self.fps
        self.index += 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts

    def close(self) -> None:
        self.cap.release()


def load_crop(path: str) -> dict | None:
    """crop.json {"x","y","w","h"} in capture pixels, or None if the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return {k: int(d[k]) for k in ("x", "y", "w", "h")}


def save_crop(path: str, x: int, y: int, w: int, h: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, f, indent=2)
    os.replace(tmp, path)


def crop_rgb(rgb: np.ndarray, crop: dict | None) -> np.ndarray:
    """Cut the game screen out of a full capture frame. crop None = use the whole frame."""
    if crop is None:
        return rgb
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    return rgb[y:y + h, x:x + w]
