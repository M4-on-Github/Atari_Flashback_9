"""Observation preprocessing shared by training, PC play and console play (docs/contracts.md §2)."""
import cv2
import numpy as np

OBS_SIZE = 84
STACK = 4
OBS_SHAPE = (STACK + 2, OBS_SIZE, OBS_SIZE)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> HxW uint8 (identical to ALE getScreenGrayscale)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def process_frame(prev_gray: np.ndarray, last_gray: np.ndarray) -> np.ndarray:
    """Max-pool the last two frames of a step (removes flicker), resize the full screen to 84x84."""
    pooled = np.maximum(prev_gray.reshape(prev_gray.shape[:2]), last_gray.reshape(last_gray.shape[:2]))
    return cv2.resize(pooled, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)


class FrameStack:
    """Last STACK frames plus two seat-indicator planes -> obs (6,84,84) uint8."""

    def __init__(self, seat: int):
        assert seat in (0, 1)
        self.buf = np.zeros(OBS_SHAPE, dtype=np.uint8)
        self.buf[STACK + seat] = 255

    def reset(self, frame84: np.ndarray) -> np.ndarray:
        self.buf[:STACK] = frame84
        return self.buf.copy()

    def push(self, frame84: np.ndarray) -> np.ndarray:
        self.buf[:STACK - 1] = self.buf[1:STACK]
        self.buf[STACK - 1] = frame84
        return self.buf.copy()
