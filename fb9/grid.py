"""Surround grid observation: one pixel per cell of the 18x38 play field, seat-relative (no multi_agent_ale_py import).

Geometry and palette come from fb9.bots. Each cell is read as the mean RGB of a 3x2 patch at its centre and assigned
the nearest palette colour, which tolerates capture-card noise. On the emulator this agrees with parse_grid.
"""
import numpy as np

from fb9.bots import BG, CELL_H, CELL_W, NCOLS, NROWS, SEAT_COLORS, WALL, X0, Y0
from fb9.preprocess import OBS_SHAPE

C_EMPTY, C_WALL, C_BLUE, C_GREEN = 0, 1, 2, 3
_PALETTE = np.stack([BG, WALL, SEAT_COLORS[0], SEAT_COLORS[1]]).astype(np.float32)   # (4,3), order = codes below
_PALETTE_CODES = np.array([C_EMPTY, C_WALL, C_BLUE, C_GREEN], dtype=np.int8)

# sample grid: rows centre-1..centre+1, cols centre-1..centre (inside the 4x9 cell)
_YS = (Y0 + CELL_H * np.arange(NROWS)[:, None, None, None] + CELL_H // 2
       + np.arange(-1, 2)[None, None, :, None])                                     # (NROWS,1,3,1)
_XS = (X0 + CELL_W * np.arange(NCOLS)[None, :, None, None] + CELL_W // 2
       + np.arange(-1, 1)[None, None, None, :])                                     # (1,NCOLS,1,2)

GRID_STACK = 2
GRID_PLANES = 3   # occupied, own head, opponent head
GRID_OBS_SHAPE = (GRID_STACK * GRID_PLANES, NROWS, NCOLS)


def classify_cells(rgb: np.ndarray) -> np.ndarray:
    """Surround screen (210,160,3) uint8 -> (NROWS, NCOLS) int8 codes C_EMPTY, C_WALL, C_BLUE, C_GREEN."""
    patch = rgb[_YS, _XS].astype(np.float32)                 # (NROWS,NCOLS,3,2,3)
    mean = patch.mean(axis=(2, 3))                           # (NROWS,NCOLS,3)
    dist = ((mean[:, :, None, :] - _PALETTE) ** 2).sum(axis=-1)   # (NROWS,NCOLS,4)
    return _PALETTE_CODES[dist.argmin(axis=-1)]


def grid_planes(codes: np.ndarray, seat: int) -> np.ndarray:
    """Cell codes -> (GRID_PLANES, NROWS, NCOLS) uint8 in {0, 255}: occupied, own head, opponent head."""
    assert seat in (0, 1)
    own_code, opp_code = (C_BLUE, C_GREEN) if seat == 0 else (C_GREEN, C_BLUE)
    planes = np.stack([codes != C_EMPTY, codes == own_code, codes == opp_code])
    return planes.astype(np.uint8) * 255


class GridStack:
    """Last GRID_STACK grid frames of one seat -> obs GRID_OBS_SHAPE uint8, oldest frame first."""

    def __init__(self, seat: int):
        assert seat in (0, 1)
        self.seat = seat
        self.buf = np.zeros(GRID_OBS_SHAPE, dtype=np.uint8)

    def reset(self, rgb: np.ndarray | None, codes: np.ndarray | None = None) -> np.ndarray:
        """rgb screen, or codes = classify_cells(rgb) when the caller already has them (shared by both seats)."""
        planes = grid_planes(classify_cells(rgb) if codes is None else codes, self.seat)
        self.buf[:] = np.tile(planes, (GRID_STACK, 1, 1))
        return self.buf.copy()

    def push(self, rgb: np.ndarray | None, codes: np.ndarray | None = None) -> np.ndarray:
        self.buf[:-GRID_PLANES] = self.buf[GRID_PLANES:]
        self.buf[-GRID_PLANES:] = grid_planes(classify_cells(rgb) if codes is None else codes, self.seat)
        return self.buf.copy()


def obs_shape(kind: str) -> tuple:
    """Observation shape for an obs kind ("pixels" or "grid")."""
    if kind == "pixels":
        return OBS_SHAPE
    if kind == "grid":
        return GRID_OBS_SHAPE
    raise ValueError(f"unknown obs kind {kind!r}")
