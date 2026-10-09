"""Scripted opponents for evaluation (plan.md §4.8). Pure numpy, no ALE import.

RandomBot: uniform random action index.
SurroundBot: reads the Surround grid from the RGB screen and steers toward the largest reachable open area.

Surround screen geometry (measured on the ALE screen, 210x160x3):
- Play field: x in [4, 156) = 38 columns of 4 px; y in [35, 197) = 18 rows of 9 px (walls/border are orange).
- Each cell is sampled at its centre: y = 35 + 9*row + 4, x = 4 + 4*col + 2. Trails are drawn in the wall colour,
  so a cell is occupied iff its centre is not background.
- Heads are 4x9 sprites in the player colour: blue (45,50,184) = seat 0 / port 1, green (92,186,92) = seat 1.
  A snake that has crashed loses its head sprite until the next round.
"""
from collections import deque

import numpy as np

# Surround action indices (docs/contracts.md §1): 0 NOOP, 1 UP, 2 RIGHT, 3 LEFT, 4 DOWN
S_UP, S_RIGHT, S_LEFT, S_DOWN = 1, 2, 3, 4
# (drow, dcol) per direction action
_DIR_DELTA = {S_UP: (-1, 0), S_RIGHT: (0, 1), S_LEFT: (0, -1), S_DOWN: (1, 0)}
_OPPOSITE = {S_UP: S_DOWN, S_DOWN: S_UP, S_LEFT: S_RIGHT, S_RIGHT: S_LEFT}

X0, CELL_W, NCOLS = 4, 4, 38
Y0, CELL_H, NROWS = 35, 9, 18
BG = np.array((184, 50, 50), dtype=np.int16)
WALL = np.array((227, 151, 89), dtype=np.int16)
SEAT_COLORS = (np.array((45, 50, 184), dtype=np.int16), np.array((92, 186, 92), dtype=np.int16))
EMPTY, WALL_CODE, OWN, OPP = 0, 1, 2, 3
RISK_PENALTY = 5   # cells next to the opponent head lose this many points of reachable area


class RandomBot:
    """Uniform random action index."""

    def __init__(self, num_actions: int, seed: int):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, rgb: np.ndarray, seat: int) -> int:
        return int(self.rng.integers(self.num_actions))


def parse_grid(rgb: np.ndarray, seat: int) -> np.ndarray:
    """Surround screen -> (NROWS, NCOLS) int8 codes: EMPTY, WALL_CODE (trail), OWN head, OPP head."""
    ys = Y0 + CELL_H * np.arange(NROWS)[:, None] + CELL_H // 2
    xs = X0 + CELL_W * np.arange(NCOLS)[None, :] + CELL_W // 2
    px = rgb[ys, xs].astype(np.int16)                       # (NROWS, NCOLS, 3)
    own = SEAT_COLORS[seat]
    opp = SEAT_COLORS[1 - seat]
    grid = np.full((NROWS, NCOLS), WALL_CODE, dtype=np.int8)  # anything not background counts as occupied
    grid[np.all(px == BG, axis=-1)] = EMPTY
    grid[np.all(px == own, axis=-1)] = OWN
    grid[np.all(px == opp, axis=-1)] = OPP
    return grid


class SurroundBot:
    """Flood-fill Surround player. Stateful only for heading and round restarts."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)   # only used to break exact ties
        self.reset()

    def reset(self) -> None:
        self.prev_head: tuple[int, int] | None = None
        self.heading: int | None = None
        self.prev_occupied = 0

    def act(self, rgb: np.ndarray, seat: int) -> int:
        grid = parse_grid(rgb, seat)
        occupied = int(np.count_nonzero(grid != EMPTY))
        own = np.argwhere(grid == OWN)
        opp = np.argwhere(grid == OPP)

        # new round: the board was cleared (trails only grow within a round)
        if occupied < self.prev_occupied:
            self.reset()
        self.prev_occupied = occupied

        if len(own) != 1:   # head not visible (crashed, or between rounds): keep going straight if we can
            return self.heading if self.heading is not None else S_UP
        head = (int(own[0][0]), int(own[0][1]))
        if self.prev_head is not None and head != self.prev_head:
            dr, dc = head[0] - self.prev_head[0], head[1] - self.prev_head[1]
            if max(abs(dr), abs(dc)) == 1 and (dr == 0 or dc == 0):
                self.heading = next(a for a, d in _DIR_DELTA.items() if d == (dr, dc))
            else:   # jumped: the board was reset under us
                self.heading = None
        self.prev_head = head

        opp_head = (int(opp[0][0]), int(opp[0][1])) if len(opp) == 1 else None
        blocked = grid != EMPTY
        blocked[head] = True
        best_action, best_score = None, None
        for action, (dr, dc) in _DIR_DELTA.items():
            if self.heading is not None and action == _OPPOSITE[self.heading]:
                continue   # never reverse into the neck
            nxt = (head[0] + dr, head[1] + dc)
            if not (0 <= nxt[0] < NROWS and 0 <= nxt[1] < NCOLS) or blocked[nxt]:
                continue
            area = _reachable_area(blocked, nxt)
            risky = opp_head is not None and max(abs(nxt[0] - opp_head[0]), abs(nxt[1] - opp_head[1])) <= 1
            score = (area - (RISK_PENALTY if risky else 0), action == self.heading)
            if best_score is None or score > best_score:
                best_action, best_score = action, score
        if best_action is None:   # boxed in: any action (we are lost anyway)
            return self.heading if self.heading is not None else S_UP
        return best_action


def _reachable_area(blocked: np.ndarray, start: tuple[int, int]) -> int:
    """Number of open cells reachable from start (4-connected), start included."""
    seen = np.zeros(blocked.shape, dtype=bool)
    seen[start] = True
    queue = deque([start])
    count = 0
    while queue:
        r, c = queue.popleft()
        count += 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < NROWS and 0 <= nc < NCOLS and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return count
