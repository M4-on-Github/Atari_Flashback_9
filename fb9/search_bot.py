"""Search-based Surround opponent (held-out yardstick, not trained against). Pure numpy + pure Python, no ALE import.

SearchBot: reads the Surround grid (fb9.bots.parse_grid), tracks both heads and headings, and picks a move with
depth-limited alpha-beta minimax over joint moves. A ply pair is resolved simultaneously: both snakes move, a shared
target cell or a move into an occupied cell / off the grid is a crash, both crashing is a draw. Leaf evaluation is
Voronoi territory (cells reached strictly first), or the flood-fill area difference (weighted) when the two snakes are
in separate regions. Iterative deepening stops on a deterministic node budget (evaluations + move resolutions), so the
chosen move depends only on the board, never on wall-clock time.
"""
import numpy as np

from fb9.bots import (EMPTY, NCOLS, NROWS, OPP, OWN, S_DOWN, S_LEFT, S_RIGHT, S_UP, _DIR_DELTA, _OPPOSITE,
                      parse_grid)

N = NROWS * NCOLS
ACTIONS = (S_UP, S_RIGHT, S_LEFT, S_DOWN)
WIN = 10_000          # terminal score magnitude; a win/loss at ply k scores WIN - k (earlier wins, later losses)
INF = 10 ** 9
SEP_WEIGHT = 2        # flood-fill area difference weight when the snakes are in separate regions
MAX_DEPTH = 40        # iterative deepening cap (ply pairs); the node budget normally stops well before this
NODE_BUDGET = 300     # evaluations + resolutions per act() beyond depth 1 (mean act() ~4 ms, see tests)


def _build_tables():
    """NXT[action][cell] -> target cell or -1 (off grid); NEIGH[cell] -> tuple of on-grid neighbour cells."""
    nxt = [None] * 5
    for a in ACTIONS:
        dr, dc = _DIR_DELTA[a]
        tab = []
        for i in range(N):
            r, c = divmod(i, NCOLS)
            nr, nc = r + dr, c + dc
            tab.append(nr * NCOLS + nc if (0 <= nr < NROWS and 0 <= nc < NCOLS) else -1)
        nxt[a] = tab
    neigh = tuple(tuple(nxt[a][i] for a in ACTIONS if nxt[a][i] >= 0) for i in range(N))
    return nxt, neigh


NXT, NEIGH = _build_tables()
# legal moves for a heading (never the reverse); the straight move comes first
MOVES = {None: ACTIONS}
for _h in ACTIONS:
    MOVES[_h] = (_h,) + tuple(a for a in ACTIONS if a not in (_h, _OPPOSITE[_h]))


def _heading_between(prev, head):
    """Action index of the single-cell step prev -> head, or None if they are not orthogonally adjacent."""
    dr, dc = head[0] - prev[0], head[1] - prev[1]
    if max(abs(dr), abs(dc)) == 1 and (dr == 0 or dc == 0):
        return next(a for a, d in _DIR_DELTA.items() if d == (dr, dc))
    return None


class _Search:
    """Alpha-beta over joint moves. State is mutated in place and restored on the way back up."""

    def __init__(self, blocked: bytearray, own_head: int, opp_head: int, own_h, opp_h):
        self.blocked = blocked
        self.p0, self.p1 = own_head, opp_head
        self.h0, self.h1 = own_h, opp_h
        self.nodes = 0
        self.limit = INF
        self.depth = 0            # deepest completed iteration

    def run(self, root_moves, straight, budget: int) -> dict:
        """Iterative deepening over our root moves. Returns {action: value} of the last completed depth."""
        values, hint = None, {}
        for depth in range(1, MAX_DEPTH + 1):
            self.limit = INF if depth == 1 else budget   # depth 1 always completes (a handful of nodes)
            order = sorted(root_moves, key=lambda a: (-hint.get(a, 0), a != straight))
            cur, alpha = {}, -INF
            for a0 in order:
                cur[a0] = self._min(a0, depth, alpha, INF, 0)
                alpha = max(alpha, cur[a0])
                if self.nodes > self.limit:
                    break
            if self.nodes > self.limit:
                break
            values, hint = cur, cur
            self.depth = depth
        return values

    def _max(self, depth, alpha, beta, ply):
        """Our node: maximise over our moves."""
        self.nodes += 1
        best = -INF
        for a0 in MOVES[self.h0]:
            v = self._min(a0, depth, alpha, beta, ply)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta or self.nodes > self.limit:
                break
        return best

    def _min(self, a0, depth, alpha, beta, ply):
        """Opponent's node for a fixed move of ours: minimise over its moves."""
        best = INF
        for a1 in MOVES[self.h1]:
            v = self._play(a0, a1, depth, alpha, beta, ply)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta or self.nodes > self.limit:
                break
        return best

    def _play(self, a0, a1, depth, alpha, beta, ply):
        blocked = self.blocked
        n0, n1 = NXT[a0][self.p0], NXT[a1][self.p1]
        c0 = n0 < 0 or blocked[n0]
        c1 = n1 < 0 or blocked[n1]
        if n0 >= 0 and n0 == n1:          # both enter the same empty cell: both crash
            c0 = c1 = True
        self.nodes += 1
        if c0 and c1:
            return 0
        if c0:
            return -(WIN - (ply + 1))
        if c1:
            return WIN - (ply + 1)

        old = (self.p0, self.p1, self.h0, self.h1)
        blocked[n0] = 1
        blocked[n1] = 1
        self.p0, self.p1, self.h0, self.h1 = n0, n1, a0, a1
        if depth == 1:
            v = self._eval()
        else:
            v = self._max(depth - 1, alpha, beta, ply + 1)
        self.p0, self.p1, self.h0, self.h1 = old
        blocked[n0] = 0
        blocked[n1] = 0
        return v

    def _eval(self) -> int:
        """Voronoi territory from both heads, level by level. Returns (own - opp) strictly-first cells, or the
        weighted area difference if no cell is reachable from both snakes (separate regions)."""
        self.nodes += 1
        blocked = self.blocked
        a, b = self.p0, self.p1
        # seen: 0 free, 1 ours, 2 theirs, 3 tie (same level, excluded), 4 our claim at the current level
        seen = bytearray(N)
        seen[a] = 1
        seen[b] = 2
        fa, fb = [a], [b]
        contact = False
        while fa or fb:
            na = []
            for x in fa:
                for y in NEIGH[x]:
                    if y == b:
                        contact = True
                        continue
                    if blocked[y]:
                        continue
                    s = seen[y]
                    if s == 0:
                        seen[y] = 4
                        na.append(y)
                    elif s == 2:
                        contact = True
            nb = []
            for x in fb:
                for y in NEIGH[x]:
                    if y == a:
                        contact = True
                        continue
                    if blocked[y]:
                        continue
                    s = seen[y]
                    if s == 0:
                        seen[y] = 2
                        nb.append(y)
                    elif s == 4:          # reached by both at the same level: tie
                        seen[y] = 3
                        contact = True
                    elif s == 1:
                        contact = True
            fa = []
            for y in na:
                if seen[y] == 4:
                    seen[y] = 1
                    fa.append(y)
            fb = nb
        diff = seen.count(1) - seen.count(2)
        return diff if contact else SEP_WEIGHT * diff


def _flood_area(blocked: bytearray, start: int) -> int:
    """Open cells reachable from start (start included), 4-connected, over the cells not marked in blocked."""
    seen = bytearray(blocked)
    seen[start] = 1
    stack = [start]
    count = 0
    while stack:
        x = stack.pop()
        count += 1
        for y in NEIGH[x]:
            if not seen[y]:
                seen[y] = 1
                stack.append(y)
    return count


class SearchBot:
    """Alpha-beta Surround player with Voronoi evaluation. Stateful for headings, round restarts and a decision cache.

    The decision is a pure function of the board and both headings, so an identical state (the head has not moved
    since the last call, which is most agent steps) returns the cached action.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)   # only used to break exact ties
        self.reset()

    def reset(self) -> None:
        self.prev_occupied = 0
        self._new_round()

    def _new_round(self) -> None:
        self.prev = [None, None]        # last seen head cell of (own, opponent)
        self.heading = [None, None]     # last move direction of (own, opponent), None if unknown
        self.cache_key = None
        self.cache_action = None
        self.cache_values = None
        self.last_values = None         # {action: value} of the last search, None if act() did not search

    def act(self, rgb: np.ndarray, seat: int) -> int:
        self.last_values = None
        grid = parse_grid(rgb, seat)
        occupied = int(np.count_nonzero(grid != EMPTY))
        if occupied < self.prev_occupied:   # the board was cleared: a new round
            self._new_round()
        self.prev_occupied = occupied

        heads = []
        for k, code in enumerate((OWN, OPP)):
            pos = np.argwhere(grid == code)
            if len(pos) == 1:
                head = (int(pos[0][0]), int(pos[0][1]))
                if self.prev[k] is not None and head != self.prev[k]:
                    self.heading[k] = _heading_between(self.prev[k], head)
                self.prev[k] = head
                heads.append(head[0] * NCOLS + head[1])   # flat cell index
            else:                             # crashed or between rounds: no head sprite
                heads.append(None)
        own_head, opp_head = heads
        own_h, opp_h = self.heading

        if own_head is None:   # our head is not visible: keep going straight if we can
            return own_h if own_h is not None else S_UP

        blocked = bytearray((grid != EMPTY).ravel().astype(np.uint8).tobytes())
        key = (bytes(blocked), own_head, opp_head, own_h, opp_h)
        if key == self.cache_key:
            self.last_values = self.cache_values
            return self.cache_action
        if opp_head is None:
            action = self._solo(blocked, own_head, own_h)
        else:
            action = self._search(blocked, own_head, opp_head, own_h, opp_h)
        self.cache_key, self.cache_action, self.cache_values = key, action, self.last_values
        return action

    def _search(self, blocked, own_head, opp_head, own_h, opp_h) -> int:
        search = _Search(blocked, own_head, opp_head, own_h, opp_h)
        values = search.run(MOVES[own_h], own_h, NODE_BUDGET)
        self.last_values = values
        best = max(values.values())
        cands = sorted(a for a, v in values.items() if v == best)
        if own_h in cands:
            return own_h
        return cands[int(self.rng.integers(len(cands)))]

    def _solo(self, blocked, own_head, own_h) -> int:
        """Opponent head not visible (it has crashed): keep as much open area reachable as possible."""
        scores = {}
        for a in MOVES[own_h]:
            n = NXT[a][own_head]
            if n >= 0 and not blocked[n]:
                scores[a] = _flood_area(blocked, n)
        if not scores:
            return own_h if own_h is not None else S_UP
        best = max(scores.values())
        cands = sorted(a for a, v in scores.items() if v == best)
        if own_h in cands:
            return own_h
        return cands[int(self.rng.integers(len(cands)))]
