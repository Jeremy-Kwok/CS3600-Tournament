"""CarrieBot — empirical imitator of the real tournament opponent Carrie.

This version is NOT trying to play well; it's trying to *behave* like the real
Carrie from bytefight so that MyBot can be tuned against a realistic proxy.

Extracted from 45 MyBot-vs-Carrie match records in Match Records/:

    Action mix (overall):
        prime:  45.0%
        plain:  26.3%
        carpet: 16.9%
        search: 11.8%
    Per-game avgs: ~6.6 carpet rolls, ~4.6 searches, 39 turns used
    Carpet length distribution (296 rolls total):
        1 -> 25%   2 -> 35%   3 -> 27%   4 -> 8%   5 -> 5%
    Avg carpet length: 2.33 — Carrie does NOT hold out for length-5+
    Real Carrie avg score vs MyBot: 42.91

    Phase behaviour:
        first-10 moves: 60% prime, 18% plain, 17% carpet,  4% search
        final-10 moves: 34% prime, 31% plain, 16% carpet, 20% search
    When ahead: less prime (38%), more search (15%), more walking (32%)
    When behind: more prime (44%), still 20% carpet

    Key implementation notes:
      * 132 of 296 carpets followed ZERO primes — Carrie walks to cold primed
        cells and rolls them. This agent explicitly searches for any adjacent
        primed run and rolls it when available.
      * She rolls length-1 frequently (25% of rolls are length 1), even though
        it's -1 points. We keep this "bug" because the real Carrie has it.
      * Search distribution: ~4.6 per game, quota ramps through the game.
      * No deep planning. Short chain building, immediate cash-in.

The goal is for this bot to post ~42 points/game against MyBot, matching real
Carrie's distribution — NOT to play optimally.
"""
import random
from collections.abc import Callable
from typing import Optional

import numpy as np

from game.board import Board
from game.move import Move
from game.enums import (
    MoveType, Cell, Direction, BOARD_SIZE,
    CARPET_POINTS_TABLE, loc_after_direction,
)


NOISE_EMISSION = np.array([
    [0.7, 0.15, 0.15],
    [0.1, 0.8,  0.1],
    [0.1, 0.1,  0.8],
    [0.5, 0.3,  0.2],
], dtype=np.float64)
DISTANCE_ERROR_PROBS = np.array([0.12, 0.7, 0.12, 0.06], dtype=np.float64)
DISTANCE_OFFSETS = np.array([-1, 0, 1, 2], dtype=np.int32)

ALL_DIRS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

# Bit shifts (engine layout: bit = y*8+x)
def _shift_up(m):    return (m >> 8) & 0x00FFFFFFFFFFFFFF
def _shift_down(m):  return (m << 8) & 0xFFFFFFFFFFFFFF00
def _shift_left(m):  return (m >> 1) & 0x7F7F7F7F7F7F7F7F
def _shift_right(m): return (m << 1) & 0xFEFEFEFEFEFEFEFE
SHIFTS = (_shift_up, _shift_right, _shift_down, _shift_left)


# Empirical carpet-length distribution (excludes len 6 & 7 — never observed)
CARPET_LEN_CHOICES = [1, 2, 3, 4, 5]
CARPET_LEN_WEIGHTS = [73, 105, 79, 25, 14]  # raw counts from 296 rolls


def pos_to_idx(p):
    return p[1] * BOARD_SIZE + p[0]


def idx_to_pos(i):
    return (i % BOARD_SIZE, i // BOARD_SIZE)


class SimpleHMM:
    """Minimal HMM rat tracker."""

    def __init__(self, T, board):
        self.T = np.asarray(T, dtype=np.float64)
        s = np.zeros(64, dtype=np.float64); s[0] = 1.0
        Tp = self.T.copy()
        for _ in range(10):
            Tp = Tp @ Tp
        s = s @ Tp
        tot = s.sum()
        self.stationary = s / tot if tot > 0 else np.ones(64) / 64.0
        self.belief = self.stationary.copy()
        self.cell_xs = (np.arange(64) % BOARD_SIZE).astype(np.int32)
        self.cell_ys = (np.arange(64) // BOARD_SIZE).astype(np.int32)
        self._dist_table = (np.abs(self.cell_xs[:, None] - self.cell_xs[None, :]) +
                            np.abs(self.cell_ys[:, None] - self.cell_ys[None, :])).astype(np.int16)
        self.cell_types = np.zeros(64, dtype=np.int32)
        self.refresh_cell_types(board)

    def refresh_cell_types(self, board):
        primed = board._primed_mask
        carpet = board._carpet_mask
        blocked = board._blocked_mask
        ct = self.cell_types
        for i in range(64):
            bit = 1 << i
            if primed & bit: ct[i] = 1
            elif carpet & bit: ct[i] = 2
            elif blocked & bit: ct[i] = 3
            else: ct[i] = 0

    def predict(self):
        self.belief = self.belief @ self.T

    def update_noise(self, noise):
        self.belief *= NOISE_EMISSION[self.cell_types, int(noise)]
        self._norm()

    def update_distance(self, wloc, reported):
        w_idx = wloc[1] * BOARD_SIZE + wloc[0]
        actual = self._dist_table[w_idx]
        lk = np.zeros(64, dtype=np.float64)
        for off, p in zip(DISTANCE_OFFSETS, DISTANCE_ERROR_PROBS):
            lk += p * (np.maximum(0, actual + off) == reported).astype(np.float64)
        self.belief *= lk
        self._norm()

    def apply_search_miss(self, loc):
        if loc is None:
            return
        idx = pos_to_idx(loc)
        if 0 <= idx < 64:
            self.belief[idx] = 0.0
        self._norm()

    def apply_search_hit(self):
        self.belief = self.stationary.copy()

    def _norm(self):
        s = self.belief.sum()
        if s > 1e-300:
            self.belief /= s
            np.maximum(self.belief, 1e-8, out=self.belief)
            self.belief /= self.belief.sum()
        else:
            self.belief = self.stationary.copy()

    def best_guess(self):
        i = int(np.argmax(self.belief))
        return idx_to_pos(i), float(self.belief[i])


class PlayerAgent:
    """CarrieBot — behavior-matching imitator of real Carrie.

    Strategy (per turn):
      1. Update rat belief.
      2. Decide action TYPE via phase/state-aware weighted roll that
         matches real Carrie's empirical distribution.
      3. Convert chosen type to a concrete Move:
           * CARPET -> find any adjacent primed run. If none, walk toward
                       the nearest primed cell (cold-roll). Length sampled
                       from empirical distribution, clamped to available.
           * PRIME  -> prime in longest-chain direction (simple).
           * SEARCH -> HMM best guess (quota-gated).
           * PLAIN  -> step toward highest chain-potential neighbour.
      4. If the chosen type isn't executable (e.g. no primes to roll),
         fall back through the priority list.

    Concrete move code below borrows from the prior CarrieBot skeleton but
    the DECISION layer is new and stats-driven.
    """

    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        self.tracker = None
        if transition_matrix is not None:
            try:
                T_np = np.asarray(transition_matrix, dtype=np.float64)
                self.tracker = SimpleHMM(T_np, board)
            except Exception:
                self.tracker = None

        self.turn_number = 0
        self.is_first_turn = True
        self.catches = 0
        self.search_count = 0
        # Seed varies per instance to desync parallel games while staying
        # deterministic within a single game seed.
        self.rng = random.Random()

    def commentate(self):
        try:
            return "CarrieBot t%d s%d c%d" % (
                self.turn_number, self.search_count, self.catches)
        except Exception:
            return "cb"

    def play(self, board: Board, sensor_data, time_left: Callable):
        noise, distance = sensor_data
        self._update_belief(board, noise, distance)
        self.is_first_turn = False
        self.turn_number += 1
        move = self._decide(board)
        return move

    # ── belief update ──────────────────────────────────────────────────────

    def _update_belief(self, board, noise, distance):
        if self.tracker is None:
            return
        self.tracker.refresh_cell_types(board)
        if self.is_first_turn:
            self.tracker.predict()
            opp_loc, opp_caught = board.opponent_search
            if opp_loc is not None:
                if opp_caught: self.tracker.apply_search_hit()
                else: self.tracker.apply_search_miss(opp_loc)
                self.tracker.predict()
        else:
            my_loc, my_caught = board.player_search
            if my_loc is not None:
                if my_caught:
                    self.catches += 1
                    self.tracker.apply_search_hit()
                else:
                    self.tracker.apply_search_miss(my_loc)
            self.tracker.predict()
            opp_loc, opp_caught = board.opponent_search
            if opp_loc is not None:
                if opp_caught: self.tracker.apply_search_hit()
                else: self.tracker.apply_search_miss(opp_loc)
            self.tracker.predict()
        worker_loc = board.player_worker.get_location()
        self.tracker.update_noise(noise)
        self.tracker.update_distance(worker_loc, distance)

    # ── phase-dependent weights ────────────────────────────────────────────

    def _action_weights(self, board):
        """Return (plain, prime, carpet, search) weights for this turn."""
        t = self.turn_number
        # Phase thresholds roughly match the 3 buckets used in the analysis.
        if t <= 10:
            w_plain, w_prime, w_carpet, w_search = 0.18, 0.60, 0.18, 0.04
        elif t <= 30:
            w_plain, w_prime, w_carpet, w_search = 0.27, 0.45, 0.17, 0.11
        else:
            w_plain, w_prime, w_carpet, w_search = 0.31, 0.34, 0.16, 0.20

        # Score-state adjustment (multiplicative, mild)
        try:
            my_pts = board.player_worker.get_points()
            opp_pts = board.opponent_worker.get_points()
        except Exception:
            my_pts = opp_pts = 0
        diff = my_pts - opp_pts
        if diff > 3:  # ahead -> more walking, more searching
            w_plain *= 1.20
            w_search *= 1.25
            w_prime *= 0.85
        elif diff < -3:  # behind -> more priming
            w_prime *= 1.15
            w_search *= 0.85
        return (w_plain, w_prime, w_carpet, w_search)

    def _pick_action_type(self, board):
        weights = self._action_weights(board)
        types = ['plain', 'prime', 'carpet', 'search']
        # Weighted sample
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for t_name, w in zip(types, weights):
            acc += w
            if r < acc:
                return t_name
        return 'prime'

    # ── decision logic ─────────────────────────────────────────────────────

    def _decide(self, board):
        my_loc = board.player_worker.get_location()
        enemy_loc = board.opponent_worker.get_location()

        chosen = self._pick_action_type(board)

        # Try the chosen action; fall through if impossible.
        order = self._fallback_order(chosen)

        for action in order:
            move = self._try_action(board, action, my_loc, enemy_loc)
            if move is not None:
                return move

        # Last resort: any valid non-len-1-carpet move.
        moves = board.get_valid_moves(exclude_search=True)
        moves = [m for m in moves
                 if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if moves:
            return moves[0]
        return Move.search((0, 0))

    def _fallback_order(self, chosen):
        # After the chosen action fails, try actions in decreasing
        # "naturalness" order matching Carrie's overall distribution.
        base = ['prime', 'plain', 'carpet', 'search']
        return [chosen] + [a for a in base if a != chosen]

    def _try_action(self, board, action, my_loc, enemy_loc):
        if action == 'carpet':
            return self._try_carpet(board, my_loc, enemy_loc)
        if action == 'prime':
            return self._try_prime(board, my_loc, enemy_loc)
        if action == 'search':
            return self._try_search(board)
        if action == 'plain':
            return self._try_plain(board, my_loc, enemy_loc)
        return None

    # ---- action executors ----

    def _sample_carpet_length(self, max_avail):
        """Sample a carpet length that matches Carrie's empirical distribution,
        clamped to what's actually available."""
        if max_avail <= 0:
            return 0
        # Restrict distribution to available lengths
        choices = [l for l in CARPET_LEN_CHOICES if l <= max_avail]
        weights = [w for l, w in zip(CARPET_LEN_CHOICES, CARPET_LEN_WEIGHTS)
                   if l <= max_avail]
        if not choices:
            return max_avail
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for l, w in zip(choices, weights):
            acc += w
            if r < acc:
                return l
        return choices[-1]

    def _try_carpet(self, board, my_loc, enemy_loc):
        # Find best adjacent primed run.
        best_dir, best_len = None, 0
        for d in ALL_DIRS:
            length = self._primed_chain_length(board, my_loc, d, enemy_loc)
            if length > best_len:
                best_dir, best_len = d, length
        if best_dir is None or best_len == 0:
            return None
        # Sample length from Carrie's distribution, clamped to what's available.
        L = self._sample_carpet_length(best_len)
        if L <= 0:
            return None
        move = Move.carpet(best_dir, L)
        if board.is_valid_move(move):
            return move
        return None

    def _try_prime(self, board, my_loc, enemy_loc):
        best_dir, best_max = None, 0
        for d in ALL_DIRS:
            if not board.is_valid_move(Move.prime(d)):
                continue
            length = self._max_prime_chain(board, my_loc, d, enemy_loc)
            if length > best_max:
                best_max, best_dir = length, d
        if best_dir is None:
            # Any valid prime at all?
            for d in ALL_DIRS:
                if board.is_valid_move(Move.prime(d)):
                    return Move.prime(d)
            return None
        return Move.prime(best_dir)

    def _try_search(self, board):
        if self.tracker is None:
            return None
        # Quota check — Carrie searches ~4.6/game.
        # Allow a little slack; early-game threshold higher, late lower.
        t = self.turn_number
        if t <= 10:
            threshold = 0.40
        elif t <= 25:
            threshold = 0.25
        else:
            threshold = 0.15
        # Budget: roughly matches 4.6 searches/game.
        budget = 2 + int(t * 0.15)  # ramps from 2 to ~8
        if self.search_count >= budget:
            return None
        loc, prob = self.tracker.best_guess()
        if prob >= threshold:
            self.search_count += 1
            return Move.search(loc)
        return None

    def _try_plain(self, board, my_loc, enemy_loc):
        # Prefer walking toward the nearest primed cell (to cold-roll it),
        # else toward the best build target, else any best 1-step plain.
        target = self._nearest_primed_cell(board, my_loc, enemy_loc)
        if target is not None:
            step = self._step_toward(board, my_loc, target, enemy_loc)
            if step is not None:
                mv = Move.plain(step)
                if board.is_valid_move(mv):
                    return mv
        target = self._best_build_target(board, my_loc, enemy_loc, max_dist=4)
        if target is not None:
            step = self._step_toward(board, my_loc, target, enemy_loc)
            if step is not None:
                mv = Move.plain(step)
                if board.is_valid_move(mv):
                    return mv
        return self._best_plain(board, my_loc, enemy_loc)

    # ---- helpers ----

    def _best_build_target(self, board, my_loc, enemy_loc, max_dist=4):
        from collections import deque
        visited = {my_loc: 0}
        q = deque([my_loc])
        candidates = []
        while q:
            cur = q.popleft()
            dist = visited[cur]
            cx, cy = cur
            cur_bit = 1 << (cy * 8 + cx)
            is_space = not ((board._primed_mask | board._carpet_mask
                             | board._blocked_mask) & cur_bit)
            if is_space and dist > 0:
                best_chain = 0
                for d in ALL_DIRS:
                    length = self._max_prime_chain(board, cur, d, enemy_loc)
                    if length > best_chain:
                        best_chain = length
                if best_chain >= 2:
                    score = best_chain * 2 - dist
                    candidates.append((score, dist, cur))
            if dist >= max_dist:
                continue
            for d in ALL_DIRS:
                nxt = loc_after_direction(cur, d)
                nx, ny = nxt
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                if nxt in visited or nxt == enemy_loc:
                    continue
                bit = 1 << (ny * 8 + nx)
                if (board._primed_mask | board._blocked_mask) & bit:
                    continue
                visited[nxt] = dist + 1
                q.append(nxt)
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][2]

    def _nearest_primed_cell(self, board, my_loc, enemy_loc):
        primed = board._primed_mask
        if not primed:
            return None
        from collections import deque
        visited = {my_loc}
        q = deque([(my_loc, 0)])
        while q:
            cur, dist = q.popleft()
            if dist >= 5:
                continue
            for d in ALL_DIRS:
                nxt = loc_after_direction(cur, d)
                nx, ny = nxt
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                if nxt in visited or nxt == enemy_loc:
                    continue
                visited.add(nxt)
                bit = 1 << (ny * 8 + nx)
                if primed & bit:
                    return nxt
                if (board._blocked_mask | board._primed_mask) & bit:
                    continue
                q.append((nxt, dist + 1))
        return None

    def _step_toward(self, board, src, dst, enemy_loc):
        from collections import deque
        if src == dst:
            return None
        visited = {src}
        q = deque()
        for d in ALL_DIRS:
            nxt = loc_after_direction(src, d)
            nx, ny = nxt
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            if nxt == enemy_loc or nxt in visited:
                continue
            bit = 1 << (ny * 8 + nx)
            if (board._blocked_mask | board._primed_mask) & bit:
                continue
            if nxt == dst:
                return d
            visited.add(nxt)
            q.append((nxt, d))
        while q:
            cur, first_d = q.popleft()
            for d in ALL_DIRS:
                nxt = loc_after_direction(cur, d)
                nx, ny = nxt
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                if nxt == enemy_loc or nxt in visited:
                    continue
                bit = 1 << (ny * 8 + nx)
                if (board._blocked_mask | board._primed_mask) & bit:
                    continue
                if nxt == dst:
                    return first_d
                visited.add(nxt)
                q.append((nxt, first_d))
        return None

    def _primed_chain_length(self, board, start, direction, enemy):
        primed = board._primed_mask
        ex, ey = enemy
        enemy_bit = 1 << (ey * BOARD_SIZE + ex)
        sx, sy = start
        cur = 1 << (sy * BOARD_SIZE + sx)
        shift = SHIFTS[int(direction)]
        length = 0
        for _ in range(7):
            cur = shift(cur)
            if not cur or (cur & enemy_bit) or not (cur & primed):
                break
            length += 1
        return length

    def _max_prime_chain(self, board, start, direction, enemy):
        primed = board._primed_mask
        carpet = board._carpet_mask
        blocked = board._blocked_mask
        ex, ey = enemy
        enemy_bit = 1 << (ey * BOARD_SIZE + ex)
        cant_stand = primed | carpet | blocked | enemy_bit
        cant_walk = primed | blocked | enemy_bit
        sx, sy = start
        cur = 1 << (sy * BOARD_SIZE + sx)
        if cur & cant_stand:
            return 0
        shift = SHIFTS[int(direction)]
        count = 0
        for _ in range(7):
            nxt = shift(cur)
            if not nxt or (nxt & cant_walk):
                break
            count += 1
            cur = nxt
            if cur & board._carpet_mask:
                break
        return count

    def _best_plain(self, board, my_loc, enemy_loc):
        best_dir, best_score = None, -1
        for d in ALL_DIRS:
            if not board.is_valid_move(Move.plain(d)):
                continue
            target = loc_after_direction(my_loc, d)
            if board.get_cell(target) != Cell.SPACE:
                score = 0
            else:
                score = sum(
                    self._max_prime_chain(board, target, dd, enemy_loc)
                    for dd in ALL_DIRS
                )
            if score > best_score:
                best_score, best_dir = score, d
        return Move.plain(best_dir) if best_dir is not None else None
