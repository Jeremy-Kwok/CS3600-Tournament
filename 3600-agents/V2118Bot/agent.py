"""
MyBot v2.3: HMM rat tracker + greedy carpet + negamax with integrated rat search.
"""
import sys
import traceback
from collections.abc import Callable
from typing import Dict, Optional, Tuple
import numpy as np

from game.board import Board
from game.move import Move
from game.enums import (
    MoveType, Cell, Direction, BOARD_SIZE,
    CARPET_POINTS_TABLE, loc_after_direction,
)


# ── Constants ──────────────────────────────────────────────────────────────────

# P(noise | cell_type):  rows = {SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3}
#                         cols = {SQUEAK=0, SCRATCH=1, SQUEAL=2}
NOISE_EMISSION = np.array([
    [0.7, 0.15, 0.15],   # SPACE
    [0.1, 0.8,  0.1],    # PRIMED
    [0.1, 0.1,  0.8],    # CARPET
    [0.5, 0.3,  0.2],    # BLOCKED
], dtype=np.float64)

DISTANCE_ERROR_PROBS = np.array([0.12, 0.7, 0.12, 0.06], dtype=np.float64)
DISTANCE_OFFSETS = np.array([-1, 0, 1, 2], dtype=np.int32)

OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}
ALL_DIRS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

# ── bitmask shift helpers (mirror Board._shift_mask_*) ────────────────────────
# Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3.
# Engine layout: bit_index = y * 8 + x. (0,0) at bit 0.

def _shift_up(m):    return (m >> 8) & 0x00FFFFFFFFFFFFFF
def _shift_down(m):  return (m << 8) & 0xFFFFFFFFFFFFFF00
def _shift_left(m):  return (m >> 1) & 0x7F7F7F7F7F7F7F7F
def _shift_right(m): return (m << 1) & 0xFEFEFEFEFEFEFEFE

# Indexed by Direction int value (UP=0, RIGHT=1, DOWN=2, LEFT=3)
SHIFTS = (_shift_up, _shift_right, _shift_down, _shift_left)

# Search EV breakeven is P > 1/3 ≈ 0.333, but the real breakeven after the
# 2-3 pt opportunity cost of a wasted carpet-build turn sits closer to 0.45.
# Albert's empirical hit rate at his floor is ~83%, ours at 0.35 was 50%.
SEARCH_MINIMAX_FLOOR = 0.45     # batch 5 values — higher volume + HMM accuracy
SEARCH_HIGH_FLOOR = 0.55       # tighter gate after budget exhausted
SEARCH_BUDGET = 6              # don't trigger tight gate too early
SEARCH_ALWAYS_THRESHOLD = 0.45 # greedy path matches minimax floor
SEARCH_CONSECUTIVE_OVERRIDE = 0.80  # allow back-to-back if very confident


def pos_to_idx(pos: Tuple[int, int]) -> int:
    return pos[1] * BOARD_SIZE + pos[0]


def idx_to_pos(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


# ── HMM Rat Tracker ───────────────────────────────────────────────────────────

class HMMRatTracker:
    """Belief vector over 64 cells tracking the hidden rat."""

    def __init__(self, T, board: Board):
        self.T = np.asarray(T, dtype=np.float64)

        # Stationary distribution: rat does 1000 headstart moves from (0,0).
        # Use repeated squaring for T^1024 ≈ T^1000 (exact, fast).
        stationary = np.zeros(64, dtype=np.float64)
        stationary[0] = 1.0
        T_pow = self.T.copy()
        for _ in range(10):  # 2^10 = 1024
            T_pow = T_pow @ T_pow
        stationary = stationary @ T_pow
        s = stationary.sum()
        self.stationary = stationary / s if s > 0 else np.ones(64) / 64.0
        self.belief = self.stationary.copy()

        self.cell_xs = (np.arange(64) % BOARD_SIZE).astype(np.int32)
        self.cell_ys = (np.arange(64) // BOARD_SIZE).astype(np.int32)
        self.cell_types = np.zeros(64, dtype=np.int32)
        # Precomputed Manhattan distance: dist_table[w_idx, r_idx] = |wx-rx|+|wy-ry|
        xs = self.cell_xs
        ys = self.cell_ys
        self._dist_table = (np.abs(xs[:, None] - xs[None, :]) +
                            np.abs(ys[:, None] - ys[None, :])).astype(np.int16)
        # Reusable buffer for likelihood (avoids per-call allocation)
        self._likelihood_buf = np.zeros(64, dtype=np.float64)
        self.refresh_cell_types(board)

    def refresh_cell_types(self, board: Board) -> None:
        # Direct mask reads — no get_cell dispatch.
        primed = board._primed_mask
        carpet = board._carpet_mask
        blocked = board._blocked_mask
        ct = self.cell_types
        # Cell enum values: SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3
        for i in range(64):
            bit = 1 << i
            if primed & bit:
                ct[i] = 1
            elif carpet & bit:
                ct[i] = 2
            elif blocked & bit:
                ct[i] = 3
            else:
                ct[i] = 0

    def predict(self) -> None:
        self.belief = self.belief @ self.T

    def update_noise(self, noise) -> None:
        emission = NOISE_EMISSION[self.cell_types, int(noise)]
        self.belief *= emission
        self._normalize()

    def update_distance(self, worker_loc: Tuple[int, int], reported: int) -> None:
        # Distance lookup via precomputed table; reuse a scratch likelihood buffer.
        w_idx = worker_loc[1] * BOARD_SIZE + worker_loc[0]
        actual = self._dist_table[w_idx]
        lk = self._likelihood_buf
        lk.fill(0.0)
        # Engine maxes negative distances to 0: max(0, actual+offset) == reported.
        # For offset=-1: equivalent to (actual==reported+1), with a special case
        # at reported==0 which also matches actual==0.
        for offset, prob in zip(DISTANCE_OFFSETS, DISTANCE_ERROR_PROBS):
            if offset < 0:
                if reported == 0:
                    lk += prob * ((actual == 0) | (actual == 1))
                else:
                    lk += prob * (actual == reported - offset)
            else:
                lk += prob * (actual == reported - offset)
        self.belief *= lk
        self._normalize()

    def apply_search_miss(self, loc: Optional[Tuple[int, int]]) -> None:
        if loc is None:
            return
        idx = pos_to_idx(loc)
        if 0 <= idx < 64:
            self.belief[idx] = 0.0
        self._normalize()

    def apply_search_hit(self) -> None:
        self.belief = self.stationary.copy()

    def _normalize(self) -> None:
        s = self.belief.sum()
        if s > 1e-300:
            self.belief /= s
            # Prevent permanent zero-traps: a cell zeroed by noisy observations
            # can never recover without a floor. 1e-8 per cell is negligible but
            # allows predict() to revive cells the rat might have moved to.
            np.maximum(self.belief, 1e-8, out=self.belief)
            self.belief /= self.belief.sum()
        else:
            self.belief = self.stationary.copy()

    def best_guess(self) -> Tuple[Tuple[int, int], float]:
        idx = int(np.argmax(self.belief))
        return idx_to_pos(idx), float(self.belief[idx])

    def top_k(self, k: int = 3):
        """Return list of (pos, prob) for top-k cells."""
        order = np.argsort(-self.belief)[:k]
        return [(idx_to_pos(int(i)), float(self.belief[i])) for i in order]


# ── Player Agent ───────────────────────────────────────────────────────────────

class PlayerAgent:
    """Greedy carpet-builder with HMM-based rat search integrated into minimax."""

    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        # ── error capture (visible in commentate) ──
        self._last_error: str = "none"
        self.init_error: str = ""
        self.minimax_failures: int = 0

        self.tracker: Optional[HMMRatTracker] = None
        if transition_matrix is not None:
            try:
                T_np = self._to_numpy_2d(transition_matrix)
                self.tracker = HMMRatTracker(T_np, board)
            except Exception as e:
                self.tracker = None
                self.init_error = f"{type(e).__name__}: {e}"

        self.is_first_turn = True
        self.turn_number = 0

        # ── search discipline state ──
        self._searched_last_turn = False

        # ── search diagnostics ──
        self.search_count = 0
        self.catches = 0
        self.search_beliefs: list = []   # belief at moment we decided to search
        self.max_belief_per_turn: list = []  # max belief each turn (for debugging)

        # ── minimax diagnostics ──
        self.max_depth_reached = 0
        self.total_nodes = 0

    @staticmethod
    def _to_numpy_2d(T) -> np.ndarray:
        """Robustly convert a transition matrix to a contiguous numpy float64 array.

        Handles numpy arrays, JAX arrays (via __array__ protocol), nested
        Python lists, and 0-d/1-d edge cases. Validates shape (64, 64).
        """
        # First attempt: direct conversion
        try:
            arr = np.asarray(T, dtype=np.float64)
        except Exception:
            arr = None

        # Fallback 1: JAX-style arrays expose .__array__()
        if arr is None or arr.ndim == 0:
            try:
                arr = np.asarray(T.__array__(), dtype=np.float64)
            except Exception:
                arr = None

        # Fallback 2: list-of-lists conversion (slow but always works)
        if arr is None or arr.ndim != 2:
            try:
                arr = np.array([[float(x) for x in row] for row in T],
                               dtype=np.float64)
            except Exception as e:
                raise ValueError(f"could not convert T to ndarray: {e}")

        if arr.shape != (BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE):
            raise ValueError(f"transition matrix has shape {arr.shape}, expected (64,64)")

        # Force contiguous to be safe
        return np.ascontiguousarray(arr)

    def commentate(self):
        try:
            e = str(self._last_error)[:80] if hasattr(self, '_last_error') else '?'
            ie = str(self.init_error)[:80] if hasattr(self, 'init_error') else '?'
            return "s%d c%d f%d d%d n%d e:%s ie:%s" % (
                getattr(self, 'search_count', 0),
                getattr(self, 'catches', 0),
                getattr(self, 'minimax_failures', 0),
                getattr(self, 'max_depth_reached', 0),
                getattr(self, 'total_nodes', 0),
                e, ie)
        except:
            return "err"

    # ─────────────────────── main entry ───────────────────────

    def play(self, board: Board, sensor_data, time_left: Callable):
        noise, distance = sensor_data
        self._update_belief(board, noise, distance)
        self.is_first_turn = False
        self.turn_number += 1

        # Record max belief this turn (for diagnostics)
        if self.tracker is not None:
            _, mb = self.tracker.best_guess()
            self.max_belief_per_turn.append(mb)

        move = self._choose_move_top(board, time_left)

        # Record search diagnostics + update consecutive-search flag
        if move is not None and move.move_type == MoveType.SEARCH:
            tag = getattr(self, '_search_path_tag', '?')
            print(f"[SEARCH-RET] turn={self.turn_number} count_before={self.search_count} tag={tag}",
                  file=sys.stderr)
            self.search_count += 1
            self._searched_last_turn = True
            if self.tracker is not None:
                _, p = self.tracker.best_guess()
                self.search_beliefs.append(p)
        else:
            self._searched_last_turn = False
        self._search_path_tag = None

        return move

    # ─────────────────────── top-level decision ───────────────────────

    def _choose_move_top(self, board: Board, time_left: Callable):
        """Minimax with greedy fallback. Captures any exception into
        self._last_error and appends a full traceback to error_log.txt
        in the working directory.
        """
        try:
            mm_move, _ = self._minimax_search(board, time_left)
            if mm_move is not None:
                return mm_move
        except Exception as e:
            self.minimax_failures += 1
            self._last_error = f"turn{self.turn_number} {type(e).__name__}: {e}"
            # stderr print — most likely to reach the bytefight logs
            try:
                print(f"[MM-FAIL] {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            except Exception:
                pass
            # File logging — in case the sandbox captures files in the working dir
            try:
                with open("error_log.txt", "a") as f:
                    f.write(f"\n=== turn {self.turn_number} ===\n")
                    traceback.print_exc(file=f)
            except Exception:
                pass
            # stdout — third channel
            try:
                print(f"[MyBot] minimax exception on turn {self.turn_number}: {e}")
            except Exception:
                pass

        return self._choose_move_greedy(board, time_left)

    # ─────────────────────── belief update ───────────────────────

    def _update_belief(self, board: Board, noise, distance: int) -> None:
        if self.tracker is None:
            return

        self.tracker.refresh_cell_types(board)

        if self.is_first_turn:
            # First rat move (always happens before any sample)
            self.tracker.predict()
            # If I'm Player B, A has already played and may have searched.
            # board.opponent_search will be populated with A's last search.
            # We need to apply that observation, then a second predict for
            # the rat move that happens between A's turn and B's sample.
            opp_loc, opp_caught = board.opponent_search
            if opp_loc is not None:
                if opp_caught:
                    self.tracker.apply_search_hit()
                else:
                    self.tracker.apply_search_miss(opp_loc)
                self.tracker.predict()
        else:
            # 1) My own search result from previous turn (belief still at T).
            my_loc, my_caught = board.player_search
            if my_loc is not None:
                if my_caught:
                    self.catches += 1
                    self.tracker.apply_search_hit()
                else:
                    self.tracker.apply_search_miss(my_loc)

            # 2) First rat move (pre-opponent).
            self.tracker.predict()

            # 3) Opponent's search.
            opp_loc, opp_caught = board.opponent_search
            if opp_loc is not None:
                if opp_caught:
                    self.tracker.apply_search_hit()
                else:
                    self.tracker.apply_search_miss(opp_loc)

            # 4) Second rat move (pre-my sample).
            self.tracker.predict()

        # Current sample
        worker_loc = board.player_worker.get_location()
        self.tracker.update_noise(noise)
        self.tracker.update_distance(worker_loc, distance)

    # ─────────────────────── board helpers ───────────────────────

    def _find_primed_chains(self, board: Board, my_loc) -> Dict[Direction, int]:
        """Bitmask walk: count consecutive PRIMED cells in each direction from my_loc."""
        primed = board._primed_mask
        ex, ey = board.opponent_worker.get_location()
        enemy_bit = 1 << (ey * BOARD_SIZE + ex)
        sx, sy = my_loc
        start_bit = 1 << (sy * BOARD_SIZE + sx)

        chains: Dict[Direction, int] = {}
        for d in ALL_DIRS:
            shift = SHIFTS[int(d)]
            length = 0
            cur = start_bit
            for _ in range(BOARD_SIZE - 1):
                cur = shift(cur)
                # Stop if off-board (cur=0), enemy, or not primed
                if not cur or (cur & enemy_bit) or not (cur & primed):
                    break
                length += 1
            chains[d] = length
        return chains

    def _max_chain_from(self, board: Board, start_loc, prime_dir, enemy_loc) -> int:
        """Bitmask walk: how many primes can we lay starting at start_loc going in prime_dir.

        Engine rules (board.py:98-106):
          - PRIME requires current cell to be SPACE (not primed/carpet)
          - Next cell must be NOT is_cell_blocked → can be SPACE or CARPET
          - So intermediate steps must walk SPACE→SPACE, but the FINAL step
            can land on CARPET (we lay one last prime, end on carpet, can't
            extend further).
        """
        ex, ey = enemy_loc
        enemy_bit = 1 << (ey * BOARD_SIZE + ex)
        # Cells we cannot stand-and-prime from (blockers for intermediate cells)
        cant_stand = (board._primed_mask | board._carpet_mask
                      | board._blocked_mask | enemy_bit)
        # Cells we cannot WALK ONTO (for the destination of any step)
        cant_walk = board._primed_mask | board._blocked_mask | enemy_bit

        sx, sy = start_loc
        cur_bit = 1 << (sy * BOARD_SIZE + sx)
        # Starting cell must be SPACE
        if cur_bit & cant_stand:
            return 0

        shift = SHIFTS[int(prime_dir)]
        count = 0
        for _ in range(BOARD_SIZE - 1):
            nxt = shift(cur_bit)
            if not nxt or (nxt & cant_walk):
                break
            count += 1
            cur_bit = nxt
            # If we just stepped onto CARPET we can't prime again from here.
            if cur_bit & board._carpet_mask:
                break
        return count

    def _area_control(self, board: Board, loc, enemy_loc) -> int:
        """Count SPACE cells reachable in 1-2 steps. Bitmask flood from loc."""
        ex, ey = enemy_loc
        enemy_bit = 1 << (ey * BOARD_SIZE + ex)
        sx, sy = loc
        start_bit = 1 << (sy * BOARD_SIZE + sx)
        # SPACE = NOT (primed | carpet | blocked) on the board.
        non_space = (board._primed_mask | board._carpet_mask
                     | board._blocked_mask | enemy_bit)
        # 1-step reachable
        ring1 = 0
        for shift in SHIFTS:
            ring1 |= shift(start_bit)
        ring1 &= ~non_space
        # 2-step reachable (excluding start cell)
        ring2 = 0
        for shift in SHIFTS:
            ring2 |= shift(ring1)
        ring2 &= ~(non_space | start_bit | ring1)
        return bin(ring1).count("1") + bin(ring2).count("1")

    def _opponent_chain_threat(self, board: Board, opp_loc, my_loc) -> float:
        mx = 0.0
        for d in ALL_DIRS:
            length = 0
            cur = opp_loc
            for _ in range(BOARD_SIZE - 1):
                cur = loc_after_direction(cur, d)
                if not board.is_valid_cell(cur) or cur == my_loc:
                    break
                if board.get_cell(cur) != Cell.PRIMED:
                    break
                length += 1
            if length >= 2:
                t = CARPET_POINTS_TABLE[length] * 0.5
                if t > mx:
                    mx = t
        return mx

    # ─────────────────────── greedy fallback ───────────────────────

    def _choose_move_greedy(self, board: Board, time_left: Callable):
        my_loc = board.player_worker.get_location()
        enemy_loc = board.opponent_worker.get_location()
        turns_left = board.player_worker.turns_left
        chains = self._find_primed_chains(board, my_loc)

        best_roll_dir, best_roll_len = None, 0
        for d, length in chains.items():
            if length >= 2 and length > best_roll_len:
                best_roll_dir, best_roll_len = d, length

        best_prime_dir, best_max_chain, best_additional, best_existing = None, 0, 0, 0
        for d in ALL_DIRS:
            if not board.is_valid_move(Move.prime(d)):
                continue
            opp = OPPOSITE_DIR[d]
            existing = chains[opp]
            additional = self._max_chain_from(board, my_loc, d, enemy_loc)
            if additional < 1:
                continue
            max_chain = min(existing + additional, BOARD_SIZE - 1)
            if max_chain > best_max_chain or (max_chain == best_max_chain and existing > best_existing):
                best_prime_dir, best_max_chain = d, max_chain
                best_additional, best_existing = additional, existing

        search_loc, search_prob = None, 0.0
        if self.tracker is not None:
            search_loc, search_prob = self.tracker.best_guess()

        # === CARRIE-MATCHING STRATEGY: fast prime→roll cycles ===
        # Carrie averages 7.2 rolls/game at length 2.3. She accepts length-2
        # rolls to keep the cycle fast. 44% of her rolls are length-2.
        # Her split: 42% prime, 28% plain, 18% carpet, 13% search.

        # 1) Always roll length-2+ immediately. Don't hold out for longer.
        #    Banked points > speculative longer chain.
        if best_roll_len >= 2:
            return Move.carpet(best_roll_dir, best_roll_len)

        # 2) Last turn: prime for +1 or search
        if turns_left <= 1:
            if best_prime_dir is not None:
                return Move.prime(best_prime_dir)
            if search_loc and self._should_search(search_prob):
                self._search_path_tag = 'greedy_lastturn'
                return Move.search(search_loc)
            return self._fallback(board, my_loc, enemy_loc, search_loc, search_prob)

        # 3) Extend chain — prime if any direction available
        if best_prime_dir is not None:
            return Move.prime(best_prime_dir)

        # 4) Search (only when nothing productive available)
        if search_loc and self._should_search(search_prob):
            self._search_path_tag = 'greedy_main'
            return Move.search(search_loc)

        # 5) Reposition
        p = self._best_plain_move(board, my_loc, enemy_loc)
        return p if p else self._fallback(board, my_loc, enemy_loc, search_loc, search_prob)

    def _should_search(self, search_prob: float) -> bool:
        """Centralised search-discipline gate. Used by greedy + fallback paths.

        Hard caps to prevent the catastrophic 17-21 search/game pattern
        seen on bytefight when the HMM tracker is miscalibrated.
        """
        # HARD CAP: never search more than 8 times per game, period.
        if self.search_count >= 8:
            return False
        # HIT RATE SHUTOFF: if we've searched 4+ times and hit rate is
        # below 30%, the HMM is clearly wrong — stop searching entirely.
        if self.search_count >= 4 and self.catches < self.search_count * 0.3:
            return False
        # Consecutive search block (unless very confident)
        if self._searched_last_turn and search_prob < SEARCH_CONSECUTIVE_OVERRIDE:
            return False
        floor = SEARCH_HIGH_FLOOR if self.search_count >= SEARCH_BUDGET \
                else SEARCH_ALWAYS_THRESHOLD
        return search_prob >= floor

    def _best_plain_move(self, board: Board, my_loc, enemy_loc):
        best_dir, best_score = None, -1
        for d in ALL_DIRS:
            if not board.is_valid_move(Move.plain(d)):
                continue
            target = loc_after_direction(my_loc, d)
            if board.get_cell(target) != Cell.SPACE:
                score = 0
            else:
                score = sum(self._max_chain_from(board, target, dd, enemy_loc) for dd in ALL_DIRS)
            if score > best_score:
                best_score, best_dir = score, d
        return Move.plain(best_dir) if best_dir is not None else None

    def _fallback(self, board, my_loc, enemy_loc, search_loc, search_prob):
        if search_loc and self._should_search(search_prob):
            self._search_path_tag = 'fallback_should'
            return Move.search(search_loc)
        moves = board.get_valid_moves(exclude_search=True)
        # Never play length-1 carpet from fallback (-1 pt)
        moves = [m for m in moves
                 if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if moves:
            return moves[0]
        moves = board.get_valid_moves(exclude_search=False)
        moves = [m for m in moves
                 if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if moves:
            # Check if first move is a search — log if so
            m0 = moves[0]
            if m0.move_type == MoveType.SEARCH:
                self._search_path_tag = 'fallback_list_search'
            return m0
        self._search_path_tag = 'fallback_zero_escape'
        return Move.search((0, 0))

    # ─────────────────────── minimax ───────────────────────

    def _minimax_search(self, board: Board, time_left: Callable):
        """Iterative-deepening negamax with search move injected at root."""
        remaining = time_left()
        try:
            print(f"[MM] turn={self.turn_number} remaining={remaining:.3f}",
                  file=sys.stderr)
        except Exception:
            pass
        turns_remaining = max(1, board.player_worker.turns_left)
        # Use ~80% of fair-share per turn, capped at 4.0s.
        budget = min(4.0, max(0.5, remaining / turns_remaining * 0.8))
        deadline = remaining - budget

        my_loc = board.player_worker.get_location()
        enemy_loc = board.opponent_worker.get_location()
        chains = self._find_primed_chains(board, my_loc)

        # ── gather candidate moves ──
        raw_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not raw_moves:
            return None, None
        # Hard-filter length-1 carpets — they always lose -1 pt and never help.
        raw_moves = [m for m in raw_moves
                     if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if not raw_moves:
            # Only length-1 carpets were available — let greedy handle this.
            return None, None

        scored = [(self._move_score(board, m, chains, my_loc, enemy_loc), m)
                  for m in raw_moves]

        # ── inject search move if discipline checks pass ──
        search_ev = 0.0
        can_search = (self.tracker is not None
                      and not self._searched_last_turn
                      and self.search_count < 8  # hard cap
                      and not (self.search_count >= 4 and self.catches < self.search_count * 0.3))
        print(f"[MM-GATE] turn={self.turn_number} sc={self.search_count} cc={self.catches} "
              f"srch_last={self._searched_last_turn} can_search={can_search}", file=sys.stderr)
        search_move_injected = None
        if can_search:
            s_loc, s_prob = self.tracker.best_guess()
            # Diminishing returns: tighten threshold after budget exhausted.
            floor = SEARCH_HIGH_FLOOR if self.search_count >= SEARCH_BUDGET \
                    else SEARCH_MINIMAX_FLOOR
            print(f"[MM-GATE] s_prob={s_prob:.3f} floor={floor:.3f}", file=sys.stderr)
            if s_prob >= floor:
                search_ev = 6.0 * s_prob - 2.0
                search_move = Move.search(s_loc)
                search_move_injected = search_move
                # Score it high enough to be tried early when EV is good
                scored.append((40 + search_ev * 10, search_move))

        scored.sort(key=lambda x: -x[0])
        moves = [m for _, m in scored][:10]

        best_move = moves[0]
        best_val = float('-inf')

        for depth in range(1, 50):
            if time_left() <= deadline:
                break
            alpha = float('-inf')
            beta = float('inf')
            current_best = None
            current_val = float('-inf')
            timed_out = False

            for move in moves:
                # ── forecast ──
                if move.move_type == MoveType.SEARCH:
                    # Custom forecast: apply expected point change manually.
                    # Round to nearest int so points stays an int (avoids any
                    # downstream type-confusion).
                    new_board = board.get_copy()
                    new_board.player_worker.points = int(
                        new_board.player_worker.points + round(search_ev)
                    )
                    new_board.end_turn()
                    new_board.reverse_perspective()
                else:
                    new_board = board.forecast_move(move, check_ok=False)
                    if new_board is None:
                        continue
                    new_board.reverse_perspective()

                try:
                    val = -self._negamax(new_board, depth - 1, -beta, -alpha,
                                         time_left, deadline)
                except TimeoutError:
                    timed_out = True
                    break

                if val > current_val:
                    current_val = val
                    current_best = move
                if val > alpha:
                    alpha = val

            if timed_out:
                break
            if current_best is not None:
                best_move = current_best
                best_val = current_val
                if depth > self.max_depth_reached:
                    self.max_depth_reached = depth
                # Move best to front for next iteration
                if best_move in moves:
                    moves.remove(best_move)
                    moves.insert(0, best_move)

        if best_move is not None and best_move.move_type == MoveType.SEARCH:
            # Tag where this search came from
            if search_move_injected is not None and best_move == search_move_injected:
                self._search_path_tag = 'minimax_injected'
            else:
                self._search_path_tag = 'minimax_UNKNOWN_SEARCH'
                print(f"[MM-UNKNOWN-SEARCH] turn={self.turn_number} sc={self.search_count} "
                      f"can_search={can_search} injected={search_move_injected}", file=sys.stderr)
        return best_move, best_val

    def _negamax(self, board: Board, depth, alpha, beta, time_left, deadline):
        self.total_nodes += 1
        if time_left() <= deadline:
            raise TimeoutError
        if depth <= 0 or board.is_game_over() or board.player_worker.turns_left <= 0:
            return self._evaluate_for_current(board)

        my_loc = board.player_worker.get_location()
        enemy_loc = board.opponent_worker.get_location()
        chains = self._find_primed_chains(board, my_loc)

        raw_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not raw_moves:
            return self._evaluate_for_current(board)
        # Filter length-1 carpets (always -1pt loss)
        raw_moves = [m for m in raw_moves
                     if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if not raw_moves:
            return self._evaluate_for_current(board)

        scored = [(self._move_score(board, m, chains, my_loc, enemy_loc), m)
                  for m in raw_moves]
        scored.sort(key=lambda x: -x[0])
        moves = [m for _, m in scored][:8]

        best = float('-inf')
        for move in moves:
            new_board = board.forecast_move(move, check_ok=False)
            if new_board is None:
                continue
            new_board.reverse_perspective()
            val = -self._negamax(new_board, depth - 1, -beta, -alpha,
                                 time_left, deadline)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best

    def _move_score(self, board, move, chains, my_loc, enemy_loc):
        """Heuristic move ordering. Philosophy: ROLL EARLY, ROLL OFTEN.

        In this game, either player can roll any primed cell. A chain you
        don't roll is a chain your opponent rolls. So:
        - CARPET length-3+ ALWAYS dominates PRIME (cash in > extend)
        - CARPET length-2 still penalized (low payoff) unless endgame
        - PRIME is scored by chain potential but never beats a rollable 3+
        - PLAIN is heavily penalized when any chain exists to roll or extend
        """
        mt = move.move_type
        tl = board.player_worker.turns_left
        best_chain = max(chains.values()) if chains else 0

        if mt == MoveType.CARPET:
            rl = move.roll_length
            if rl == 1:
                return -1000
            # Compute roll-end position for repositioning bonus
            end_loc = my_loc
            shift_fn = SHIFTS[int(move.direction)]
            cur_bit = 1 << (my_loc[1] * BOARD_SIZE + my_loc[0])
            for _ in range(rl):
                cur_bit = shift_fn(cur_bit)
            if cur_bit:
                end_idx = (cur_bit.bit_length() - 1)
                end_loc = (end_idx % BOARD_SIZE, end_idx // BOARD_SIZE)
            # Bonus for landing near buildable space (reduces future PLAINs)
            reposition_bonus = 0
            non_space = board._primed_mask | board._carpet_mask | board._blocked_mask
            ex, ey = enemy_loc
            non_space |= 1 << (ey * BOARD_SIZE + ex)
            end_bit = 1 << (end_loc[1] * BOARD_SIZE + end_loc[0])
            # Count adjacent SPACE cells from roll endpoint
            adj_space = 0
            for sf in SHIFTS:
                nb = sf(end_bit)
                if nb and not (nb & non_space):
                    adj_space += 1
            reposition_bonus = adj_space * 3  # up to +12

            # ALL carpet rolls length-2+ score 200+, beating any prime.
            # Carrie rolls length-2 in 44% of her carpets and wins. Don't
            # penalize short rolls — bank the points immediately.
            return 200 + CARPET_POINTS_TABLE[rl] * 5 + reposition_bonus

        if mt == MoveType.PRIME:
            d = move.direction
            existing = chains[OPPOSITE_DIR[d]]
            additional = self._max_chain_from(board, my_loc, d, enemy_loc)
            mc = existing + additional
            if mc > BOARD_SIZE - 1:
                mc = BOARD_SIZE - 1
            pv = CARPET_POINTS_TABLE[mc] if mc >= 2 else 0
            score = 50 + pv * 3 + existing * 3
            # Build-away bonus
            target = loc_after_direction(my_loc, d)
            tx, ty = target
            if 0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE:
                ox, oy = enemy_loc
                if abs(tx - ox) + abs(ty - oy) > abs(my_loc[0] - ox) + abs(my_loc[1] - oy):
                    score += 5
            return score

        if mt == MoveType.PLAIN:
            target = loc_after_direction(my_loc, move.direction)
            tx, ty = target
            if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE):
                return -50
            bit = 1 << (ty * BOARD_SIZE + tx)
            if (board._primed_mask | board._blocked_mask) & bit:
                return -50
            if board._carpet_mask & bit:
                return -15  # carpet traversal
            # Check if this PLAIN move reaches a position to steal opponent chains.
            # From the target cell, count rollable primed chains (any primed cells
            # are stealable, regardless of who primed them).
            steal_len = self._best_steal_from(board, target, my_loc)
            if steal_len >= 3:
                # Stealing a length-3+ chain is worth as much as building our own
                return 150 + CARPET_POINTS_TABLE[steal_len] * 3
            if steal_len >= 2:
                return 80
            # If we have an in-progress chain, don't wander
            if best_chain >= 2:
                return -40
            pot = 0
            for dd in ALL_DIRS:
                v = self._max_chain_from(board, target, dd, enemy_loc)
                if v > pot:
                    pot = v
            if pot < 3:
                return -20
            return 10 + pot * 3
        return 0

    def _best_steal_from(self, board, loc, my_loc) -> int:
        """Best carpet roll length available from loc (counting any primed cells).

        Used to detect steal opportunities — walking to loc and rolling
        opponent's primed chain next turn.
        """
        primed = board._primed_mask
        if not primed:
            return 0
        lx, ly = loc
        start_bit = 1 << (ly * BOARD_SIZE + lx)
        my_bit = 1 << (my_loc[1] * BOARD_SIZE + my_loc[0])
        best = 0
        for d in ALL_DIRS:
            shift = SHIFTS[int(d)]
            length = 0
            cur = start_bit
            for _ in range(BOARD_SIZE - 1):
                cur = shift(cur)
                if not cur or (cur & my_bit) or not (cur & primed):
                    break
                length += 1
            if length > best:
                best = length
        return best

    # ─────────────────────── evaluation ───────────────────────

    def _evaluate_for_current(self, board):
        """Clean score-differential eval.

        Philosophy: points already banked are safe. Future potential is
        discounted because either player can roll primed cells. We let
        the move ordering handle the "roll early" logic — the eval just
        needs to be accurate, not nudge behavior.
        """
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        my_loc = board.player_worker.get_location()
        opp_loc = board.opponent_worker.get_location()
        my_tl = board.player_worker.turns_left
        opp_tl = board.opponent_worker.turns_left

        my_pot = self._future_potential_points(board, my_loc, opp_loc, my_tl)
        opp_pot = self._future_potential_points(board, opp_loc, my_loc, opp_tl)

        my_area = self._area_control(board, my_loc, opp_loc)
        opp_area = self._area_control(board, opp_loc, my_loc)

        # Potential is heavily discounted — banked points are 7× more valuable
        # than speculative chains. This makes the search prefer rolling over
        # extending, because rolling converts potential → banked points.
        return (my_pts - opp_pts) + 0.15 * (my_pot - opp_pot) + 0.1 * (my_area - opp_area)

    def _prime_ownership(self, board, my_loc, opp_loc) -> float:
        """For each primed cell on the board, whoever is closer is the
        probable roller. Returns net "ownership" (positive = I own more).

        Each cell contributes: +1 if I'm clearly closer, -1 if opponent is,
        weighted by distance (close cells matter more).
        """
        primed = board._primed_mask
        if not primed:
            return 0.0
        mx, my_y = my_loc
        ox, oy = opp_loc
        balance = 0.0
        for i in range(64):
            bit = 1 << i
            if not (primed & bit):
                continue
            cx, cy = i % 8, i // 8
            my_d = abs(cx - mx) + abs(cy - my_y)
            opp_d = abs(cx - ox) + abs(cy - oy)
            # Cells very close to either player matter most
            min_d = min(my_d, opp_d)
            # Weight: distance-1 = 1.0, distance-2 = 0.7, ..., dist-7 ≈ 0.0
            weight = max(0.0, 1.0 - min_d / 8.0)
            if my_d < opp_d:
                balance += weight
            elif opp_d < my_d:
                balance -= weight
            # If tied, neither side gets it
        return balance

    def _plan_progress_bonus(self, board, loc, enemy_loc, turns):
        """Score how close we are to completing a length-5+ rollable chain.

        For each direction, look at: existing primed chain behind us +
        space ahead. If a 5+ chain is buildable in time, the eventual
        carpet payoff (CARPET_POINTS_TABLE[L]) is partially credited based
        on (primes_done / primes_total). 2-prime distance to a length-5
        chain credits ~60% of the 10-pt payoff = +6.
        """
        if turns <= 1:
            return 0.0
        best = 0.0
        for prime_dir in ALL_DIRS:
            opp_d = OPPOSITE_DIR[prime_dir]
            existing = 0
            cur = loc
            for _ in range(BOARD_SIZE - 1):
                cur = loc_after_direction(cur, opp_d)
                if not board.is_valid_cell(cur) or cur == enemy_loc:
                    break
                if board.get_cell(cur) != Cell.PRIMED:
                    break
                existing += 1
            additional = self._max_chain_from(board, loc, prime_dir, enemy_loc)
            max_L = min(existing + additional, BOARD_SIZE - 1)
            if max_L < 5:
                continue
            primes_needed = max_L - existing
            if primes_needed + 1 > turns:
                # Can we still finish a 5-chain?
                feasible_L = min(max_L, existing + turns - 1)
                if feasible_L < 5:
                    continue
                primes_needed = feasible_L - existing
                target_L = feasible_L
            else:
                target_L = max_L
            payoff = CARPET_POINTS_TABLE[target_L]
            primes_total = max(1, target_L - 0)
            progress = existing / primes_total  # 0..1
            credit = payoff * (0.4 + 0.5 * progress)
            if credit > best:
                best = credit
        return best

    def _region_overlap_penalty(self, board, my_loc, opp_loc) -> float:
        """Penalty if our worker is contesting the same board region as opponent.

        Cheap proxy: opponent has primed chains in some direction, and we're
        on the same row/column projecting into that direction.
        """
        opp_chains = self._find_primed_chains(board, opp_loc)
        max_chain = max(opp_chains.values()) if opp_chains else 0
        if max_chain < 2:
            return 0.0
        # Manhattan-distance proximity penalty
        d = abs(my_loc[0] - opp_loc[0]) + abs(my_loc[1] - opp_loc[1])
        if d >= 5:
            return 0.0
        return (5 - d) * 0.4 * (max_chain / 4.0)

    def _future_potential_points(self, board, loc, enemy_loc, turns):
        if turns <= 0:
            return 0

        primed = board._primed_mask
        carpet = board._carpet_mask
        blocked = board._blocked_mask
        ex, ey = enemy_loc
        enemy_bit = 1 << (ey * BOARD_SIZE + ex)
        non_space = primed | carpet | blocked | enemy_bit

        sx, sy = loc
        loc_bit = 1 << (sy * BOARD_SIZE + sx)

        reposition_cost = 0
        check_loc = loc
        check_bit = loc_bit
        # If our cell is not SPACE, we'd need to step off first.
        if loc_bit & non_space:
            reposition_cost = 1
            best_adj, best_adj_score, best_adj_bit = None, -1, 0
            for d in ALL_DIRS:
                shift = SHIFTS[int(d)]
                adj_bit = shift(loc_bit)
                if not adj_bit or (adj_bit & non_space):
                    continue
                # adj is SPACE; score it by total reachable chain length in 4 dirs
                ax = (adj_bit.bit_length() - 1) % BOARD_SIZE
                ay = (adj_bit.bit_length() - 1) // BOARD_SIZE
                adj = (ax, ay)
                sc = sum(self._max_chain_from(board, adj, dd, enemy_loc)
                         for dd in ALL_DIRS)
                if sc > best_adj_score:
                    best_adj_score = sc
                    best_adj = adj
                    best_adj_bit = adj_bit
            if best_adj is not None:
                check_loc = best_adj
                check_bit = best_adj_bit
            else:
                return 0

        effective_turns = turns - reposition_cost
        if effective_turns <= 0:
            return 0

        best = 0
        for prime_dir in ALL_DIRS:
            opp_d = OPPOSITE_DIR[prime_dir]
            opp_shift = SHIFTS[int(opp_d)]
            # Bitmask walk back along opp direction counting PRIMED cells
            existing = 0
            cur = check_bit
            for _ in range(BOARD_SIZE - 1):
                cur = opp_shift(cur)
                if not cur or (cur & enemy_bit) or not (cur & primed):
                    break
                existing += 1
            additional = self._max_chain_from(board, check_loc, prime_dir, enemy_loc)
            max_possible = existing + additional
            if max_possible >= 2:
                best_L = min(max_possible, existing + effective_turns - 1, BOARD_SIZE - 1)
                if best_L >= 2:
                    new_primes = max(0, best_L - existing)
                    # KEY FIX: if the chain is already rollable (new_primes == 0),
                    # don't credit the carpet payoff here — it belongs in my_pts
                    # once actually rolled. Crediting it as "potential" makes the
                    # search think extending is free and never rolls.
                    if new_primes == 0:
                        pts = 0
                    else:
                        pts = new_primes + CARPET_POINTS_TABLE[best_L]
                else:
                    pts = 1 if effective_turns >= 1 and additional >= 1 else 0
            else:
                pts = 1 if effective_turns >= 1 and additional >= 1 else 0
            if pts > best:
                best = pts
        return best
