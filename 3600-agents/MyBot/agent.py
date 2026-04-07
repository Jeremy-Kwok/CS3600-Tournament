"""
MyBot: HMM rat tracker + greedy carpet-building strategy.
Milestone 1 target: beat Yolanda (random bot) consistently.
"""
from collections.abc import Callable
from typing import Dict, Optional, Tuple
import numpy as np

from game.board import Board
from game.move import Move
from game.enums import (
    MoveType, Cell, Direction, BOARD_SIZE,
    CARPET_POINTS_TABLE, loc_after_direction,
)


# P(noise | cell_type) where cell_type in {SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3}
# and noise in {SQUEAK=0, SCRATCH=1, SQUEAL=2}.
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


def pos_to_idx(pos: Tuple[int, int]) -> int:
    return pos[1] * BOARD_SIZE + pos[0]


def idx_to_pos(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


class HMMRatTracker:
    """Tracks rat location as a belief vector over 64 cells."""

    def __init__(self, T, board: Board):
        self.T = np.asarray(T, dtype=np.float64)

        # Stationary distribution after ~1000 headstart moves from (0,0).
        stationary = np.zeros(64, dtype=np.float64)
        stationary[0] = 1.0
        for _ in range(200):
            stationary = stationary @ self.T
        s = stationary.sum()
        self.stationary = stationary / s if s > 0 else np.ones(64) / 64.0
        self.belief = self.stationary.copy()

        # Coords & cached cell types (refreshed each turn).
        self.cell_xs = (np.arange(64) % BOARD_SIZE).astype(np.int32)
        self.cell_ys = (np.arange(64) // BOARD_SIZE).astype(np.int32)
        self.cell_types = np.zeros(64, dtype=np.int32)
        self.refresh_cell_types(board)

    def refresh_cell_types(self, board: Board) -> None:
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self.cell_types[y * BOARD_SIZE + x] = int(board.get_cell((x, y)))

    def predict(self) -> None:
        self.belief = self.belief @ self.T

    def update_noise(self, noise) -> None:
        emission = NOISE_EMISSION[self.cell_types, int(noise)]
        self.belief = self.belief * emission
        self._normalize()

    def update_distance(self, worker_loc: Tuple[int, int], reported: int) -> None:
        wx, wy = worker_loc
        actual = np.abs(self.cell_xs - wx) + np.abs(self.cell_ys - wy)
        likelihood = np.zeros(64, dtype=np.float64)
        for offset, prob in zip(DISTANCE_OFFSETS, DISTANCE_ERROR_PROBS):
            expected = np.maximum(0, actual + offset)
            likelihood = likelihood + prob * (expected == reported).astype(np.float64)
        self.belief = self.belief * likelihood
        self._normalize()

    def apply_search_miss(self, loc: Optional[Tuple[int, int]]) -> None:
        if loc is None:
            return
        idx = pos_to_idx(loc)
        if 0 <= idx < 64:
            self.belief[idx] = 0.0
        self._normalize()

    def apply_search_hit(self) -> None:
        # Rat respawned at (0,0) with HEADSTART_MOVES -> stationary again.
        self.belief = self.stationary.copy()

    def _normalize(self) -> None:
        s = self.belief.sum()
        if s > 0:
            self.belief = self.belief / s
        else:
            self.belief = self.stationary.copy()

    def best_guess(self) -> Tuple[Tuple[int, int], float]:
        idx = int(np.argmax(self.belief))
        return idx_to_pos(idx), float(self.belief[idx])


class PlayerAgent:
    """Greedy carpet-builder with HMM-based rat search."""

    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        self.tracker: Optional[HMMRatTracker] = None
        if transition_matrix is not None:
            try:
                T_np = np.asarray(transition_matrix, dtype=np.float64)
                self.tracker = HMMRatTracker(T_np, board)
            except Exception:
                self.tracker = None

        self.is_first_turn = True
        self.turn_number = 0
        self.search_count = 0
        self.catches = 0
        self.max_depth_reached = 0
        self.total_nodes = 0

    def commentate(self) -> str:
        msg = (f"MyBot v1 | turns: {self.turn_number} | searches: {self.search_count} "
               f"| catches: {self.catches} | max_depth: {self.max_depth_reached} "
               f"| nodes: {self.total_nodes}")
        if self.tracker is not None:
            _, p = self.tracker.best_guess()
            msg += f" | final max belief: {p:.3f}"
        return msg

    def play(self, board: Board, sensor_data, time_left: Callable):
        noise, distance = sensor_data

        self._update_belief(board, noise, distance)
        self.is_first_turn = False
        self.turn_number += 1

        move = self._choose_move_top(board, time_left)

        if move is not None and move.move_type == MoveType.SEARCH:
            self.search_count += 1

        return move

    # ---------------------- Top-level move selection ----------------------

    def _choose_move_top(self, board: Board, time_left: Callable):
        """High-confidence search first, then minimax, then greedy fallback."""
        search_loc = None
        search_prob = 0.0
        if self.tracker is not None:
            search_loc, search_prob = self.tracker.best_guess()
        if search_loc is not None and search_prob >= 0.65:
            return Move.search(search_loc)

        try:
            mm_move, _ = self._minimax_search(board, time_left)
            if mm_move is not None:
                return mm_move
        except Exception:
            pass

        return self._choose_move(board, time_left)

    # ---------------------- Belief update ----------------------

    def _update_belief(self, board: Board, noise, distance: int) -> None:
        if self.tracker is None:
            return

        self.tracker.refresh_cell_types(board)

        if self.is_first_turn:
            # Approximately stationary; predict once for the rat move before sample.
            self.tracker.predict()
        else:
            # Belief state currently represents rat at my previous play time.
            # Timeline between plays: [my_search resolved @ T] -> rat.move
            #   -> opp samples -> [opp_search resolved] -> rat.move -> my sample.
            # Apply updates in that temporal order.

            # 1) My own search from turn T (result learned now, belief still at T).
            my_loc, my_caught = board.player_search
            if my_loc is not None:
                if my_caught:
                    self.catches += 1
                    self.tracker.apply_search_hit()
                else:
                    self.tracker.apply_search_miss(my_loc)

            # 2) First rat move (pre-opponent).
            self.tracker.predict()

            # 3) Opponent's search at T+1 (we observe it now).
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

    # ---------------------- Board analysis helpers ----------------------

    def _find_primed_chains(self, board: Board, my_loc) -> Dict[Direction, int]:
        """Count adjacent PRIMED cells extending in each direction from my_loc."""
        enemy_loc = board.opponent_worker.get_location()
        chains: Dict[Direction, int] = {}
        for d in ALL_DIRS:
            length = 0
            cur = my_loc
            for _ in range(BOARD_SIZE - 1):
                cur = loc_after_direction(cur, d)
                if not board.is_valid_cell(cur):
                    break
                if cur == enemy_loc:
                    break
                if board.get_cell(cur) != Cell.PRIMED:
                    break
                length += 1
            chains[d] = length
        return chains

    def _max_chain_from(self, board: Board, start_loc, prime_dir, enemy_loc) -> int:
        """How many consecutive primes can we chain starting at start_loc going in prime_dir?

        Each iteration requires: current cell is SPACE, next cell is not BLOCKED/PRIMED/enemy.
        After the move, we're at the next cell (which becomes the new "current" cell).
        """
        count = 0
        cur = start_loc
        for _ in range(BOARD_SIZE - 1):
            if not board.is_valid_cell(cur):
                break
            if board.get_cell(cur) != Cell.SPACE:
                break
            if cur == enemy_loc:
                break
            next_loc = loc_after_direction(cur, prime_dir)
            if not board.is_valid_cell(next_loc):
                break
            if next_loc == enemy_loc:
                break
            next_cell = board.get_cell(next_loc)
            if next_cell == Cell.BLOCKED or next_cell == Cell.PRIMED:
                break
            count += 1
            cur = next_loc
        return count

    # ---------------------- Move selection ----------------------

    def _choose_move(self, board: Board, time_left: Callable):
        my_loc = board.player_worker.get_location()
        enemy_loc = board.opponent_worker.get_location()
        turns_left = board.player_worker.turns_left

        chains = self._find_primed_chains(board, my_loc)

        # Best immediate carpet roll (length >= 2; length 1 is -1pt)
        best_roll_dir = None
        best_roll_len = 0
        for d, length in chains.items():
            if length >= 2 and length > best_roll_len:
                best_roll_dir = d
                best_roll_len = length

        # Best prime direction (maximize max reachable chain; tiebreak on longest existing chain)
        best_prime_dir = None
        best_max_chain = 0
        best_additional = 0
        best_existing = 0
        for d in ALL_DIRS:
            if not board.is_valid_move(Move.prime(d)):
                continue
            opp = OPPOSITE_DIR[d]
            existing = chains[opp]
            additional = self._max_chain_from(board, my_loc, d, enemy_loc)
            if additional < 1:
                continue
            max_chain = min(existing + additional, BOARD_SIZE - 1)
            if (max_chain > best_max_chain
                    or (max_chain == best_max_chain and existing > best_existing)):
                best_prime_dir = d
                best_max_chain = max_chain
                best_additional = additional
                best_existing = existing

        # Search consideration
        search_loc = None
        search_prob = 0.0
        if self.tracker is not None:
            search_loc, search_prob = self.tracker.best_guess()

        # === Decision tree ===

        # 1) Full chain (7) -> always roll.
        if best_roll_len >= 7:
            return Move.carpet(best_roll_dir, best_roll_len)

        # 2) Last turn: cash in.
        if turns_left <= 1:
            if best_roll_len >= 2:
                return Move.carpet(best_roll_dir, best_roll_len)
            if best_prime_dir is not None:
                return Move.prime(best_prime_dir)  # at least +1
            if search_loc is not None and search_prob >= 0.35:
                return Move.search(search_loc)
            return self._fallback_move(board, my_loc, enemy_loc, search_loc, search_prob)

        # 3) Extend the chain when possible (marginal analysis says always worth it).
        if best_prime_dir is not None:
            turns_to_finish = best_additional + (1 if best_max_chain >= 2 else 0)
            if turns_to_finish <= turns_left or best_max_chain >= 5:
                return Move.prime(best_prime_dir)
            # Not enough turns to fully finish; roll existing if worthwhile.
            if best_roll_len >= 2:
                return Move.carpet(best_roll_dir, best_roll_len)
            return Move.prime(best_prime_dir)

        # 4) Can't prime; roll whatever we have.
        if best_roll_len >= 2:
            return Move.carpet(best_roll_dir, best_roll_len)

        # 5) High-confidence search.
        if search_loc is not None and search_prob >= 0.5:
            return Move.search(search_loc)

        # 6) Plain move to reposition for priming.
        plain = self._best_plain_move(board, my_loc, enemy_loc)
        if plain is not None:
            return plain

        return self._fallback_move(board, my_loc, enemy_loc, search_loc, search_prob)

    def _best_plain_move(self, board: Board, my_loc, enemy_loc):
        """Pick a plain move that lands on a cell with the most prime-chain potential."""
        best_dir = None
        best_score = -1
        for d in ALL_DIRS:
            if not board.is_valid_move(Move.plain(d)):
                continue
            target = loc_after_direction(my_loc, d)
            # Only SPACE cells allow future primes.
            if board.get_cell(target) != Cell.SPACE:
                score = 0
            else:
                score = 0
                for dd in ALL_DIRS:
                    score += self._max_chain_from(board, target, dd, enemy_loc)
            if score > best_score:
                best_score = score
                best_dir = d
        if best_dir is not None:
            return Move.plain(best_dir)
        return None

    def _fallback_move(self, board, my_loc, enemy_loc, search_loc, search_prob):
        if search_loc is not None and search_prob >= 0.25:
            return Move.search(search_loc)
        moves = board.get_valid_moves(exclude_search=False)
        if moves:
            return moves[0]
        return Move.search((0, 0))

    # ---------------------- Minimax (negamax) search ----------------------

    def _minimax_search(self, board: Board, time_left: Callable):
        """Iterative-deepening negamax. Returns (best_move, best_value) or (None, None)."""
        remaining = time_left()
        turns_remaining = max(1, board.player_worker.turns_left)
        # Use 60% of average remaining time per turn, clamped [0.3s, 2s].
        budget = min(2.0, max(0.3, remaining / turns_remaining * 0.6))
        deadline = remaining - budget

        my_loc = board.player_worker.get_location()
        enemy_loc = board.opponent_worker.get_location()
        chains = self._find_primed_chains(board, my_loc)

        raw_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not raw_moves:
            return None, None

        scored = [(self._move_score(board, m, chains, my_loc, enemy_loc), m) for m in raw_moves]
        scored.sort(key=lambda x: -x[0])
        moves = [m for _, m in scored][:8]

        best_move = moves[0]
        best_val = float('-inf')

        for depth in range(1, 6):
            if time_left() <= deadline:
                break
            alpha = float('-inf')
            beta = float('inf')
            current_best = None
            current_val = float('-inf')
            timed_out = False
            for move in moves:
                new_board = board.forecast_move(move, check_ok=False)
                if new_board is None:
                    continue
                new_board.reverse_perspective()
                try:
                    val = -self._negamax(new_board, depth - 1, -beta, -alpha, time_left, deadline)
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
                if best_move in moves:
                    moves.remove(best_move)
                    moves.insert(0, best_move)
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

        scored = [(self._move_score(board, m, chains, my_loc, enemy_loc), m) for m in raw_moves]
        scored.sort(key=lambda x: -x[0])
        moves = [m for _, m in scored][:6]

        best = float('-inf')
        for move in moves:
            new_board = board.forecast_move(move, check_ok=False)
            if new_board is None:
                continue
            new_board.reverse_perspective()
            val = -self._negamax(new_board, depth - 1, -beta, -alpha, time_left, deadline)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best

    def _move_score(self, board, move, chains, my_loc, enemy_loc):
        """Heuristic score for move ordering (higher = tried first)."""
        if move.move_type == MoveType.CARPET:
            return 100 + CARPET_POINTS_TABLE.get(move.roll_length, 0)
        if move.move_type == MoveType.PRIME:
            opp = OPPOSITE_DIR[move.direction]
            existing = chains.get(opp, 0)
            additional = self._max_chain_from(board, my_loc, move.direction, enemy_loc)
            max_chain = min(existing + additional, BOARD_SIZE - 1)
            return 50 + max_chain * 5 + additional
        if move.move_type == MoveType.PLAIN:
            target = loc_after_direction(my_loc, move.direction)
            if not board.is_valid_cell(target) or board.get_cell(target) != Cell.SPACE:
                return 5
            pot = 0
            for d in ALL_DIRS:
                pot = max(pot, self._max_chain_from(board, target, d, enemy_loc))
            return 20 + pot * 3
        return 0

    def _evaluate_for_current(self, board):
        """Score differential from player_worker's perspective (current turn)."""
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        my_loc = board.player_worker.get_location()
        opp_loc = board.opponent_worker.get_location()
        my_tl = board.player_worker.turns_left
        opp_tl = board.opponent_worker.turns_left

        my_pot = self._future_potential_points(board, my_loc, opp_loc, my_tl)
        opp_pot = self._future_potential_points(board, opp_loc, my_loc, opp_tl)

        return (my_pts + my_pot) - (opp_pts + opp_pot)

    def _future_potential_points(self, board, loc, enemy_loc, turns):
        """Best single-chain payoff achievable from this position within turns."""
        if turns <= 0:
            return 0
        best = 0
        for prime_dir in ALL_DIRS:
            opp_d = OPPOSITE_DIR[prime_dir]
            # Existing primed chain in direction opp_d
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
            max_possible = existing + additional
            if max_possible >= 2:
                best_L = min(max_possible, existing + turns - 1, BOARD_SIZE - 1)
                if best_L >= 2:
                    new_primes = max(0, best_L - existing)
                    pts = new_primes + CARPET_POINTS_TABLE[best_L]
                else:
                    pts = min(turns, additional)
            else:
                pts = min(turns, additional)
            if pts > best:
                best = pts
        return best
