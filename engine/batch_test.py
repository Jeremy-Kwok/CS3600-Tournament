"""Run a batch of games to evaluate an agent vs a baseline."""
import os
import pathlib
import sys
import time
import random

from gameplay import play_game


def main():
    if len(sys.argv) < 4:
        print(f"Usage: python3 {sys.argv[0]} <player_a> <player_b> <n_games> [seed]")
        sys.exit(1)

    player_a = sys.argv[1]
    player_b = sys.argv[2]
    n_games = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    random.seed(seed)

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")

    wins_a = 0
    wins_b = 0
    ties = 0
    errors = 0
    score_diffs = []

    start_total = time.perf_counter()
    for game_idx in range(n_games):
        # alternate who goes first to remove first-mover bias
        if game_idx % 2 == 0:
            pa, pb = player_a, player_b
        else:
            pa, pb = player_b, player_a

        t0 = time.perf_counter()
        try:
            result = play_game(
                play_directory,
                play_directory,
                pa,
                pb,
                display_game=False,
                delay=0.0,
                clear_screen=False,
                record=False,
                limit_resources=False,
            )
        except Exception as e:
            print(f"Game {game_idx}: ERROR {e}")
            errors += 1
            continue

        final_board, _, _, _, msg_a, msg_b = result
        winner = final_board.get_winner()
        reason = final_board.get_win_reason()

        # Resolve A/B scores. At game end, reverse_perspective is NOT called.
        # After the final apply_move: is_player_a_turn has been flipped.
        # is_player_a_turn==True  -> B just moved, player_worker=B, opponent_worker=A
        # is_player_a_turn==False -> A just moved, player_worker=A, opponent_worker=B
        if final_board.is_player_a_turn:
            a_score = final_board.opponent_worker.get_points()
            b_score = final_board.player_worker.get_points()
        else:
            a_score = final_board.player_worker.get_points()
            b_score = final_board.opponent_worker.get_points()

        if pa == player_a:
            our_score, their_score = a_score, b_score
        else:
            our_score, their_score = b_score, a_score

        # Winner translation: ResultArbiter (PLAYER_A=0, PLAYER_B=1, TIE=2, ERROR=3)
        our_is_a = (pa == player_a)
        if winner == 0:  # PLAYER_A won
            if our_is_a:
                wins_a += 1
                outcome = "WIN"
            else:
                wins_b += 1
                outcome = "LOSS"
        elif winner == 1:  # PLAYER_B won
            if our_is_a:
                wins_b += 1
                outcome = "LOSS"
            else:
                wins_a += 1
                outcome = "WIN"
        elif winner == 2:
            ties += 1
            outcome = "TIE"
        else:
            errors += 1
            outcome = "ERROR"

        score_diffs.append(our_score - their_score)
        dt = time.perf_counter() - t0

        print(f"Game {game_idx+1:2d}: {outcome} | ours={our_score:3d} theirs={their_score:3d} diff={our_score-their_score:+4d} | {pa}(A)vs{pb}(B) | {reason.name if hasattr(reason,'name') else reason} | {dt:.1f}s")

    total_dt = time.perf_counter() - start_total
    played = n_games - errors
    print("\n" + "=" * 60)
    print(f"Results for {player_a} vs {player_b}:")
    print(f"  Wins: {wins_a}/{played} ({100*wins_a/max(1,played):.1f}%)")
    print(f"  Losses: {wins_b}/{played}")
    print(f"  Ties: {ties}/{played}")
    print(f"  Errors: {errors}")
    if score_diffs:
        avg_diff = sum(score_diffs) / len(score_diffs)
        print(f"  Avg score diff: {avg_diff:+.1f}")
    print(f"  Total time: {total_dt:.1f}s ({total_dt/max(1,played):.1f}s/game)")


if __name__ == "__main__":
    main()
