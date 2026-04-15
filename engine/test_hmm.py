"""Test harness for HMMRatTracker.

Simulates 80 turns of the engine's per-turn rat schedule and compares the
tracker's belief against ground truth. Reports:
 - top-1 accuracy
 - rank of true cell in belief
 - mean belief mass on true cell
 - catch rate of a greedy search policy at various P-thresholds
 - divergence points (turns where rank > 5)
"""
import os
import sys
import pickle
import random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "3600-agents", "MyBot"))

from game.board import Board
from game.rat import Rat, HEADSTART_MOVES
from game.enums import BOARD_SIZE, Cell, Noise
import agent as mybot  # noqa: E402


def load_T(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # pkl files contain either a single matrix or a list/dict of them
    if isinstance(data, dict):
        T = next(iter(data.values()))
    elif isinstance(data, list):
        T = data[0]
    else:
        T = data
    T = np.asarray(T, dtype=np.float64)
    # row-normalize just in case
    T = T / T.sum(axis=1, keepdims=True)
    return T


def run_trial(T, n_turns=80, seed=0, verbose=False):
    random.seed(seed)
    np.random.seed(seed)

    # Ground-truth rat
    rat = Rat(T.tolist())
    rat.spawn()  # 1000 headstart moves

    # Minimal board-like object for tracker init
    board = Board()  # default ctor
    tracker = mybot.HMMRatTracker(T, board)

    # Compare stationary
    naive = np.zeros(64)
    naive[0] = 1.0
    for _ in range(HEADSTART_MOVES):
        naive = naive @ T
    tvd = 0.5 * np.abs(naive - tracker.stationary).sum()

    results = {
        "stationary_tvd_vs_1000_iters": float(tvd),
        "true_rank_per_turn": [],
        "true_prob_per_turn": [],
        "max_belief_per_turn": [],
        "top1_correct": 0,
        "turns": n_turns,
    }

    # Simulate turns. The engine does: rat.move(), then sample.
    # Our tracker's first-turn branch (is_first_turn=True, not Player B) does:
    # predict() once, then noise/distance update. That matches.
    worker_loc = (0, 0)  # fixed worker for test
    is_first = True

    for turn in range(n_turns):
        # Engine: rat moves, then samples
        rat.move()
        noise = rat.make_noise(board)
        dist = rat.estimate_distance(worker_loc)

        # Tracker: predict then update
        if is_first:
            tracker.predict()
            is_first = False
        else:
            # No own/opp searches in this minimal harness → just one predict
            # (we're skipping the double-predict-for-opponent sequence to
            # isolate the core HMM math). This models a symmetric solo game.
            tracker.predict()
        tracker.update_noise(int(noise))
        tracker.update_distance(worker_loc, dist)

        # Score
        true_idx = rat.position[1] * BOARD_SIZE + rat.position[0]
        order = np.argsort(-tracker.belief)
        rank = int(np.where(order == true_idx)[0][0])
        results["true_rank_per_turn"].append(rank)
        results["true_prob_per_turn"].append(float(tracker.belief[true_idx]))
        results["max_belief_per_turn"].append(float(tracker.belief.max()))
        if rank == 0:
            results["top1_correct"] += 1

        if verbose and rank > 5:
            print(f"  turn {turn}: true={rat.position} rank={rank} "
                  f"p_true={tracker.belief[true_idx]:.3f} "
                  f"max={tracker.belief.max():.3f}")

    return results


def run_catch_sim(T, n_trials=100, n_turns=80, threshold=0.35):
    """Simulate catch rate of a greedy searcher."""
    hits = 0
    searches = 0
    for s in range(n_trials):
        random.seed(s * 17 + 1)
        np.random.seed(s * 17 + 1)
        rat = Rat(T.tolist())
        rat.spawn()
        board = Board()
        tracker = mybot.HMMRatTracker(T, board)
        worker_loc = (3, 3)  # center worker
        is_first = True
        for turn in range(n_turns):
            rat.move()
            noise = rat.make_noise(board)
            dist = rat.estimate_distance(worker_loc)
            if is_first:
                tracker.predict()
                is_first = False
            else:
                tracker.predict()
            tracker.update_noise(int(noise))
            tracker.update_distance(worker_loc, dist)
            guess_pos, p = tracker.best_guess()
            if p >= threshold:
                searches += 1
                if guess_pos == rat.position:
                    hits += 1
                # reset belief as if search ended (miss zeros, hit respawns)
                if guess_pos == rat.position:
                    rat.spawn()
                    tracker.apply_search_hit()
                else:
                    tracker.apply_search_miss(guess_pos)
    return hits, searches, (hits / searches if searches else 0.0)


def main():
    pkl = os.path.join(HERE, "transition_matrices", "bigloop.pkl")
    T = load_T(pkl)
    print(f"Loaded T from {os.path.basename(pkl)}: shape={T.shape}, "
          f"row_sum_range=[{T.sum(1).min():.4f}, {T.sum(1).max():.4f}]")

    # 1) Stationary comparison
    r = run_trial(T, n_turns=80, seed=42, verbose=True)
    print(f"\n[1] Stationary TVD (T^1024 vs 1000 iters from (0,0)): "
          f"{r['stationary_tvd_vs_1000_iters']:.6e}")

    # 2) Tracking accuracy
    ranks = np.array(r["true_rank_per_turn"])
    probs = np.array(r["true_prob_per_turn"])
    print(f"\n[2] Tracking accuracy over 80 turns (seed 42):")
    print(f"    top-1 correct:    {r['top1_correct']}/80 = {r['top1_correct']/80:.1%}")
    print(f"    mean rank of true cell: {ranks.mean():.2f}")
    print(f"    median rank: {int(np.median(ranks))}")
    print(f"    p90 rank: {int(np.percentile(ranks, 90))}")
    print(f"    mean p(true): {probs.mean():.3f}")
    print(f"    p(true) first 10 turns: "
          f"{[f'{p:.2f}' for p in probs[:10]]}")
    print(f"    p(true) last 10 turns:  "
          f"{[f'{p:.2f}' for p in probs[-10:]]}")

    # 3) Multi-seed summary
    print(f"\n[3] Multi-seed summary (20 seeds, 80 turns each):")
    all_ranks = []
    all_probs = []
    all_top1 = []
    for s in range(20):
        r = run_trial(T, n_turns=80, seed=s)
        all_ranks.extend(r["true_rank_per_turn"])
        all_probs.extend(r["true_prob_per_turn"])
        all_top1.append(r["top1_correct"] / 80)
    all_ranks = np.array(all_ranks)
    all_probs = np.array(all_probs)
    print(f"    mean top-1 per seed: {np.mean(all_top1):.1%}  "
          f"(min {min(all_top1):.1%}, max {max(all_top1):.1%})")
    print(f"    mean rank of true: {all_ranks.mean():.2f}")
    print(f"    mean p(true):      {all_probs.mean():.3f}")
    print(f"    fraction turns with rank=0:   {(all_ranks == 0).mean():.1%}")
    print(f"    fraction turns with rank<=2:  {(all_ranks <= 2).mean():.1%}")
    print(f"    fraction turns with rank>10:  {(all_ranks > 10).mean():.1%}")

    # 4) Catch rate at various thresholds
    print(f"\n[4] Greedy searcher hit rate (100 trials, 80 turns, worker=(3,3)):")
    for th in [0.25, 0.35, 0.45, 0.55, 0.65]:
        hits, searches, rate = run_catch_sim(T, n_trials=100, n_turns=80, threshold=th)
        print(f"    P>={th:.2f}: {hits}/{searches} hits "
              f"({rate:.1%}), {searches/100:.1f} searches/game")

    # 5) Belief entropy over time — drift toward uniform?
    print(f"\n[5] Belief entropy over time (seed 42, should stay below log2(64)=6):")
    r2 = run_trial(T, n_turns=80, seed=42)
    rat = Rat(T.tolist()); rat.spawn()
    tracker = mybot.HMMRatTracker(T, Board())
    random.seed(42); np.random.seed(42)
    # re-sim to snapshot entropy
    ents = []
    is_first = True
    for t in range(80):
        rat.move()
        noise = rat.make_noise(Board())
        dist = rat.estimate_distance((0, 0))
        if is_first: tracker.predict(); is_first = False
        else: tracker.predict()
        tracker.update_noise(int(noise))
        tracker.update_distance((0, 0), dist)
        b = tracker.belief
        ent = -np.sum(b * np.log2(b + 1e-12))
        ents.append(ent)
    ents = np.array(ents)
    print(f"    entropy first 5: {[f'{e:.2f}' for e in ents[:5]]}")
    print(f"    entropy last 5:  {[f'{e:.2f}' for e in ents[-5:]]}")
    print(f"    mean entropy turn 0-20: {ents[:20].mean():.2f}")
    print(f"    mean entropy turn 60-80: {ents[60:80].mean():.2f}")


if __name__ == "__main__":
    main()
