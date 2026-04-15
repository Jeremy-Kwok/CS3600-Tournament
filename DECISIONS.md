# MyBot Decision Log

A running log of every change shipped to bytefight, with the reasoning, the data that motivated it, and what happened. New entries go at the bottom.

Format for each entry:
- **Version + commit hash**
- **Hypothesis** — what I thought was wrong
- **Change** — what I actually did
- **Evidence** — the data I used to decide
- **Outcome** — bytefight scrim results
- **Lesson** — what I'd do differently

## Workflow rules I follow

- Don't ship a change without a specific data-backed hypothesis.
- Don't trust any result under 30 games — score diff stdev is ~10–18 points,
  so 10-game batches have ±4–6 pts of noise on the mean.
- Local testing against our CarrieBot clone is a very weak proxy for real
  Carrie on bytefight (we've seen 35-point swings between local and real).
  Treat local only as a catastrophe detector.
- Commit every shipped version to git so we can bisect. Include the
  `MyBot.zip` in the commit so anyone can see exactly what was uploaded.
- Test search EV math from first principles: each hit is +4, each miss is
  −2, so breakeven at P = 1/3. Every search threshold decision is anchored
  to this.

---

## v2 — `f9469bb` — "cap bypass fix + HMM ring-ambiguity gate"

**Hypothesis:** Two separate bugs were costing us whole games.
1. The search cap of 8 was not actually being enforced on bytefight:
   match 74 showed s=32, c=1, with MyBot standing still at (2,7) searching
   its own cell 32 times and donating 62 points to the opponent.
2. On games where the HMM belief locked onto the wrong cell of a
   Manhattan-distance ring from a stationary worker, hit rate collapsed
   from ~73% to ~33% and never recovered.

**Change:**
- `play()` now intercepts any returned SEARCH move when `search_count >= 8`
  or when the hit-rate shutoff triggers, and replaces it with the best
  non-search move (`_pick_nonsearch_move`). This is a belt-and-suspenders
  defense that catches any code path the cap missed.
- `HMMRatTracker.ring_peers()` counts cells on the same Manhattan-distance
  ring as the top-belief cell that hold ≥40% of top's probability. If
  `peers >= 2`, search is blocked — this is the disambiguation the HMM
  can't do from observations alone when the worker is stationary.
- Added a stationary-turn counter that requires very high confidence
  (≥0.75) to search once the worker has been still for 3+ turns.
- Fixed a subtle Player-B first-turn predict bug: when we're Player B
  and Player A didn't search on her first turn (the common case), the
  rat has still moved twice before our first sample, so we need two
  `predict()` calls regardless of A's action.

**Evidence:**
- 45 prior bytefight matches showed mean diff −14.9, with win rate
  correlating perfectly with hit rate (73% in wins, 33% in losses).
- Match 74 rollout showed MyBot searched on 32 of 40 turns from the same
  position. Cap was 8 — it failed.
- HMM subagent replay found that on sparse transition matrices
  (`quadloops.pkl`), a stationary worker has top-1 accuracy of 40% ±
  18.8% with a 0%-accuracy worst case. A moving worker had 65% ± 6.8%
  with no games below 54%.

**Outcome (50 games vs Carrie):**
- Mean diff: **−2.86** (improved from −14.9, +12.0 swing)
- Stdev: **10.0** (was 18.3, halved)
- Min diff: **−25** (was −78 — disasters eliminated)
- Max searches/game: **8** (cap held, no bypasses)
- Hit rate: **58.6%** overall (was 33% in losses)
- Win rate: 32% (was ~27%)

**Lesson:** The bimodal hit-rate failure was a real root cause, not a
symptom. Measuring per-game hit rate surfaced it; a global average
would have hidden the bimodality.

---

## v3 — `9b6c5ca` — "length-2 chain extension (REGRESSED, reverted)"

**Hypothesis:** After v2, the remaining gap vs Carrie is in carpet points
(12.2 vs 18.1 per game). Carrie has 2.2× more length-4+ rolls than us.
The fix: stop rolling length-2 chains immediately; extend them first.

**Change:**
- `_choose_move_greedy` line 621: instead of unconditionally rolling any
  length-2+ chain, check if priming in the opposite direction would
  extend the chain AND the opponent is ≥3 cells away AND turns_left ≥ 3.
  If all true, extend; otherwise roll.
- `_move_score`: gave PRIME moves that extend an existing chain a score
  of ~200 (comparable to rolling) so minimax would actually explore the
  extension branch.
- `_future_potential_points`: removed the `if new_primes == 0: pts = 0`
  line that was zeroing out potential for ready-to-roll chains.

**Evidence:**
- Independently verified: 77% of MyBot's length-2 rolls had a free
  neighbor for extension.
- 63% of those extendable rolls happened with ≥5 turns remaining (no
  time pressure excuse).
- Local sanity batch: 10/10 vs CarrieBot at +37.5 avg.

**Outcome (60 games vs Carrie):**
- Mean diff: **−5.77** (worse by 2.9 points vs v2)
- Cells stolen by Carrie: **289** (was 193, +50%)
- MyBot self-roll rate: **53.7%** (was 59.5%)
- Carrie length-3 rolls: **133** (was 94, +41%)
- Carrie avg score: **19.6** (was 18.1, +1.5)

**What went wrong:** The "opponent ≥3 cells away = safe to extend"
check was too permissive. Carrie moves 2–3 cells per our turn pair, so
by the time we've extended length-2 → length-3 and are ready to roll,
she's adjacent. Extensions exposed our primes. We spent 24% more turns
priming but 50% more of those primes got stolen mid-build.

**Lesson:** A "safe distance" has to account for how much the opponent
can CLOSE during our build time, not just the current distance. More
generally: extension strategies require modeling the opponent's
reachable set over N turns, not a simple distance gate. Heuristics
that look right on paper can backfire against a reactive opponent.

---

## v2-revert — `6b11729` — "revert v3, back to f9469bb baseline"

**Hypothesis:** v3 is measurably worse on every dimension (stolen count,
self-roll rate, Carrie's carpet production, mean diff). Reverting is the
correct move even if I don't yet know the replacement.

**Change:** `git checkout f9469bb -- 3600-agents/MyBot/agent.py`, rebuild
the zip.

**Evidence:**
- 60 games of v3 data with n=60, SE≈1.4. Regression ~2σ.
- All secondary metrics moved in the same direction (worse), not just
  the noisy final score.

**Outcome:** Restored the −2.86 baseline. The bytefight submission was
replaced with this zip.

**Lesson:** Don't fight the data when you see a coherent regression
signal across multiple independent metrics. Revert fast; investigate
from a known-good baseline.

---

## v4 — `f70eea2` — "lower search threshold 0.45 → 0.40"

**Hypothesis:** Our HMM is already MORE accurate than Carrie's (56.7%
vs 52.4% hit rate from match records). We search less than we should
given the EV math — every belief > 33% is +EV per the rules.

**Change:** One line each in `agent.py`:
```
SEARCH_MINIMAX_FLOOR = 0.40     # was 0.45
SEARCH_ALWAYS_THRESHOLD = 0.40  # was 0.45
```
Kept: ring-ambiguity gate, stationary-turn penalty, hard cap of 8, and
the hit-rate shutoff. All disaster protections remain active.

**Evidence:**
- From 50 v2 matches (measured via ground-truth rat positions in the
  match JSONs): MyBot 56.7% hit rate at 5.1 searches/game; Carrie
  52.4% hit rate at 5.0 searches/game. We're accuracy-competitive but
  under-searching.
- EV math: 6P − 2 per search. At P=0.40, EV = +0.4/search. At P=0.45,
  EV = +0.7/search. Dropping the floor lets through searches that were
  already +EV but getting rejected.
- Subagent replay across 4 thresholds showed net expected gain of
  +0.7 to +1.5 pts/game from the drop to 0.40, with hit rate holding
  at ~58% (in replay).
- Local sanity batch: 10/10 vs CarrieBot at +38.9 avg (within noise
  of v2's +42.9 — no disasters).

**Why this change is safer than v3's:** search is a single-action
decision. There's no multi-turn liability like extending a chain.
Worst case per extra search is −2 pts. If the bot makes 2 extra bad
searches per game, the regression is −4 pts/game — smaller than v3's
was, and easier to detect in the first batch of scrims.

**Outcome:** TBD — pending 50-game scrim results.

**Lesson (if positive):** First-principles math on game rules beats
tuning heuristics by intuition. We should have tested this before v3.

**Lesson (if negative):** Even low-risk changes need to be tested on
bytefight, not locally. Plan the revert path up front.

---

## How I'm measuring success

The stated grading rubric is rank-based: above Carrie in final ELO = 90%,
above Albert = 80%, above George = 70%. ELO is reset before the final
tournament. To rank above Carrie I need roughly 50% win rate against her
in the final, which translates to a mean score diff of approximately 0.

Current position after v2/v4:
- ~−3 mean diff → implied ELO ~1790 vs Carrie's 1921 → ~130 ELO short
- Need to close ~130 ELO before the final tournament
- 1 ELO point ≈ ~0.02 pts/game mean diff improvement at high-end ranges
- Budget: ~+2–3 pts/game to reach parity

The search-threshold change (v4) is expected to get us ~1/3 of the way
there. Remaining ~2 pts/game will come from a second targeted fix
(candidates: improving steal rate, reducing orphaned primes, tightening
move ordering). Each will be shipped only after verifying against
bytefight data.
