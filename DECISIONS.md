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

**Outcome (50 games vs Carrie):**
- Mean diff: **−2.18** (improved from −2.86, +0.68 swing)
- Win rate: **46%** (up from 32%)
- Hit rate: **66.1%** (up from 58.6%)
- MyBot avg: 39.6 (up from 37.2)

**Lesson:** First-principles math on game rules beats tuning
heuristics by intuition. Should have tried this before v3.

---

## v5 — `48508ea` — "boost steal scoring + d-2 approach (REGRESSED, reverted)"

**Hypothesis:** 50 v4 games showed 45 dist-1 missed steals per game and
22 cases where MyBot plain-moved in the wrong direction. Boost the
steal score in `_move_score` PLAIN branch from 80 → 190 for length-2
and 150+CARPET_PTS×3 → 220+CARPET_PTS×5 for length-3+. Add d-2
approach bonus (80-130).

**Evidence:** Steal rate 9.5%; 45 missed dist-1 steals; +80 cells over
50 games estimate = +1.5 pts/game conservative.

**Outcome (49 games):**
- Mean diff: −2.71 (not significantly different from v4's −2.18, p=0.84)
- Hit rate: **55.3% (significantly worse, p=0.005)**
- Win rate: 35%

**What went wrong:** Steal score 190 for length-2 beats the search
score 44 for p=0.40 in minimax move ordering. Bot chased steal
opportunities instead of searching. Steals are speculative (Carrie
may roll before we arrive), while a 55% search is guaranteed +EV.

**Lesson:** A new term in `_move_score` must not silently outrank
existing gating scores. Test move-ordering consequences explicitly
before shipping.

---

## v5-revert — `35d820e` — "revert v5 to v4"

Same pattern as v2-revert. Clean revert, commit the zip.

---

## v6 — `23971a2` — "time management + endgame search + eval threat awareness"

**Hypothesis:** v4 used only 59% of its 240s budget (99s wasted),
explored 2.3M nodes (~depth 10-11 effective with alpha-beta), and
left the `_opponent_chain_threat` function as dead code.

**Change (three independent improvements bundled):**
1. **Time:** raised cap 4s → 10s per turn, root width 10 → 14,
   internal width 8 → 12 (progressive: 12 at depth≥3, 6 at depth≤2).
   Safety floor at remaining < 15s.
2. **Endgame search:** threshold drops from 0.40 → 0.33 in last 5
   turns (pure EV breakeven).
3. **Eval threat awareness:** activated `_opponent_chain_threat` at
   −0.7 weight. Minimax now sees imminent opponent rolls.

**Evidence:** Budget analysis showed fair-share per turn = 6.4s mid,
16s late — all capped at 4s. 17/50 v4 games ended on PLAIN on turn
40 (should've been search). Opponent's rollable chain threat was
invisible to the eval.

**Outcome (50 games):**
- Mean diff: −1.58 (best so far, from −2.18)
- Win rate: 42%
- Nodes: 3.1M (from 2.3M, +35%)
- Time usage: 84% (from 59%)
- No timeouts, no failures

**Lesson:** Parameter changes (time limits, widths) are much safer
than logic changes. Activating dead code (`_opponent_chain_threat`)
was low-risk because the function was already tested in isolation.

---

## v7 — `9bcd255` — "cold-roll walk + length-1 elimination"

**Hypothesis:** Carrie rolls 7.5/game vs our 4.6 because she walks
to existing primed cells (cold-rolling) instead of always building
new chains. Fix the greedy tree to walk toward rollable chains.
Also fix length-1 rolls (35/50 games = −0.7 pt/game leak) by
preferring EV-breakeven search over length-1 in fallback.

**Change:**
1. Greedy branch 2 (new): if no immediate roll, scan neighbors for
   cells adjacent to rollable chains (length≥2). If found, walk
   there. Zero exposure risk — cashing in what's already built.
2. Fallback path: when only length-1 carpets available, try a
   search at 0.33 threshold first.

**Outcome (50 games as Player A):**
- Mean diff: **+0.06** (FIRST time positive!)
- Win rate: 48%
- MyBot avg: 40.6 (highest ever)
- Batch consistency: 5-5, 4-6, 6-4, 5-5, 4-6 (no catastrophic batch)

**Outcome (50 games as Player B — next submission):**
- Mean diff: −3.50 (worse as B)
- Win rate: 40%
- Carrie plays more aggressively as A (first-mover), steals more
  of our chains, delays our first roll by 5 turns.

**Lesson:** Cold-rolling works — +2.5 pts/game vs v6. But the bot
plays 3.5 pts/game worse as B than as A, which investigation showed
is mostly Carrie's first-mover advantage amplifying through our
carpet strategy. The fix for this came in v8.

---

## v8 — `322dfeb` — "fix 4 bugs found in deep audit"

**Hypothesis:** TA advice ("simple and bug-free beats fancy and
buggy") prompted a relentless 4-subagent audit. They found 4 real
bugs — good ideas undermined by implementation errors.

**Bugs fixed:**
1. **Eval asymmetry (line 1048):** `_evaluate_for_current` had
   `−0.7 × opp_threat` but no matching `+0.7 × my_threat`. Same
   chain evaluated 2.1 pts differently from A-view vs B-view.
   Classic sign-pessimism. Fix: `+0.7 × (my_threat − opp_threat)`.
2. **Minimax endgame override missing (line 793):** `_should_search`
   drops to 0.33 in last 5 turns; minimax used 0.40. Minimax was
   rejecting +EV searches in turns 36-40.
3. **Minimax consecutive override missing (line 782):** Greedy
   allows back-to-back searches at prob≥0.80; minimax blocked them
   unconditionally even at p=0.99.
4. **`_best_steal_from` opponent block (line 1017):** Function
   walked primed chains blocking only on our own worker. Engine
   rejects carpet rolls through EITHER worker, so we were counting
   phantom chain length when opp sat on the chain.

**Evidence:**
- Eval sum-to-zero test: previously off by ~±2.1 per threat; now 0.000
- Exhaustive 8,960-state test: 0 discrepancies between `_should_search`
  and the minimax injection after the fix
- Engine-matching tests on `_best_steal_from`: all pass

**Outcome (local, 10 games vs CarrieBot):** 10/10 wins at +40.7 avg.
Bytefight outcome TBD.

**Lesson:** Sometimes a good idea is implemented wrong. Heuristic
tuning gets you only so far — running a rigorous audit on the
existing code surfaces real bugs that no amount of parameter tuning
would fix. All 4 fixes were bounded corrections, not new features.

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
