# DATA_DECISIONS.md

Every design decision, with its reason. Fill in as you go, not retrospectively.

A decision with no recorded reason is a Q&A question you cannot answer.

---

## Template

```
## Decision N: [name]
**Date:**
**Options considered:**
**Chosen:**
**Reason:**
**Consequence / what this costs:**
```

---

## Decision 1: Replacement All-Stars
**Date:** 2026-08-26
**Options considered:** (a) label every row of `All-Star Selections.csv` as y=1;
(b) drop `replaced=True` rows from the dataset; (c) relabel `replaced=True` as y=0
**Chosen:** (a) — all 677 filtered rows are y=1.
**Reason:** The file records who was *named* an All-Star, so that is what the label
means. Dropping the `replaced=True` rows (b) would delete injury/reputation
selections such as Grant Hill 2001, Yao Ming 2011 and Kobe Bryant 2014 — exactly
the near-boundary cases the Criterion C analysis needs. Relabelling as y=0 (c) is
factually wrong: Damian Lillard 2020 was voted in; calling that season "not
selected" creates poisoned negatives.
**Consequence / what this costs:** The positive class mixes three selection
mechanisms — voted-in players, injured selectees (`replaced=True`, 81 of 677, 12%)
and commissioner-appointed replacements — under one label. This label ambiguity is
acknowledged in the journal's Limitations section. `replaced=True` players average
lower value and minutes (VORP 3.15 vs 4.33, MP 2104 vs 2575), consistent with
injury-shortened seasons.

### Note: the original assumption about `replaced` was backwards
PROJECT_SPEC §4.3 originally assumed the boolean flagged the commissioner-appointed
*replacement*, and proposed excluding those as "not voted in". Verified against a
known case before deciding: in 2020, Devin Booker was appointed as the injury
replacement for Damian Lillard. The data shows **Lillard `replaced=True`, Booker
`replaced=False`** — the flag marks the originally selected player who *was
replaced*. Appointed replacements are indistinguishable from ordinary selections in
this file, so the originally proposed "exclude appointees" option is not
implementable at all. §4.3 corrected on 2026-08-26.

## Decision 2: Multi-team seasons (2TM / 3TM / 4TM / 5TM)
**Date:** 2026-08-26
**Options considered:** (a) keep the season-combined TM row, drop per-team stint
rows; (b) keep per-team stint rows, drop the combined row; (c) keep only the
stint with the most minutes
**Chosen:** (a), matching the pattern `^\d+TM$`. The data contains 2TM (1,461),
3TM (97), 4TM (2) and 5TM (1) rows — the filter matches the digit rather than the
two literals the spec happened to name.
**Reason:** All-Star selection considers the whole season, so season-total
statistics are the right unit. (b) turns one player-season into 2–3 partial rows
and would double-count traded All-Stars after the label join. (c) silently
discards part of the season's production and availability.
**Consequence / what this costs:** Team identity is lost for 1,561 player-seasons
(9.8% of rows), including 17 All-Star seasons (Harden 2021 and 2022, Durant 2023,
Iverson 2007/2009/2010, ...). If the week-2 interaction feature
(production × team win rate) is added, traded players will need a stated
convention — e.g. a minutes-weighted average across stints. Also: combined rows
can legitimately show up to g=85, because the two teams' schedules are offset at
trade time — so the spec's "no g > 82" sanity check applies to single-team rows
only. The real double-counting guard is asserting one row per (player_id, season).

## Decision 3: Eligibility threshold
**Date:** 2026-08-26
**Options considered:** none / g ≥ 20 / g ≥ 40 / mp ≥ 500 / mp ≥ 1000, each
measured on the deduplicated 12,667-row dataset before choosing.
**Chosen:** mp ≥ 500 (total minutes across the season).
**Reason:** Minutes is a better sample-size proxy than games — a 60-game bench
player can have fewer minutes than a 30-game starter. Advanced metrics on tiny
minute totals are unstable. At mp ≥ 500 all 50 rows with a missing candidate
feature are removed, so no imputation is ever needed or defended.
**Consequence / what this costs:** 8,867 player-seasons remain with 673 of 677
positives (7.6% positive rate). Four All-Stars are excluded: Grant Hill 2001
(4 g, 133 mp), Alonzo Mourning 2001 (13 g, 306 mp), Yao Ming 2011 (5 g, 91 mp),
Kobe Bryant 2014 (6 g, 177 mp). All four are `replaced=True` injury selections
chosen on reputation — their on-court statistics that season cannot support the
selection, so a model given only current-season statistics has no basis for them.
`build_dataset()` reports these exclusions at run time (never silent), and the
Limitations section states that the model does not cover reputation picks with
under 500 minutes.

## Decision 4: Season range
**Chosen:** 2000-2025 (proposed)
**Reason:** 1999 lockout season had no All-Star Game. 2026 incomplete. Selection
format has changed materially since 1947; a narrower window keeps the label meaning
approximately constant.
**Consequence:** ~26 seasons, ~12,000 player-seasons. Findings do not generalise to
earlier eras.

## Decision 5: Feature set
**Date:** 2026-08-26
**Options considered:** (a) all 11 spec candidates; (b) one column per concept;
(c) either plus Per 100 Poss counting stats
**Chosen:** (b) — six features: `mp`, `per`, `usg_percent`, `ws_48`, `vorp`, `age`.
**Reason for each column:**
- `mp` — availability/visibility: selection requires being on the floor; subsumes
  `g` and `gs` (gs ~ mp r = 0.82)
- `per` — per-minute production quality (efficiency)
- `usg_percent` — offensive role and prominence
- `ws_48` — rate-based value estimate
- `vorp` — cumulative value over the season
- `age` — career arc / reputation proxy
**Excluded and why:** `ws`, `bpm`, `obpm`, `dbpm` are correlated composites of the
same production already carried by `ws_48` and `vorp` (measured: vorp ~ bpm
r = 0.92, vorp ~ ws r = 0.92, ws ~ ws_48 r = 0.83, bpm ~ per r = 0.86); including
them gives unstable, uninterpretable coefficients in a linear model. Component
percentages (`orb_percent`, `stl_percent`, `tov_percent`, `x3p_ar`, `f_tr`, etc.)
describe playing style rather than value. Per 100 Poss columns not joined: the
per-X files are redundant encodings of the same production (spec §3.6), and using
one would undercut the stated reason for excluding the others.
**Verification before committing:** correlation matrix of the six on the built
dataset — no pair above |r| = 0.85 (max 0.81: per ~ ws_48 and per ~ vorp).
**Consequence / what this costs:** any signal genuinely unique to the excluded
columns (e.g. defensive value in `dbpm`) is unavailable to the model; every fitted
weight is explainable in one sentence.

## Decision 6: Split boundaries
**Date:** 2026-08-26
**Options considered:** spec suggestion 2000–17 / 2018–21 / 2022–25
(5,977 / 1,415 / 1,477 rows; 460 / 107 / 106 positives) vs a later split
2000–19 / 2020–22 / 2023–25 (6,691 / 1,076 / 1,102 rows; 515 / 79 / 79 positives)
**Chosen:** the spec suggestion — train 2000–2017, validation 2018–2021,
test 2022–2025.
**Reason:** validation drives the decision threshold (Decision 10) and early
stopping (Decision 8), and ~107 positives is already thin; dropping to 79 makes
those estimates noisier. Extra training positives help a linear model less than
stable validation helps the evaluation.
**Consequence / what this costs:** 55 fewer training positives than the later
split; the test window ends at 2025, so conclusions are about the current
selection era.

## Decision 7: Learning rate
**Values tried:**
**Chosen:**
**What happened at each:**

## Decision 8: Iterations / stopping criterion
**Chosen:**
**Reason:**

## Decision 9: L2 regularisation
**Used:** yes / no
**Lambda:**
**Reason:**

## Decision 10: Decision threshold (tau)
**Chosen:**
**Precision at this threshold:**
**Recall at this threshold:**
**Reason for this trade-off:**

---

## Criterion C investigation

**Date:** 2026-08-26. Model: five features (mp, per, usg_percent, ws_48, age),
lr=1.0, 5000 iters, τ=0.40 chosen on validation. Test split 2022–2025.
Team win rates come from `Team Summaries.csv` — used for ERROR ANALYSIS ONLY,
never as a model feature (state this in the journal: three files feed the
model, a fourth is read to study its errors).

### High-confidence false positives
(top 10, descending ŷ; win rate is the player's team, minutes-weighted for 2TM)

| Player | Season | ŷ | Team win rate | Age | G |
|---|---|---|---|---|---|
| Jimmy Butler | 2023 | 0.98 | .537 | 33 | 64 |
| Anthony Davis | 2023 | 0.94 | .524 | 29 | 56 |
| Domantas Sabonis | 2024 | 0.80 | .561 | 27 | 82 |
| DeMar DeRozan | 2024 | 0.78 | .476 | 34 | 79 |
| Luka Dončić | 2025 | 0.73 | .550 (DAL+LAL) | 25 | 50 |
| Jimmy Butler | 2024 | 0.70 | .561 | 34 | 60 |
| Trae Young | 2023 | 0.70 | .500 | 24 | 73 |
| Kawhi Leonard | 2023 | 0.67 | .537 | 31 | 52 |
| James Harden | 2023 | 0.63 | .659 | 33 | 58 |
| Kyrie Irving | 2024 | 0.60 | .610 | 31 | 58 |

### High-confidence false negatives
(top 10, ascending ŷ)

| Player | Season | ŷ | Team win rate | Age | Note |
|---|---|---|---|---|---|
| Draymond Green | 2022 | 0.001 | .646 | 31 | replaced=True; dbpm 4.6 |
| Andrew Wiggins | 2022 | 0.02 | .646 | 26 | fan-vote STARTER |
| Scottie Barnes | 2024 | 0.04 | .305 | 22 | |
| Fred VanVleet | 2022 | 0.10 | .585 | 27 | |
| Jarrett Allen | 2022 | 0.12 | .537 | 23 | |
| Anthony Edwards | 2023 | 0.12 | .512 | 21 | |
| Julius Randle | 2024 | 0.13 | .610 | 29 | replaced=True |
| Jaren Jackson Jr. | 2023 | 0.14 | .622 | 23 | dbpm 2.0 |
| LaMelo Ball | 2022 | 0.14 | .524 | 20 | |
| Paolo Banchero | 2024 | 0.16 | .573 | 21 | |

### Pattern observed

**The team-success hypothesis (PROJECT_SPEC §7.2) is REFUTED on this data.**
Win rate does not separate the groups: false positives mean .552 (8/10 above
.500), false negatives mean .556 (9/10 above .500); hand-computed
Mann-Whitney U = 58/100 where 50 means no separation. Both groups sit between
the test-split means for non-All-Stars (.483) and All-Stars (.594). The
predicted failure mode — "elite production on losing teams, predicted highly,
not selected" — does not describe these errors: 8 of 10 false positives were
on playoff teams.

What actually separates the groups is **career trajectory**:

- False positives average **age 30.1** — established stars (Butler, DeRozan,
  Kawhi, Harden, Kyrie) putting up All-Star-calibre full-season numbers in
  seasons the voters passed them over. Secondary signal: games played —
  FP group 50–64 games for 7 of 10 vs a true-positive median of 69.
  Selection happens MID-season; a full-season stat line partly built after
  the break is invisible to February voters (the temporal-leakage issue of
  REQUIREMENTS §2.4 surfacing as a concrete error pattern).
- False negatives average **age 24.3** — ascending young stars (Edwards,
  Barnes, LaMelo, Banchero, ages 20–22) selected on trajectory and
  narrative with production the model reads as ordinary (group mean PER
  18.4 vs 24.3 for true positives; ws_48 0.12 vs 0.19), plus two
  defence-first selections (Draymond dbpm 4.6, JJJ 2.0) whose value sits
  partly in the defensive columns Decision 5 excluded, and one popularity
  pick (Wiggins, voted starter).

### Why the model cannot capture it

The model's fitted age weight is +0.59: globally, being older goes with
selection because veterans have the minutes and production. But at the
decision boundary the effect runs the OTHER way — young players are selected
ahead of their current-season numbers, old players are held to a higher bar.
A single linear term must be monotone: one sign of one weight for all
players. "Age helps through the accumulation of production, but hurts at the
margin relative to trajectory" is an interaction (age × production, or a
non-monotone transform of age), and interactions are outside the hypothesis
space of a linear model unless constructed explicitly as features
(PROJECT_SPEC §5). The defensive selections are a second, simpler gap: the
information (dbpm, dws) was deliberately excluded by Decision 5, so no
weight assignment could recover it — a measured cost of the
interpretability trade, not a failure of the optimiser.

### Follow-up candidates (optional, per tasks.md)
- An explicit age × production interaction feature, or age-squared, to test
  whether extending the hypothesis space closes the gap. Either result is a
  finding.
- Report errors per conference-season: selection competes within a
  conference, which per-row loss ignores.
