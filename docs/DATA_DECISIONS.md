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
**Chosen:**
**Reason for each column:**
**Excluded and why:**

## Decision 6: Split boundaries
**Chosen:**
**Reason:**

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

### High-confidence false positives
(top 10, with names and seasons)

### High-confidence false negatives
(top 10, with names and seasons)

### Pattern observed

### Why the model cannot capture it
