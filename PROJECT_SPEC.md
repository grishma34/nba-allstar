# PROJECT_SPEC.md

**Subject:** Advanced Data Analytics Algorithms / Machine Learning, UTS
**Assessment:** A2 — Study, Implement and Present a Machine Learning Model
**Due:** Sunday 28 September, 23:59 (3-day extension available via Canvas, no more)
**Grade depends on A3:** the journal alone earns nothing. A2 is graded *through* the
peer-viva presentation. Non-attendance at A3 = zero for both A2 and A3.

---

## 1. Project category

**Hybrid — Option 2 framing, Option 1 implementation.**

The specification permits this explicitly:

> "While you may define your effort as a mixed type, you will not be penalised for
> crossing boundaries between categories."

- **Option 2 side:** a real, non-trivial, publicly available dataset; a practical
  prediction task; a loss function that is both computationally effective and
  practically meaningful.
- **Option 1 side:** the learning algorithm is implemented from scratch. No
  `sklearn.linear_model.LogisticRegression`. Sigmoid, log loss, gradient and
  parameter update are all hand-written.

**Why the hybrid rather than pure Option 2.** Criterion B's failure conditions are
entirely about being unable to explain your own implementation under questioning.
Writing the algorithm by hand makes that failure mode structurally impossible — you
cannot write a gradient update without understanding it.

---

## 2. The task

Predict whether an NBA player was selected as an All-Star in a given season, using
only that season's on-court statistics.

Binary classification. One row = one player-season.

---

## 3. Input specification

Target level: **"Clear"** per Table 1 of the assignment spec. This is what separates
Fair (65%) from Pass (50%) on Criterion A.

### 3.1 What the model receives

A vector `x ∈ ℝ^d` of standardised numeric features describing one player's
performance across one regular season.

### 3.2 Feature source

`Advanced.csv` from the Kaggle "NBA Stats (1947-present)" dataset, joined optionally
to `Per 100 Poss.csv`. Both keyed on `player_id` (Basketball-Reference player slug)
and `season`.

### 3.3 Candidate features

Confirm exact column names against the downloaded file before committing.

| Group | Columns | Rationale |
|---|---|---|
| Volume / availability | `g`, `gs`, `mp` | Selection requires visibility; a player who missed most of the season is rarely chosen. |
| Efficiency | `per`, `ts_percent` | Per-minute production quality. |
| Usage | `usg_percent` | Proxy for offensive role and prominence. |
| Value | `ws`, `ws_48`, `bpm`, `vorp` | Composite value estimates. |
| Context | `age` | Reputation and career-arc effects. |

### 3.4 Preprocessing (all must be documented)

1. **League filter.** NBA only. `lg == "NBA"`. The ABA ran a separate All-Star game
   under a different selection process; mixing them conflates two labels.
2. **Season filter.** 2000–2025 inclusive.
   - 1999 had no All-Star Game (lockout) — every label would legitimately be 0.
   - 2026 is the current season and incomplete.
   - Selection format has changed materially over 75 years; a narrower window keeps
     the label meaning roughly constant.
3. **Multi-team seasons.** A team value matching `^\d+TM$` (2TM through 5TM all
   occur) marks a combined row for a traded player, alongside per-team rows.
   **Keep the combined row, drop the per-team rows.** Rationale: All-Star selection
   considers the whole season, not a partial stint. Sanity checks after this step:
   exactly one row per (player_id, season), and no *single-team* row with `g > 82`.
   Combined rows can legitimately reach g = 85, because the two teams' schedules
   are offset at trade time.
4. **Eligibility filter.** Minimum games and/or minutes threshold. State the value
   chosen and why. This removes players with negligible playing time whose advanced
   statistics are unstable on tiny samples.
5. **Standardisation.** Zero mean, unit variance, **fit on the training split only**
   and applied to validation and test. Fitting on the full dataset leaks test
   information into training.

### 3.5 Explicitly excluded — leakage

These columns exist in the dataset and must **not** be used:

| Source | Why excluded |
|---|---|
| `End of Season Teams.csv` (All-NBA) | Selected by the same voter pool in the same season. Encodes the label, not a predictor. |
| `Player Award Shares.csv` (MVP, DPOY etc.) | Same reasoning — award voting reflects the same reputation signal that drives All-Star selection. |

This is the same failure mode as a post-outcome variable in a churn dataset: near-
perfect association with the target, zero operational value, because the information
is not available at the moment a prediction would need to be made.

### 3.6 The 18 unused files

Of 22 CSVs in the dataset, four are used: three feed player rows and labels, and
`Team Summaries.csv` feeds one team-context feature (win rate — added by the
Decision 5 amendment after a controlled experiment) plus the error analysis. The
exclusions fall into three distinct categories and this should be stated in the
report:

- **Not relevant** — team totals, opponent statistics, draft history.
- **Leakage** — award and end-of-season-team files (§3.5).
- **Redundant** — `Player Per Game.csv`, `Per 36 Minutes.csv` and `Per 100 Poss.csv`
  are three encodings of the same underlying production. Using several introduces
  multicollinearity without adding information.

---

## 4. Output specification

### 4.1 What the model produces

A single real number `ŷ ∈ (0, 1)`.

Formally: `ŷ = σ(wᵀx + b)` where `σ(z) = 1 / (1 + e^(−z))`.

**Interpretation.** `ŷ` is the model's estimated probability that this player-season
resulted in All-Star selection. `ŷ = 0.8` means the model assigns 80% probability.

### 4.2 Converting probability to a decision

A threshold `τ` maps probability to a binary prediction: selected if `ŷ ≥ τ`.

`τ = 0.5` is the default but is **not** appropriate here. Roughly 24–30 players are
selected from 450–550 eligible each season — a positive rate near 5%. A model
predicting "no" for everyone achieves ~95% accuracy while being useless. Threshold
choice must be justified against precision/recall, not assumed.

### 4.3 Label definition

From `All-Star Selections.csv`. Columns confirmed against the downloaded file:
`player`, `player_id`, `team`, `season`, `lg`, `replaced`.

**`replaced` marks the originally selected player who was replaced** (usually
through injury) — *not* the replacement. Verified against a known case: in 2020,
Devin Booker was appointed as the injury replacement for Damian Lillard; the data
shows Lillard with `replaced = True` and Booker with `replaced = False`.
Commissioner-appointed replacements are therefore indistinguishable from ordinary
selections in this file, and the originally proposed option of excluding
appointees is not implementable with this data.

**Decision 1 (see `docs/DATA_DECISIONS.md`):** every player-season present in this
file is labelled `y = 1` — the label means "was named an All-Star that season".
Every other eligible player-season in the filtered set is `y = 0`. The positive
class consequently mixes voted-in players, injured selectees and appointed
replacements; this label ambiguity is acknowledged in the journal's Limitations
section.

---

## 5. Model

Logistic regression, implemented from scratch.

| Component | Definition |
|---|---|
| Hypothesis space | All functions `f(x) = σ(wᵀx + b)` over `w ∈ ℝ^d`, `b ∈ ℝ`. One hypothesis per parameter setting. |
| Loss | Binary cross-entropy (log loss): `L = −(1/n) Σ [ yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ) ]` |
| Optimiser | Batch gradient descent. `w ← w − η ∇_w L`, `b ← b − η ∇_b L` |
| Regularisation | L2 optional. If used, state the penalty and justify λ. |

**Known limitation to state up front:** the hypothesis space contains only linear
decision boundaries in feature space. It cannot represent interactions unless they
are constructed explicitly as features. This matters — see §7.

---

## 6. Evaluation

### 6.1 Split

Chronological, not random. Train on earlier seasons, validate and test on later ones.

**Why chronological.** A random split places players from the same season on both
sides of the split. Selection is a *comparative* process — only ~24 slots exist per
season — so a random split leaks information about how competitive a given season was.
Chronological splitting also mirrors the deployment scenario: predicting a future
season from past ones.

Suggested: train 2000–2017, validation 2018–2021, test 2022–2025. Adjust and justify.

### 6.2 Metrics

Accuracy is **not** a primary metric here — see §4.2. Report:

- **Log loss** — the training objective; report on all three splits.
- **Precision, recall, F1** on the positive class.
- **Confusion matrix** — with named examples of false positives and false negatives.
- **ROC-AUC and/or PR-AUC.** PR-AUC is more informative under this level of imbalance.
- **Calibration.** Of players given `ŷ ≈ 0.7`, roughly 70% should have been selected.

### 6.3 Baseline

Report at least one. Without a baseline, a metric has no meaning.

- Majority class (always predict "not selected").
- Single-feature model, e.g. VORP alone.

---

## 7. Criterion C — loss function vs task objective

**This is where the marks are.** The 80–100% band asks for "specific edge cases where
the loss minimises but the task objective fails, and specific monitoring or mitigation
strategies."

### 7.1 The mismatch

Log loss averages over every player-season equally. Roughly 95% of those are obvious
non-selections — bench players nobody would consider. The optimiser can drive average
loss down substantially by being confidently correct on cases nobody cares about.

The cases that carry the meaning of the task are the ~5% near the boundary: borderline
selections, snubs, reputation picks. Log loss barely registers them.

### 7.2 Concrete failure mode to demonstrate

A player with elite per-100 and advanced metrics on a losing team, predicted highly
but not selected.

Log loss punishes that confident error heavily, so the optimiser reduces the weight on
production features to avoid it — which degrades predictions for every other player.
The model cannot learn *"production matters, conditional on team success"* because
that is an interaction, and it lies outside the hypothesis space of a linear model
(§5).

That single example connects hypothesis space, loss function and optimiser. It is the
core of both the report and the presentation.

### 7.3 Mitigations to discuss (implement at least one if time permits)

- Class weighting or resampling to raise the loss contribution of positives.
- An explicit interaction feature (production × team win rate) to extend the
  hypothesis space.
- A per-season ranking evaluation — since selection is comparative, evaluating
  "did the top 24 predicted players match the actual 24" is closer to the real
  objective than per-row probability.

### 7.4 Framing discipline

Name specific players when discussing errors — the rubric wants specific edge cases.
Describe **what the model output and what the data shows**. Do not make claims about
merit or who deserved selection. "The model assigned 0.87 and the player was not
selected; players in this group share a profile of high per-100 production on
sub-.500 teams" is analysis. "This player was robbed" is not.

---

## 8. Deliverables

1. **PDF journal**, named `GRISHMAKATTELA_24956106_2026_UTS_ML_Journal.pdf`
   *(confirm the exact naming convention against the spec before submitting)*
2. **Public Colab notebook URL** as plain text in the PDF. Must be self-contained:
   environment setup, data download, preprocessing, training, evaluation.
3. **Implementation log** — challenges and solutions, AI tool use, and an explicit
   **Knowledge Gaps** section (see `REQUIREMENTS.md` §6).
4. **A3 presentation** — 5 minutes plus 10 minutes Q&A.

---

## 9. Out of scope

Deliberately excluded, and each exclusion is defensible:

- **AWS / cloud deployment** — deliverable is a notebook; adds risk, earns nothing.
- **Front end / UI** — the spec warns that undefended cosmetic material can be
  assessed as misuse of AI. Build only if everything else is finished.
- **Extensive EDA** — the spec explicitly warns this "does NOT add value" and
  "can distract from the main focus."
- **Neural networks / ensembles** — a model you cannot fully explain scores worse
  than a simple one you can.
