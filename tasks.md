# tasks.md

Ordered build steps. Tick as completed. Do not jump ahead — later steps assume
decisions made in earlier ones.

**Deadline:** Sunday 28 September, 23:59
**Presentation:** A3 peer viva — date from your Canvas group page

---

## Week 1 — data and model

### Setup
- [x] Create repo *(local only — push to GitHub deferred to week 2)*
- [x] `requirements.txt` — numpy, pandas, matplotlib only
- [x] Download the three CSVs to `data/` (git-ignored)
- [ ] Upload the three CSVs somewhere with a stable public URL, for the notebook
      *(deferred to week 2, alongside the Colab port)*
- [ ] Record the Kaggle dataset version number in the README

### Confirm the data — before writing any code
- [x] Open `All-Star Selections.csv`. Confirm all six column names, especially the
      boolean one (`replaced`?) — confirmed: `player, player_id, team, season, lg,
      replaced`. `replaced` is True for 81 of 677 filtered selections (12%)
- [x] Open `Advanced.csv`. List all 30 columns — confirmed, includes all candidate
      features from PROJECT_SPEC §3.3
- [x] Check whether `All-Star Selections.csv` uses the same `player_id` slug format
      as `Advanced.csv` — confirmed
- [x] Count rows per season after filtering to NBA and 2000–2025 — 15,893 rows total;
      1,561 are 2TM/3TM combined rows (9.8%); max g is 85 overall, 82 excluding TM
      rows. Missing values: ts_percent 92; per/usg_percent/ws_48/bpm 5 each
- [x] Count All-Star selections per season — 677 total (4.3% positive), roughly
      24–28 per season

### `src/data.py`
- [x] `load_raw()`
- [x] Filter `lg == "NBA"`
- [x] Filter seasons 2000–2025
- [x] **DECISION 2:** resolve multi-team rows — keep combined `^\d+TM$` row,
      drop stints. Recorded in DATA_DECISIONS.md
- [x] **DECISION 3:** eligibility threshold — mp ≥ 500; drops 4 named All-Stars,
      reported at run time. Recorded
- [x] **DECISION 1:** replacement All-Stars — all 677 labelled y=1; `replaced`
      marks the player who WAS replaced, not the appointee. Recorded
- [x] Join labels on `player_id` + `season` → `y` (all 677 match; hard error
      if any selection has no stats row)
- [x] Assertion: one row per (player_id, season); no *single-team* row with
      `g > 82` — combined TM rows can legitimately reach 85 (see Decision 2)
- [x] Print class balance — 8,869 rows, 673 positive (7.6% after eligibility;
      5.3% before)

### `src/features.py`
- [x] **DECISION 5:** feature set — mp, per, usg_percent, ws_48, vorp, age
      (one per concept; all pairs |r| ≤ 0.81, verified). Recorded
- [x] `chronological_split()` — **DECISION 6:** 2000–17 / 2018–21 / 2022–25
      (spec suggestion; keeps val at 107 positives). Recorded
- [x] `standardise()` — fit on train only. Verify val/test are not used in
      fitting *(structural: only X_train enters the mean/std computation;
      smoke test confirms train cols are exactly 0/1 and val cols are not)*

### `src/model.py` — the core, hand-written
- [x] `sigmoid()` — with z clipped to ±500 to avoid float64 overflow warnings
- [x] `predict_proba()`
- [x] `log_loss()` with eps clipping — verified finite on exact 0/1 inputs
- [x] `gradients()` — with the derivation in the docstring; spot-checked
      against finite differences (agreement ~1e-8, formal test to follow)

### `tests/test_model.py`
- [ ] sigmoid boundaries
- [ ] log loss ≈ 0 on perfect predictions
- [ ] log loss large on confidently wrong predictions
- [ ] **gradient check vs numerical finite differences** ← the important one
- [ ] loss decreases on a small separable synthetic dataset

### `src/train.py`
- [x] `fit()` — gradient descent loop (batch GD, w=0 start, optional L2)
- [x] Record train and validation loss per iteration
- [ ] **DECISION 7:** learning rate — swept 0.01–10.0; 0.1–3.0 all converge
      to the same optimum (convex loss), 10.0 oscillates. lr=1.0 used
      provisionally — *record pending user sign-off*
- [ ] **DECISION 8:** iterations — val loss bottoms at ~iter 457 (0.09628)
      then drifts up to 0.09715 (mild overfit); n_iters=5000 used
      provisionally — early stopping worth discussing. *Pending sign-off*

**End of week 1: a model that trains and a loss curve that goes down.**

---

## Week 2 — evaluation, and the Criterion C material

### `src/evaluate.py`
- [x] Log loss on train / val / test — `loss_table()`
- [x] Confusion matrix — hand-written, verified on a hand-computed example
- [x] Precision, recall, F1 — zero-division returns 0 (all-negative baseline
      scores 0/0/0 instead of crashing)
- [ ] **DECISION 10:** decision threshold τ, with the trade-off documented
      *(τ is a required argument everywhere — no default; decision pending
      until there is a trained model)*
- [x] PR curve and ROC curve — plus ROC-AUC (trapezoid) and PR-AUC (average
      precision); verified: perfect ranking → 1.0, random → 0.5 / base rate
- [x] Calibration table — predicted probability vs observed rate, bucketed;
      verified on the constant-rate baseline
- [x] Baselines: constant training-rate probability (log-loss-optimal
      constant; majority class is its thresholded twin). Single-feature
      baseline = train.fit() on a one-column X — needs `src/train.py`
- [x] `named_errors()` — top false positives and false negatives with player
      names and seasons; row-alignment assert against meta

### The Criterion C investigation — this is where the marks are
- [x] Pull the 10 highest-confidence false positives. What do they have in common?
      — established stars, mean age 30.1, 7/10 played ≤ 64 games
- [x] Pull the 10 highest-confidence false negatives. Same question.
      — ascending young stars (mean age 24.3) + defensive picks + one
      fan-vote starter
- [x] Test the hypothesis: is team success the missing variable? — **NO.**
      FP mean win rate .552 vs FN .556, U = 58/100. Hypothesis refuted;
      the separator is career trajectory (age), not team success
- [x] Write it up in `docs/DATA_DECISIONS.md` — the pattern, and why a linear model
      cannot capture it
- [x] **Optional, if time:** add an interaction feature and measure whether it helps.
      Either result is a finding. — DONE: win_rate main effect helps (log loss
      −9%, recall +0.10); per×(wr−.5) interaction adds nothing. Hard errors
      (trajectory/defence/popularity) unchanged. See DATA_DECISIONS.md

### Colab port — do this now, not in week 4
- [x] Build the thin notebook — 17 cells (9 code, 8 markdown); all code cells
      verified executing top-to-bottom locally; Colab setup lines present but
      commented until the repo is pushed and data hosted
- [ ] Verify it runs from a fresh kernel, top to bottom
- [ ] Verify the link opens while signed out of Google
- [ ] Time how long a cold run takes

**End of week 2: full evaluation, and a named, evidenced failure mode.**

---

## Week 3 — the journal

Structure follows the spec's required components.

- [ ] **Problem Definition** — from PROJECT_SPEC §3 and §4. Input/output at the
      "Clear" level of Table 1
- [ ] **Machine Learning Approach**
  - [ ] Model choice and justification
  - [ ] Hypothesis space — what the model can and cannot represent
  - [ ] Loss function, with the formula and what it penalises
  - [ ] Optimiser, hyperparameters, and how they were chosen
  - [ ] Evaluation metrics and why accuracy is not among the primary ones
- [ ] **Results** — tables, loss curves, confusion matrix, calibration plot,
      baseline comparison
- [ ] **Discussion**
  - [ ] Loss function vs task objective (PROJECT_SPEC §7)
  - [ ] Named examples of the failure mode
  - [ ] Limitations — temporal leakage, linear hypothesis space, label ambiguity
  - [ ] Future work grounded in the observed failures, not generic
- [ ] **Implementation Log**
  - [ ] Challenges and solutions
  - [ ] AI tool use — what was accepted, what was rejected, what was modified
  - [ ] **Knowledge Gaps** — from `docs/KNOWLEDGE_GAPS.md`
- [ ] Plain-text Colab URL in the PDF
- [ ] Filename: `GRISHMAKATTELA_24956106_2026_UTS_ML_Journal.pdf`
      *(confirm the exact convention against the spec)*

- [ ] **Submit by Sunday 28 September**

---

## Week 4 — presentation

5 minutes plus 10 minutes Q&A. Overtime is penalised: −1 per minute past 5:00,
terminated at 7:00 with a −2 penalty.

- [ ] Slides — few, plain. Everything on them is yours to defend
- [ ] Open with input/output, not with background. The spec recommends this
      explicitly and gives a counter-example of the wrong approach
- [ ] Two minutes on the part you genuinely invested in — the Criterion C analysis
- [ ] Practise switching between slides and code. Not fullscreen
- [ ] Pre-run the Colab notebook shortly before the session
- [ ] Rehearse to time. Five minutes is short
- [ ] Prepare for the obvious questions:
  - [ ] Why logistic regression and not something more powerful?
  - [ ] Why log loss? What does it penalise?
  - [ ] Why not accuracy?
  - [ ] Why a chronological split rather than random?
  - [ ] Why were those 19 files excluded?
  - [ ] How do you know your gradients are correct?
  - [ ] What is the model's biggest failure mode, and why?
- [ ] Leave deliberate hooks for questions — the spec calls these
      "baits-for-questions"

### After the presentation
- [ ] Post peer feedback on the Canvas group page **within 24 hours** — comments
      only, no scores. This contributes to the A3 grade
- [ ] Submit final peer evaluations as a PDF **after** 24 hours
- [ ] A3 due 31 October

---

## Running notes

Keep these current as you go, not retrospectively:

- `docs/DATA_DECISIONS.md` — every choice and its reason
- `docs/KNOWLEDGE_GAPS.md` — anything not fully understood, plus how it was verified
