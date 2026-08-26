# ARCHITECTURE.md

## Design principle

Real code lives in importable modules. The notebook is thin — it fetches data,
imports modules, runs them, displays results.

Three reasons this matters here:

1. **Porting to Colab is a copy, not a rewrite.** A monolithic notebook has to be
   rebuilt; modules just get cloned.
2. **The presentation needs both.** The spec recommends against fullscreen so the
   audience can see slides and code together. Clean, navigable files beat scrolling
   a 400-cell notebook.
3. **Criterion B rewards knowing where things are.** The rubric explicitly penalises
   a student who "needs to read through the code/comments on-site during the
   presentation to find how many channels are used in a layer / the loss function
   definition." One file per concept means you can answer instantly.

---

## Repository layout

```
nba-allstar/
├── README.md                   # what this is, how to run it
├── PROJECT_SPEC.md             # task definition, written to Criterion A
├── claude.md                   # instructions for Claude Code
├── tasks.md                    # ordered build steps, tick off as done
├── requirements.txt            # numpy, pandas, matplotlib — nothing else
│
├── docs/
│   ├── ARCHITECTURE.md         # this file
│   ├── REQUIREMENTS.md         # constraints, bans, decisions to make
│   ├── DATA_DECISIONS.md       # every preprocessing choice + justification
│   └── KNOWLEDGE_GAPS.md       # running log — feeds the journal
│
├── src/
│   ├── __init__.py
│   ├── data.py                 # load, filter, join, clean
│   ├── features.py             # select, engineer, standardise
│   ├── model.py                # sigmoid, forward pass, loss, gradients
│   ├── train.py                # gradient descent loop
│   ├── evaluate.py             # metrics, confusion matrix, calibration
│   └── plots.py                # figures for the report
│
├── notebooks/
│   └── allstar.ipynb           # thin — the submitted artefact
│
├── data/
│   └── .gitkeep                # raw CSVs live here, git-ignored
│
└── tests/
    └── test_model.py           # sanity checks (see below)
```

---

## Data flow

```
Kaggle CSVs
   Advanced.csv ─────────┐
   All-Star Selections ──┤
   Per 100 Poss.csv ─────┘
        │
        ▼
   data.py
     · filter lg == "NBA"
     · filter season 2000–2025
     · resolve multi-team rows (keep 2TM/3TM combined, drop per-team)
     · apply eligibility threshold (min games / minutes)
     · left-join labels on (player_id, season) → y ∈ {0,1}
     · assert: no player-season with g > 82
        │
        ▼
   features.py
     · select feature columns
     · optional engineered features (e.g. production × team success)
     · chronological split: train / val / test
     · standardise — fit on TRAIN ONLY, apply to val and test
        │
        ▼
   model.py
     · sigmoid(z)
     · predict_proba(X, w, b)      →  ŷ
     · log_loss(y, ŷ)              →  scalar
     · gradients(X, y, ŷ)          →  ∂L/∂w, ∂L/∂b
        │
        ▼
   train.py
     · initialise w, b
     · loop: forward → loss → gradients → update
     · record loss history for train and val
     · optional early stopping on validation loss
        │
        ▼
   evaluate.py
     · log loss on all three splits
     · precision / recall / F1 at chosen threshold
     · confusion matrix, with names attached to errors
     · PR curve, ROC curve
     · calibration: predicted probability vs observed rate, bucketed
     · baselines: majority class, single-feature
        │
        ▼
   plots.py → figures → report + slides
```

---

## Module contracts

Keep these signatures stable — the notebook and the tests depend on them.

### `data.py`

```python
load_raw(data_dir: str) -> tuple[DataFrame, DataFrame, DataFrame]
    """Read the three CSVs. No transformation."""

build_dataset(advanced, allstar, per100, *, season_min, season_max,
              min_games, include_replacements) -> DataFrame
    """Filter, resolve multi-team rows, join labels.
    Returns one row per eligible player-season with a `y` column."""
```

### `features.py`

```python
select_features(df, feature_cols) -> tuple[ndarray, ndarray, DataFrame]
    """Returns X, y, and a metadata frame carrying player/season/team
    so errors can be named later."""

chronological_split(df, train_end, val_end) -> tuple[DataFrame, ...]

standardise(X_train, X_val, X_test) -> tuple[ndarray, ndarray, ndarray, dict]
    """Fit mean/std on train only. Returns the fitted params too."""
```

### `model.py`

Every function here must be hand-written. No sklearn.

```python
sigmoid(z: ndarray) -> ndarray
predict_proba(X, w, b) -> ndarray
log_loss(y_true, y_pred, eps=1e-15) -> float
gradients(X, y_true, y_pred) -> tuple[ndarray, float]
```

### `train.py`

```python
fit(X_train, y_train, X_val, y_val, *, lr, n_iters,
    l2=0.0, verbose=True) -> tuple[ndarray, float, dict]
    """Returns w, b, and a history dict of train/val loss per iteration."""
```

### `evaluate.py`

```python
confusion(y_true, y_pred_binary) -> dict
precision_recall_f1(y_true, y_pred_binary) -> tuple[float, float, float]
calibration_table(y_true, y_prob, n_bins=10) -> DataFrame
named_errors(meta, y_true, y_prob, threshold, top_n=10) -> DataFrame
    """Highest-confidence false positives and false negatives, WITH
    player names and seasons. This feeds the Criterion C discussion."""
```

---

## Sanity checks (`tests/test_model.py`)

Cheap to write, and they demonstrate verification — which the spec's Knowledge Gaps
clause specifically asks about ("how you verified its correctness").

- `sigmoid(0) == 0.5`; large positive → ~1; large negative → ~0
- `log_loss` on perfect predictions ≈ 0
- `log_loss` on confidently wrong predictions is large
- Gradient check: compare analytic gradients against a numerical finite-difference
  approximation on a small random matrix. **This is the single most valuable test** —
  it proves the calculus is right, and it is an excellent thing to be able to
  point at during Q&A.
- Loss decreases monotonically on a tiny separable synthetic dataset

---

## Notebook structure (`notebooks/allstar.ipynb`)

Keep it under ~20 cells.

1. Markdown: task, input, output — lifted from PROJECT_SPEC §3 and §4
2. `!git clone` the repo (or pip install requirements)
3. Download the three CSVs from a stable public URL
4. `from src import data, features, model, train, evaluate, plots`
5. Build dataset — print shape, class balance
6. Split and standardise — print split sizes and date ranges
7. Train — show loss curves
8. Evaluate — metrics table, confusion matrix
9. Named errors — the table that drives the Criterion C discussion
10. Calibration plot
11. Markdown: findings and limitations

---

## Porting to Colab

Do this in **week two**, not the final week.

- Confirm the notebook runs top to bottom from a fresh kernel
- Confirm the Colab link opens **while signed out** — the spec requires public access
- Pre-run shortly before the presentation; the spec warns that kernel startup and
  data download eat into the 15-minute slot
