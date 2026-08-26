# nba-allstar

Predicting NBA All-Star selection from a season's on-court statistics, with
logistic regression **implemented from scratch** (numpy only — no sklearn).

University assessment (UTS, Advanced Data Analytics Algorithms / ML, A2).
The design rationale lives in the docs:

- `PROJECT_SPEC.md` — task definition, input/output spec
- `docs/DATA_DECISIONS.md` — every design decision, its reason, and the
  Criterion C error investigation
- `docs/ARCHITECTURE.md` — repository layout and module contracts
- `docs/REQUIREMENTS.md` — hard constraints (hand-written algorithm, permitted
  libraries, leakage rules)

## Run it

**Notebook** (self-contained — clones this repo and downloads the data release
if needed): [`notebooks/allstar.ipynb`](notebooks/allstar.ipynb)

**Locally:**

```bash
pip install -r requirements.txt        # numpy, pandas, matplotlib — nothing else
# put the four CSVs in data/ (or let the notebook download them), then e.g.
python -m src.data                     # build + verify the dataset
```

## Data

Kaggle "NBA Stats (1947–present)" — dataset version: **56+ (TODO: confirm the
exact version number from the Kaggle download page)**.

Four of the 22 CSVs are used: `Advanced.csv` (features),
`All-Star Selections.csv` (labels), `Team Summaries.csv` (team win rate),
`Per 100 Poss.csv` (loaded, unused to date). A snapshot of these four is
attached to the `data-v1` GitHub release so the notebook is reproducible at a
pinned data version; the CSVs themselves are git-ignored.
