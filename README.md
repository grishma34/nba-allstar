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
# put the three CSVs in data/ (or let the notebook download them), then e.g.
python -m src.data                     # build + verify the dataset
python -m pytest tests/                # 30 unit tests (pure logic, no CSVs needed)
```

## Data

**Source and credit:** [NBA Stats (1947–present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)
by **Sumitro Datta**, published on Kaggle (data originally scraped from
[Basketball-Reference](https://www.basketball-reference.com/)). See the Kaggle
page for licence terms. Dataset version used: **56**.

Three of the 22 CSVs are used: `Advanced.csv` (features),
`All-Star Selections.csv` (labels), and `Team Summaries.csv` (team win rate).
The [`data-v1` release](https://github.com/grishma34/nba-allstar/releases/tag/data-v1)
snapshot also contains `Per 100 Poss.csv`, kept while the feature set was
still an open decision; the final set takes nothing from it and the code no
longer reads it. The notebook downloads the snapshot from the release so it
stays reproducible at a pinned data version; the CSVs themselves are
git-ignored.
