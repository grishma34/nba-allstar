# REQUIREMENTS.md

Hard constraints. Each traces back to a line in the assignment specification or to a
named rubric band. Nothing here is stylistic preference.

---

## 1. Implementation constraints

### 1.1 The learning algorithm is hand-written

**Banned:**
- `sklearn.linear_model.LogisticRegression` or any other sklearn estimator
- `statsmodels`, `scipy.optimize` for fitting
- Any library that computes the loss, the gradient, or the parameter update

**Required, written by hand:**
- `sigmoid`
- forward pass / `predict_proba`
- `log_loss`
- analytic gradients with respect to `w` and `b`
- the gradient descent update loop

**Why.** The Option 1 half of the hybrid requires algorithm implementation, not
library use. The spec is blunt about the alternative:

> "this criterion is considered 'Failed' if a model is merely used rather than
> studied and built, e.g. via importing a function from the scikit-learn library"

### 1.2 Permitted libraries

| Library | Use |
|---|---|
| `numpy` | array maths, linear algebra |
| `pandas` | CSV loading, joins, filtering |
| `matplotlib` | figures |

`sklearn.metrics` for confusion matrix / ROC is a grey area. **Prefer hand-written
metrics** — they are ten lines each and remove the question entirely. If any sklearn
utility is used, it must be for evaluation or splitting only, never for fitting, and
it must be declared in the journal.

### 1.3 Self-contained notebook

Spec requirement, quoted:

> "Be self-contained, including environment setup, data downloading, and preprocessing."

Consequences:
- No reading from local paths (`/mnt/c/Users/...`) in the submitted notebook
- Data fetched at runtime from a stable public URL
- Any `pip install` happens inside the notebook
- Must run top to bottom from a fresh kernel with no manual steps

### 1.4 Public accessibility

> "Be publicly accessible (not private or restricted)."

Verify by opening the link in a private browsing window while signed out of Google.

---

## 2. Data constraints

### 2.1 Dataset

Kaggle: "NBA Stats (1947-present)", version 56 or later. Record the version used.

Four files only:
- `Advanced.csv` — features
- `All-Star Selections.csv` — labels
- `Per 100 Poss.csv` — optional additional features (not used to date)
- `Team Summaries.csv` — team win rate, added by the Decision 5 amendment
  after a controlled experiment (see DATA_DECISIONS.md); also used for the
  Criterion C error analysis

### 2.2 Not a toy dataset

Spec: *"Option 2 must use real, non-trivial datasets (no toy datasets like Iris or
Titanic)."* Web-scraped Basketball-Reference data across 26 seasons and ~12,000
player-seasons satisfies this comfortably.

### 2.3 Leakage — hard exclusions

Never use as features:
- All-NBA team selections (`End of Season Teams.csv`)
- MVP / DPOY / any award voting (`Player Award Shares.csv`)

Both are produced by the same voter pool in the same season and encode the label
rather than predict it.

### 2.4 Temporal leakage — must be discussed

All-Star selection happens **mid-season**. Full-season statistics therefore contain
information from after the selection was made.

This is not necessarily fatal — but it **must be named and reasoned about** in the
journal. Options:
- Accept it, and state the consequence: the model is not usable for live prediction
  during a season, only for retrospective analysis.
- Restrict to a pre-break subset if the data allows it.

Naming this puts the project ahead of most submissions. Not naming it and being asked
about it in Q&A is the bad outcome.

---

## 3. Decisions that must be made and documented

Record each in `docs/DATA_DECISIONS.md` **as you make it**, with the reasoning. These
are exactly what Criterion B's "why it is done this way" band is testing.

| # | Decision | Options | Status |
|---|---|---|---|
| 1 | Replacement All-Stars | include as positive / exclude | ☐ |
| 2 | Multi-team seasons | keep 2TM combined row / keep per-team rows | ☐ |
| 3 | Eligibility threshold | min games — 20? 40? min minutes? | ☐ |
| 4 | Season range | 2000–2025 confirmed? | ☐ |
| 5 | Feature set | which advanced metrics, and why each | ☐ |
| 6 | Split boundaries | train/val/test season cutoffs | ☐ |
| 7 | Learning rate | value, and how chosen | ☐ |
| 8 | Iterations / stopping | fixed count or early stopping on val loss | ☐ |
| 9 | L2 regularisation | used? λ value? justification? | ☐ |
| 10 | Decision threshold | value of τ, and the precision/recall trade-off | ☐ |

A decision with no recorded reason is a Q&A question you cannot answer.

---

## 4. Code style — comments

Every non-trivial block gets a comment explaining **why**, not what.

Bad:
```python
# multiply X by w
z = X @ w + b
```

Good:
```python
# Linear combination of features. This is the model's raw score (the "logit").
# It can be any real number; the sigmoid below squashes it into (0, 1) so it
# can be read as a probability.
z = X @ w + b
```

For any line implementing a formula, name the formula:

```python
# Gradient of binary cross-entropy w.r.t. the weights.
# dL/dw = (1/n) * Xᵀ(ŷ - y)
# The (ŷ - y) term is the prediction error per sample; multiplying by Xᵀ
# distributes that error back across the features that produced it.
dw = (X.T @ (y_pred - y_true)) / n
```

**Why this matters beyond readability.** During Q&A you may need to point at a line
and explain it. Comments written for your future self are the notes you present from.

---

## 5. Testing

Minimum set in `tests/test_model.py`:

1. `sigmoid` boundary behaviour
2. `log_loss` = ~0 for perfect predictions
3. `log_loss` large for confidently wrong predictions
4. **Gradient check** — analytic gradients vs numerical finite differences
5. Loss decreases on a small separable synthetic dataset

Item 4 is the important one. It is direct evidence that the hand-written calculus is
correct, and it is a strong thing to be able to show if challenged on the
implementation.

---

## 6. Knowledge Gaps — required journal section

Spec requirement, quoted in full:

> "If there are components you used but do not fully understand, explicitly state
> this and explain: Why the component is necessary. How you verified its correctness
> (e.g., testing, cross-referencing documentation). What attempts you made to
> understand the underlying mechanism.
>
> **Note: Documenting your thought process and verification attempts will protect you
> from penalties if questioned during the presentation.**"

Maintain `docs/KNOWLEDGE_GAPS.md` throughout. Add an entry the moment something is
unclear, not retrospectively.

Format:

```
## [Component]
**What it does:** ...
**Why it's necessary:** ...
**What I don't fully understand:** ...
**How I verified it works:** ...
**What I did to understand it:** ...
```

This section is an asset, not an admission. The A3 criteria list *"I don't know.
ChatGPT states so. I can ask for references to check the correctness"* as a **good
response** worth credit, and a confident vague answer as worth zero.

---

## 7. AI tool use

Permitted and encouraged. Must be documented and critically reviewed.

Record in the implementation log:
- which tool, for what
- what was accepted, what was rejected and why
- where generated code was modified, and the reason

The rubric does not penalise AI use. It penalises not understanding the result:

> "an implementation is generated by AI tools and the student cannot demonstrate
> BASIC understanding of the model structure, algorithm, and key components"

The countermeasure is §1.1 (hand-written algorithm), §4 (comments), §5 (tests) and
§6 (knowledge gaps).

---

## 8. Scope discipline — what NOT to build

| Excluded | Reason |
|---|---|
| AWS / cloud deployment | Deliverable is a notebook. Adds failure modes, earns nothing. |
| Web UI | Spec warns undefended cosmetic material may be assessed as AI misuse. |
| Extensive EDA | Spec: *"does NOT add value to either A2 or A3... can distract from the main focus."* |
| Neural networks, ensembles, boosting | A model you cannot fully explain scores below a simple one you can. |
| More than three CSVs | Every added file is another thing to justify in Q&A. |

Expected workload per the spec: **20–40 hours** for a mid-level result. Scope to that.
