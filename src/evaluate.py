"""
Evaluation: metrics, curves, calibration, baselines, and named errors.

Every metric here is hand-written (REQUIREMENTS.md §1.2 — using
sklearn.metrics would reopen the "merely used a library" question for ten
lines of arithmetic each).

Accuracy is deliberately absent. With ~7% positives, predicting "no" for
everyone scores ~93% accuracy while being useless; every metric in this
module is chosen to stay meaningful under that imbalance (PROJECT_SPEC
§4.2, §6.2).

The decision threshold τ is a parameter everywhere it appears — never a
default. Choosing it is Decision 10 and must be justified against the
precision/recall trade-off, not assumed to be 0.5.
"""

import numpy as np
import pandas as pd

from src.model import log_loss


def binarise(y_prob, threshold):
    """
    Convert probabilities to hard 0/1 predictions: predict 1 iff ŷ >= τ.

    Kept as a named function so the threshold application is visible at
    call sites instead of buried in comparisons.

    Args:
        y_prob: ndarray (n,) of predicted probabilities.
        threshold: τ, the decision threshold (Decision 10).

    Returns:
        ndarray (n,) of 0.0/1.0 hard predictions.
    """
    return (y_prob >= threshold).astype(np.float64)


def confusion(y_true, y_pred_binary):
    """
    The four cells of the confusion matrix.

    tp — All-Stars the model predicted as All-Stars
    fp — non-All-Stars the model predicted as All-Stars ("false alarms")
    fn — All-Stars the model missed ("misses")
    tn — non-All-Stars correctly left out

    Under this imbalance tn is huge and nearly meaningless; the story of
    the model lives in the other three cells.

    Args:
        y_true: ndarray (n,) of 0/1 labels.
        y_pred_binary: ndarray (n,) of hard 0/1 predictions (from binarise).

    Returns:
        dict with keys "tp", "fp", "fn", "tn" (ints).
    """
    y_true = np.asarray(y_true)
    y_pred_binary = np.asarray(y_pred_binary)
    tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(y_true, y_pred_binary):
    """
    Precision, recall and F1 on the positive (All-Star) class.

        precision = tp / (tp + fp)   "of those the model named, how many
                                      really were All-Stars?"
        recall    = tp / (tp + fn)   "of the real All-Stars, how many did
                                      the model find?"
        F1        = harmonic mean of the two — punishes lopsidedness, so a
                    model cannot score well by maxing one and ignoring the
                    other.

    Zero-division convention: if the model predicts nobody (tp + fp = 0),
    precision is defined as 0 here rather than NaN, and likewise recall if
    there are no true positives in the split. This makes the all-negative
    majority-class baseline score 0/0/0 instead of crashing — which is the
    honest description of a model that finds no one.

    Args:
        y_true: ndarray (n,) of 0/1 labels.
        y_pred_binary: ndarray (n,) of hard 0/1 predictions.

    Returns:
        (precision, recall, f1) — three floats in [0, 1].
    """
    cells = confusion(y_true, y_pred_binary)
    tp, fp, fn = cells["tp"], cells["fp"], cells["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def _rank_curve_counts(y_true, y_prob):
    """
    Shared machinery for the PR and ROC curves.

    Sort samples by predicted probability, most confident first, and count
    cumulative true/false positives as the threshold sweeps downward: the
    k-th entry describes the hard classifier "predict 1 for the top k".

    Assumes predicted probabilities are effectively tie-free, which holds
    for float64 sigmoid outputs of continuous features; ties would merge
    into intermediate points slightly out of order but leave endpoints and
    areas essentially unchanged.

    Returns:
        (tp_cum, fp_cum, n_pos, n_neg, thresholds) — thresholds are the
        sorted probabilities, so curve point k corresponds to τ just low
        enough to include the top k samples.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    order = np.argsort(-y_prob)          # descending confidence
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)         # positives captured so far
    fp_cum = np.cumsum(1.0 - y_sorted)   # negatives swept up so far
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y_sorted) - n_pos)
    return tp_cum, fp_cum, n_pos, n_neg, y_prob[order]


def pr_curve(y_true, y_prob):
    """
    Precision-recall curve: one (precision, recall) point per threshold.

    PR is the informative curve under heavy imbalance (PROJECT_SPEC §6.2):
    both axes are computed from the model's behaviour on the ~7% positive
    class alone, so the flood of easy true negatives cannot flatter it —
    unlike ROC, where the false positive rate divides by ~8,200 negatives.

    Args:
        y_true: ndarray (n,) of 0/1 labels.
        y_prob: ndarray (n,) of predicted probabilities.

    Returns:
        (precision, recall, thresholds) — ndarrays of equal length, ordered
        from the strictest threshold (low recall) to the loosest.
    """
    tp_cum, fp_cum, n_pos, _, thresholds = _rank_curve_counts(y_true, y_prob)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    return precision, recall, thresholds


def roc_curve(y_true, y_prob):
    """
    ROC curve: true positive rate vs false positive rate per threshold.

        TPR = tp / n_pos   (identical to recall)
        FPR = fp / n_neg   (share of non-All-Stars wrongly named)

    Reported alongside PR because it is the conventional curve, but see
    pr_curve for why it is the less informative of the two here.

    Args:
        y_true: ndarray (n,) of 0/1 labels.
        y_prob: ndarray (n,) of predicted probabilities.

    Returns:
        (fpr, tpr, thresholds) — ndarrays ordered from strict to loose.
    """
    tp_cum, fp_cum, n_pos, n_neg, thresholds = _rank_curve_counts(y_true, y_prob)
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg
    return fpr, tpr, thresholds


def roc_auc(y_true, y_prob):
    """
    Area under the ROC curve, by the trapezoidal rule with the (0, 0)
    origin prepended.

    Interpretation: the probability that a randomly chosen All-Star is
    ranked above a randomly chosen non-All-Star. 0.5 is coin-flipping,
    1.0 is perfect ranking.

    Returns:
        float in [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))


def pr_auc(y_true, y_prob):
    """
    Area under the PR curve, computed as average precision:

        AP = Σ (Rᵢ − Rᵢ₋₁) · Pᵢ

    Each true positive, as the threshold sweeps down, contributes the
    precision at the moment it was captured, weighted by the recall step
    it adds (1/n_pos). This step-sum is used instead of trapezoids because
    precision is not monotone in recall and linear interpolation between
    PR points is known to overstate the area.

    Baseline for comparison: a random-ranking model scores approximately
    the positive rate (~0.076 here), NOT 0.5.

    Returns:
        float in (0, 1].
    """
    precision, recall, _ = pr_curve(y_true, y_prob)
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


def loss_table(split_probs):
    """
    Log loss (the training objective) reported per split, side by side.

    The gap between train and validation/test loss is the overfitting
    signal; the gap between a split's loss and the constant-rate baseline
    (baseline_constant_rate) is how much the features actually bought.

    Args:
        split_probs: dict mapping split name -> (y_true, y_prob), e.g.
            {"train": (y_tr, p_tr), "val": (y_va, p_va), "test": ...}

    Returns:
        DataFrame with columns split, n, positives, log_loss.
    """
    rows = []
    for name, (y_true, y_prob) in split_probs.items():
        rows.append({
            "split": name,
            "n": len(y_true),
            "positives": int(np.sum(y_true)),
            "log_loss": log_loss(np.asarray(y_true, dtype=np.float64),
                                 np.asarray(y_prob, dtype=np.float64)),
        })
    return pd.DataFrame(rows)


def calibration_table(y_true, y_prob, n_bins=10):
    """
    Are the probabilities honest? Bucket predictions into equal-width bins
    and compare the mean predicted probability in each bin against the
    observed selection rate.

    A calibrated model has the two columns tracking each other: of the
    players given ŷ ≈ 0.7, roughly 70% should actually have been selected
    (PROJECT_SPEC §6.2). Systematic gaps mean the probabilities are
    over- or under-confident even if the ranking is good.

    Bins are equal-width in predicted probability. Under this imbalance
    the low bins hold thousands of rows and the high bins a handful —
    the count column is printed so sparse bins are read with due caution
    rather than hidden by resampling.

    Args:
        y_true: ndarray (n,) of 0/1 labels.
        y_prob: ndarray (n,) of predicted probabilities.
        n_bins: number of equal-width bins over [0, 1].

    Returns:
        DataFrame with one row per non-empty bin: bin range, count,
        mean predicted probability, observed positive rate.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Final bin closes at 1.0 inclusive so ŷ = 1.0 lands somewhere.
        if i < n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue  # an empty bin has no observed rate to report
        rows.append({
            "bin": f"[{lo:.1f}, {hi:.1f})" if i < n_bins - 1 else f"[{lo:.1f}, {hi:.1f}]",
            "count": count,
            "mean_predicted": float(y_prob[in_bin].mean()),
            "observed_rate": float(y_true[in_bin].mean()),
        })
    return pd.DataFrame(rows)


def baseline_constant_rate(y_train, n):
    """
    The no-information baseline: predict the same probability — the
    TRAINING split's positive rate — for every sample.

    Why this constant: among all constant predictions, the base rate
    minimises expected log loss, so this is the strongest model that uses
    no features at all. Any feature-based model must beat it to have
    learnt anything. Its hard-prediction twin is the majority class
    (thresholding this constant at any τ above the base rate predicts
    nobody), which scores precision = recall = F1 = 0.

    The rate comes from the training split only — using an evaluation
    split's own rate would leak label information into the baseline.

    The other required baseline — a single-feature model — is not built
    here: it is the same hand-written model trained on a one-column X via
    train.fit(), needing no extra code.

    Args:
        y_train: ndarray of training labels (rate is fitted on these only).
        n: length of the evaluation split to predict for.

    Returns:
        ndarray (n,) filled with the training positive rate.
    """
    rate = float(np.mean(y_train))
    return np.full(n, rate, dtype=np.float64)


def named_errors(meta, y_true, y_prob, threshold, top_n=10):
    """
    The model's most confident mistakes, WITH NAMES — the raw material of
    the Criterion C analysis (PROJECT_SPEC §7).

    False positives: non-All-Stars given the HIGHEST probabilities — the
    model was sure they'd be selected and they were not ("snubs", by the
    model's reckoning). False negatives: All-Stars given the LOWEST
    probabilities — selected despite statistics the model reads as
    ordinary (reputation picks, injury selectees).

    Aggregate metrics say how often the model is wrong; this table says
    WHO it is wrong about, which is where the pattern (and the hypothesis
    that team success is the missing variable) becomes visible.

    Args:
        meta: metadata frame from features.select_features(), row-aligned
              with y_true/y_prob (player, player_id, season, team,
              replaced, y).
        y_true: ndarray (n,) of 0/1 labels for the same rows.
        y_prob: ndarray (n,) of predicted probabilities.
        threshold: τ (Decision 10) — defines which side of the line each
                   error sits on.
        top_n: how many of each error type to return.

    Returns:
        DataFrame: top_n false positives (descending ŷ) then top_n false
        negatives (ascending ŷ), with an error_type column.
    """
    # meta carries its own y column; if it disagrees with y_true the rows
    # are misaligned and every name in the table would be wrong.
    assert np.array_equal(np.asarray(meta["y"], dtype=np.float64),
                          np.asarray(y_true, dtype=np.float64)), (
        "meta and y_true disagree — rows are misaligned")

    frame = meta.copy().reset_index(drop=True)
    frame["y_prob"] = np.asarray(y_prob, dtype=np.float64)

    is_fp = (frame["y"] == 0) & (frame["y_prob"] >= threshold)
    is_fn = (frame["y"] == 1) & (frame["y_prob"] < threshold)

    false_positives = (frame[is_fp]
                       .sort_values("y_prob", ascending=False)
                       .head(top_n)
                       .assign(error_type="false_positive"))
    false_negatives = (frame[is_fn]
                       .sort_values("y_prob", ascending=True)
                       .head(top_n)
                       .assign(error_type="false_negative"))

    columns = ["error_type", "player", "season", "team", "replaced",
               "y", "y_prob"]
    return pd.concat([false_positives, false_negatives])[columns].reset_index(drop=True)
