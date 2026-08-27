"""Tests for src/evaluate.py — the hand-written metrics.

The curve and AUC expectations are hand-computed on four-sample examples
(small enough to walk through the rank sweep by hand), plus the two
boundary cases every metric must survive: a perfect classifier and the
all-negative majority-class baseline.
"""

import numpy as np

import src.evaluate as ev


def test_binarise_threshold_is_inclusive():
    # Predict 1 iff ŷ >= τ — a probability exactly at τ counts as positive.
    out = ev.binarise(np.array([0.2, 0.35, 0.5]), threshold=0.35)
    assert np.array_equal(out, np.array([0.0, 1.0, 1.0]))


def test_confusion_hand_computed_example():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    assert ev.confusion(y_true, y_pred) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_precision_recall_f1_hand_computed_example():
    # tp=2, fp=1, fn=1 → P = 2/3, R = 2/3, F1 = 2/3
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    p, r, f1 = ev.precision_recall_f1(y_true, y_pred)
    assert np.isclose(p, 2 / 3) and np.isclose(r, 2 / 3) and np.isclose(f1, 2 / 3)


def test_all_negative_baseline_scores_zero_without_crashing():
    # The zero-division convention: predicting nobody must give 0/0/0,
    # not NaN — this is the majority-class baseline's score.
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.zeros(4)
    assert ev.precision_recall_f1(y_true, y_pred) == (0.0, 0.0, 0.0)


def test_perfect_classifier_scores_one():
    y_true = np.array([1, 0, 1, 0])
    p, r, f1 = ev.precision_recall_f1(y_true, y_true.copy())
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_roc_auc_is_one_for_perfect_ranking():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.2, 0.1])  # every positive above every negative
    assert np.isclose(ev.roc_auc(y, prob), 1.0)


def test_roc_auc_is_zero_for_reversed_ranking():
    y = np.array([0, 0, 1])
    prob = np.array([0.9, 0.8, 0.1])  # the positive is ranked dead last
    assert np.isclose(ev.roc_auc(y, prob), 0.0)


def test_pr_auc_is_one_for_perfect_ranking():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.2, 0.1])
    assert np.isclose(ev.pr_auc(y, prob), 1.0)


def test_pr_auc_hand_computed_for_worst_ranking():
    # Positives ranked last. Average precision: the first positive is
    # captured at rank 3 (precision 1/3), the second at rank 4 (precision
    # 2/4); each contributes a recall step of 1/2:
    #   AP = (1/2)(1/3) + (1/2)(1/2) = 5/12
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.9, 0.8, 0.2, 0.1])
    assert np.isclose(ev.pr_auc(y, prob), 5 / 12)


def test_calibration_table_accounts_for_every_sample():
    rng = np.random.default_rng(1)
    prob = rng.random(500)              # covers all ten bins
    y = (rng.random(500) < prob).astype(np.float64)
    table = ev.calibration_table(y, prob)
    assert table["count"].sum() == 500
    assert ((table["observed_rate"] >= 0) & (table["observed_rate"] <= 1)).all()
    assert ((table["mean_predicted"] >= 0) & (table["mean_predicted"] <= 1)).all()


def test_baseline_constant_rate_uses_training_rate_only():
    y_train = np.array([1, 0, 0, 0])    # rate 0.25
    baseline = ev.baseline_constant_rate(y_train, n=7)
    assert baseline.shape == (7,)
    assert np.all(baseline == 0.25)
