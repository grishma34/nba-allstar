"""Tests for src/model.py — the hand-written forward pass, loss and gradients.

Every expected value here is either a mathematical identity (σ(0) = 0.5,
loss of 0.5-everywhere = ln 2) or hand-computed on an example small enough
to verify with a calculator. The gradient check is the load-bearing test:
it compares the analytic gradients against central finite differences, which
verifies the calculus independently of the code that implements it.
"""

import numpy as np

from src.model import sigmoid, predict_proba, log_loss, gradients


def test_sigmoid_at_zero():
    assert sigmoid(np.array([0.0])) == 0.5


def test_sigmoid_symmetry():
    z = np.array([-3.0, -1.0, 0.5, 2.0, 7.0])
    assert np.allclose(sigmoid(-z), 1.0 - sigmoid(z))


def test_sigmoid_extreme_inputs_stay_finite():
    # z = ±1000 would overflow exp() without the internal clip.
    out = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] >= 0.0 and out[1] <= 1.0
    assert out[0] < 1e-6 and out[1] > 1 - 1e-6


def test_predict_proba_matches_manual_computation():
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    w = np.array([0.5, -0.25])
    b = 0.1
    expected = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    assert np.allclose(predict_proba(X, w, b), expected)


def test_log_loss_of_half_everywhere_is_ln2():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.full(4, 0.5)
    assert np.isclose(log_loss(y, p), np.log(2.0))


def test_log_loss_finite_on_exact_zero_and_one():
    # The epsilon clip is exactly for this case: log(0) would be -inf.
    y = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])  # maximally wrong AND saturated
    assert np.isfinite(log_loss(y, p))


def test_log_loss_hand_computed_example():
    # loss = -(log 0.8 + log(1 - 0.3)) / 2, computed by hand
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert np.isclose(log_loss(y, p), expected)


def test_log_loss_eps_choice_is_irrelevant_away_from_saturation():
    # For predictions inside every clip range the eps value cannot matter.
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.7, 0.2, 0.9])
    losses = {eps: log_loss(y, p, eps=eps) for eps in (1e-10, 1e-15, 1e-20)}
    assert len(set(losses.values())) == 1


def test_gradients_match_finite_differences():
    # The central-difference check: perturb each parameter by ±h and compare
    # the measured slope of the loss against the analytic gradient.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = (rng.random(50) < 0.3).astype(np.float64)
    w = rng.normal(scale=0.5, size=4)
    b = 0.3
    h = 1e-6

    y_pred = predict_proba(X, w, b)
    dw, db = gradients(X, y, y_pred)

    for j in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h
        w_minus[j] -= h
        numeric = (log_loss(y, predict_proba(X, w_plus, b))
                   - log_loss(y, predict_proba(X, w_minus, b))) / (2 * h)
        assert abs(numeric - dw[j]) / max(abs(numeric), abs(dw[j])) < 1e-6
    numeric_b = (log_loss(y, predict_proba(X, w, b + h))
                 - log_loss(y, predict_proba(X, w, b - h))) / (2 * h)
    assert abs(numeric_b - db) / max(abs(numeric_b), abs(db)) < 1e-6


def test_gradients_zero_when_predictions_equal_labels():
    # ∂L/∂w = (1/n)Xᵀ(ŷ−y): a perfect fit has zero gradient everywhere.
    X = np.array([[1.0, 2.0], [3.0, -1.0]])
    y = np.array([1.0, 0.0])
    dw, db = gradients(X, y, y.copy())
    assert np.allclose(dw, 0.0) and np.isclose(db, 0.0)
