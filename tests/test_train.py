"""Tests for src/train.py — the batch gradient descent loop."""

import numpy as np

from src.train import fit


def _separable_data():
    # Two clearly separated clusters: descent must drive the loss far below
    # the coin-flip starting point.
    rng = np.random.default_rng(3)
    X_pos = rng.normal(loc=+2.0, size=(40, 2))
    X_neg = rng.normal(loc=-2.0, size=(40, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.r_[np.ones(40), np.zeros(40)]
    return X, y


def test_first_recorded_loss_is_coin_flip():
    # Zero initialisation predicts 0.5 for everyone, so the loss recorded
    # before the first update must be ln 2 ≈ 0.6931 — the sanity anchor
    # every loss curve starts from.
    X, y = _separable_data()
    _, _, history = fit(X, y, X, y, lr=0.5, n_iters=5, verbose=False)
    assert np.isclose(history["train_loss"][0], np.log(2.0))


def test_loss_descends_on_separable_data():
    X, y = _separable_data()
    _, _, history = fit(X, y, X, y, lr=0.5, n_iters=500, verbose=False)
    assert len(history["train_loss"]) == 500
    assert len(history["val_loss"]) == 500
    assert history["train_loss"][-1] < 0.05  # far below the 0.693 start


def test_l2_shrinks_the_weights():
    X, y = _separable_data()
    w_free, _, _ = fit(X, y, X, y, lr=0.5, n_iters=500, l2=0.0, verbose=False)
    w_reg, _, _ = fit(X, y, X, y, lr=0.5, n_iters=500, l2=50.0, verbose=False)
    assert np.linalg.norm(w_reg) < np.linalg.norm(w_free)
