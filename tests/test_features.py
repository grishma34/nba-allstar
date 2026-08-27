"""Tests for src/features.py — split, selection, standardisation.

The leakage test is the one that matters: validation and test must be
transformed with the TRAINING split's mean and std, never their own.
"""

import numpy as np
import pandas as pd

from src.features import select_features, chronological_split, standardise


def _toy_dataset():
    rng = np.random.default_rng(2)
    n = 120
    return pd.DataFrame({
        "player": [f"Player {i}" for i in range(n)],
        "player_id": [f"p{i:03d}" for i in range(n)],
        "season": rng.integers(2000, 2026, size=n),
        "team": ["AAA"] * n,
        "replaced": [False] * n,
        "y": (rng.random(n) < 0.1).astype(np.float64),
        "feat_a": rng.normal(50, 10, size=n),
        "feat_b": rng.normal(0.5, 0.1, size=n),
    })


def test_chronological_split_is_disjoint_and_complete():
    df = _toy_dataset()
    train, val, test = chronological_split(df, train_end=2017, val_end=2021)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)
    assert set(train.index).isdisjoint(test.index)


def test_chronological_split_boundary_seasons_land_correctly():
    # Boundaries are inclusive on the earlier side: season == train_end is
    # training, season == val_end is validation.
    df = _toy_dataset()
    train, val, test = chronological_split(df, train_end=2017, val_end=2021)
    assert train["season"].max() <= 2017
    assert 2017 < val["season"].min() and val["season"].max() <= 2021
    assert test["season"].min() > 2021


def test_standardise_gives_train_zero_mean_unit_std():
    df = _toy_dataset()
    train, val, test = chronological_split(df, train_end=2017, val_end=2021)
    X_tr, _, _ = select_features(train, ["feat_a", "feat_b"])
    X_va, _, _ = select_features(val, ["feat_a", "feat_b"])
    X_te, _, _ = select_features(test, ["feat_a", "feat_b"])
    X_tr_s, _, _, _ = standardise(X_tr, X_va, X_te)
    assert np.allclose(X_tr_s.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(X_tr_s.std(axis=0), 1.0, atol=1e-12)


def test_standardise_applies_train_stats_to_other_splits():
    # Leakage check: the validation split must be transformed with the
    # training mean/std — its own statistics must play no part.
    df = _toy_dataset()
    train, val, test = chronological_split(df, train_end=2017, val_end=2021)
    X_tr, _, _ = select_features(train, ["feat_a", "feat_b"])
    X_va, _, _ = select_features(val, ["feat_a", "feat_b"])
    X_te, _, _ = select_features(test, ["feat_a", "feat_b"])
    _, X_va_s, X_te_s, params = standardise(X_tr, X_va, X_te)
    assert np.allclose(X_va_s, (X_va - params["mean"]) / params["std"])
    assert np.allclose(X_te_s, (X_te - params["mean"]) / params["std"])
    assert np.allclose(params["mean"], X_tr.mean(axis=0))
    assert np.allclose(params["std"], X_tr.std(axis=0))


def test_select_features_keeps_rows_aligned():
    df = _toy_dataset()
    X, y, meta = select_features(df, ["feat_b", "feat_a"])
    assert X.shape == (len(df), 2)
    # column order follows the requested list, not the frame
    assert np.allclose(X[:, 0], df["feat_b"]) and np.allclose(X[:, 1], df["feat_a"])
    assert np.array_equal(y, df["y"].to_numpy())
    assert list(meta["player"]) == list(df["player"])
