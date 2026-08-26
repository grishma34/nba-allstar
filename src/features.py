"""
Select features, split chronologically, and standardise.

This module is deliberately decision-free machinery: WHICH columns form the
feature set (Decision 5) and WHERE the split boundaries fall (Decision 6)
are passed in by the caller. Keeping the choices out of this file means they
stay visible at the call site and traceable to docs/DATA_DECISIONS.md.
"""

import numpy as np

# Carried alongside X and y so every prediction can later be traced back to
# a person and a season. `replaced` rides along because injury selections
# are likely to matter in the error analysis.
META_COLS = ["player", "player_id", "season", "team", "replaced", "y"]


def select_features(df, feature_cols):
    """
    Split the dataset frame into a numeric feature matrix, a label vector,
    and a row-aligned metadata frame.

    The metadata frame exists so that errors can be NAMED later — "the model
    assigned 0.87 and the player was not selected" needs the player and
    season attached to the row. The Criterion C analysis depends on this.

    Args:
        df: dataset from data.build_dataset(), one row per player-season.
        feature_cols: list of column names to use as features (Decision 5 —
                      the reason for each column is recorded there).

    Returns:
        X:    ndarray, shape (n, d), float64 — one row per player-season,
              columns in feature_cols order.
        y:    ndarray, shape (n,), float64 0/1 labels. Float rather than int
              because y enters the loss and gradient arithmetic directly.
        meta: DataFrame with player, player_id, season, team, replaced, y —
              same row order as X and y.
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    meta = df[META_COLS].reset_index(drop=True)

    # The eligibility threshold (Decision 3, mp >= 500) was verified to
    # remove every row with a missing candidate feature. A NaN here means
    # the threshold or the feature list changed without re-checking
    # missingness — fail loudly rather than let NaN poison the training
    # arithmetic silently (NaN survives every multiply and add).
    assert not np.isnan(X).any(), (
        "NaNs in the feature matrix — the eligibility threshold no longer "
        "covers the missing values of this feature set; revisit Decision 3")
    return X, y, meta


def chronological_split(df, train_end, val_end):
    """
    Split by season: train has season <= train_end, validation has
    train_end < season <= val_end, test has season > val_end.

    Why chronological and not random (Decision 6 records the boundaries;
    the principle is fixed): All-Star selection is COMPARATIVE — roughly 24
    slots exist per season, so the rows of one season are not independent.
    A random split would put players from the same season on both sides,
    leaking how competitive that season was into training. Chronological
    splitting also mirrors deployment: predicting a future season from past
    ones.

    Args:
        df: dataset from data.build_dataset().
        train_end: last season included in training.
        val_end:   last season included in validation (must be > train_end).

    Returns:
        (train_df, val_df, test_df) — disjoint by construction, and every
        row of df lands in exactly one of them.
    """
    assert train_end < val_end, "boundaries out of order"

    train_df = df[df["season"] <= train_end]
    val_df = df[(df["season"] > train_end) & (df["season"] <= val_end)]
    test_df = df[df["season"] > val_end]

    # An empty split would make the downstream loss/metric code divide by
    # zero somewhere far from the actual mistake — catch it here instead.
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0, (
        "a split is empty — check the boundary seasons against the data")
    return train_df, val_df, test_df


def standardise(X_train, X_val, X_test):
    """
    Scale every feature to zero mean and unit variance:

        z = (x - mean_train) / std_train

    The mean and std are computed FROM THE TRAINING SPLIT ONLY and then
    applied unchanged to validation and test. Fitting them on the full
    dataset would leak information: the test rows would have shifted the
    very numbers used to transform the training rows, so training would
    "know" something about data it is never supposed to see. Structurally,
    only X_train appears in the mean/std computation below — validation and
    test cannot influence the fit.

    Why standardise at all: the raw features live on wildly different
    scales (mp is in the thousands, ts_percent is around 0.5). Gradient
    descent takes one shared learning rate across all weights; with
    unscaled features the loss surface is a long narrow valley and the
    step size that suits one weight overshoots another. Equal scales also
    make the fitted weights comparable to each other when interpreting the
    model.

    Args:
        X_train, X_val, X_test: feature matrices from select_features(),
        already split. Columns must be in the same order in all three.

    Returns:
        (X_train_s, X_val_s, X_test_s, params) where params is
        {"mean": ndarray (d,), "std": ndarray (d,)} — kept so any future
        row can be transformed exactly the same way, and so the
        standardisation can be inspected or undone.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # A zero std means the feature is constant on the training split —
    # dividing by it would produce inf/NaN. A constant feature also carries
    # no information, so the right response is to remove it from Decision 5,
    # not to paper over it with an epsilon.
    assert (std > 0).all(), (
        "a feature is constant on the training split — remove it from the "
        "feature set rather than dividing by zero")

    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std
    params = {"mean": mean, "std": std}
    return X_train_s, X_val_s, X_test_s, params


if __name__ == "__main__":
    # Smoke test only: python -m src.features (from the repo root).
    # The column list and boundaries below are the spec's SUGGESTIONS, used
    # to exercise the machinery — they are NOT Decisions 5 and 6, which are
    # still open in docs/DATA_DECISIONS.md.
    from src.data import load_raw, build_dataset

    advanced, allstar, _ = load_raw("data")
    dataset = build_dataset(advanced, allstar, verbose=False)

    smoke_cols = ["g", "gs", "mp", "per", "ts_percent", "usg_percent",
                  "ws", "ws_48", "bpm", "vorp", "age"]
    train_df, val_df, test_df = chronological_split(
        dataset, train_end=2017, val_end=2021)

    X_tr, y_tr, meta_tr = select_features(train_df, smoke_cols)
    X_va, y_va, meta_va = select_features(val_df, smoke_cols)
    X_te, y_te, meta_te = select_features(test_df, smoke_cols)
    X_tr, X_va, X_te, params = standardise(X_tr, X_va, X_te)

    for name, X, y, df_ in [("train", X_tr, y_tr, train_df),
                            ("val", X_va, y_va, val_df),
                            ("test", X_te, y_te, test_df)]:
        seasons = f"{df_['season'].min()}-{df_['season'].max()}"
        print(f"{name:5s} {seasons}: X {X.shape}, positives {int(y.sum())} "
              f"({100 * y.mean():.1f}%)")
    print("train means ~0:", np.allclose(X_tr.mean(axis=0), 0),
          "| train stds ~1:", np.allclose(X_tr.std(axis=0), 1))
    print("val col means (not exactly 0, transformed with TRAIN params):",
          np.round(X_va.mean(axis=0)[:4], 3))
