"""
The logistic regression model, hand-written with numpy.

Nothing in this file is imported from a machine learning library — that is
the point (REQUIREMENTS.md §1.1). Four functions:

    sigmoid(z)                 squash a raw score into (0, 1)
    predict_proba(X, w, b)     the forward pass:  ŷ = σ(Xw + b)
    log_loss(y_true, y_pred)   binary cross-entropy, the training objective
    gradients(X, y_true, y_pred)   ∂L/∂w and ∂L/∂b, derived in the docstring

The hypothesis space this defines: every function of the form
f(x) = σ(wᵀx + b) for some w ∈ ℝ^d, b ∈ ℝ — one hypothesis per parameter
setting. All decision boundaries are linear in feature space; interactions
between features (e.g. "production matters MORE on winning teams") are
outside the space unless built explicitly as features. That limitation is
central to the Criterion C analysis (PROJECT_SPEC §7).
"""

import numpy as np


def sigmoid(z):
    """
    The logistic function:

        σ(z) = 1 / (1 + e^(−z))

    Maps any real number into (0, 1), which is what lets a linear score be
    read as a probability. σ(0) = 0.5; large positive z → 1; large negative
    z → 0. It is smooth and differentiable everywhere, and its derivative
    has the convenient form σ(z)(1 − σ(z)) — used in the gradient
    derivation below.

    Numerical note: for z below about −745, e^(−z) overflows float64 and
    numpy emits a warning before returning inf (the result 1/inf = 0 is
    actually correct). Clipping z to ±500 avoids the overflow while
    changing nothing observable: σ(±500) is already within 10^−217 of the
    limit values 1 and 0.

    Args:
        z: ndarray of raw scores (logits), any shape.

    Returns:
        ndarray, same shape — σ(z), values strictly in (0, 1).
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def predict_proba(X, w, b):
    """
    The forward pass — the model itself:

        ŷ = σ(Xw + b)

    Two steps. First the linear combination z = Xw + b: each feature is
    multiplied by its weight and summed, plus the bias. This is the model's
    raw score for each player-season — any real number, higher meaning
    "more All-Star-like" once trained. Then the sigmoid squashes each score
    into (0, 1) so it can be read as an estimated probability of selection.

    Args:
        X: ndarray (n, d) — standardised feature matrix.
        w: ndarray (d,)   — one weight per feature.
        b: float          — bias; the score of an all-average player, since
                            standardised features make "average" the zero
                            vector.

    Returns:
        ndarray (n,) — estimated selection probability per player-season.
    """
    z = X @ w + b
    return sigmoid(z)


def log_loss(y_true, y_pred, eps=1e-15):
    """
    Binary cross-entropy loss — the training objective:

        L = −(1/n) · Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]

    Reads as: for each sample, if the true label is 1 we are penalised by
    −log(ŷ) — small when ŷ is near 1, exploding as ŷ approaches 0. If the
    true label is 0, the mirrored term −log(1−ŷ) applies instead. Only one
    of the two terms is ever non-zero for a given sample.

    The practical consequence: being CONFIDENTLY WRONG is punished far more
    heavily than being uncertain. Predicting 0.99 on a negative sample
    costs about 4.6; predicting 0.6 costs about 0.9. This matters for the
    Criterion C analysis — a confident error on one snubbed star can
    outweigh many small errors on obvious non-selections.

    eps clips predictions into [eps, 1−eps] before the log, because
    log(0) = −inf would make the loss undefined and break training.

    Args:
        y_true: ndarray (n,) of 0/1 labels.
        y_pred: ndarray (n,) of predicted probabilities in (0, 1).
        eps:    clipping bound for numerical stability.

    Returns:
        float — mean loss over the n samples. Lower is better; 0 is a
        perfect, fully confident model.
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    per_sample_loss = -(y_true * np.log(y_pred_clipped)
                        + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
    return float(per_sample_loss.mean())


def gradients(X, y_true, y_pred):
    """
    Gradients of the log loss with respect to w and b.

    Derivation (chain rule through loss → sigmoid → linear score), for one
    sample and then averaged. With ŷ = σ(z) and z = xᵀw + b:

      1. Loss term:      L = −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]
         ∂L/∂ŷ = −y/ŷ + (1−y)/(1−ŷ)

      2. Sigmoid:        ∂ŷ/∂z = σ(z)·(1−σ(z)) = ŷ·(1−ŷ)

      3. Multiply (chain rule) and simplify:
         ∂L/∂z = [−y/ŷ + (1−y)/(1−ŷ)] · ŷ(1−ŷ)
               = −y·(1−ŷ) + (1−y)·ŷ
               = ŷ − y

         The messy fractions cancel exactly because the log in the loss is
         the inverse of the exponential in the sigmoid — this clean
         "prediction minus truth" error term is WHY log loss is the
         standard pairing for sigmoid outputs.

      4. Linear score:   ∂z/∂w = x   and   ∂z/∂b = 1, so per sample
         ∂L/∂w = (ŷ − y)·x     and     ∂L/∂b = (ŷ − y)

    Averaged over all n samples, in matrix form:

        ∂L/∂w = (1/n) · Xᵀ(ŷ − y)
        ∂L/∂b = (1/n) · Σ(ŷ − y)

    Interpretation: (ŷ − y) is the signed prediction error per sample.
    Multiplying by Xᵀ distributes each sample's error back across the
    feature values that produced it — features that were large when the
    model was wrong get their weights corrected hardest.

    Correctness of this calculus is verified against numerical finite
    differences in tests/test_model.py.

    Args:
        X:      ndarray (n, d) — feature matrix used for the predictions.
        y_true: ndarray (n,)   — 0/1 labels.
        y_pred: ndarray (n,)   — predicted probabilities from predict_proba.

    Returns:
        (dw, db):
            dw — ndarray (d,), gradient of mean loss w.r.t. each weight.
            db — float, gradient of mean loss w.r.t. the bias.
    """
    n = X.shape[0]
    error = y_pred - y_true          # (ŷ − y), shape (n,)
    dw = (X.T @ error) / n           # (1/n)·Xᵀ(ŷ − y), shape (d,)
    db = float(error.mean())         # (1/n)·Σ(ŷ − y)
    return dw, db
