"""
The gradient descent training loop.

This is the optimiser: repeatedly compute predictions, measure the loss,
compute the gradients, and step the parameters downhill:

    w ← w − η·∂L/∂w        b ← b − η·∂L/∂b

Batch gradient descent — every iteration uses the full training set. With
~6,000 rows and a handful of features the full-batch gradient is cheap, and
determinism makes every run reproducible without seed bookkeeping.

Hyperparameters (learning rate, iteration count, L2 strength) are caller
arguments: choosing them is Decisions 7-9 in docs/DATA_DECISIONS.md.
"""

import numpy as np

from src.model import predict_proba, log_loss, gradients


def fit(X_train, y_train, X_val, y_val, *, lr, n_iters, l2=0.0, verbose=True):
    """
    Train logistic regression by batch gradient descent.

    Each iteration:
      1. Forward pass: ŷ = σ(X_train·w + b)
      2. Record the current train and validation loss (so the loss curves
         show the state BEFORE this iteration's update)
      3. Gradients: ∂L/∂w = (1/n)·Xᵀ(ŷ−y), ∂L/∂b = (1/n)·Σ(ŷ−y)
      4. Step downhill: w ← w − η·∂L/∂w, b ← b − η·∂L/∂b

    The validation set is only ever FORWARDED through the model to measure
    generalisation per iteration — its gradients are never computed, so it
    cannot influence the parameters.

    Initialisation is w = 0, b = 0. Unlike neural networks there is no
    symmetry to break: the log loss of logistic regression is convex, so
    gradient descent reaches the same global optimum from any start; zeros
    also make the first iteration's loss exactly the loss of "predict 0.5
    for everyone" (≈0.693), a useful sanity anchor for the loss curve.

    Optional L2 regularisation adds (λ/2n)·‖w‖² to the loss, which adds
    (λ/n)·w to ∂L/∂w — each step then also shrinks the weights toward
    zero. The bias b is not penalised: it encodes the base selection rate,
    and shrinking it would push predictions toward 0.5 rather than toward
    the honest prior. Whether λ > 0 is used is Decision 9.

    Args:
        X_train: ndarray (n, d) — standardised training features.
        y_train: ndarray (n,)   — 0/1 training labels.
        X_val:   ndarray (m, d) — standardised validation features.
        y_val:   ndarray (m,)   — 0/1 validation labels.
        lr:      η, the learning rate (Decision 7 — chosen by trying
                 several values and recording what happened).
        n_iters: number of full-batch iterations (Decision 8).
        l2:      λ, L2 penalty strength; 0.0 disables it (Decision 9).
        verbose: print the losses ten times over the run.

    Returns:
        (w, b, history):
            w — ndarray (d,), fitted weights.
            b — float, fitted bias.
            history — {"train_loss": [...], "val_loss": [...]}, one entry
            per iteration, for the loss curves and stopping analysis.
    """
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0.0
    history = {"train_loss": [], "val_loss": []}

    for iteration in range(n_iters):
        # Forward pass on both splits with the CURRENT parameters.
        y_pred_train = predict_proba(X_train, w, b)
        y_pred_val = predict_proba(X_val, w, b)

        train_loss = log_loss(y_train, y_pred_train)
        val_loss = log_loss(y_val, y_pred_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and iteration % max(1, n_iters // 10) == 0:
            print(f"  iter {iteration:5d}: train loss {train_loss:.4f}, "
                  f"val loss {val_loss:.4f}")

        # Gradients of the mean log loss (derivation in model.gradients).
        dw, db = gradients(X_train, y_train, y_pred_train)

        # L2 term: gradient of (λ/2n)·‖w‖² is (λ/n)·w. Bias not penalised
        # (see docstring).
        if l2 > 0.0:
            dw = dw + (l2 / n) * w

        # The update itself — step against the gradient, scaled by η.
        w = w - lr * dw
        b = b - lr * db

    return w, b, history
