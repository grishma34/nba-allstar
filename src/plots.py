"""
Figures for the report and slides. Each function returns a matplotlib
Figure; the caller decides where to save or show it.

Design rules applied throughout (they are checks, not taste):
- Colors come from the Okabe-Ito colorblind-safe palette. The two series
  hues are its blue/vermillion pair (orange was rejected: contrast 2.25:1
  on white is too weak for thin marks; vermillion gives 3.87:1).
- Color never carries identity alone: every series also differs by line
  style or marker shape, and reference lines are labelled.
- Reference/comparison lines (random baseline, perfect calibration, the
  decision threshold) are recessive gray dashes — context, not data.
- One axis per chart. Grids are light and sit behind the data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito hues (validated colorblind-safe palette; Okabe & Ito 2008).
BLUE = "#0072B2"        # primary series
VERMILLION = "#D55E00"  # second series / positive class
GRAY = "#7F7F7F"        # reference lines only — never a data series
INK = "#333333"         # text; text never wears a series color


def _style(ax):
    """Recessive frame: data in front, scaffolding barely visible."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)  # grid behind the marks, never through them
    ax.tick_params(colors=INK, labelsize=9)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("0.7")


def loss_curves(history):
    """
    Training and validation log loss per iteration — the optimiser's story.

    How to read it: both curves start at log(2) ≈ 0.693 (w = 0 predicts 0.5
    for everyone) and fall as gradient descent works. The gap between the
    curves is the overfitting signal; the marked validation minimum is the
    point early stopping would pick (Decision 8 discussion).

    Args:
        history: dict from train.fit() with "train_loss" and "val_loss"
                 lists, one entry per iteration.

    Returns:
        matplotlib Figure.
    """
    train = np.asarray(history["train_loss"])
    val = np.asarray(history["val_loss"])
    iters = np.arange(len(train))

    fig, ax = plt.subplots(figsize=(7, 4))
    # Line style doubles the color coding so the pair survives grayscale
    # printing and every form of colorblindness.
    ax.plot(iters, train, color=BLUE, linewidth=2, label="train")
    ax.plot(iters, val, color=VERMILLION, linewidth=2, linestyle="--",
            label="validation")

    best = int(np.argmin(val))
    ax.plot(best, val[best], marker="o", markersize=7, color=VERMILLION,
            markerfacecolor="white", markeredgewidth=2)
    ax.annotate(f"val min: {val[best]:.4f} @ iter {best}",
                xy=(best, val[best]), xytext=(best, val[best] + 0.06),
                fontsize=9, color=INK, ha="left")

    ax.set_xlabel("iteration", color=INK)
    ax.set_ylabel("log loss", color=INK)
    ax.set_title("Gradient descent: loss per iteration", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def pr_plot(precision, recall, auc=None, base_rate=None):
    """
    Precision-recall curve — the informative curve under ~7% positives.

    How to read it: each point is one threshold; moving right (more recall)
    costs precision. The gray line is what random ranking achieves —
    precision equal to the base rate at every recall, NOT 0.5.

    Args:
        precision, recall: arrays from evaluate.pr_curve().
        auc: optional PR-AUC to print in the title.
        base_rate: optional positive rate, drawn as the random baseline.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, color=BLUE, linewidth=2)
    if base_rate is not None:
        ax.axhline(base_rate, color=GRAY, linewidth=1.2, linestyle="--")
        ax.annotate(f"random = base rate ({base_rate:.3f})",
                    xy=(0.02, base_rate + 0.02), fontsize=8.5, color=GRAY)
    title = "Precision-recall (test)"
    if auc is not None:
        title += f" — PR-AUC {auc:.3f}"
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel("recall", color=INK)
    ax.set_ylabel("precision", color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    return fig


def roc_plot(fpr, tpr, auc=None):
    """
    ROC curve, with the chance diagonal for reference.

    Reported alongside PR because it is conventional; under this imbalance
    it flatters the model (the false positive rate divides by ~1,350
    negatives), which is why PR is the primary curve.

    Args:
        fpr, tpr: arrays from evaluate.roc_curve().
        auc: optional ROC-AUC to print in the title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.2, linestyle="--")
    ax.annotate("random", xy=(0.55, 0.5), fontsize=8.5, color=GRAY,
                rotation=38)
    title = "ROC (test)"
    if auc is not None:
        title += f" — ROC-AUC {auc:.3f}"
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel("false positive rate", color=INK)
    ax.set_ylabel("true positive rate", color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    return fig


def calibration_plot(table):
    """
    Calibration: mean predicted probability vs observed selection rate per
    bin, against the y = x line of perfect calibration.

    How to read it: points above the diagonal are bins where the model is
    UNDER-confident (more selections happened than it predicted); below,
    over-confident. Marker area scales with bin count and each point is
    labelled with n, because the low bins hold thousands of rows and the
    high bins a handful — equal-sized dots would hide that asymmetry.

    Args:
        table: DataFrame from evaluate.calibration_table().

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.2, linestyle="--")
    ax.annotate("perfect calibration", xy=(0.62, 0.57), fontsize=8.5,
                color=GRAY, rotation=38)

    counts = table["count"].to_numpy()
    # sqrt keeps the AREA proportional to count, which is how dot size is read
    sizes = 30 + 300 * np.sqrt(counts / counts.max())
    ax.scatter(table["mean_predicted"], table["observed_rate"], s=sizes,
               color=BLUE, alpha=0.75, edgecolor="white", linewidth=1,
               zorder=3)
    for _, row in table.iterrows():
        ax.annotate(f"n={int(row['count'])}",
                    xy=(row["mean_predicted"], row["observed_rate"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color=INK)

    ax.set_title("Calibration (test)", color=INK, fontsize=11)
    ax.set_xlabel("mean predicted probability (bin)", color=INK)
    ax.set_ylabel("observed selection rate (bin)", color=INK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _style(ax)
    fig.tight_layout()
    return fig


def winrate_scatter(win_rate, y_prob, y_true, threshold=None):
    """
    Predicted probability against team win rate, coloured by outcome — the
    Criterion C figure.

    How to read it: if team success were the missing variable, the model's
    errors would separate horizontally (false positives left, false
    negatives right). They do not — selected and non-selected players mix
    across the win-rate axis at every probability level, which is the
    visual form of the refuted hypothesis in DATA_DECISIONS.md. The dashed
    line is the decision threshold τ: selected players below it are the
    false negatives, non-selected players above it the false positives.

    Args:
        win_rate: ndarray (n,) — team win rate per player-season.
        y_prob: ndarray (n,) — predicted probabilities.
        y_true: ndarray (n,) — 0/1 labels.
        threshold: optional τ, drawn as a horizontal reference line.

    Returns:
        matplotlib Figure.
    """
    win_rate = np.asarray(win_rate, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    selected = np.asarray(y_true, dtype=np.float64) == 1

    fig, ax = plt.subplots(figsize=(7, 5))
    # Marker shape doubles the color coding (circle vs triangle); the
    # majority class is drawn first and translucent so the ~7% positives
    # stay visible on top of it.
    ax.scatter(win_rate[~selected], y_prob[~selected], s=14, marker="o",
               color=BLUE, alpha=0.30, linewidth=0, label="not selected")
    ax.scatter(win_rate[selected], y_prob[selected], s=34, marker="^",
               color=VERMILLION, alpha=0.85, edgecolor="white",
               linewidth=0.5, label="All-Star")

    if threshold is not None:
        ax.axhline(threshold, color=GRAY, linewidth=1.2, linestyle="--")
        # x in axes fraction, y in data coords — the win-rate axis rarely
        # starts at 0, so a data-coordinate x would fall outside the plot.
        ax.annotate(f"τ = {threshold:.2f}",
                    xy=(0.99, threshold + 0.015),
                    xycoords=("axes fraction", "data"),
                    fontsize=8.5, color=GRAY, ha="right")

    ax.set_title("Predicted probability vs team win rate (test)",
                 color=INK, fontsize=11)
    ax.set_xlabel("team win rate (minutes-weighted for traded players)",
                  color=INK)
    ax.set_ylabel("predicted All-Star probability", color=INK)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig
