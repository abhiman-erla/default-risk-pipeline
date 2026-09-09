"""Evaluation metrics and diagnostic plots.

Primary metric is PR-AUC (average precision) given ~20% positives. Everything
else supports the operating-point and business framing the README reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Metrics:
    pr_auc: float  # average precision -- primary
    roc_auc: float
    brier: float  # calibration error (lower = better)
    ks: float  # max separation between good/bad score CDFs
    recall_at_precision: float
    threshold_at_precision: float
    precision_target: float
    expected_loss_avoided: float
    realized_loss_avoided: float
    n: int
    positive_rate: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    y = y_true[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def _recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target: float
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more element than thresholds
    ok = precision[:-1] >= target
    if not ok.any():
        return 0.0, 1.0
    idx = np.argmax(recall[:-1][ok])  # best recall among precision-satisfying points
    candidate_recall = recall[:-1][ok][idx]
    candidate_thr = thresholds[ok][idx]
    return float(candidate_recall), float(candidate_thr)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    loan_amount: pd.Series | np.ndarray | None = None,
    precision_target: float = 0.5,
) -> Metrics:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)

    recall_at_p, thr = _recall_at_precision(y, p, precision_target)

    amount = np.ones_like(p) if loan_amount is None else np.asarray(loan_amount, dtype=float)
    declined = p >= thr
    expected_loss_avoided = float(np.sum(p[declined] * amount[declined]))
    realized_loss_avoided = float(np.sum(amount[declined & (y == 1)]))

    return Metrics(
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ks=_ks_statistic(y, p),
        recall_at_precision=recall_at_p,
        threshold_at_precision=float(thr),
        precision_target=float(precision_target),
        expected_loss_avoided=expected_loss_avoided,
        realized_loss_avoided=realized_loss_avoided,
        n=len(y),
        positive_rate=float(y.mean()),
    )


# --------------------------------------------------------------------------
# Plots -- returned as matplotlib figures so callers decide where to save
# --------------------------------------------------------------------------
def pr_curve_figure(y_true: np.ndarray, y_prob: np.ndarray) -> Figure:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2)
    ax.axhline(np.mean(y_true), ls="--", c="grey", lw=1, label="base rate")
    ax.set(xlabel="recall", ylabel="precision", title=f"PR curve (AP={ap:.3f})")
    ax.legend()
    fig.tight_layout()
    return fig


def calibration_figure(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> Figure:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label="model")
    ax.set(
        xlabel="mean predicted probability", ylabel="observed frequency", title="Reliability curve"
    )
    ax.legend()
    fig.tight_layout()
    return fig
