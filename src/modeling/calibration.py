from __future__ import annotations

"""Calibration helpers for pregame win-probability models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class CalibrationCurve:
    prob_pred: np.ndarray
    prob_true: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class CalibrationEvaluation:
    method: str
    brier: float
    log_loss: float


class PlattScaler:
    """Simple Platt scaling wrapper (logistic regression on raw probabilities)."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=2000)

    def fit(self, p_raw: Iterable[float], y_true: Iterable[int]) -> "PlattScaler":
        X = np.asarray(list(p_raw), dtype=float).reshape(-1, 1)
        y = np.asarray(list(y_true), dtype=int)
        self.model.fit(X, y)
        return self

    def predict(self, p_raw: Iterable[float]) -> np.ndarray:
        X = np.asarray(list(p_raw), dtype=float).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_raw: Iterable[float], y_true: Iterable[int]) -> "IsotonicCalibrator":
        x = np.asarray(list(p_raw), dtype=float)
        y = np.asarray(list(y_true), dtype=int)
        self.model.fit(x, y)
        return self

    def predict(self, p_raw: Iterable[float]) -> np.ndarray:
        x = np.asarray(list(p_raw), dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)


def evaluate_calibration(y_true: Iterable[int], p_pred: Iterable[float], method: str) -> CalibrationEvaluation:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(p_pred), dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return CalibrationEvaluation(
        method=method,
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.column_stack([1 - p, p]))),
    )


def calibration_curve_bins(*, y_true: Iterable[float], p_pred: Iterable[float], n_bins: int = 10) -> CalibrationCurve:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_pred), dtype=float)

    if y.shape != p.shape:
        raise ValueError("y_true and p_pred must have same shape")

    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(p, edges, right=True)
    idx = np.clip(idx, 1, int(n_bins))

    prob_pred = np.zeros(int(n_bins), dtype=float)
    prob_true = np.zeros(int(n_bins), dtype=float)
    count = np.zeros(int(n_bins), dtype=int)

    for b in range(1, int(n_bins) + 1):
        mask = idx == b
        c = int(np.sum(mask))
        count[b - 1] = c
        if c <= 0:
            prob_pred[b - 1] = float("nan")
            prob_true[b - 1] = float("nan")
            continue
        prob_pred[b - 1] = float(np.mean(p[mask]))
        prob_true[b - 1] = float(np.mean(y[mask]))

    return CalibrationCurve(prob_pred=prob_pred, prob_true=prob_true, count=count)


def calibration_curve_df(curve: CalibrationCurve) -> pd.DataFrame:
    return pd.DataFrame({"prob_pred": curve.prob_pred, "prob_true": curve.prob_true, "count": curve.count})


def save_reliability_plot(*, curve: CalibrationCurve, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = calibration_curve_df(curve).dropna().copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
    ax.plot(df["prob_pred"], df["prob_true"], "o-", label="Model")

    for x, y, c in zip(df["prob_pred"], df["prob_true"], df["count"], strict=False):
        ax.annotate(str(int(c)), (float(x), float(y)), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
