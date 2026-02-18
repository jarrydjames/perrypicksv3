from __future__ import annotations

"""Independent pregame win-probability modeling (not derived from margin)."""

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class WinProbabilityConfig:
    n_splits: int = 8
    random_state: int = 42


class WinProbabilityPregameModel:
    def __init__(self, config: WinProbabilityConfig | None = None):
        self.config = config or WinProbabilityConfig()
        self.models = {
            "logistic": LogisticRegression(max_iter=2000),
            "gboost": GradientBoostingClassifier(random_state=self.config.random_state),
        }
        self.fitted_models: dict[str, object] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WinProbabilityPregameModel":
        for name, model in self.models.items():
            model.fit(X, y)
            self.fitted_models[name] = model

        calibrated = CalibratedClassifierCV(
            estimator=self.fitted_models["gboost"],
            method="isotonic",
            cv=TimeSeriesSplit(n_splits=min(self.config.n_splits, 5)),
        )
        calibrated.fit(X, y)
        self.fitted_models["gboost_calibrated"] = calibrated
        return self

    def predict_proba(self, X: np.ndarray, model_name: str = "gboost_calibrated") -> np.ndarray:
        if model_name not in self.fitted_models:
            raise KeyError(f"Unknown model {model_name}. Fit model before prediction.")
        return self.fitted_models[model_name].predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray, model_name: str = "gboost_calibrated") -> dict[str, float]:
        p = self.predict_proba(X, model_name=model_name)
        return {
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.column_stack([1 - p, p]))),
        }
