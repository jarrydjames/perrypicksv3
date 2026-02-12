from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.modeling.base import BaseTwoHeadModel, TwoHeadFitResult
from src.modeling.types_model import TrainedHead
from src.modeling.uncertainty import sigma_from_residuals


def _with_imputer(est):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", est),
    ])


class LightGBMTwoHeadModel(BaseTwoHeadModel):
    name = "lightgbm"
    version = "1"

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        feature_version: str = "v1",
    ):
        super().__init__(feature_version=feature_version)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self._fit: TwoHeadFitResult | None = None

    def fit(self, X: np.ndarray, feature_names: List[str], y_total: np.ndarray, y_margin: np.ndarray) -> "LightGBMTwoHeadModel":
        mt = _with_imputer(
            LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=0,
            )
        )
        mm = _with_imputer(
            LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=0,
            )
        )

        mt.fit(X, y_total)
        mm.fit(X, y_margin)

        res_t = y_total - mt.predict(X)
        res_m = y_margin - mm.predict(X)

        self._fit = TwoHeadFitResult(
            total=TrainedHead(features=list(feature_names), model=mt, residual_sigma=sigma_from_residuals(res_t)),
            margin=TrainedHead(features=list(feature_names), model=mm, residual_sigma=sigma_from_residuals(res_m)),
        )
        return self

    def predict_heads(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fit:
            raise RuntimeError("Model not fit")
        mt = self._fit.total.model
        mm = self._fit.margin.model
        return (mt.predict(X), mm.predict(X))

    def trained_heads(self) -> TwoHeadFitResult:
        if not self._fit:
            raise RuntimeError("Model not fit")
        return self._fit
