from __future__ import annotations

"""Stacked pregame ensemble for spread and total prediction."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.modeling.elo_pregame import EloPregameModel


@dataclass(frozen=True)
class StackedPregameConfig:
    n_splits: int = 8
    random_state: int = 42


class StackedPregameEnsemble:
    def __init__(self, config: StackedPregameConfig | None = None):
        self.config = config or StackedPregameConfig()
        self.base_models = self._build_base_models()
        self.meta_model = Ridge(alpha=1.0)
        self.fitted_base_models: dict[str, object] = {}
        self.fitted_meta_model: object | None = None
        self.elo_model: EloPregameModel | None = None

    def _build_base_models(self) -> dict[str, object]:
        models: dict[str, object] = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=3.0),
            "gbr": GradientBoostingRegressor(random_state=self.config.random_state),
        }
        optional: list[tuple[str, str, Callable[[], object]]] = [
            ("xgboost", "xgboost", lambda: __import__("xgboost").XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=self.config.random_state)),
            ("catboost", "catboost", lambda: __import__("catboost").CatBoostRegressor(iterations=350, depth=6, learning_rate=0.05, loss_function="MAE", verbose=False, random_seed=self.config.random_state)),
            ("lightgbm", "lightgbm", lambda: __import__("lightgbm").LGBMRegressor(n_estimators=350, learning_rate=0.04, max_depth=-1, num_leaves=31, subsample=0.85, colsample_bytree=0.85, random_state=self.config.random_state)),
        ]
        for name, module_name, builder in optional:
            try:
                __import__(module_name)
                models[name] = builder()
            except Exception:
                continue
        return models

    @staticmethod
    def _elo_features(elo: EloPregameModel, frame: pd.DataFrame) -> np.ndarray:
        vals = [elo.features_for_matchup(r["home_team"], r["away_team"])["elo_net_rating_proxy"] for _, r in frame.iterrows()]
        return np.asarray(vals, dtype=float)

    def fit_predict_oof(self, X: np.ndarray, y: np.ndarray, game_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        splitter = TimeSeriesSplit(n_splits=self.config.n_splits)
        oof = {name: np.full(len(X), np.nan, dtype=float) for name in self.base_models}
        oof["elo"] = np.full(len(X), np.nan, dtype=float)

        for train_idx, test_idx in splitter.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            for name, model in self.base_models.items():
                fitted = clone(model)
                fitted.fit(X_train, y_train)
                oof[name][test_idx] = fitted.predict(X_test)

            elo_fold = EloPregameModel().fit(game_frame.iloc[train_idx]) if len(train_idx) > 0 else EloPregameModel()
            oof["elo"][test_idx] = self._elo_features(elo_fold, game_frame.iloc[test_idx])

        return oof

    def fit(self, X: np.ndarray, y: np.ndarray, game_frame: pd.DataFrame) -> "StackedPregameEnsemble":
        oof = self.fit_predict_oof(X, y, game_frame)
        ordered = sorted(oof)
        base_matrix = np.column_stack([oof[name] for name in ordered])

        valid_rows = ~np.isnan(base_matrix).any(axis=1)
        self.meta_model.fit(base_matrix[valid_rows], y[valid_rows])
        self.fitted_meta_model = self.meta_model

        for name, model in self.base_models.items():
            trained = clone(model)
            trained.fit(X, y)
            self.fitted_base_models[name] = trained

        self.elo_model = EloPregameModel().fit(game_frame)
        return self

    def predict(self, X: np.ndarray, game_frame: pd.DataFrame) -> np.ndarray:
        if self.fitted_meta_model is None:
            raise RuntimeError("Must fit stacked ensemble before predict().")
        if self.elo_model is None:
            raise RuntimeError("Elo model not fitted. Call fit() first.")

        preds = [self.fitted_base_models[name].predict(X) for name in sorted(self.base_models)]
        preds.append(self._elo_features(self.elo_model, game_frame))
        meta_X = np.column_stack(preds)
        return self.fitted_meta_model.predict(meta_X)

    @staticmethod
    def score_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_error(y_true, y_pred))
