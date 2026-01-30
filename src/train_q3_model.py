"""Train Q3 models - follows same pipeline as halftime training.

This script trains Q3 two-head models (total + margin) using the same
training infrastructure, calibration, and quantile regression as halftime.

Key differences from halftime training:
- Uses Q3 dataset (data/processed/q3_team_v2.parquet)
- Targets are q3_total and q3_margin (but still predicts game outcomes)
- Models stored in models_v3/q3/ (separate from halftime models)
"""

from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.training_loader import TrainingDataSpec, load_training_df
from src.modeling.feature_columns import feature_columns
from src.modeling.sklearn_models import GBTTwoHeadModel, RandomForestTwoHeadModel, RidgeTwoHeadModel

# Q3-specific targets
TARGET_TOTAL = "q3_total"
TARGET_MARGIN = "q3_margin"


def train_q3_models(
    parquet_path: Path,
    out_dir: Path,
    *,
    include_xgb: bool = False,
    include_cat: bool = False,
) -> None:
    """
    Train Q3 two-head models following same methodology as halftime.
    
    Args:
        parquet_path: Path to Q3 training parquet
        out_dir: Output directory for models (models_v3/q3/)
        include_xgb: Include XGBoost models (backtest-only)
        include_cat: Include CatBoost models (backtest-only)
    """
    out_dir = out_dir / "q3"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Q3 dataset
    df = load_training_df(TrainingDataSpec(path=parquet_path))
    feats = feature_columns(df)
    
    X = df[feats].to_numpy(dtype=float)
    y_total = df[TARGET_TOTAL].to_numpy(dtype=float)
    y_margin = df[TARGET_MARGIN].to_numpy(dtype=float)
    
    # Feature version: bump when feature engineering changes
    feature_ver = "v3_q3"
    
    # Same model types as halftime
    models = [
        RidgeTwoHeadModel(alpha=2.0, feature_version=feature_ver),
        RandomForestTwoHeadModel(feature_version=feature_ver),
        GBTTwoHeadModel(feature_version=feature_ver),
    ]
    
    # Backtest-only optional models
    if include_xgb:
        from src.modeling.xgb_models import XGBoostTwoHeadModel
        models.append(XGBoostTwoHeadModel(feature_version=feature_ver))
    
    if include_cat:
        from src.modeling.cat_models import CatBoostTwoHeadModel
        models.append(CatBoostTwoHeadModel(feature_version=feature_ver))
    
    # Quantile model (same as halftime)
    def q_model(alpha: float) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=float(alpha),
                        random_state=0,
                    ),
                ),
            ]
        )
    
    for m in models:
        m.fit(X, feats, y_total, y_margin)
        heads = m.trained_heads()
        
        # Quantile models for distribution-free 80% intervals (same as halftime)
        q10_total = q_model(0.10).fit(X, y_total)
        q90_total = q_model(0.90).fit(X, y_total)
        q10_margin = q_model(0.10).fit(X, y_margin)
        q90_margin = q_model(0.90).fit(X, y_margin)
        
        # Joint model for residual covariance (same as halftime)
        joint = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=500,
                        min_samples_leaf=30,
                        random_state=0,
                    ),
                ),
            ]
        ).fit(X, np.vstack([y_total, y_margin]).T)
        
        joint_pred = joint.predict(X)
        res_t = y_total - joint_pred[:, 0]
        res_m = y_margin - joint_pred[:, 1]
        cov = np.cov(np.vstack([res_t, res_m]))  # 2x2
        
        # Save model (same structure as halftime)
        payload = {
            "model_name": m.name,
            "model_version": m.version,
            "feature_version": m.feature_version,
            "features": feats,
            "joint": {"model": joint, "residual_cov": cov.tolist()},
            "total": {
                "model": heads.total.model,
                "residual_sigma": heads.total.residual_sigma,
                "q10_model": q10_total,
                "q90_model": q90_total,
            },
            "margin": {
                "model": heads.margin.model,
                "residual_sigma": heads.margin.residual_sigma,
                "q10_model": q10_margin,
                "q90_model": q90_margin,
            },
        }
        
        # Save to models_v3/q3/ directory
        out_path = out_dir / f"{m.name}_twohead.joblib"
        joblib.dump(payload, out_path)
        print(f"Saved Q3 model: {out_path}")


def main() -> None:
    """CLI entry point for training Q3 models."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/q3_team_v2.parquet"),
        help="Path to Q3 training parquet",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models_v3"),
        help="Output directory for Q3 models",
    )
    ap.add_argument(
        "--include-xgb",
        action="store_true",
        help="Backtest-only: train XGBoost two-head",
    )
    ap.add_argument(
        "--include-cat",
        action="store_true",
        help="Backtest-only: train CatBoost two-head",
    )
    args = ap.parse_args()
    
    train_q3_models(
        args.data,
        args.out_dir,
        include_xgb=args.include_xgb,
        include_cat=args.include_cat,
    )


if __name__ == "__main__":
    main()
