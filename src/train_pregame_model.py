"""Train pregame models - follows same pipeline as halftime/Q3 training.

This script trains pregame two-head models (total + margin) using the same
training infrastructure, calibration, and quantile regression as halftime/Q3.

Key differences from halftime/Q3:
- Uses pregame dataset (data/processed/pregame_team_v2.parquet)
- No game state features (only team stats)
- Models stored in models_v3/pregame/ (separate from halftime/Q3)
- Predicts same targets: final total, final margin
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

# Pregame targets (same as halftime/Q3 - final outcomes)
TARGET_TOTAL = "total"
TARGET_MARGIN = "margin"


def train_pregame_models(
    parquet_path: Path,
    out_dir: Path,
    *,
    include_xgb: bool = False,
    include_cat: bool = False,
) -> None:
    """
    Train pregame two-head models following same methodology as halftime/Q3.
    
    Args:
        parquet_path: Path to pregame training parquet
        out_dir: Output directory for models (models_v3/pregame/)
        include_xgb: Include XGBoost models (backtest-only)
        include_cat: Include CatBoost models (backtest-only)
    """
    out_dir = out_dir / "pregame"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pregame dataset
    df = load_training_df(TrainingDataSpec(path=parquet_path))
    feats = feature_columns(df, ignore={"game_id", "home_tri", "away_tri"})
    
    X = df[feats].to_numpy(dtype=float)
    y_total = df[TARGET_TOTAL].to_numpy(dtype=float)
    y_margin = df[TARGET_MARGIN].to_numpy(dtype=float)
    
    # Feature version: bump when feature engineering changes
    feature_ver = "v3_pregame"
    
    # Same model types as halftime/Q3
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
    
    # Quantile model (same as halftime/Q3)
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
        
        # Quantile models for distribution-free 80% intervals (same as halftime/Q3)
        q10_total = q_model(0.1)
        q90_total = q_model(0.9)
        q10_margin = q_model(0.1)
        q90_margin = q_model(0.9)
        
        q10_total.fit(X, y_total)
        q90_total.fit(X, y_total)
        q10_margin.fit(X, y_margin)
        q90_margin.fit(X, y_margin)
        
        # Build model dict (same structure as halftime/Q3)
        model_dict = {
            "features": feats,
            "feature_version": feature_ver,
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
        
        # Save model (same naming as halftime/Q3)
        model_name = f"{m.__class__.__name__.replace('TwoHeadModel', '').lower()}_twohead.joblib"
        model_path = out_dir / model_name
        joblib.dump(model_dict, model_path)
        print(f"Saved pregame model: {model_path}")
    
    print(f"\\nPregame models trained successfully!")
    print(f"Features: {len(feats)}")
    print(f"Training samples: {len(df)}")
    print(f"Output directory: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train pregame two-head models")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/pregame_team_v2.parquet"),
        help="Path to pregame training parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models_v3"),
        help="Output directory for models",
    )
    parser.add_argument(
        "--include-xgb",
        action="store_true",
        help="Include XGBoost models (backtest-only)",
    )
    parser.add_argument(
        "--include-cat",
        action="store_true",
        help="Include CatBoost models (backtest-only)",
    )
    args = parser.parse_args()
    
    train_pregame_models(
        args.parquet,
        args.out_dir,
        include_xgb=args.include_xgb,
        include_cat=args.include_cat,
    )


if __name__ == "__main__":
    main()
