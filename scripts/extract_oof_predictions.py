#!/usr/bin/env python3
"""
Extract Out-of-Fold Predictions from Production Run

This script extracts OOF predictions from the 51-fold production run
without re-running the entire backtest. It:
1. Loads fold metrics to get exact hyperparameters
2. Re-trains models with those hyperparameters (fast, no tuning)
3. Generates predictions on test sets
4. Saves all OOF predictions to parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.modeling.backtest_utils import FoldSpec, iter_walkforward_indices
from src.modeling.feature_columns import feature_columns
from src.modeling.xgb_models import XGBoostTwoHeadModel
from src.modeling.cat_models import CatBoostTwoHeadModel


# Configuration
DATA_PATH = Path("data/processed/halftime_with_temporal_features_total.parquet")
METRICS_PATH = Path("reports/champion_runs/latest/halftime_fold_metrics.csv")
OUTPUT_PATH = Path("data/processed/halftime_oof_predictions.parquet")
TARGET_TOTAL = "h2_total"
TARGET_MARGIN = "h2_margin"
FEATURE_VERSION = "v1"


def load_fold_metrics() -> pd.DataFrame:
    """Load the production run fold metrics."""
    return pd.read_csv(METRICS_PATH)


def load_data() -> pd.DataFrame:
    """Load the halftime dataset."""
    return pd.read_parquet(DATA_PATH)


def get_fold_indices(n: int) -> List[tuple]:
    """Get train/test indices for each fold (same as production run)."""
    # Match the production run configuration
    spec = FoldSpec(train_min=800, test_size=200, step_size=200)
    return list(iter_walkforward_indices(n, spec=spec))


def parse_model_params(params_str: str) -> Dict[str, Any]:
    """Parse JSON params string."""
    return json.loads(params_str)


def create_model(model_type: str, params: Dict[str, Any]) -> Any:
    """Create a model instance with given parameters."""
    if model_type == "xgboost":
        # Remove any keys not accepted by XGBoostTwoHeadModel
        valid_keys = {
            "n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "min_child_weight", "reg_lambda", "n_jobs"
        }
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        filtered_params.setdefault("n_jobs", -1)
        return XGBoostTwoHeadModel(feature_version=FEATURE_VERSION, **filtered_params)
    
    elif model_type == "catboost":
        # Remove any keys not accepted by CatBoostTwoHeadModel
        valid_keys = {
            "iterations", "learning_rate", "depth", "l2_leaf_reg",
            "subsample", "random_seed"
        }
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        filtered_params.setdefault("random_seed", 42)
        return CatBoostTwoHeadModel(feature_version=FEATURE_VERSION, **filtered_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def extract_oof_predictions() -> pd.DataFrame:
    """Extract out-of-fold predictions for all folds and models."""
    
    print("Loading data...")
    df = load_data()
    metrics_df = load_fold_metrics()
    
    # Prepare features
    feat_cols = feature_columns(df)
    available_feats = [c for c in feat_cols if c in df.columns]
    
    # Filter to only numeric columns (exclude datetime, object, etc.)
    numeric_feats = []
    for col in available_feats:
        # Check if column is datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        # Check if column is numeric
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
            numeric_feats.append(col)
        elif df[col].dtype == 'bool':
            numeric_feats.append(col)
        else:
            # Try to convert to numeric, skip if fails
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():
                    numeric_feats.append(col)
            except:
                pass
    
    available_feats = numeric_feats
    print(f"Numeric features: {len(available_feats)} / {len(feat_cols)}")
    print(f"Excluded non-numeric: {set(feat_cols) - set(available_feats)}")
    
    X = df[available_feats].values
    
    # Replace NaN with 0 for training
    X = np.nan_to_num(X, nan=0.0)
    y_total = df[TARGET_TOTAL].values
    y_margin = df[TARGET_MARGIN].values
    
    # Get game IDs if available
    if "game_id" in df.columns:
        game_ids = df["game_id"].values
    elif "game_date" in df.columns and "home_team_id" in df.columns:
        # Create composite game_id
        game_ids = df["game_date"].astype(str) + "_" + df["home_team_id"].astype(str)
    else:
        # Use row index as game_id
        game_ids = np.arange(len(df))
    
    # Get fold indices
    n = len(df)
    fold_indices = get_fold_indices(n)
    
    print(f"\nDataset: {n} samples")
    print(f"Features: {len(available_feats)}")
    print(f"Folds: {len(fold_indices)}")
    
    # Collect all predictions
    all_predictions = []
    
    for fold_id, (train_idx, test_idx) in enumerate(fold_indices, start=1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_id}/{len(fold_indices)}")
        print(f"{'='*60}")
        print(f"Train: {len(train_idx)} samples")
        print(f"Test: {len(test_idx)} samples")
        
        # Get test set data
        X_test = X[test_idx]
        y_total_test = y_total[test_idx]
        y_margin_test = y_margin[test_idx]
        game_ids_test = game_ids[test_idx]
        
        # Extract true win label
        y_win_test = (y_margin_test > 0).astype(float)
        
        # Process each model
        for model_type in ["catboost", "xgboost"]:
            print(f"\n  Processing {model_type}...")
            
            # Get model parameters for this fold
            model_metrics = metrics_df[
                (metrics_df["fold"] == fold_id) & 
                (metrics_df["model"] == model_type)
            ]
            
            if len(model_metrics) == 0:
                print(f"    WARNING: No metrics found for {model_type} fold {fold_id}")
                continue
            
            model_metrics = model_metrics.iloc[0]
            params = parse_model_params(model_metrics["params"])
            
            print(f"    Parameters: {params}")
            
            # Create and train model
            model = create_model(model_type, params)
            
            # Get training data
            X_train = X[train_idx]
            y_total_train = y_total[train_idx]
            y_margin_train = y_margin[train_idx]
            
            # Train model
            model.fit(X_train, available_feats, y_total_train, y_margin_train)
            
            # Generate predictions
            mu_total, mu_margin = model.predict_heads(X_test)
            
            # Get win probability
            trained_heads = model.trained_heads()
            sig_margin = trained_heads.margin.residual_sigma
            
            # 🦖 REPTAR: Compute win probability using CORRECT formula
            # P(home wins) = P(H1_margin + H2_margin > 0)
            #              = P(H2_margin > -H1_margin)
            #              = 1 - CDF(-H1_margin | mu_H2, sigma)
            h1_margin = X_test['h1_margin'].values
            from scipy.stats import norm
            p_win = 1 - norm.cdf(-h1_margin, loc=mu_margin, scale=sig_margin)
            
            # Store predictions
            for i in range(len(test_idx)):
                all_predictions.append({
                    "game_id": game_ids_test[i],
                    "fold_id": fold_id,
                    "y_total_true": y_total_test[i],
                    "y_margin_true": y_margin_test[i],
                    "y_win_true": y_win_test[i],
                    "model": model_type,
                    "total_pred": mu_total[i],
                    "margin_pred": mu_margin[i],
                    "win_prob": p_win[i],
                })
            
            print(f"    Generated {len(test_idx)} predictions")
    
    # Create DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Folds: {predictions_df['fold_id'].nunique()}")
    print(f"Models: {predictions_df['model'].unique().tolist()}")
    
    return predictions_df


def main():
    """Main entry point."""
    print("="*60)
    print("EXTRACTING OUT-OF-FOLD PREDICTIONS")
    print("="*60)
    
    # Extract predictions
    predictions_df = extract_oof_predictions()
    
    # Save to parquet
    print(f"\nSaving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"✅ Saved {len(predictions_df)} predictions to {OUTPUT_PATH}")
    
    # Print sample
    print("\nSample predictions:")
    print(predictions_df.head(10))


if __name__ == "__main__":
    main()
