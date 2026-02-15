#!/usr/bin/env python3
"""
Production Model Backtest - Complete Pipeline

This script:
1. Uses the latest date in the dataset as the test date
2. Trains a production model on all data before that date
3. Generates halftime predictions for all games on that date
4. Reports comprehensive metrics
"""

from __future__ import annotations

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modeling.cat_models import CatBoostTwoHeadModel
from src.modeling.feature_columns import feature_columns


# Configuration
DATA_PATH = Path("data/processed/halftime_with_temporal_features_total.parquet")
METRICS_PATH = Path("reports/champion_runs/latest/halftime_fold_metrics.csv")
OUTPUT_DIR = Path("reports/backtest")
TARGET_FOLD = 51  # Use latest production fold


def load_data() -> pd.DataFrame:
    """Load the halftime dataset."""
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)
    df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    return df


def get_test_date(df: pd.DataFrame) -> datetime:
    """Get the test date (latest date in dataset)."""
    latest_date = df['game_date'].max()
    print(f"\nTest date: {latest_date.date()}")
    return latest_date


def split_train_test(df: pd.DataFrame, test_date: datetime) -> tuple:
    """Split data into train (before test date) and test (on test date)."""
    
    train_df = df[df['game_date'] < test_date].copy()
    test_df = df[df['game_date'] == test_date].copy()
    
    print(f"\nTrain set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games on {test_date.date()}")
    
    return train_df, test_df


def load_production_params() -> Dict:
    """Load production CatBoost hyperparameters from fold 51."""
    
    metrics_df = pd.read_csv(METRICS_PATH)
    
    fold_metrics = metrics_df[
        (metrics_df["fold"] == TARGET_FOLD) & 
        (metrics_df["model"] == "catboost")
    ]
    
    if len(fold_metrics) == 0:
        raise ValueError(f"No CatBoost metrics found for fold {TARGET_FOLD}")
    
    params_str = fold_metrics.iloc[0]["params"]
    params = json.loads(params_str)
    
    print(f"\nProduction parameters (fold {TARGET_FOLD}):")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return params


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix and targets."""
    
    # Get feature columns
    feat_cols = feature_columns(df)
    
    # Filter to numeric columns only
    numeric_feats = []
    for col in feat_cols:
        if col in df.columns:
            if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
                numeric_feats.append(col)
            elif df[col].dtype == 'bool':
                numeric_feats.append(col)
    
    print(f"\nFeatures: {len(numeric_feats)}")
    
    X = df[numeric_feats].values
    X = np.nan_to_num(X, nan=0.0)
    
    y_total = df['h2_total'].values
    y_margin = df['h2_margin'].values
    
    return X, y_total, y_margin, numeric_feats


def train_production_model(
    X_train: np.ndarray,
    y_total_train: np.ndarray,
    y_margin_train: np.ndarray,
    params: Dict,
    feature_names: List[str],
) -> CatBoostTwoHeadModel:
    """Train production model with exact hyperparameters."""
    
    print("\nTraining production model...")
    
    model = CatBoostTwoHeadModel(feature_version="v1", **params)
    model.fit(X_train, feature_names, y_total_train, y_margin_train)
    
    print("✅ Model trained successfully")
    
    return model


def generate_predictions(
    model: CatBoostTwoHeadModel,
    X_test: np.ndarray,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Generate predictions for test set."""
    
    print("\nGenerating predictions...")
    
    mu_total, mu_margin = model.predict_heads(X_test)
    
    # Get win probability
    trained_heads = model.trained_heads()
    sig_margin = trained_heads.margin.residual_sigma
    
    # 🦖 REPTAR: Compute win probability using CORRECT formula
    # P(home wins) = P(H1_margin + H2_margin > 0)
    #              = P(H2_margin > -H1_margin)
    #              = 1 - norm.cdf(-H1_margin, loc=H2_margin, scale=sigma)
    h1_margin = X_test['h1_margin'].values
    p_win = 1 - norm.cdf(-h1_margin, loc=mu_margin, scale=sig_margin)
    
    print(f"  Predictions generated for {len(mu_total)} games")
    
    return {
        "pred_total": mu_total,
        "pred_margin": mu_margin,
        "pred_win_prob": p_win,
    }


def compute_metrics(
    y_true_total: np.ndarray,
    y_pred_total: np.ndarray,
    y_true_margin: np.ndarray,
    y_pred_margin: np.ndarray,
    y_true_win: np.ndarray,
    y_pred_win_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    
    metrics = {}
    
    # Total metrics
    metrics["mae_total"] = mean_absolute_error(y_true_total, y_pred_total)
    metrics["rmse_total"] = np.sqrt(mean_squared_error(y_true_total, y_pred_total))
    
    # Margin metrics
    metrics["mae_margin"] = mean_absolute_error(y_true_margin, y_pred_margin)
    metrics["rmse_margin"] = np.sqrt(mean_squared_error(y_true_margin, y_pred_margin))
    
    # Winner metrics
    pred_winner = (y_pred_margin > 0).astype(int)
    actual_winner = (y_true_margin > 0).astype(int)
    metrics["win_accuracy"] = np.mean(pred_winner == actual_winner)
    
    # Brier score
    metrics["brier_score"] = brier_score_loss(y_true_win, y_pred_win_prob)
    
    return metrics


def interpret_results(metrics: Dict[str, float], n_games: int) -> str:
    """Interpret backtest results."""
    
    interpretation = []
    interpretation.append("\n" + "="*60)
    interpretation.append("PERFORMANCE INTERPRETATION")
    interpretation.append("="*60)
    
    # Totals interpretation
    interpretation.append("\nTOTAL POINTS:")
    if metrics["mae_total"] <= 8.0:
        interpretation.append(f"  ✅ STRONG - MAE {metrics['mae_total']:.2f} ≤ 8.0")
        totals_rating = "Strong"
    elif metrics["mae_total"] <= 10.0:
        interpretation.append(f"  ⚠️  ACCEPTABLE - MAE {metrics['mae_total']:.2f} in [8, 10]")
        totals_rating = "Acceptable"
    else:
        interpretation.append(f"  ❌ NEEDS INVESTIGATION - MAE {metrics['mae_total']:.2f} > 10.0")
        totals_rating = "Needs Investigation"
    
    # Winner interpretation
    interpretation.append("\nWINNER PREDICTION:")
    if metrics["win_accuracy"] >= 0.60:
        interpretation.append(f"  ✅ STRONG - Accuracy {metrics['win_accuracy']*100:.1f}% ≥ 60%")
        winner_rating = "Strong"
    elif metrics["win_accuracy"] >= 0.55:
        interpretation.append(f"  ⚠️  ACCEPTABLE - Accuracy {metrics['win_accuracy']*100:.1f}% in [55%, 60%)")
        winner_rating = "Acceptable"
    else:
        interpretation.append(f"  ❌ NEEDS INVESTIGATION - Accuracy {metrics['win_accuracy']*100:.1f}% < 55%")
        winner_rating = "Needs Investigation"
    
    # Overall assessment
    interpretation.append("\n" + "="*60)
    interpretation.append("OVERALL ASSESSMENT")
    interpretation.append("="*60)
    
    if totals_rating == "Strong" and winner_rating == "Strong":
        interpretation.append("\n✅ PERFORMANCE MATCHES EXPECTATIONS")
        interpretation.append("Model is performing well on out-of-sample data.")
    elif totals_rating == "Needs Investigation" or winner_rating == "Needs Investigation":
        interpretation.append("\n❌ PERFORMANCE BELOW EXPECTATIONS")
        interpretation.append("Model may need retraining or feature updates.")
    else:
        interpretation.append("\n⚠️  PERFORMANCE ACCEPTABLE")
        interpretation.append("Model is performing adequately but could be improved.")
    
    return "\n".join(interpretation)


def main():
    """Main entry point."""
    
    print("="*60)
    print("PRODUCTION MODEL BACKTEST")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Get test date (latest date in dataset)
    test_date = get_test_date(df)
    
    # Step 3: Split train/test
    train_df, test_df = split_train_test(df, test_date)
    
    if len(test_df) == 0:
        print(f"\n❌ No games found on test date {test_date.date()}")
        return
    
    # Step 4: Load production parameters
    params = load_production_params()
    
    # Step 5: Prepare features
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)
    
    X_train, y_total_train, y_margin_train, feature_names = prepare_features(train_df)
    X_test, y_total_test, y_margin_test, _ = prepare_features(test_df)
    
    # Step 6: Train model
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model = train_production_model(
        X_train, y_total_train, y_margin_train, params, feature_names
    )
    
    # Step 7: Generate predictions
    print("\n" + "="*60)
    print("PREDICTION GENERATION")
    print("="*60)
    
    predictions = generate_predictions(model, X_test, feature_names)
    
    # Step 8: Compute metrics
    y_win_test = (y_margin_test > 0).astype(float)
    
    metrics = compute_metrics(
        y_total_test,
        predictions["pred_total"],
        y_margin_test,
        predictions["pred_margin"],
        y_win_test,
        predictions["pred_win_prob"],
    )
    
    # Step 9: Create results table
    results_df = pd.DataFrame({
        "game_id": test_df["game_id"].values if "game_id" in test_df.columns else range(len(test_df)),
        "home_team_id": test_df["home_team_id"].values,
        "away_team_id": test_df["away_team_id"].values,
        "pred_total": predictions["pred_total"],
        "actual_total": y_total_test,
        "total_error": predictions["pred_total"] - y_total_test,
        "pred_margin": predictions["pred_margin"],
        "actual_margin": y_margin_test,
        "margin_error": predictions["pred_margin"] - y_margin_test,
        "pred_win_prob": predictions["pred_win_prob"],
        "pred_winner": (predictions["pred_margin"] > 0).astype(int),
        "actual_winner": (y_margin_test > 0).astype(int),
        "correct_winner": ((predictions["pred_margin"] > 0) == (y_margin_test > 0)).astype(int),
    })
    
    # Step 10: Print results
    print("\n" + "="*60)
    print("PER-GAME RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Step 11: Print metrics
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    print(f"\nNumber of Games: {len(test_df)}")
    print(f"\nTOTAL POINTS:")
    print(f"  MAE: {metrics['mae_total']:.2f}")
    print(f"  RMSE: {metrics['rmse_total']:.2f}")
    print(f"\nMARGIN:")
    print(f"  MAE: {metrics['mae_margin']:.2f}")
    print(f"  RMSE: {metrics['rmse_margin']:.2f}")
    print(f"\nWINNER:")
    print(f"  Accuracy: {metrics['win_accuracy']*100:.1f}%")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    # Step 12: Interpret results
    interpretation = interpret_results(metrics, len(test_df))
    print(interpretation)
    
    # Step 13: Save results
    results_path = OUTPUT_DIR / f"backtest_{test_date.date()}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to {results_path}")
    
    metrics_path = OUTPUT_DIR / f"metrics_{test_date.date()}.json"
    with open(metrics_path, 'w') as f:
        json.dump({**metrics, "n_games": len(test_df), "test_date": str(test_date.date())}, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
