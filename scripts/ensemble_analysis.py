#!/usr/bin/env python3
"""
Ensemble Analysis: CatBoost vs XGBoost

This script evaluates whether an ensemble of CatBoost and XGBoost provides
statistically meaningful improvements over individual models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Configuration
OOF_PATH = Path("data/processed/halftime_oof_predictions.parquet")
OUTPUT_DIR = Path("reports/ensemble_analysis")

# Ensemble weights to test
WEIGHT_CONFIGS = [
    ("catboost_100", 1.0, 0.0),
    ("catboost_75", 0.75, 0.25),
    ("catboost_60", 0.60, 0.40),
    ("balanced_50", 0.50, 0.50),
    ("xgboost_60", 0.40, 0.60),
    ("xgboost_75", 0.25, 0.75),
    ("xgboost_100", 0.0, 1.0),
]


def load_oof_predictions() -> pd.DataFrame:
    """Load out-of-fold predictions."""
    return pd.read_parquet(OOF_PATH)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_margin: np.ndarray = None,
    y_pred_margin: np.ndarray = None,
    y_true_win: np.ndarray = None,
    y_pred_win_prob: np.ndarray = None,
) -> Dict[str, float]:
    """Compute comprehensive metrics for predictions."""
    
    metrics = {}
    
    # Total metrics
    metrics["mae_total"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse_total"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["r2_total"] = r2_score(y_true, y_pred)
    
    # Margin metrics (if provided)
    if y_true_margin is not None and y_pred_margin is not None:
        metrics["mae_margin"] = mean_absolute_error(y_true_margin, y_pred_margin)
        metrics["rmse_margin"] = np.sqrt(mean_squared_error(y_true_margin, y_pred_margin))
    
    # Win probability metrics (if provided)
    if y_true_win is not None and y_pred_win_prob is not None:
        # Brier score
        metrics["brier_win"] = brier_score_loss(y_true_win, y_pred_win_prob)
        
        # Log loss
        try:
            # Clip probabilities to avoid log(0)
            y_pred_win_prob_clipped = np.clip(y_pred_win_prob, 1e-10, 1 - 1e-10)
            metrics["log_loss"] = log_loss(y_true_win, y_pred_win_prob_clipped)
        except:
            metrics["log_loss"] = np.nan
        
        # Calibration (ECE - Expected Calibration Error)
        metrics["ece"] = compute_ece(y_true_win, y_pred_win_prob)
    
    return metrics


def compute_ece(y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def compute_stability_metrics(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """Compute stability metrics across folds."""
    
    # Group by fold and compute MAE per fold
    fold_mae = predictions_df.groupby("fold_id").apply(
        lambda x: mean_absolute_error(x["y_total_true"], x["total_pred"])
    )
    
    # Group by fold and compute Brier per fold
    fold_brier = predictions_df.groupby("fold_id").apply(
        lambda x: brier_score_loss(x["y_win_true"], x["win_prob"])
    )
    
    return {
        "mean_mae_total": fold_mae.mean(),
        "std_mae_total": fold_mae.std(),
        "mean_brier": fold_brier.mean(),
        "std_brier": fold_brier.std(),
    }


def create_ensemble_predictions(
    oof_df: pd.DataFrame,
    cat_weight: float,
    xgb_weight: float,
) -> pd.DataFrame:
    """Create ensemble predictions with given weights."""
    
    # Pivot to get CatBoost and XGBoost predictions side by side
    pivot_df = oof_df.pivot_table(
        index=["game_id", "fold_id", "y_total_true", "y_margin_true", "y_win_true"],
        columns="model",
        values=["total_pred", "margin_pred", "win_prob"],
        aggfunc="first"
    ).reset_index()
    
    # Flatten column names
    pivot_df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in pivot_df.columns]
    pivot_df = pivot_df.rename(columns={
        "game_id_": "game_id",
        "fold_id_": "fold_id",
        "y_total_true_": "y_total_true",
        "y_margin_true_": "y_margin_true",
        "y_win_true_": "y_win_true",
    })
    
    # Create ensemble predictions
    ensemble_df = pd.DataFrame({
        "game_id": pivot_df["game_id"],
        "fold_id": pivot_df["fold_id"],
        "y_total_true": pivot_df["y_total_true"],
        "y_margin_true": pivot_df["y_margin_true"],
        "y_win_true": pivot_df["y_win_true"],
        "total_pred": cat_weight * pivot_df["total_pred_catboost"] + xgb_weight * pivot_df["total_pred_xgboost"],
        "margin_pred": cat_weight * pivot_df["margin_pred_catboost"] + xgb_weight * pivot_df["margin_pred_xgboost"],
        "win_prob": cat_weight * pivot_df["win_prob_catboost"] + xgb_weight * pivot_df["win_prob_xgboost"],
    })
    
    return ensemble_df


def paired_t_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    name1: str,
    name2: str,
) -> Dict[str, float]:
    """Perform paired t-test between two error arrays."""
    
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    return {
        "comparison": f"{name1} vs {name2}",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "mean_diff": np.mean(errors1 - errors2),
        "std_diff": np.std(errors1 - errors2),
    }


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
) -> Dict[str, float]:
    """Perform Diebold-Mariano test for predictive accuracy."""
    
    # Compute loss differential
    d = errors1 ** 2 - errors2 ** 2
    
    # Mean of loss differential
    d_mean = np.mean(d)
    
    # Compute variance with Newey-West correction
    n = len(d)
    
    # Autocovariance up to lag h
    gamma = [np.cov(d[i:], d[:n-i])[0, 1] if i < n else 0 for i in range(h + 1)]
    
    # Variance estimate
    var_d = gamma[0] + 2 * sum(gamma[1:])
    
    # DM statistic
    if var_d > 0:
        dm_stat = d_mean / np.sqrt(var_d / n)
    else:
        dm_stat = 0.0
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def simulate_betting(
    predictions_df: pd.DataFrame,
    total_edge_threshold: float = 2.0,
    margin_edge_threshold: float = 1.5,
    win_edge_threshold: float = 0.03,
) -> Dict[str, float]:
    """Simulate betting performance."""
    
    results = {
        "totals_bets": 0,
        "totals_wins": 0,
        "totals_roi": 0.0,
        "spreads_bets": 0,
        "spreads_wins": 0,
        "spreads_roi": 0.0,
        "moneyline_bets": 0,
        "moneyline_wins": 0,
        "moneyline_roi": 0.0,
    }
    
    # For now, we'll use a simplified simulation
    # In practice, you'd need actual betting lines to compute edge
    
    # Totals: Bet when prediction differs significantly from average
    total_mean = predictions_df["y_total_true"].mean()
    total_edge = np.abs(predictions_df["total_pred"] - total_mean)
    total_bets_mask = total_edge >= total_edge_threshold
    
    if total_bets_mask.sum() > 0:
        # Assume -110 odds (win 100 on 110 bet)
        total_correct = (
            ((predictions_df["total_pred"] > total_mean) & (predictions_df["y_total_true"] > total_mean)) |
            ((predictions_df["total_pred"] < total_mean) & (predictions_df["y_total_true"] < total_mean))
        )
        
        total_bets = total_bets_mask.sum()
        total_wins = (total_bets_mask & total_correct).sum()
        
        # ROI calculation: wins * 100 - losses * 110
        total_profit = total_wins * 100 - (total_bets - total_wins) * 110
        total_investment = total_bets * 110
        
        results["totals_bets"] = int(total_bets)
        results["totals_wins"] = int(total_wins)
        results["totals_win_rate"] = total_wins / total_bets if total_bets > 0 else 0
        results["totals_roi"] = total_profit / total_investment if total_investment > 0 else 0
    
    # Spreads: Similar logic
    margin_mean = predictions_df["y_margin_true"].mean()
    margin_edge = np.abs(predictions_df["margin_pred"] - margin_mean)
    margin_bets_mask = margin_edge >= margin_edge_threshold
    
    if margin_bets_mask.sum() > 0:
        margin_correct = (
            ((predictions_df["margin_pred"] > margin_mean) & (predictions_df["y_margin_true"] > margin_mean)) |
            ((predictions_df["margin_pred"] < margin_mean) & (predictions_df["y_margin_true"] < margin_mean))
        )
        
        margin_bets = margin_bets_mask.sum()
        margin_wins = (margin_bets_mask & margin_correct).sum()
        
        margin_profit = margin_wins * 100 - (margin_bets - margin_wins) * 110
        margin_investment = margin_bets * 110
        
        results["spreads_bets"] = int(margin_bets)
        results["spreads_wins"] = int(margin_wins)
        results["spreads_win_rate"] = margin_wins / margin_bets if margin_bets > 0 else 0
        results["spreads_roi"] = margin_profit / margin_investment if margin_investment > 0 else 0
    
    # Moneyline: Use win probability
    win_prob_mean = predictions_df["y_win_true"].mean()
    win_edge = np.abs(predictions_df["win_prob"] - win_prob_mean)
    win_bets_mask = win_edge >= win_edge_threshold
    
    if win_bets_mask.sum() > 0:
        win_bets = win_bets_mask.sum()
        win_wins = ((predictions_df["win_prob"] > win_prob_mean) & (predictions_df["y_win_true"] == 1) |
                    (predictions_df["win_prob"] < win_prob_mean) & (predictions_df["y_win_true"] == 0))
        win_wins_count = (win_bets_mask & win_wins).sum()
        
        win_profit = win_wins_count * 100 - (win_bets - win_wins_count) * 110
        win_investment = win_bets * 110
        
        results["moneyline_bets"] = int(win_bets)
        results["moneyline_wins"] = int(win_wins_count)
        results["moneyline_win_rate"] = win_wins_count / win_bets if win_bets > 0 else 0
        results["moneyline_roi"] = win_profit / win_investment if win_investment > 0 else 0
    
    return results


def evaluate_all_ensembles() -> pd.DataFrame:
    """Evaluate all ensemble configurations."""
    
    print("Loading OOF predictions...")
    oof_df = load_oof_predictions()
    
    print(f"\nTotal predictions: {len(oof_df)}")
    print(f"Folds: {oof_df['fold_id'].nunique()}")
    print(f"Models: {oof_df['model'].unique().tolist()}")
    
    results = []
    
    for name, cat_weight, xgb_weight in WEIGHT_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} (CatBoost={cat_weight:.2f}, XGBoost={xgb_weight:.2f})")
        print(f"{'='*60}")
        
        # Create ensemble predictions
        ensemble_df = create_ensemble_predictions(oof_df, cat_weight, xgb_weight)
        
        # Compute metrics
        metrics = compute_metrics(
            ensemble_df["y_total_true"].values,
            ensemble_df["total_pred"].values,
            ensemble_df["y_margin_true"].values,
            ensemble_df["margin_pred"].values,
            ensemble_df["y_win_true"].values,
            ensemble_df["win_prob"].values,
        )
        
        # Compute stability metrics
        stability = compute_stability_metrics(ensemble_df)
        
        # Simulate betting
        betting = simulate_betting(ensemble_df)
        
        # Combine all results
        result = {
            "name": name,
            "catboost_weight": cat_weight,
            "xgboost_weight": xgb_weight,
            **metrics,
            **stability,
            **betting,
        }
        
        results.append(result)
        
        print(f"  MAE Total: {metrics['mae_total']:.4f}")
        print(f"  RMSE Total: {metrics['rmse_total']:.4f}")
        print(f"  R² Total: {metrics['r2_total']:.4f}")
        print(f"  MAE Margin: {metrics['mae_margin']:.4f}")
        print(f"  RMSE Margin: {metrics['rmse_margin']:.4f}")
        print(f"  Brier: {metrics['brier_win']:.6f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Stability (Std MAE): {stability['std_mae_total']:.4f}")
        print(f"  Stability (Std Brier): {stability['std_brier']:.6f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df


def perform_statistical_tests(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests between best ensemble and individual models."""
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # Get CatBoost and XGBoost predictions separately
    cat_df = oof_df[oof_df["model"] == "catboost"].copy()
    xgb_df = oof_df[oof_df["model"] == "xgboost"].copy()
    
    # Sort by game_id and fold_id to ensure alignment
    cat_df = cat_df.sort_values(["game_id", "fold_id"])
    xgb_df = xgb_df.sort_values(["game_id", "fold_id"])
    
    # Compute absolute errors
    cat_errors_total = np.abs(cat_df["y_total_true"].values - cat_df["total_pred"].values)
    xgb_errors_total = np.abs(xgb_df["y_total_true"].values - xgb_df["total_pred"].values)
    
    # Test CatBoost vs XGBoost
    print("\n1. CatBoost vs XGBoost (Total Points)")
    t_test_result = paired_t_test(cat_errors_total, xgb_errors_total, "CatBoost", "XGBoost")
    print(f"   t-statistic: {t_test_result['t_statistic']:.4f}")
    print(f"   p-value: {t_test_result['p_value']:.6f}")
    print(f"   Significant: {t_test_result['significant']}")
    print(f"   Mean diff: {t_test_result['mean_diff']:.4f}")
    
    # Diebold-Mariano test
    dm_result = diebold_mariano_test(cat_errors_total, xgb_errors_total, h=1)
    print(f"\n   Diebold-Mariano test:")
    print(f"   DM statistic: {dm_result['dm_statistic']:.4f}")
    print(f"   p-value: {dm_result['p_value']:.6f}")
    print(f"   Significant: {dm_result['significant']}")
    
    # Test best ensemble vs CatBoost
    print("\n2. Best Ensemble (50/50) vs CatBoost")
    ensemble_50_df = create_ensemble_predictions(oof_df, 0.5, 0.5)
    ensemble_50_df = ensemble_50_df.sort_values(["game_id", "fold_id"])
    ensemble_errors_total = np.abs(ensemble_50_df["y_total_true"].values - ensemble_50_df["total_pred"].values)
    
    # The ensemble has one row per game (combined CatBoost+XGBoost)
    # CatBoost has duplicate rows (one per model type)
    # So we need to use only unique game-level catboost errors
    cat_df_unique = cat_df.drop_duplicates(subset=["game_id", "fold_id"])
    cat_df_unique = cat_df_unique.sort_values(["game_id", "fold_id"])
    cat_errors_total_unique = np.abs(cat_df_unique["y_total_true"].values - cat_df_unique["total_pred"].values)
    
    t_test_ensemble = paired_t_test(ensemble_errors_total, cat_errors_total_unique, "Ensemble_50", "CatBoost")
    print(f"   t-statistic: {t_test_ensemble['t_statistic']:.4f}")
    print(f"   p-value: {t_test_ensemble['p_value']:.6f}")
    print(f"   Significant: {t_test_ensemble['significant']}")
    
    dm_ensemble = diebold_mariano_test(ensemble_errors_total, cat_errors_total_unique, h=1)
    print(f"\n   Diebold-Mariano test:")
    print(f"   DM statistic: {dm_ensemble['dm_statistic']:.4f}")
    print(f"   p-value: {dm_ensemble['p_value']:.6f}")
    print(f"   Significant: {dm_ensemble['significant']}")
    
    # Create summary
    test_results = pd.DataFrame([
        {
            "test": "Paired t-test (CatBoost vs XGBoost)",
            **t_test_result,
        },
        {
            "test": "Diebold-Mariano (CatBoost vs XGBoost)",
            **dm_result,
            "comparison": "CatBoost vs XGBoost",
        },
        {
            "test": "Paired t-test (Ensemble_50 vs CatBoost)",
            **t_test_ensemble,
        },
        {
            "test": "Diebold-Mariano (Ensemble_50 vs CatBoost)",
            **dm_ensemble,
            "comparison": "Ensemble_50 vs CatBoost",
        },
    ])
    
    return test_results


def main():
    """Main entry point."""
    
    print("="*60)
    print("ENSEMBLE ANALYSIS: CATBOOST VS XGBOOST")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load OOF predictions
    oof_df = load_oof_predictions()
    
    # Evaluate all ensembles
    results_df = evaluate_all_ensembles()
    
    # Save results
    results_path = OUTPUT_DIR / "ensemble_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Saved ensemble comparison to {results_path}")
    
    # Perform statistical tests
    test_results = perform_statistical_tests(oof_df)
    
    # Save test results
    test_path = OUTPUT_DIR / "statistical_tests.csv"
    test_results.to_csv(test_path, index=False)
    print(f"✅ Saved statistical tests to {test_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(results_df[["name", "mae_total", "rmse_total", "brier_win", "std_mae_total"]].to_string(index=False))
    
    # Determine best model
    best_idx = results_df["mae_total"].idxmin()
    best_model = results_df.loc[best_idx]
    
    print(f"\n🏆 BEST MODEL: {best_model['name']}")
    print(f"   MAE Total: {best_model['mae_total']:.4f}")
    print(f"   RMSE Total: {best_model['rmse_total']:.4f}")
    print(f"   Brier: {best_model['brier_win']:.6f}")
    print(f"   Stability: {best_model['std_mae_total']:.4f}")
    
    # Check viability criteria
    catboost_row = results_df[results_df["name"] == "catboost_100"].iloc[0]
    best_mae_improvement = catboost_row["mae_total"] - best_model["mae_total"]
    brier_equal_or_better = best_model["brier_win"] <= catboost_row["brier_win"]
    stability_improved = best_model["std_mae_total"] < catboost_row["std_mae_total"]
    
    print(f"\nVIABILITY CRITERIA:")
    print(f"  1. MAE improvement ≥ 0.05: {best_mae_improvement:.4f} {'✅' if best_mae_improvement >= 0.05 else '❌'}")
    print(f"  2. Brier ≤ CatBoost: {brier_equal_or_better} {'✅' if brier_equal_or_better else '❌'}")
    print(f"  3. Stability improved: {stability_improved} {'✅' if stability_improved else '❌'}")
    
    if best_mae_improvement >= 0.05 and brier_equal_or_better and stability_improved:
        print(f"\n✅ ENSEMBLE IS VIABLE - Recommend {best_model['name']}")
    else:
        print(f"\n❌ ENSEMBLE NOT VIABLE - Recommend sticking with CatBoost")


if __name__ == "__main__":
    main()
