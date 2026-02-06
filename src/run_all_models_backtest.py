"""
Comprehensive backtest: ALL models (Ridge, RandomForest, GBT) with walkforward split.
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import pandas as pd
import numpy as np
from pathlib import Path

# Import model classes
from src.modeling.walkforward_backtest import run_backtest, default_models
from src.modeling.backtest_utils import FoldSpec

def main():
    print("=" * 80)
    print("COMPREHENSIVE BACKTEST: ALL MODELS (RIDGE, RF, GBT)")
    print("=" * 80)
    
    # Walkforward spec
    spec = FoldSpec(train_min=500, test_size=200, step_size=200)
    
    print(f"\nBacktest spec:")
    print(f"  Min train size: {spec.train_min}")
    print(f"  Test size: {spec.test_size}")
    print(f"  Step size: {spec.step_size}")
    
    # Run with all models
    print(f"\nRunning backtest with ALL models (Ridge, RF, GBT)...")
    
    results = run_backtest(
        parquet_path=Path('data/processed/halftime_with_temporal_features_total.parquet'),
        box_dir=Path('data/raw/box'),
        out_csv=Path('reports/comprehensive_backtest.csv'),
        spec=spec,
        drop_market_priors=True,  # Drop market priors to focus on basketball features
        run_roi=False,
        pi_method='normal',
        calibration=False,
        include_xgb=False,
        include_cat=False
    )
    
    # Get final metrics (last row - aggregated)
    final_metrics = results[-1]
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS (ALL MODELS)")
    print("=" * 80)
    
    # Find best model by total_mae
    best_idx = final_metrics['total_mae'].idxmin()
    best_model = final_metrics.loc[best_idx, 'model']
    best_mae = final_metrics.loc[best_idx, 'total_mae']
    
    print(f"\nBest Model: {best_model}")
    print(f"  Total MAE: {best_mae:.4f}")
    
    print(f"\nAll models ranked by Total MAE:")
    print(f"{'Model':20} | MAE   | RMSE  | Margin MAE | Margin RMSE")
    print("-" * 80)
    
    for _, row in final_metrics.iterrows():
        is_best = row['model'] == best_model
        marker = "✅ BEST" if is_best else ""
        print(f"{row['model']:20} | {row['total_mae']:6.4f} | {row['total_rmse']:6.4f} | {row['margin_mae']:10.4f} | {row['margin_rmse']:10.4f}  {marker}")
    
    # Save to CSV
    final_metrics.to_csv('reports/comprehensive_backtest_summary.csv', index=False)
    
    print("\n" + "=" * 80)
    print("Results saved to: reports/comprehensive_backtest.csv")
    print("Summary saved to: reports/comprehensive_backtest_summary.csv")
    print("=" * 80)

if __name__ == '__main__':
    main()
