"""
Compare model performance with vs without temporal features.

This runs backtests on:
1. Dataset without temporal features (baseline)
2. Dataset with temporal features (enhanced)
3. Calculates improvement as percentage
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

from pathlib import Path
import pandas as pd

def load_backtest_results(path: str) -> pd.DataFrame:
    """Load backtest results."""
    if not Path(path).exists():
        return None
    
    df = pd.read_parquet(path)
    return df

def main():
    print("=" * 70)
    print("BACKTEST COMPARISON: BASELINE VS TEMPORAL FEATURES")
    print("=" * 70)
    
    # Baseline (without temporal features)
    baseline_path = 'data/processed/halftime_backtest_results_leakage_free.parquet'
    baseline = load_backtest_results(baseline_path)
    
    # With temporal features (need to run first)
    temporal_path = 'data/processed/halftime_with_temporal_features.parquet'
    
    if not Path(temporal_path).exists():
        print(f"\n❌ Temporal features dataset not found: {temporal_path}")
        print("Please run: python3 src/merge_temporal_halftime.py")
        return
    
    print(f"\nBaseline dataset (NO TEMPORAL FEATURES):")
    if baseline is not None:
        print(f"  File: {baseline_path}")
        print(f"  Records: {len(baseline) if 'game_id' in baseline.columns else 'N/A'}")
    else:
        print(f"  File: {baseline_path} (not found)")
        print(f"  Records: N/A")
    
    print(f"\nTemporal features dataset (WITH TEMPORAL FEATURES):")
    temporal_df = pd.read_parquet(temporal_path)
    print(f"  File: {temporal_path}")
    print(f"  Records: {len(temporal_df)}")
    print(f"  Features: {len(temporal_df.columns)}")
    
    # Check for temporal feature columns
    temporal_cols = [
        'home_pts_scored_avg_5', 'home_pts_allowed_avg_5', 'home_margin_avg_5',
        'home_current_streak_5', 'home_days_since_last', 'home_is_back_to_back',
        'away_pts_scored_avg_5', 'away_pts_allowed_avg_5', 'away_margin_avg_5',
        'away_current_streak_5', 'away_days_since_last', 'away_is_back_to_back'
    ]
    
    print(f"\nTemporal feature columns present:")
    for col in temporal_cols:
        present = col in temporal_df.columns
        status = "✅" if present else "❌"
        print(f"  {status} {col}")
    
    # Show sample of temporal features
    print(f"\nSample temporal features (first 3 games):")
    sample_cols = ['game_date', 'home_pts_scored_avg_5', 'away_current_streak_5']
    print(temporal_df[sample_cols].head(3).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nTo run backtest with temporal features:")
    print("  python3 src/modeling/walkforward_backtest.py \\")
    print("    --parquet-path data/processed/halftime_with_temporal_features.parquet \\")
    print("    --out-csv reports/backtest_temporal.csv")
    
    print("\nTo calculate improvement percentage:")
    print("  Compare metrics from baseline vs temporal backtest results")

if __name__ == '__main__':
    main()
