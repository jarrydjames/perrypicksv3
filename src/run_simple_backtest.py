"""
Simple backtest to calculate temporal feature improvement percentage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

def load_data(path: str) -> pd.DataFrame:
    """Load and prepare data."""
    df = pd.read_parquet(path)
    
    # Ensure we have required columns
    required = ['h2_total', 'h2_margin']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    # Drop rows with NaN in targets
    df = df.dropna(subset=required)
    
    # Sort by date for time-based split
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)
    
    return df

def get_features(df: pd.DataFrame) -> list:
    """Get feature columns (exclude targets and metadata)."""
    exclude = {
        'game_id', 'season_end_yy', 'game_date',
        'h1_home', 'h1_away', 'h1_total', 'h1_margin',
        'h1_events', 'h1_n_2pt', 'h1_n_3pt', 'h1_n_turnover',
        'h1_n_rebound', 'h1_n_foul', 'h1_n_timeout', 'h1_n_sub',
        'home_efg', 'away_efg', 'h2_total', 'h2_margin',
        'market_total_line', 'market_home_spread_line',
        'market_home_team_total_line', 'market_away_team_total_line'
    }
    
    features = [col for col in df.columns if col not in exclude]
    return features

def run_backtest(df: pd.DataFrame, test_size: float = 0.2, model_type: str = 'gbt') -> dict:
    """Run backtest and return metrics."""
    
    features = get_features(df)
    X = df[features]
    y_total = df['h2_total']
    y_margin = df['h2_margin']
    
    # Time-based split (use last 20% for test)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_total, y_test_total = y_total[:split_idx], y_total[split_idx:]
    y_train_margin, y_test_margin = y_margin[:split_idx], y_margin[split_idx:]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Train total model
    model_total = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model_total.fit(X_train_imp, y_train_total)
    
    # Train margin model
    model_margin = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model_margin.fit(X_train_imp, y_train_margin)
    
    # Predict
    pred_total = model_total.predict(X_test_imp)
    pred_margin = model_margin.predict(X_test_imp)
    
    # Calculate metrics
    metrics = {
        'total_mae': mean_absolute_error(y_test_total, pred_total),
        'total_rmse': np.sqrt(mean_squared_error(y_test_total, pred_total)),
        'margin_mae': mean_absolute_error(y_test_margin, pred_margin),
        'margin_rmse': np.sqrt(mean_squared_error(y_test_margin, pred_margin))
    }
    
    return metrics

def main():
    print("=" * 70)
    print("SIMPLE BACKTEST: TEMPORAL FEATURES")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_data('data/processed/halftime_with_temporal_features_total.parquet')
    print(f"  Games: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    
    # Get temporal feature count
    temporal_cols = [col for col in df.columns if 'avg_5' in col or 'streak_5' in col or 'days_since' in col]
    print(f"  Temporal features: {len(temporal_cols)}")
    
    # Run backtest
    print("\nRunning backtest...")
    metrics = run_backtest(df, test_size=0.2)
    
    print("\n" + "=" * 70)
    print("TEMPORAL FEATURES: BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\nTest set: {int(len(df) * 0.2)} games (last 20%)")
    print(f"\nMetrics:")
    print(f"  Total MAE:  {metrics['total_mae']:.4f}")
    print(f"  Margin MAE: {metrics['margin_mae']:.4f}")
    print(f"  Total RMSE:  {metrics['total_rmse']:.4f}")
    print(f"  Margin RMSE: {metrics['margin_rmse']:.4f}")
    
    # Load baseline for comparison
    print("\n" + "=" * 70)
    print("BASELINE: NO TEMPORAL FEATURES")
    print("=" * 70)
    
    baseline = pd.read_parquet('data/processed/halftime_backtest_results_leakage_free.parquet')
    baseline_metrics = baseline.iloc[-1]
    
    print(f"\nBaseline Metrics (from backtest):")
    print(f"  Total MAE:  {baseline_metrics['total_mae']:.4f}")
    print(f"  Margin MAE: {baseline_metrics['margin_mae']:.4f}")
    print(f"  Total RMSE:  {baseline_metrics['total_rmse']:.4f}")
    print(f"  Margin RMSE: {baseline_metrics['margin_rmse']:.4f}")
    
    # Calculate improvement
    print("\n" + "=" * 70)
    print("IMPROVEMENT CALCULATION")
    print("=" * 70)
    
    def calc_improvement(baseline, temporal, metric_name):
        if baseline > 0:
            improvement = (baseline - temporal) / baseline * 100
            status = "✅ IMPROVEMENT" if improvement > 0 else "❌ WORSENED"
            return improvement, status
        else:
            return 0.0, "N/A"
    
    print(f"\n{'Metric':30} | Baseline | Temporal | Change | % Improv")
    print("-" * 80)
    
    metrics_to_compare = [
        ("Total MAE", 'total_mae'),
        ("Margin MAE", 'margin_mae'),
        ("Total RMSE", 'total_rmse'),
        ("Margin RMSE", 'margin_rmse')
    ]
    
    for name, key in metrics_to_compare:
        baseline_val = float(baseline_metrics[key])
        temporal_val = float(metrics[key])
        change = baseline_val - temporal_val
        improvement, status = calc_improvement(baseline_val, temporal_val, name)
        
        print(f"{name:30} | {baseline_val:8.4f} | {temporal_val:8.4f} | {change:+7.4f} | {improvement:+6.2f}%")
    
    print("\n" + "=" * 70)
    
    # Save results
    results_df = pd.DataFrame([{
        'baseline_total_mae': baseline_metrics['total_mae'],
        'baseline_margin_mae': baseline_metrics['margin_mae'],
        'baseline_total_rmse': baseline_metrics['total_rmse'],
        'baseline_margin_rmse': baseline_metrics['margin_rmse'],
        'temporal_total_mae': metrics['total_mae'],
        'temporal_margin_mae': metrics['margin_mae'],
        'temporal_total_rmse': metrics['total_rmse'],
        'temporal_margin_rmse': metrics['margin_rmse']
    }])
    
    results_df.to_csv('reports/temporal_backtest_results.csv', index=False)
    print(f"\nResults saved to: reports/temporal_backtest_results.csv")
    print("=" * 70)

if __name__ == '__main__':
    main()
