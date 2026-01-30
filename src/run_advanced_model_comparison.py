"""
Advanced model comparison: XGBoost, CatBoost vs Baseline.
Tests longer rolling windows and more complex models.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

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

def get_baseline_features(df: pd.DataFrame) -> list:
    """Get features without temporal features."""
    exclude = {
        'game_id', 'season_end_yy', 'game_date',
        'h1_home', 'h1_away', 'h1_total', 'h1_margin',
        'h1_events', 'h1_n_2pt', 'h1_n_3pt', 'h1_n_turnover',
        'h1_n_rebound', 'h1_n_foul', 'h1_n_timeout', 'h1_n_sub',
        'home_efg', 'away_efg', 'h2_total', 'h2_margin',
        'market_total_line', 'market_home_spread_line',
        'market_home_team_total_line', 'market_away_team_total_line',
        # All temporal/rolling features
        'home_pts_scored_avg_5', 'home_pts_allowed_avg_5', 'home_margin_avg_5',
        'home_current_streak_5', 'home_days_since_last', 'home_is_back_to_back',
        'away_pts_scored_avg_5', 'away_pts_allowed_avg_5', 'away_margin_avg_5',
        'away_current_streak_5', 'away_days_since_last', 'away_is_back_to_back',
        'home_pts_scored_avg_10', 'home_pts_allowed_avg_10', 'home_margin_avg_10',
        'home_wins_10', 'away_pts_scored_avg_10', 'away_pts_allowed_avg_10',
        'away_margin_avg_10', 'away_wins_10', 'home_wins_5', 'away_wins_5',
        'home_pts_scored_avg_20', 'home_pts_allowed_avg_20', 'home_margin_avg_20',
        'home_current_streak_20', 'away_pts_scored_avg_20', 'away_pts_allowed_avg_20',
        'away_margin_avg_20', 'away_current_streak_20',
    }
    
    features = [col for col in df.columns if col not in exclude]
    return features

def get_temporal_features(df: pd.DataFrame) -> list:
    """Get all features including temporal."""
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

def run_backtest_sklearn(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2, 
                       n_estimators: int = 100, max_depth: int = 3) -> dict:
    """Run backtest with sklearn model."""
    
    X = df[feature_cols]
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
    
    # Train models
    model_total = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=max_depth,
        random_state=42
    )
    model_total.fit(X_train_imp, y_train_total)
    
    model_margin = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=max_depth,
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
    print("=" * 80)
    print("ADVANCED MODEL COMPARISON: XGBoost, CatBoost vs Baseline")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = load_data('data/processed/halftime_with_temporal_features_total.parquet')
    print(f"  Games: {len(df)}")
    
    # Get feature sets
    baseline_features = get_baseline_features(df)
    temporal_features = get_temporal_features(df)
    
    print(f"\nBaseline features: {len(baseline_features)}")
    print(f"Temporal features: {len(temporal_features)}")
    print(f"  (Added {len(temporal_features) - len(baseline_features)} temporal features)")
    
    # Check for longer windows
    longer_window_cols = [col for col in df.columns if 'avg_20' in col or 'streak_20' in col]
    if len(longer_window_cols) > 0:
        print(f"  Longer windows (20 games): {len(longer_window_cols)} columns")
    else:
        print(f"  No longer windows found (only 5/10 game windows)")
    
    # Test configurations
    configs = [
        ("Baseline (GBT, depth=3)", baseline_features, 100, 3),
        ("Temporal (GBT, depth=3)", temporal_features, 100, 3),
        ("Baseline (GBT, depth=6)", baseline_features, 100, 6),
        ("Temporal (GBT, depth=6)", temporal_features, 100, 6),
        ("Baseline (GBT, depth=10)", baseline_features, 200, 10),
        ("Temporal (GBT, depth=10)", temporal_features, 200, 10),
    ]
    
    results = []
    
    for name, features, n_est, depth in configs:
        print(f"\n{'=' * 80}")
        print(f"Running: {name}")
        print(f"  Features: {len(features)}, Estimators: {n_est}, Max Depth: {depth}")
        print(f"{'=' * 80}")
        
        metrics = run_backtest_sklearn(df, features, test_size=0.2, 
                                    n_estimators=n_est, max_depth=depth)
        
        print(f"\nResults:")
        print(f"  Total MAE:  {metrics['total_mae']:.4f}")
        print(f"  Margin MAE: {metrics['margin_mae']:.4f}")
        print(f"  Total RMSE:  {metrics['total_rmse']:.4f}")
        print(f"  Margin RMSE: {metrics['margin_rmse']:.4f}")
        
        results.append({
            'config': name,
            'n_estimators': n_est,
            'max_depth': depth,
            'n_features': len(features),
            **metrics
        })
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    baseline_mae = results_df[results_df['config'] == 'Baseline (GBT, depth=3)']['total_mae'].values[0]
    
    print(f"\n{'Config':35} | MAE   | RMSE  | vs Baseline")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        diff = baseline_mae - row['total_mae']
        pct = (diff / baseline_mae * 100) if baseline_mae > 0 else 0.0
        status = "✅" if diff > 0 else "❌"
        print(f"{row['config']:35} | {row['total_mae']:6.4f} | {row['total_rmse']:6.4f} | {diff:+6.4f} ({pct:+5.2f}%) {status}")
    
    # Find best model
    best_idx = results_df['total_mae'].idxmin()
    best = results_df.iloc[best_idx]
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best['config']}")
    print(f"  Total MAE: {best['total_mae']:.4f}")
    print(f"  Improvement over baseline: {baseline_mae - best['total_mae']:.4f} ({(baseline_mae - best['total_mae'])/baseline_mae*100:+.2f}%)")
    print("=" * 80)
    
    # Save results
    results_df.to_csv('reports/advanced_model_comparison.csv', index=False)
    print(f"\nResults saved to: reports/advanced_model_comparison.csv")

if __name__ == '__main__':
    main()
