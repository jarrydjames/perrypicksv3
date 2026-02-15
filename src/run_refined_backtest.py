"""
Backtest with refined temporal features - target: match 48hr CatBoost tuning accuracy.
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
        'final_total', 'final_margin',  # These are targets, not features!
        'market_total_line', 'market_home_spread_line',
        'market_home_team_total_line', 'market_away_team_total_line',
        'home_tri', 'away_tri', 'home_team_id', 'away_team_id'
    }
    
    features = [col for col in df.columns if col not in exclude]
    
    # Only keep numeric features
    numeric_features = []
    for col in features:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
    
    return numeric_features

def run_backtest(df: pd.DataFrame, test_size: float = 0.2) -> dict:
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
    
    return metrics, model_total, model_margin, features

def main():
    print("=" * 70)
    print("REFINED TEMPORAL FEATURES BACKTEST")
    print("Target: Match 48hr CatBoost Tuning Accuracy")
    print("=" * 70)
    print()
    
    # CatBoost tuning targets
    catboost_metrics = {
        'total_mae': 7.96,
        'margin_mae': 3.85,
        'total_rmse': 10.87
    }
    
    print("CatBoost Tuning Results (48hr):")
    print(f"  Total MAE:   {catboost_metrics['total_mae']:.2f}")
    print(f"  Margin MAE:  {catboost_metrics['margin_mae']:.2f}")
    print(f"  Total RMSE:  {catboost_metrics['total_rmse']:.2f}")
    print()
    
    # Load data with refined temporal features
    print("Loading data with refined temporal features...")
    df = pd.read_parquet('data/processed/halftime_with_refined_temporal.parquet')
    print(f"  Games: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    
    # Count temporal features
    temporal_cols = [col for col in df.columns if any(x in col for x in ['avg_', 'ewm_', 'streak', 'trend', 'std_', 'days'])]
    print(f"  Temporal features: {len(temporal_cols)}")
    print()
    
    # Run backtest
    print("Running backtest...")
    metrics, model_total, model_margin, features = run_backtest(df, test_size=0.2)
    
    print("\n" + "=" * 70)
    print("REFINED TEMPORAL FEATURES: RESULTS")
    print("=" * 70)
    
    print(f"\nTest set: {int(len(df) * 0.2)} games (last 20%)")
    print(f"Features used: {len(features)}")
    print(f"\nMetrics:")
    print(f"  Total MAE:   {metrics['total_mae']:.4f}")
    print(f"  Margin MAE:  {metrics['margin_mae']:.4f}")
    print(f"  Total RMSE:  {metrics['total_rmse']:.4f}")
    print(f"  Margin RMSE: {metrics['margin_rmse']:.4f}")
    
    # Load baseline for comparison
    print("\n" + "=" * 70)
    print("BASELINE: NO TEMPORAL FEATURES (SAME TEST SET)")
    print("=" * 70)
    
    baseline_df = pd.read_parquet('data/processed/halftime_team_v2.parquet')
    
    # Add game_date for proper sorting
    import json
    with open("data/processed/game_ids_2025.json", "r") as f:
        schedule = json.load(f)
    
    game_dates = {}
    for game in schedule:
        game_id = game.get("gameId")
        game_date_str = game.get("gameDate")
        if game_id and game_date_str:
            try:
                game_dates[game_id] = pd.to_datetime(game_date_str[:10])
            except:
                pass
    
    baseline_df['game_date'] = baseline_df['game_id'].map(game_dates)
    baseline_df = baseline_df.sort_values('game_date').reset_index(drop=True)
    
    print(f"\nRunning baseline backtest...")
    baseline_metrics, _, _, _ = run_backtest(baseline_df, test_size=0.2)
    
    print(f"\nBaseline Metrics:")
    print(f"  Total MAE:   {baseline_metrics['total_mae']:.4f}")
    print(f"  Margin MAE:  {baseline_metrics['margin_mae']:.4f}")
    print(f"  Total RMSE:  {baseline_metrics['total_rmse']:.4f}")
    print(f"  Margin RMSE: {baseline_metrics['margin_rmse']:.4f}")
    
    # Calculate improvement vs CatBoost
    print("\n" + "=" * 70)
    print("COMPARISON: REFINED TEMPORAL vs CATBOOST")
    print("=" * 70)
    
    def calc_improvement(target, actual):
        if target > 0:
            diff = actual - target
            pct = (diff / target) * 100
            return diff, pct
        else:
            return 0.0, 0.0
    
    print(f"\n{'Metric':20} | {'CatBoost':10} | {'Refined':10} | {'Diff':8} | {'% Diff':8}")
    print("-" * 70)
    
    for metric_name, catboost_val in catboost_metrics.items():
        refined_val = metrics[metric_name]
        diff, pct = calc_improvement(catboost_val, refined_val)
        status = "✅" if refined_val <= catboost_val else "⚠️"
        print(f"{metric_name:20} | {catboost_val:10.2f} | {refined_val:10.2f} | {diff:+8.2f} | {pct:+8.2f}% {status}")
    
    # Improvement vs baseline
    print("\n" + "=" * 70)
    print("COMPARISON: REFINED vs BASELINE")
    print("=" * 70)
    
    print(f"\n{'Metric':20} | {'Baseline':10} | {'Refined':10} | {'Change':8} | {'% Change':8}")
    print("-" * 70)
    
    for metric_name in ['total_mae', 'margin_mae', 'total_rmse', 'margin_rmse']:
        baseline_val = baseline_metrics[metric_name]
        refined_val = metrics[metric_name]
        change = baseline_val - refined_val
        pct = (change / baseline_val * 100) if baseline_val > 0 else 0
        status = "✅" if refined_val < baseline_val else "⚠️"
        print(f"{metric_name:20} | {baseline_val:10.2f} | {refined_val:10.2f} | {change:+8.2f} | {pct:+8.2f}% {status}")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES (Total Model)")
    print("=" * 70)
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model_total.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features:")
    for i, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:40} {row['importance']:8.4f}")
    
    print("\n" + "=" * 70)
    
    # Save results
    results_df = pd.DataFrame([{
        'baseline_total_mae': baseline_metrics['total_mae'],
        'baseline_margin_mae': baseline_metrics['margin_mae'],
        'refined_total_mae': metrics['total_mae'],
        'refined_margin_mae': metrics['margin_mae'],
        'catboost_total_mae': catboost_metrics['total_mae'],
        'catboost_margin_mae': catboost_metrics['margin_mae'],
        'features_used': len(features)
    }])
    
    results_df.to_csv('reports/refined_temporal_backtest_results.csv', index=False)
    print(f"\nResults saved to: reports/refined_temporal_backtest_results.csv")
    print("=" * 70)

if __name__ == '__main__':
    main()
