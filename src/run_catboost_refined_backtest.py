"""
Backtest with refined temporal features using CatBoost (same as 48hr tuning).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
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

def run_catboost_backtest(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Run backtest with CatBoost and return metrics."""
    from catboost import CatBoostRegressor
    
    features = get_features(df)
    X = df[features]
    y_total = df['h2_total']
    y_margin = df['h2_margin']
    
    # Time-based split (use last 20% for test)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_total, y_test_total = y_total[:split_idx], y_total[split_idx:]
    y_train_margin, y_test_margin = y_margin[:split_idx], y_margin[split_idx:]
    
    # Train total model with CatBoost
    model_total = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    model_total.fit(X_train, y_train_total)
    
    # Train margin model with CatBoost
    model_margin = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    model_margin.fit(X_train, y_train_margin)
    
    # Predict
    pred_total = model_total.predict(X_test)
    pred_margin = model_margin.predict(X_test)
    
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
    print("CATBOOST BACKTEST: REFINED TEMPORAL FEATURES")
    print("Target: Match 48hr CatBoost Tuning Accuracy")
    print("=" * 70)
    print()
    
    # CatBoost tuning targets (from original training)
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
    df_refined = pd.read_parquet('data/processed/halftime_with_refined_temporal.parquet')
    print(f"  Games: {len(df_refined)}")
    print(f"  Features: {len(df_refined.columns)}")
    
    # Load original dataset (what CatBoost was trained on)
    print("\nLoading original temporal features dataset...")
    df_original = pd.read_parquet('data/processed/halftime_with_temporal_features_total.parquet')
    print(f"  Games: {len(df_original)}")
    print(f"  Features: {len(df_original.columns)}")
    print()
    
    # Run backtest on original (baseline)
    print("Running backtest on ORIGINAL temporal features...")
    original_metrics, _, _, original_features = run_catboost_backtest(df_original, test_size=0.2)
    
    print("\n" + "=" * 70)
    print("ORIGINAL TEMPORAL FEATURES (CATBOOST MODEL)")
    print("=" * 70)
    
    print(f"\nTest set: {int(len(df_original) * 0.2)} games (last 20%)")
    print(f"Features used: {len(original_features)}")
    print(f"\nMetrics:")
    print(f"  Total MAE:   {original_metrics['total_mae']:.4f}")
    print(f"  Margin MAE:  {original_metrics['margin_mae']:.4f}")
    print(f"  Total RMSE:  {original_metrics['total_rmse']:.4f}")
    print(f"  Margin RMSE: {original_metrics['margin_rmse']:.4f}")
    
    # Run backtest on refined
    print("\n" + "=" * 70)
    print("REFINED TEMPORAL FEATURES (CATBOOST MODEL)")
    print("=" * 70)
    
    print("\nRunning backtest on REFINED temporal features...")
    refined_metrics, model_total, model_margin, refined_features = run_catboost_backtest(df_refined, test_size=0.2)
    
    print(f"\nTest set: {int(len(df_refined) * 0.2)} games (last 20%)")
    print(f"Features used: {len(refined_features)}")
    print(f"\nMetrics:")
    print(f"  Total MAE:   {refined_metrics['total_mae']:.4f}")
    print(f"  Margin MAE:  {refined_metrics['margin_mae']:.4f}")
    print(f"  Total RMSE:  {refined_metrics['total_rmse']:.4f}")
    print(f"  Margin RMSE: {refined_metrics['margin_rmse']:.4f}")
    
    # Compare all three
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs REFINED vs 48HR TARGET")
    print("=" * 70)
    
    print(f"\n{'Metric':20} | {'48hr Target':12} | {'Original':12} | {'Refined':12} | {'Best':12}")
    print("-" * 80)
    
    for metric_name in ['total_mae', 'margin_mae', 'total_rmse']:
        target = catboost_metrics.get(metric_name, 0)
        orig = original_metrics[metric_name]
        ref = refined_metrics[metric_name]
        
        # Determine best (closest to target or lowest)
        best_val = min(target, orig, ref)
        best_name = '48hr' if best_val == target else 'Original' if best_val == orig else 'Refined'
        
        status = "✅" if ref <= target else "⚠️"
        
        print(f"{metric_name:20} | {target:12.2f} | {orig:12.2f} | {ref:12.2f} | {best_name:12} {status}")
    
    # Feature importance for refined model
    print("\n" + "=" * 70)
    print("TOP 20 MOST IMPORTANT FEATURES (REFINED MODEL)")
    print("=" * 70)
    
    importance_df = pd.DataFrame({
        'feature': refined_features,
        'importance': model_total.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:45} {row['importance']:8.4f}")
    
    # Check how many refined features are actually useful
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # Count features by category
    temporal_features = [f for f in refined_features if any(x in f for x in ['avg_', 'ewm_', 'streak', 'trend', 'std_', 'days'])]
    diff_features = [f for f in refined_features if f.startswith('diff_')]
    home_features = [f for f in refined_features if f.startswith('home_') and f not in temporal_features]
    away_features = [f for f in refined_features if f.startswith('away_') and f not in temporal_features]
    
    print(f"\nFeature categories:")
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"  Differential features: {len(diff_features)}")
    print(f"  Home stats: {len(home_features)}")
    print(f"  Away stats: {len(away_features)}")
    
    # Top 20 by importance
    top_20_features = importance_df.head(20)['feature'].tolist()
    top_temporal = [f for f in top_20_features if f in temporal_features]
    top_diff = [f for f in top_20_features if f in diff_features]
    
    print(f"\nTop 20 feature breakdown:")
    print(f"  Temporal in top 20: {len(top_temporal)}")
    print(f"  Differential in top 20: {len(top_diff)}")
    
    print("\n" + "=" * 70)
    
    # Save results
    results_df = pd.DataFrame([{
        'original_total_mae': original_metrics['total_mae'],
        'original_margin_mae': original_metrics['margin_mae'],
        'refined_total_mae': refined_metrics['total_mae'],
        'refined_margin_mae': refined_metrics['margin_mae'],
        'catboost_target_total_mae': catboost_metrics['total_mae'],
        'catboost_target_margin_mae': catboost_metrics['margin_mae'],
        'original_features': len(original_features),
        'refined_features': len(refined_features)
    }])
    
    results_df.to_csv('reports/catboost_refined_backtest_results.csv', index=False)
    print(f"\nResults saved to: reports/catboost_refined_backtest_results.csv")
    print("=" * 70)

if __name__ == '__main__':
    main()
