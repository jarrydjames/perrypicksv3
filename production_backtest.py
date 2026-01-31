"""
Production Backtest - Real Out-of-Sample Evaluation
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.statistical.block_bootstrap import block_bootstrap
from src.statistical.diebold_mariano import diebold_mariano_test

print("=" * 80)
print("PRODUCTION BACKTEST")
print("=" * 80)
print("")

# Load dataset
df = pd.read_parquet('data/processed/halftime_with_temporal_features_total.parquet')
print(f"Loaded dataset: {len(df)} rows")

# Features
h1_features = [col for col in df.columns if col.startswith('h1_')]

# Sort
sort_cols = ['season_end_yy', 'game_id']
df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

# Unique games
unique_games = df_sorted[['game_id']].drop_duplicates()
games = unique_games['game_id'].values
n_games = len(games)
print(f"Unique games: {n_games}")

# Walkforward CV
MIN_TRAIN = 500
TEST_SIZE = 200
STEP = 200

print(f"\nGenerating folds (min_train={MIN_TRAIN}, test_size={TEST_SIZE}, step={STEP})...")

folds = []
for start_idx in range(0, n_games - MIN_TRAIN - TEST_SIZE + 1, STEP):
    train_start = 0
    train_end = start_idx + MIN_TRAIN
    test_start = train_end
    test_end = test_start + TEST_SIZE
    
    if test_end > n_games:
        break
    
    fold_id = len(folds)
    fold_info = {
        'fold_id': fold_id,
        'train_games': games[train_start:train_end],
        'test_games': games[test_start:test_end],
        'train_size': train_end - train_start,
        'test_size': TEST_SIZE,
    }
    folds.append(fold_info)
    
print(f"Total folds: {len(folds)}\n")

# Train models on each fold
baseline_results = []
xgb_results = []

print("\nTraining models on folds...")
for fold in folds:
    fold_id = fold['fold_id']
    
    train_mask = df_sorted['game_id'].isin(fold['train_games'])
    test_mask = df_sorted['game_id'].isin(fold['test_games'])
    
    X_train = df_sorted.loc[train_mask, h1_features].values
    y_train = df_sorted.loc[train_mask, 'h2_total'].values
    
    X_test = df_sorted.loc[test_mask, h1_features].values
    y_test = df_sorted.loc[test_mask, 'h2_total'].values
    
    # Baseline
    baseline_model = Ridge(alpha=2.0, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # Store results as simple arrays (not dicts)
    for i, (y_true, y_pred_baseline, y_pred_xgb) in enumerate(zip(y_test, baseline_pred, xgb_pred)):
        baseline_results.append({'game_id': fold_id * 10000 + i, 'y_true': y_true, 'y_pred': y_pred_baseline, 'error': abs(y_true - y_pred_baseline)})
        xgb_results.append({'game_id': fold_id * 10000 + i, 'y_true': y_true, 'y_pred': y_pred_xgb, 'error': abs(y_true - y_pred_xgb)})
    
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    print(f"Fold {fold_id}: Baseline MAE={baseline_mae:.4f}, XGBoost MAE={xgb_mae:.4f}")

print("\nAll folds trained.")

# Create DataFrames
baseline_df = pd.DataFrame(baseline_results)
xgb_df = pd.DataFrame(xgb_results)

# Merge for paired comparison on game_id
merged = baseline_df.merge(xgb_df, on=['game_id'], suffixes=('_baseline', '_xgb'))

# Compute deltas
merged['delta'] = merged['error_xgb'] - merged['error_baseline']

print(f"\nPaired observations: {len(merged)}")

# Extract arrays for Diebold-Mariano
loss_baseline = merged['error_baseline'].values
loss_xgb = merged['error_xgb'].values
deltas = merged['delta'].values

# Block bootstrap
block_size = 200
n_bootstraps = 1000

print(f"\nRunning block bootstrap (block_size={block_size}, resamples={n_bootstraps})...")
bootstrap_results = block_bootstrap(
    deltas,
    block_size=block_size,
    n_bootstraps=n_bootstraps,
)

bootstrap_mean = bootstrap_results['mean_diff']
ci_lower = bootstrap_results['ci_lower']
ci_upper = bootstrap_results['ci_upper']
p_improvement = bootstrap_results['p_improvement']

print(f"Block bootstrap results:")
print(f"  Mean difference: {bootstrap_mean:.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  P(improvement): {p_improvement:.4f}")

# Diebold-Mariano test
print(f"\nRunning Diebold-Mariano test...")
dm_results = diebold_mariano_test(
    loss_baseline,
    loss_xgb,
)

dm_stat = dm_results['dm_statistic']
p_value = dm_results['p_value']

print(f"Diebold-Mariano results:")
print(f"  DM statistic: {dm_stat:.4f}")
print(f"  P-value: {p_value:.6e}")

# Decision
mean_delta = deltas.mean()
practical_threshold_1pct = 0.01 * loss_baseline.mean()
practical_threshold_10pts = 0.10

print(f"\nGO / NO-GO Decision:")
print(f"  Mean delta: {mean_delta:.4f}")
print(f"  95% CI upper < 0: {ci_upper < 0}")
print(f"  Practical (1%): {mean_delta <= -practical_threshold_1pct}")
print(f"  Practical (0.10 pts): {mean_delta <= -practical_threshold_10pts}")

ci_upper_ok = ci_upper < 0
practical_ok = mean_delta <= -practical_threshold_1pct or mean_delta <= -practical_threshold_10pts
decision = "GO" if ci_upper_ok and practical_ok else "NO-GO"

print(f"\nFINAL DECISION: {decision}")
if decision == "GO":
    print("RECOMMENDATION: DEPLOY XGBoost to production")
else:
    print("RECOMMENDATION: Do NOT deploy XGBoost - no statistically significant improvement")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
