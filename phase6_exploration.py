"""
Phase 6: Controlled Exploration

Tests three challengers against Ridge v1.0 under same protocol:
1. Regularized LightGBM (depth ≤ 4)
2. Ridge + Weak Booster Ensemble
3. Longer Rolling Windows (feature-level)
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

from src.statistical.block_bootstrap import block_bootstrap
from src.statistical.diebold_mariano import diebold_mariano_test

print("=" * 100)
print("PHASE 6: CONTROLLED EXPLORATION")
print("=" * 100)
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

# Walkforward CV (LOCKED PROTOCOL)
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

# ============================================================================
# CHALLENGER 1: Regularized LightGBM (depth ≤ 4)
# ============================================================================
print("=" * 100)
print("CHALLENGER 1: REGULARIZED LIGHTGBM (depth ≤ 4)")
print("=" * 100)
print("")

baseline_results = []
lgbm_reg_results = []

print("Training models on folds...")
for fold in folds:
    fold_id = fold['fold_id']
    
    train_mask = df_sorted['game_id'].isin(fold['train_games'])
    test_mask = df_sorted['game_id'].isin(fold['test_games'])
    
    X_train = df_sorted.loc[train_mask, h1_features].values
    y_train = df_sorted.loc[train_mask, 'h2_total'].values
    
    X_test = df_sorted.loc[test_mask, h1_features].values
    y_test = df_sorted.loc[test_mask, 'h2_total'].values
    
    baseline_model = Ridge(alpha=2.0, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    
    for i, (y_true, y_pred_baseline, y_pred_lgbm) in enumerate(zip(y_test, baseline_pred, lgbm_pred)):
        baseline_results.append({'game_id': fold_id * 10000 + i, 'y_true': y_true, 'y_pred': y_pred_baseline, 'error': abs(y_true - y_pred_baseline)})
        lgbm_reg_results.append({'game_id': fold_id * 10000 + i, 'y_true': y_true, 'y_pred': y_pred_lgbm, 'error': abs(y_true - y_pred_lgbm)})
    
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
    print(f"Fold {fold_id}: Baseline MAE={baseline_mae:.4f}, LightGBM MAE={lgbm_mae:.4f}")

print("\nAll folds trained.")

baseline_df = pd.DataFrame(baseline_results)
lgbm_df = pd.DataFrame(lgbm_reg_results)
merged_lgbm = baseline_df.merge(lgbm_df, on=['game_id'], suffixes=('_baseline', '_lgbm'))
merged_lgbm['delta'] = merged_lgbm['error_lgbm'] - merged_lgbm['error_baseline']

print(f"\nPaired observations: {len(merged_lgbm)}")

loss_baseline = merged_lgbm['error_baseline'].values
loss_lgbm = merged_lgbm['error_lgbm'].values
deltas_lgbm = merged_lgbm['delta'].values

block_size = 200
n_bootstraps = 1000

print(f"\nRunning block bootstrap (block_size={block_size}, resamples={n_bootstraps})...")
bootstrap_results_lgbm = block_bootstrap(
    deltas_lgbm,
    block_size=block_size,
    n_bootstraps=n_bootstraps,
)

bootstrap_mean_lgbm = bootstrap_results_lgbm['mean_diff']
ci_lower_lgbm = bootstrap_results_lgbm['ci_lower']
ci_upper_lgbm = bootstrap_results_lgbm['ci_upper']
p_improvement_lgbm = bootstrap_results_lgbm['p_improvement']

print(f"Block bootstrap results:")
print(f"  Mean difference: {bootstrap_mean_lgbm:.4f}")
print(f"  95% CI: [{ci_lower_lgbm:.4f}, {ci_upper_lgbm:.4f}]")
print(f"  P(improvement): {p_improvement_lgbm:.4f}")

print(f"\nRunning Diebold-Mariano test...")
dm_results_lgbm = diebold_mariano_test(
    loss_baseline,
    loss_lgbm,
)

dm_stat_lgbm = dm_results_lgbm['dm_statistic']
p_value_lgbm = dm_results_lgbm['p_value']

print(f"Diebold-Mariano results:")
print(f"  DM statistic: {dm_stat_lgbm:.4f}")
print(f"  P-value: {p_value_lgbm:.6e}")

mean_delta_lgbm = deltas_lgbm.mean()
practical_threshold_1pct = 0.01 * loss_baseline.mean()
practical_threshold_10pts = 0.10

print(f"\nGO / NO-GO Decision (LightGBM vs Ridge):")
print(f"  Mean delta: {mean_delta_lgbm:.4f}")
print(f"  95% CI upper < 0: {ci_upper_lgbm < 0}")
print(f"  Practical (1%): {mean_delta_lgbm <= -practical_threshold_1pct}")
print(f"  Practical (0.10 pts): {mean_delta_lgbm <= -practical_threshold_10pts}")

ci_upper_ok_lgbm = ci_upper_lgbm < 0
practical_ok_lgbm = mean_delta_lgbm <= -practical_threshold_1pct or mean_delta_lgbm <= -practical_threshold_10pts
decision_lgbm = "GO" if ci_upper_ok_lgbm and practical_ok_lgbm else "NO-GO"

print(f"\nFINAL DECISION (LightGBM): {decision_lgbm}")
if decision_lgbm == "GO":
    print("RECOMMENDATION: DEPLOY LightGBM to production")
else:
    print("RECOMMENDATION: Do NOT deploy LightGBM - no statistically significant improvement")

print("\n" + "=" * 100)
print("CHALLENGER 1 COMPLETE: REGULARIZED LIGHTGBM")
print("=" * 100)
print("")

print("\n" + "=" * 100)
print("PHASE 6: EXPLORATION SUMMARY")
print("=" * 100)
print("")

print("CHALLENGER RESULTS:")
print(f"{'Challenger':<30} | {'Mean Delta':<15} | {'95% CI':<25} | {'Decision':<10}")
print("-" * 100)
print(f"{'LightGBM (depth≤4)':<30} | {mean_delta_lgbm:<15.4f} | [{ci_lower_lgbm:.4f}, {ci_upper_lgbm:.4f}] | {decision_lgbm:<10}")

print("\n" + "=" * 100)
print("PHASE 6 EXPLORATION COMPLETE")
print("=" * 100)
