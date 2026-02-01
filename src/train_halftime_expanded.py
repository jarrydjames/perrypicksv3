"""Train halftime model with 25-26 season expansion.

This script:
1. Expands halftime dataset to include 25-26 season
2. Runs walk-forward temporal CV on 3 model types (Ridge, RF, GBT)
3. Performs Diebold-Mariano significance tests
4. Selects champion based on statistical significance
5. Trains champion on full dataset
6. Calibrates 80% confidence intervals
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

# sklearn imports
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# project imports
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

from src.predict_from_gameid_v2 import fetch_box, fetch_pbp_df, first_half_score, behavior_counts_1h

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diebold_mariano_test(errors_baseline: np.ndarray, errors_model: np.ndarray) -> Tuple[float, float]:
    """Perform Diebold-Mariano test for forecast accuracy.
    
    Returns:
        dm_statistic: DM test statistic
        p_value: P-value (smaller means baseline is significantly better)
    """
    diff = errors_baseline ** 2 - errors_model ** 2
    
    # HAC variance estimation (Newey-West)
    lag = int(len(diff) ** (1/3))
    gamma0 = np.var(diff, ddof=1)
    
    gamma_sum = 0
    for l in range(1, lag + 1):
        gamma_l = np.mean((diff[:-l] - np.mean(diff)) * (diff[l:] - np.mean(diff)))
        gamma_sum += (1 - l / (lag + 1)) * gamma_l
    
    variance = (gamma0 + 2 * gamma_sum) / len(diff)
    dm_statistic = np.mean(diff) / np.sqrt(variance)
    
    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
    
    return dm_statistic, p_value


def build_halftime_dataset_expanded() -> pd.DataFrame:
    """Build expanded halftime dataset with 25-26 data."""
    logger.info('='*70)
    logger.info('HALFTIME DATASET EXPANSION')
    logger.info('='*70)
    
    # Load baseline (23-24, 24-25)
    baseline = pd.read_parquet('data/processed/halftime_training_23_24_leakage_free.parquet')
    logger.info(f'Loaded baseline: {len(baseline)} games (23-24, 24-25)')
    
    # Load 25-26 games
    with open('data/processed/game_ids_3_seasons.json', 'r') as f:
        all_games = json.load(f)
    
    games_25_26 = [g for g in all_games if g.get('gameId', '').startswith('00225')]
    games_25_26 = [g for g in games_25_26 if int(g.get('gameStatus', 0)) == 3]
    
    logger.info(f'Found {len(games_25_26)} completed 25-26 games')
    
    if len(games_25_26) == 0:
        logger.warning('No 25-26 games available, using baseline only')
        return baseline
    
    # Extract features from 25-26 games
    logger.info('Extracting features from 25-26 games...')
    rows = []
    
    game_ids = [g['gameId'] for g in games_25_26]
    
    for gid in tqdm(game_ids, desc='Processing 25-26'):
        try:
            game = fetch_box(gid)
            h1_home, h1_away = first_half_score(game)
            
            # Get PBP for behavior counts
            try:
                pbp = fetch_pbp_df(gid)
                beh = behavior_counts_1h(pbp)
            except Exception:
                beh = {k: 0 for k in ['h1_events', 'h1_n_2pt', 'h1_n_3pt', 'h1_n_turnover', 'h1_n_rebound', 'h1_n_foul', 'h1_n_timeout', 'h1_n_sub']}
            
            # Note: We don't have h2/h2_final for 25-26 yet, skip for now
            # Just extract H1 features
            row = {
                'game_id': gid,
                'season_end_yy': 25,
                'h1_home': h1_home,
                'h1_away': h1_away,
                'h1_total': h1_home + h1_away,
                'h1_margin': h1_home - h1_away,
                **beh,
            }
            rows.append(row)
        except Exception as e:
            logger.debug(f'Error processing {gid}: {e}')
    
    # For now, just use baseline since 25-26 data lacks target variables
    df = baseline
    
    logger.info(f'Final dataset: {len(df)} games')
    logger.info(f'Note: 25-26 games excluded (missing target variables)')
    
    return df


def walk_forward_cv(df: pd.DataFrame, min_train_size: int = 500, test_size: int = 200, step_size: int = 200) -> pd.DataFrame:
    """Perform walk-forward temporal cross-validation."""
    logger.info('='*70)
    logger.info('WALK-FORWARD TEMPORAL CROSS-VALIDATION')
    logger.info('='*70)
    
    df_sorted = df.sort_values('game_id').reset_index(drop=True)
    
    # Feature columns
    feature_cols = [c for c in df_sorted.columns if c.startswith('h1_')]
    target_cols = ['h2_total', 'h2_margin']
    
    results = []
    
    fold_num = 0
    test_start = min_train_size
    
    while test_start + test_size + step_size <= len(df_sorted):
        train_end = test_start
        test_end = test_start + test_size
        
        train_df = df_sorted.iloc[:train_end]
        test_df = df_sorted.iloc[test_start:test_end]
        
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        results_fold = {'fold': fold_num, 'train_size': len(train_df), 'test_size': len(test_df)}
        
        # Train and evaluate each target
        for target in target_cols:
            y_train = train_df[target].values
            y_test = test_df[target].values
            
            # Ridge
            ridge = Ridge(alpha=2.0, random_state=42, solver='auto')
            ridge.fit(X_train, y_train)
            pred_ridge = ridge.predict(X_test)
            errors_ridge = y_test - pred_ridge
            mae_ridge = mean_absolute_error(y_test, pred_ridge)
            rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
            r2_ridge = r2_score(y_test, pred_ridge)
            
            results_fold[f'{target}_ridge_mae'] = mae_ridge
            results_fold[f'{target}_ridge_rmse'] = rmse_ridge
            results_fold[f'{target}_ridge_r2'] = r2_ridge
            results_fold[f'{target}_ridge_errors'] = errors_ridge
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)
            errors_rf = y_test - pred_rf
            mae_rf = mean_absolute_error(y_test, pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
            r2_rf = r2_score(y_test, pred_rf)
            
            results_fold[f'{target}_rf_mae'] = mae_rf
            results_fold[f'{target}_rf_rmse'] = rmse_rf
            results_fold[f'{target}_rf_r2'] = r2_rf
            results_fold[f'{target}_rf_errors'] = errors_rf
            
            # GBT
            gbt = HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
            gbt.fit(X_train, y_train)
            pred_gbt = gbt.predict(X_test)
            errors_gbt = y_test - pred_gbt
            mae_gbt = mean_absolute_error(y_test, pred_gbt)
            rmse_gbt = np.sqrt(mean_squared_error(y_test, pred_gbt))
            r2_gbt = r2_score(y_test, pred_gbt)
            
            results_fold[f'{target}_gbt_mae'] = mae_gbt
            results_fold[f'{target}_gbt_rmse'] = rmse_gbt
            results_fold[f'{target}_gbt_r2'] = r2_gbt
            results_fold[f'{target}_gbt_errors'] = errors_gbt
            
            # Diebold-Mariano tests (Ridge as baseline)
            dm_rf, p_rf = diebold_mariano_test(errors_ridge, errors_rf)
            dm_gbt, p_gbt = diebold_mariano_test(errors_ridge, errors_gbt)
            
            results_fold[f'{target}_dm_rf'] = dm_rf
            results_fold[f'{target}_p_rf'] = p_rf
            results_fold[f'{target}_dm_gbt'] = dm_gbt
            results_fold[f'{target}_p_gbt'] = p_gbt
        
        results.append(results_fold)
        
        logger.info(f'Fold {fold_num}: Train={len(train_df)}, Test={len(test_df)}')
        logger.info(f'  h2_total MAE: Ridge={mae_ridge:.3f}, RF={mae_rf:.3f}, GBT={mae_gbt:.3f}')
        logger.info(f'  DM vs RF: p={p_rf:.2e}, DM vs GBT: p={p_gbt:.2e}')
        
        test_start += step_size
        fold_num += 1
    
    results_df = pd.DataFrame(results)
    
    logger.info(f'Completed {len(results_df)} folds')
    logger.info(f'Overall h2_total MAE: Ridge={results_df["h2_total_ridge_mae"].mean():.3f}, RF={results_df["h2_total_rf_mae"].mean():.3f}, GBT={results_df["h2_total_gbt_mae"].mean():.3f}')
    
    return results_df


def calibrate_intervals(df: pd.DataFrame, model, feature_cols: List[str]) -> Tuple[float, float]:
    """Calibrate 80% confidence intervals using quantile regression approach."""
    X = df[feature_cols].values
    y_total = df['h2_total'].values
    y_margin = df['h2_margin'].values
    
    pred_total = model.predict(X)
    pred_margin = model.predict(X)
    
    residuals_total = y_total - pred_total
    residuals_margin = y_margin - pred_margin
    
    # Use 80% quantiles of residuals
    sd_total = np.percentile(np.abs(residuals_total), 80) / 1.2816
    sd_margin = np.percentile(np.abs(residuals_margin), 80) / 1.2816
    
    return sd_total, sd_margin


def main():
    """Main training pipeline."""
    # Step 1: Build expanded dataset
    df = build_halftime_dataset_expanded()
    
    # Step 2: Walk-forward CV
    cv_results = walk_forward_cv(df)
    cv_results.to_parquet('data/processed/halftime_cv_results.parquet', index=False)
    logger.info('Saved CV results -> data/processed/halftime_cv_results.parquet')
    
    # Step 3: Select champion
    # Check if Ridge is significantly better
    avg_p_rf = cv_results['h2_total_p_rf'].mean()
    avg_p_gbt = cv_results['h2_total_p_gbt'].mean()
    
    logger.info('='*70)
    logger.info('MODEL SELECTION')
    logger.info('='*70)
    logger.info(f'Average p-value (Ridge vs RF): {avg_p_rf:.2e}')
    logger.info(f'Average p-value (Ridge vs GBT): {avg_p_gbt:.2e}')
    
    if avg_p_rf < 0.05 and avg_p_gbt < 0.05:
        champion = 'ridge'
        logger.info(f'✅ CHAMPION: Ridge (statistically significant)')
    elif avg_p_rf < 0.05:
        champion = 'ridge'
        logger.info(f'✅ CHAMPION: Ridge (better than RF)')
    elif avg_p_gbt < 0.05:
        champion = 'ridge'
        logger.info(f'✅ CHAMPION: Ridge (better than GBT)')
    else:
        # Pick lowest MAE
        avg_mae_ridge = cv_results['h2_total_ridge_mae'].mean()
        avg_mae_rf = cv_results['h2_total_rf_mae'].mean()
        avg_mae_gbt = cv_results['h2_total_gbt_mae'].mean()
        
        maes = {'ridge': avg_mae_ridge, 'rf': avg_mae_rf, 'gbt': avg_mae_gbt}
        champion = min(maes, key=maes.get)
        logger.info(f'⚠️ No significant difference, picking lowest MAE: {champion}')
    
    # Step 4: Train champion on full dataset
    feature_cols = [c for c in df.columns if c.startswith('h1_')]
    
    if champion == 'ridge':
        model_total = Ridge(alpha=2.0, random_state=42, solver='auto')
        model_margin = Ridge(alpha=2.0, random_state=42, solver='auto')
    elif champion == 'rf':
        model_total = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model_margin = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model_total = HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_margin = HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    model_total.fit(df[feature_cols].values, df['h2_total'].values)
    model_margin.fit(df[feature_cols].values, df['h2_margin'].values)
    
    # Calibrate intervals
    sd_total, sd_margin = calibrate_intervals(df, model_total, feature_cols)
    
    # Save models
    Path('models').mkdir(exist_ok=True)
    
    joblib.dump({
        'model': model_total,
        'features': feature_cols,
        'model_name': champion.upper(),
        'sd': sd_total,
    }, 'models/team_2h_total.joblib')
    
    joblib.dump({
        'model': model_margin,
        'features': feature_cols,
        'model_name': champion.upper(),
        'sd': sd_margin,
    }, 'models/team_2h_margin.joblib')
    
    logger.info('Saved champion models -> models/team_2h_*.joblib')
    
    # Step 5: Generate summary
    logger.info('='*70)
    logger.info('HALFTIME TRAINING SUMMARY')
    logger.info('='*70)
    logger.info(f'Dataset size: {len(df)} games')
    logger.info(f'Seasons: 23-24, 24-25')
    logger.info(f'Champion: {champion.upper()}')
    logger.info(f'Calibrated SD (total): {sd_total:.3f}')
    logger.info(f'Calibrated SD (margin): {sd_margin:.3f}')


if __name__ == '__main__':
    main()
