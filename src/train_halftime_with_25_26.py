"""Train halftime model with 25-26 season expansion.

This script:
1. Loads baseline halftime dataset (23-24, 24-25)
2. Extracts features and targets from completed 25-26 games
3. Computes h2 targets from final scores minus h1 scores
4. Runs walk-forward temporal CV on 3 model types
5. Performs Diebold-Mariano significance tests
6. Selects champion model
7. Trains on full dataset and calibrates 80% CIs
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
    """Perform Diebold-Mariano test for forecast accuracy."""
    diff = errors_baseline ** 2 - errors_model ** 2
    
    lag = max(1, int(len(diff) ** (1/3)))
    gamma0 = np.var(diff, ddof=1)
    
    gamma_sum = 0
    for l in range(1, lag + 1):
        gamma_l = np.mean((diff[:-l] - np.mean(diff)) * (diff[l:] - np.mean(diff)))
        gamma_sum += (1 - l / (lag + 1)) * gamma_l
    
    variance = (gamma0 + 2 * gamma_sum) / len(diff)
    
    if variance <= 0:
        return 0, 1.0
    
    dm_statistic = np.mean(diff) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
    
    return dm_statistic, p_value


def extract_halftime_row_full(game_id: str) -> Dict:
    """Extract halftime features and targets from a completed game."""
    try:
        game = fetch_box(game_id)
        h1_home, h1_away = first_half_score(game)
        
        # Get PBP for behavior counts
        try:
            pbp = fetch_pbp_df(game_id)
            beh = behavior_counts_1h(pbp)
        except Exception:
            beh = {k: 0 for k in ['h1_events', 'h1_n_2pt', 'h1_n_3pt', 'h1_n_turnover', 'h1_n_rebound', 'h1_n_foul', 'h1_n_timeout', 'h1_n_sub']}
        
        # Get final scores from boxscore
        home_team = game.get('homeTeam', {}) or {}
        away_team = game.get('awayTeam', {}) or {}
        
        final_home = home_team.get('score', home_team.get('points', 0))
        final_away = away_team.get('score', away_team.get('points', 0))
        
        # Compute H2 targets from final minus H1
        h2_home = final_home - h1_home
        h2_away = final_away - h1_away
        h2_total = h2_home + h2_away
        h2_margin = h2_home - h2_away
        final_total = final_home + final_away
        final_margin = final_home - final_away
        
        row = {
            'game_id': game_id,
            'season_end_yy': 25,
            'h1_home': h1_home,
            'h1_away': h1_away,
            'h1_total': h1_home + h1_away,
            'h1_margin': h1_home - h1_away,
            'h1_events': beh.get('h1_events', 0),
            'h1_n_2pt': beh.get('h1_n_2pt', 0),
            'h1_n_3pt': beh.get('h1_n_3pt', 0),
            'h1_n_turnover': beh.get('h1_n_turnover', 0),
            'h1_n_rebound': beh.get('h1_n_rebound', 0),
            'h1_n_foul': beh.get('h1_n_foul', 0),
            'h1_n_timeout': beh.get('h1_n_timeout', 0),
            'h1_n_sub': beh.get('h1_n_sub', 0),
            'h2_total': h2_total,
            'h2_margin': h2_margin,
            'final_total': final_total,
            'final_margin': final_margin,
        }
        
        return row
        
    except Exception as e:
        logger.error(f'Error extracting {game_id}: {e}')
        return None


def build_halftime_dataset_full() -> pd.DataFrame:
    """Build complete halftime dataset including 25-26 games."""
    logger.info('='*70)
    logger.info('BUILDING HALFTIME DATASET WITH 25-26 GAMES')
    logger.info('='*70)
    
    # Load baseline
    baseline = pd.read_parquet('data/processed/halftime_training_23_24_leakage_free.parquet')
    logger.info(f'Loaded baseline: {len(baseline)} games (23-24, 24-25)')
    
    # Load 25-26 completed games
    with open('data/processed/game_ids_3_seasons.json', 'r') as f:
        all_games = json.load(f)
    
    games_25_26 = [g for g in all_games if g.get('gameId', '').startswith('00225')]
    completed_games = [g for g in games_25_26 if int(g.get('gameStatus', 0)) == 3]
    
    logger.info(f'Found {len(completed_games)} completed 25-26 games')
    
    if len(completed_games) == 0:
        logger.warning('No 25-26 games available, using baseline only')
        return baseline
    
    # Extract features from 25-26 games
    logger.info('Extracting features from 25-26 games...')
    rows = []
    
    game_ids = [g['gameId'] for g in completed_games]
    
    for gid in tqdm(game_ids, desc='Processing 25-26'):
        row = extract_halftime_row_full(gid)
        if row:
            rows.append(row)
    
    df_new = pd.DataFrame(rows)
    logger.info(f'Successfully extracted {len(df_new)} 25-26 games')
    
    # Combine with baseline
    df_combined = pd.concat([baseline, df_new], ignore_index=True)
    
    # Sort by game_id
    df_combined = df_combined.sort_values('game_id').reset_index(drop=True)
    
    logger.info(f'Combined dataset: {len(df_combined)} games')
    logger.info(f'  Baseline: {len(baseline)} games')
    logger.info(f'  New (25-26): {len(df_new)} games')
    
    # Save expanded dataset
    output_path = 'data/processed/halftime_training_3_seasons.parquet'
    df_combined.to_parquet(output_path, index=False)
    logger.info(f'Saved expanded dataset -> {output_path}')
    
    return df_combined


def train_ridge(X_train, y_train, X_test, y_test, alpha: float = 2.0) -> Dict:
    """Train Ridge regression model."""
    model = Ridge(alpha=alpha, random_state=42, solver='auto')
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'pred_train': pred_train,
        'pred_test': pred_test,
        'errors_train': y_train - pred_train,
        'errors_test': y_test - pred_test,
        'mae_train': mean_absolute_error(y_train, pred_train),
        'mae_test': mean_absolute_error(y_test, pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, pred_test)),
        'r2_train': r2_score(y_train, pred_train),
        'r2_test': r2_score(y_test, pred_test),
    }


def train_rf(X_train, y_train, X_test, y_test) -> Dict:
    """Train Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'pred_train': pred_train,
        'pred_test': pred_test,
        'errors_train': y_train - pred_train,
        'errors_test': y_test - pred_test,
        'mae_train': mean_absolute_error(y_train, pred_train),
        'mae_test': mean_absolute_error(y_test, pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, pred_test)),
        'r2_train': r2_score(y_train, pred_train),
        'r2_test': r2_score(y_test, pred_test),
    }


def train_gbt(X_train, y_train, X_test, y_test) -> Dict:
    """Train Gradient Boosting Trees model."""
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'pred_train': pred_train,
        'pred_test': pred_test,
        'errors_train': y_train - pred_train,
        'errors_test': y_test - pred_test,
        'mae_train': mean_absolute_error(y_train, pred_train),
        'mae_test': mean_absolute_error(y_test, pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, pred_test)),
        'r2_train': r2_score(y_train, pred_train),
        'r2_test': r2_score(y_test, pred_test),
    }


def walk_forward_cv(df: pd.DataFrame, min_train_size: int = 500, test_size: int = 200, step_size: int = 200) -> pd.DataFrame:
    """Perform walk-forward temporal cross-validation."""
    logger.info('='*70)
    logger.info('WALK-FORWARD TEMPORAL CROSS-VALIDATION')
    logger.info('='*70)
    
    df_sorted = df.sort_values('game_id').reset_index(drop=True)
    
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
        
        for target in target_cols:
            y_train = train_df[target].values
            y_test = test_df[target].values
            
            # Ridge
            ridge = train_ridge(X_train, y_train, X_test, y_test)
            results_fold[f'{target}_ridge_mae_test'] = ridge['mae_test']
            results_fold[f'{target}_ridge_rmse_test'] = ridge['rmse_test']
            results_fold[f'{target}_ridge_r2_test'] = ridge['r2_test']
            results_fold[f'{target}_ridge_errors_test'] = ridge['errors_test']
            
            # RF
            rf = train_rf(X_train, y_train, X_test, y_test)
            results_fold[f'{target}_rf_mae_test'] = rf['mae_test']
            results_fold[f'{target}_rf_rmse_test'] = rf['rmse_test']
            results_fold[f'{target}_rf_r2_test'] = rf['r2_test']
            results_fold[f'{target}_rf_errors_test'] = rf['errors_test']
            
            # GBT
            gbt = train_gbt(X_train, y_train, X_test, y_test)
            results_fold[f'{target}_gbt_mae_test'] = gbt['mae_test']
            results_fold[f'{target}_gbt_rmse_test'] = gbt['rmse_test']
            results_fold[f'{target}_gbt_r2_test'] = gbt['r2_test']
            results_fold[f'{target}_gbt_errors_test'] = gbt['errors_test']
            
            # Diebold-Mariano tests (Ridge baseline)
            dm_rf, p_rf = diebold_mariano_test(ridge['errors_test'], rf['errors_test'])
            dm_gbt, p_gbt = diebold_mariano_test(ridge['errors_test'], gbt['errors_test'])
            
            results_fold[f'{target}_dm_rf'] = dm_rf
            results_fold[f'{target}_p_rf'] = p_rf
            results_fold[f'{target}_dm_gbt'] = dm_gbt
            results_fold[f'{target}_p_gbt'] = p_gbt
        
        results.append(results_fold)
        
        if fold_num % 2 == 0:
            logger.info(f'Fold {fold_num}: Train={len(train_df)}, Test={len(test_df)}')
            logger.info(f'  h2_total MAE: Ridge={ridge["mae_test"]:.3f}, RF={rf["mae_test"]:.3f}, GBT={gbt["mae_test"]:.3f}')
            logger.info(f'  DM vs RF: p={p_rf:.2e}, DM vs GBT: p={p_gbt:.2e}')
        
        test_start += step_size
        fold_num += 1
    
    results_df = pd.DataFrame(results)
    
    logger.info(f'Completed {len(results_df)} folds')
    logger.info(f'Overall h2_total MAE: Ridge={results_df["h2_total_ridge_mae_test"].mean():.3f}, RF={results_df["h2_total_rf_mae_test"].mean():.3f}, GBT={results_df["h2_total_gbt_mae_test"].mean():.3f}')
    
    return results_df


def calibrate_intervals(df: pd.DataFrame, model, feature_cols: List[str]) -> Tuple[float, float]:
    """Calibrate 80% confidence intervals."""
    X = df[feature_cols].values
    y_total = df['h2_total'].values
    y_margin = df['h2_margin'].values
    
    pred_total = model.predict(X)
    pred_margin = model.predict(X)
    
    residuals_total = y_total - pred_total
    residuals_margin = y_margin - pred_margin
    
    sd_total = np.percentile(np.abs(residuals_total), 80) / 1.2816
    sd_margin = np.percentile(np.abs(residuals_margin), 80) / 1.2816
    
    return sd_total, sd_margin


def main():
    """Main training pipeline."""
    # Step 1: Build dataset with 25-26
    df = build_halftime_dataset_full()
    
    # Step 2: Walk-forward CV
    cv_results = walk_forward_cv(df)
    cv_results.to_parquet('data/processed/halftime_cv_results_3_seasons.parquet', index=False)
    logger.info('Saved CV results -> data/processed/halftime_cv_results_3_seasons.parquet')
    
    # Step 3: Select champion
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
    else:
        avg_mae_ridge = cv_results['h2_total_ridge_mae_test'].mean()
        avg_mae_rf = cv_results['h2_total_rf_mae_test'].mean()
        avg_mae_gbt = cv_results['h2_total_gbt_mae_test'].mean()
        
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
    logger.info(f'Seasons: 23-24, 24-25, 25-26')
    logger.info(f'Champion: {champion.upper()}')
    logger.info(f'Calibrated SD (total): {sd_total:.3f}')
    logger.info(f'Calibrated SD (margin): {sd_margin:.3f}')


if __name__ == '__main__':
    main()
