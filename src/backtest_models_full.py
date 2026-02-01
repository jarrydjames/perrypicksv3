"""Universal backtest script for all model types (Pregame, Q3, Halftime).

This script performs:
1. Walk-forward temporal cross-validation
2. Training of 3 model types: Ridge, Random Forest, GBT
3. Diebold-Mariano significance testing
4. Model selection based on statistical significance
5. Complete readout generation
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# sklearn imports
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# project imports
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

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
    
    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
    
    return dm_statistic, p_value


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
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
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
    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
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


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    min_train_size: int = 500,
    test_size: int = 200,
    step_size: int = 200,
) -> pd.DataFrame:
    """Perform walk-forward temporal cross-validation."""
    logger.info('='*70)
    logger.info('WALK-FORWARD TEMPORAL CROSS-VALIDATION')
    logger.info('='*70)
    logger.info(f'Min train size: {min_train_size}')
    logger.info(f'Test size: {test_size}')
    logger.info(f'Step size: {step_size}')
    
    df_sorted = df.sort_values('game_id').reset_index(drop=True)
    
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
        
        results_fold = {
            'fold': fold_num,
            'train_size': len(train_df),
            'test_size': len(test_df),
        }
        
        # Train and evaluate each target
        for target in target_cols:
            y_train = train_df[target].values
            y_test = test_df[target].values
            
            # Ridge
            ridge = train_ridge(X_train, y_train, X_test, y_test)
            results_fold[f'{target}_ridge_mae_train'] = ridge['mae_train']
            results_fold[f'{target}_ridge_mae_test'] = ridge['mae_test']
            results_fold[f'{target}_ridge_rmse_test'] = ridge['rmse_test']
            results_fold[f'{target}_ridge_r2_test'] = ridge['r2_test']
            results_fold[f'{target}_ridge_errors_test'] = ridge['errors_test']
            
            # Random Forest
            rf = train_rf(X_train, y_train, X_test, y_test)
            results_fold[f'{target}_rf_mae_train'] = rf['mae_train']
            results_fold[f'{target}_rf_mae_test'] = rf['mae_test']
            results_fold[f'{target}_rf_rmse_test'] = rf['rmse_test']
            results_fold[f'{target}_rf_r2_test'] = rf['r2_test']
            results_fold[f'{target}_rf_errors_test'] = rf['errors_test']
            
            # GBT
            gbt = train_gbt(X_train, y_train, X_test, y_test)
            results_fold[f'{target}_gbt_mae_train'] = gbt['mae_train']
            results_fold[f'{target}_gbt_mae_test'] = gbt['mae_test']
            results_fold[f'{target}_gbt_rmse_test'] = gbt['rmse_test']
            results_fold[f'{target}_gbt_r2_test'] = gbt['r2_test']
            results_fold[f'{target}_gbt_errors_test'] = gbt['errors_test']
            
            # Diebold-Mariano tests (Ridge as baseline)
            dm_rf, p_rf = diebold_mariano_test(ridge['errors_test'], rf['errors_test'])
            dm_gbt, p_gbt = diebold_mariano_test(ridge['errors_test'], gbt['errors_test'])
            
            results_fold[f'{target}_dm_rf'] = dm_rf
            results_fold[f'{target}_p_rf'] = p_rf
            results_fold[f'{target}_dm_gbt'] = dm_gbt
            results_fold[f'{target}_p_gbt'] = p_gbt
        
        results.append(results_fold)
        
        if fold_num % 2 == 0:
            logger.info(f'Fold {fold_num}: Train={len(train_df)}, Test={len(test_df)}')
            logger.info(f'  {target_cols[0]} MAE: Ridge={ridge["mae_test"]:.3f}, RF={rf["mae_test"]:.3f}, GBT={gbt["mae_test"]:.3f}')
            logger.info(f'  DM vs RF: p={p_rf:.2e}, DM vs GBT: p={p_gbt:.2e}')
        
        test_start += step_size
        fold_num += 1
    
    results_df = pd.DataFrame(results)
    
    logger.info(f'Completed {len(results_df)} folds')
    
    for target in target_cols:
        logger.info(f'{target}:')
        logger.info(f'  Ridge MAE (test): {results_df[f"{target}_ridge_mae_test"].mean():.3f} ± {results_df[f"{target}_ridge_mae_test"].std():.3f}')
        logger.info(f'  RF MAE (test): {results_df[f"{target}_rf_mae_test"].mean():.3f} ± {results_df[f"{target}_rf_mae_test"].std():.3f}')
        logger.info(f'  GBT MAE (test): {results_df[f"{target}_gbt_mae_test"].mean():.3f} ± {results_df[f"{target}_gbt_mae_test"].std():.3f}')
    
    return results_df


def select_champion(cv_results: pd.DataFrame, target: str = 'total') -> str:
    """Select champion model based on statistical significance."""
    avg_p_rf = cv_results[f'{target}_p_rf'].mean()
    avg_p_gbt = cv_results[f'{target}_p_gbt'].mean()
    
    avg_mae_ridge = cv_results[f'{target}_ridge_mae_test'].mean()
    avg_mae_rf = cv_results[f'{target}_rf_mae_test'].mean()
    avg_mae_gbt = cv_results[f'{target}_gbt_mae_test'].mean()
    
    logger.info('='*70)
    logger.info('MODEL SELECTION')
    logger.info('='*70)
    logger.info(f'DM P-values (Ridge vs RF): {avg_p_rf:.2e}')
    logger.info(f'DM P-values (Ridge vs GBT): {avg_p_gbt:.2e}')
    logger.info(f'Average MAE: Ridge={avg_mae_ridge:.3f}, RF={avg_mae_rf:.3f}, GBT={avg_mae_gbt:.3f}')
    
    # Selection logic
    if avg_p_rf < 0.05 and avg_mae_ridge < avg_mae_rf:
        champion = 'ridge'
        logger.info('✅ CHAMPION: Ridge (statistically better than RF)')
    elif avg_p_gbt < 0.05 and avg_mae_ridge < avg_mae_gbt:
        champion = 'ridge'
        logger.info('✅ CHAMPION: Ridge (statistically better than GBT)')
    else:
        # Pick lowest MAE
        maes = {'ridge': avg_mae_ridge, 'rf': avg_mae_rf, 'gbt': avg_mae_gbt}
        champion = min(maes, key=maes.get)
        logger.info(f'⚠️ No statistical significance, picking lowest MAE: {champion}')
    
    return champion


def generate_readout(
    model_type: str,
    cv_results: pd.DataFrame,
    champion: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
) -> str:
    """Generate comprehensive readout."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines = []
    lines.append('='*70)
    lines.append(f'{model_type.upper()} MODEL BACKTEST READOUT')
    lines.append('='*70)
    lines.append(f'Timestamp: {timestamp}')
    lines.append('')
    
    lines.append('DATASET SUMMARY')
    lines.append('-'*70)
    lines.append(f'Total games: {len(df)}')
    lines.append(f'Features ({len(feature_cols)}): {", ".join(feature_cols[:5])}...')
    lines.append(f'Targets: {", ".join(target_cols)}')
    lines.append('')
    
    lines.append('CROSS-VALIDATION RESULTS')
    lines.append('-'*70)
    lines.append(f'Folds: {len(cv_results)}')
    lines.append('')
    
    for target in target_cols:
        lines.append(f'{target.upper()} TARGET:')
        lines.append('')
        lines.append('  Model           | MAE (test)    | RMSE (test)   | R² (test)')
        lines.append('  ' + '-'*68)
        lines.append(f'  Ridge           | {cv_results[f"{target}_ridge_mae_test"].mean():6.3f}         | {cv_results[f"{target}_ridge_rmse_test"].mean():6.3f}        | {cv_results[f"{target}_ridge_r2_test"].mean():.4f}')
        lines.append(f'  Random Forest   | {cv_results[f"{target}_rf_mae_test"].mean():6.3f}         | {cv_results[f"{target}_rf_rmse_test"].mean():6.3f}        | {cv_results[f"{target}_rf_r2_test"].mean():.4f}')
        lines.append(f'  GBT             | {cv_results[f"{target}_gbt_mae_test"].mean():6.3f}         | {cv_results[f"{target}_gbt_rmse_test"].mean():6.3f}        | {cv_results[f"{target}_gbt_r2_test"].mean():.4f}')
        lines.append('')
        lines.append('  DIEBOLD-MARIANO TEST (Ridge as baseline):')
        lines.append(f'  Ridge vs RF:   DM={cv_results[f"{target}_dm_rf"].mean():6.3f},  P-value={cv_results[f"{target}_p_rf"].mean():.2e}')
        lines.append(f'  Ridge vs GBT:  DM={cv_results[f"{target}_dm_gbt"].mean():6.3f},  P-value={cv_results[f"{target}_p_gbt"].mean():.2e}')
        lines.append('')
    
    lines.append('CHAMPION MODEL')
    lines.append('-'*70)
    lines.append(f'Selected: {champion.upper()}')
    lines.append('')
    
    # Significance interpretation
    target = target_cols[0]
    avg_p_rf = cv_results[f'{target}_p_rf'].mean()
    avg_p_gbt = cv_results[f'{target}_p_gbt'].mean()
    
    if avg_p_rf < 0.05 and avg_p_gbt < 0.05:
        significance = 'HIGH - Ridge is statistically superior to both RF and GBT'
    elif avg_p_rf < 0.05 or avg_p_gbt < 0.05:
        significance = 'MODERATE - Ridge is statistically better than at least one competitor'
    else:
        significance = 'LOW - No significant difference, selected based on lowest MAE'
    
    lines.append(f'Statistical Significance: {significance}')
    lines.append('')
    
    lines.append('='*70)
    
    return '\n'.join(lines)


def train_champion(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    champion: str,
    output_prefix: str,
) -> Dict:
    """Train champion model on full dataset."""
    logger.info('='*70)
    logger.info('TRAINING CHAMPION ON FULL DATASET')
    logger.info('='*70)
    
    X = df[feature_cols].values
    
    models = {}
    
    for target in target_cols:
        y = df[target].values
        
        if champion == 'ridge':
            model = Ridge(alpha=2.0, random_state=42, solver='auto')
        elif champion == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
        
        model.fit(X, y)
        models[target] = model
        
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        
        logger.info(f'{target}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.4f}')
    
    # Save models
    Path('models_v3').mkdir(exist_ok=True)
    Path(f'models_v3/{output_prefix}').mkdir(exist_ok=True)
    
    model_info = {
        'champion': champion,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'models': {},
    }
    
    for target, model in models.items():
        model_path = f'models_v3/{output_prefix}/{champion}_{target}.joblib'
        joblib.dump({'model': model, 'features': feature_cols}, model_path)
        model_info['models'][target] = model_path
        logger.info(f'Saved {champion} {target} model -> {model_path}')
    
    return model_info


def backtest_pregame():
    """Run backtest for pregame model."""
    logger.info('='*70)
    logger.info('PREGAME MODEL BACKTEST')
    logger.info('='*70)
    
    df = pd.read_parquet('data/processed/pregame_team_v2.parquet')
    logger.info(f'Loaded {len(df)} games')
    
    feature_cols = [c for c in df.columns if c in ['home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
                                                      'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp',
                                                      'home_fga', 'home_fgm', 'away_fga', 'away_fgm']]
    target_cols = ['total', 'margin']
    
    cv_results = walk_forward_cv(df, feature_cols, target_cols, min_train_size=1000, test_size=200, step_size=200)
    champion = select_champion(cv_results, 'total')
    
    readout = generate_readout('Pregame', cv_results, champion, df, feature_cols, target_cols)
    print(readout)
    
    model_info = train_champion(df, feature_cols, target_cols, champion, 'pregame')
    
    cv_results.to_parquet('data/processed/pregame_cv_results.parquet', index=False)
    
    with open('data/processed/pregame_readout.txt', 'w') as f:
        f.write(readout)
    
    return cv_results, champion, readout


def backtest_q3():
    """Run backtest for Q3 model."""
    logger.info('='*70)
    logger.info('Q3 MODEL BACKTEST')
    logger.info('='*70)
    
    df = pd.read_parquet('data/processed/q3_team_v2.parquet')
    logger.info(f'Loaded {len(df)} games')
    
    feature_cols = [c for c in df.columns if c.startswith('q3_') or c.endswith('_efg') or c.endswith('_ftr') or c.endswith('_tpar') or c.endswith('_tor') or c.endswith('_orbp')]
    target_cols = ['total', 'margin']
    
    cv_results = walk_forward_cv(df, feature_cols, target_cols, min_train_size=500, test_size=200, step_size=200)
    champion = select_champion(cv_results, 'total')
    
    readout = generate_readout('Q3', cv_results, champion, df, feature_cols, target_cols)
    print(readout)
    
    model_info = train_champion(df, feature_cols, target_cols, champion, 'q3')
    
    cv_results.to_parquet('data/processed/q3_cv_results.parquet', index=False)
    
    with open('data/processed/q3_readout.txt', 'w') as f:
        f.write(readout)
    
    return cv_results, champion, readout


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['pregame', 'q3', 'all'], default='all')
    args = parser.parse_args()
    
    if args.model in ['pregame', 'all']:
        logger.info('Running Pregame backtest...')
        backtest_pregame()
    
    if args.model in ['q3', 'all']:
        logger.info('Running Q3 backtest...')
        backtest_q3()


if __name__ == '__main__':
    main()
