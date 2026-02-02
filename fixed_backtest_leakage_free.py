"""Leakage-free historical backtest with strict temporal constraints.

This script fixes the data leakage issue in the original backtest by:
1. Sorting by DATE (not game_id)
2. Ensuring strict temporal separation
3. Only using pregame features (season averages)

Expected outcome: Much higher MAE than 3.51, but realistic.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# sklearn imports
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def walk_forward_cv_strict(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    min_train_size: int = 500,
    test_size: int = 200,
    step_size: int = 200,
) -> pd.DataFrame:
    """
    Perform walk-forward temporal cross-validation with STRICT temporal constraints.
    
    CRITICAL FIX: Sort by DATE and ensure no future data in training.
    """
    logger.info('='*70)
    logger.info('WALK-FORWARD TEMPORAL CROSS-VALIDATION (STRICT, LEAKAGE-FREE)')
    logger.info('='*70)
    logger.info(f'Min train size: {min_train_size}')
    logger.info(f'Test size: {test_size}')
    logger.info(f'Step size: {step_size}')
    
    # FIX: Sort by DATE, not game_id!
    df_sorted = df.sort_values('game_date').reset_index(drop=True)
    logger.info(f"Sorted {len(df_sorted)} games by date")
    
    results = []
    fold_num = 0
    
    # Start with minimum training size
    train_end_idx = min_train_size
    
    while train_end_idx + test_size + step_size <= len(df_sorted):
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_size
        
        # CRITICAL: Ensure temporal separation by checking dates
        train_df = df_sorted.iloc[:train_end_idx]
        test_df = df_sorted.iloc[test_start_idx:test_end_idx]
        
        # Verify no date overlap
        train_dates = train_df['game_date'].unique()
        test_dates = test_df['game_date'].unique()
        
        max_train_date = train_dates.max()
        min_test_date = test_dates.min()
        
        if max_train_date >= min_test_date:
            logger.warning(f"Fold {fold_num}: Date overlap detected! Train max: {max_train_date}, Test min: {min_test_date}")
            # Skip this fold
            train_end_idx += step_size
            fold_num += 1
            continue
        
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        results_fold = {
            'fold': fold_num,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_date_range': (train_dates.min(), train_dates.max()),
            'test_date_range': (test_dates.min(), test_dates.max()),
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
        
        results.append(results_fold)
        
        if fold_num % 2 == 0:
            logger.info(f"Fold {fold_num}: Train={len(train_df)}, Test={len(test_df)}")
            logger.info(f"  Train dates: {train_dates.min()} to {train_dates.max()}")
            logger.info(f"  Test dates: {test_dates.min()} to {test_dates.max()}")
            logger.info(f"  {target_cols[0]} MAE: Ridge={ridge['mae_test']:.3f}, RF={rf['mae_test']:.3f}, GBT={gbt['mae_test']:.3f}")
        
        train_end_idx += step_size
        fold_num += 1
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"Completed {len(results_df)} folds")
    
    for target in target_cols:
        logger.info(f"{target}:")
        logger.info(f"  Ridge MAE (test): {results_df[f'{target}_ridge_mae_test'].mean():.3f} ± {results_df[f'{target}_ridge_mae_test'].std():.3f}")
        logger.info(f"  RF MAE (test): {results_df[f'{target}_rf_mae_test'].mean():.3f} ± {results_df[f'{target}_rf_mae_test'].std():.3f}")
        logger.info(f"  GBT MAE (test): {results_df[f'{target}_gbt_mae_test'].mean():.3f} ± {results_df[f'{target}_gbt_mae_test'].std():.3f}")
    
    return results_df


def load_pregame_dataset() -> pd.DataFrame:
    """Load pregame dataset from parquet file."""
    try:
        df = pd.read_parquet('data/processed/pregame_team_v2.parquet')
        logger.info(f"Loaded {len(df)} games from pregame dataset")
        
        # Check if date column exists
        if 'game_date' not in df.columns and 'date' in df.columns:
            df['game_date'] = pd.to_datetime(df['date'])
        elif 'game_date' not in df.columns:
            logger.error("No date column found in dataset!")
            return None
        
        return df
    except Exception as e:
        logger.error(f"Error loading pregame dataset: {e}")
        return None


def backtest_pregame_leakage_free():
    """Run leakage-free backtest for pregame model."""
    logger.info('='*70)
    logger.info('PREGAME MODEL BACKTEST - LEAKAGE-FREE VERSION')
    logger.info('='*70)
    
    df = load_pregame_dataset()
    
    if df is None or len(df) == 0:
        logger.error("Failed to load dataset")
        return None
    
    # Check for required columns
    required_features = ['home_efg', 'away_efg', 'home_ftr', 'away_ftr', 
                       'home_tpar', 'away_tpar', 'home_tor', 'away_tor',
                       'home_orbp', 'away_orbp', 'home_fga', 'away_fga',
                       'home_fgm', 'away_fgm']
    required_targets = ['total', 'margin']
    
    missing_features = [f for f in required_features if f not in df.columns]
    missing_targets = [t for t in required_targets if t not in df.columns]
    
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        return None
    
    if missing_targets:
        logger.error(f"Missing targets: {missing_targets}")
        return None
    
    feature_cols = required_features
    target_cols = required_targets
    
    logger.info(f"Features ({len(feature_cols)}): {', '.join(feature_cols[:5])}...")
    logger.info(f"Targets: {target_cols}")
    
    # Run walk-forward CV with strict temporal constraints
    cv_results = walk_forward_cv_strict(df, feature_cols, target_cols, 
                                         min_train_size=500, 
                                         test_size=200, 
                                         step_size=200)
    
    if cv_results is None or len(cv_results) == 0:
        logger.error("No folds completed")
        return None
    
    # Select champion (lowest MAE on test)
    avg_mae_ridge = cv_results['total_ridge_mae_test'].mean()
    avg_mae_rf = cv_results['total_rf_mae_test'].mean()
    avg_mae_gbt = cv_results['total_gbt_mae_test'].mean()
    
    logger.info('='*70)
    logger.info('MODEL SELECTION (LOWEST TEST MAE)')
    logger.info('='*70)
    logger.info(f'Average MAE: Ridge={avg_mae_ridge:.3f}, RF={avg_mae_rf:.3f}, GBT={avg_mae_gbt:.3f}')
    
    champion = 'ridge'
    if avg_mae_rf < avg_mae_ridge:
        champion = 'rf'
    if avg_mae_gbt < min(avg_mae_ridge, avg_mae_rf):
        champion = 'gbt'
    
    logger.info(f'CHAMPION: {champion.upper()}')
    
    # Generate readout
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines = []
    lines.append('='*70)
    lines.append(f'PREGAME MODEL BACKTEST - LEAKAGE-FREE')
    lines.append('='*70)
    lines.append(f'Timestamp: {timestamp}')
    lines.append('')
    lines.append('DATASET SUMMARY')
    lines.append('-'*70)
    lines.append(f'Total games: {len(df)}')
    lines.append(f'Features ({len(feature_cols)}): {", ".join(feature_cols[:5])}...')
    lines.append(f'Targets: {target_cols}')
    lines.append('')
    lines.append('CROSS-VALIDATION RESULTS (STRICT TEMPORAL)')
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
    
    lines.append('CHAMPION MODEL')
    lines.append('-'*70)
    lines.append(f'Selected: {champion.upper()}')
    lines.append('')
    lines.append('='*70)
    
    readout = '\n'.join(lines)
    print(readout)
    
    # Save results
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    cv_results.to_parquet(output_dir / 'pregame_cv_leakage_free.parquet', index=False)
    
    with open(output_dir / 'pregame_readout_leakage_free.txt', 'w') as f:
        f.write(readout)
    
    logger.info(f"Saved results to {output_dir}")
    
    return cv_results, champion, readout


def main():
    """Main entry point."""
    result = backtest_pregame_leakage_free()
    
    if result is None:
        logger.error("Backtest failed")
        return
    
    logger.info("="*70)
    logger.info("LEAKAGE-FREE BACKTEST COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
