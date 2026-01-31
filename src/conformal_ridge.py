"""
Conformal Calibration for Ridge Regression

Implements calibration-set conformal prediction intervals
to provide uncertainty estimates for Ridge predictions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def conformal_calibration_set(
    y_true_calib: np.ndarray,
    y_pred_calib: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.10,
) -> tuple:
    """
    Perform calibration-set conformal prediction intervals.
    
    Uses calibration set to compute conformal quantile, then applies to test set.
    
    Args:
        y_true_calib: Ground truth values for calibration set
        y_pred_calib: Point predictions for calibration set
        y_true_test: Ground truth values for test set
        y_pred_test: Point predictions for test set
        alpha: Target error rate (e.g., 0.10 for 90% intervals)
    
    Returns:
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        quantile: Conformal quantile used
    """
    # Compute calibration residuals
    calib_residuals = np.abs(y_true_calib - y_pred_calib)
    
    # Compute conformal quantile from calibration set
    # For 90% coverage, use 90th percentile (1 - alpha)
    quantile = np.quantile(calib_residuals, 1 - alpha)
    
    # Apply conformal interval to test predictions
    lower_bounds = y_pred_test - quantile
    upper_bounds = y_pred_test + quantile
    
    return lower_bounds, upper_bounds, quantile


def evaluate_coverage(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> dict:
    """
    Evaluate coverage of prediction intervals.
    
    Args:
        y_true: Ground truth values
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
    
    Returns:
        Dictionary with coverage metrics
    """
    # Check if each ground truth falls within interval
    in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    
    # Coverage rate
    coverage = np.mean(in_interval)
    
    # Interval widths
    widths = upper_bounds - lower_bounds
    mean_width = np.mean(widths)
    std_width = np.std(widths)
    
    return {
        'coverage': coverage,
        'in_interval': in_interval,
        'mean_width': mean_width,
        'std_width': std_width,
        'widths': widths,
    }


def conformal_ridge_walkforward(
    df: pd.DataFrame,
    h1_features: list,
    fold_indices: list,
    alpha: float = 0.10,
    calib_size: int = 500,
) -> dict:
    """
    Perform conformal calibration on Ridge predictions across all folds.
    
    Uses calibration set approach (last calib_size games of training set).
    
    Args:
        df: DataFrame with h1 features and h2_total target
        h1_features: List of h1 feature column names
        fold_indices: List of fold indices (each has train_games, test_games)
        alpha: Target error rate (0.10 for 90% intervals)
        calib_size: Size of calibration set from training data
    
    Returns:
        Dictionary with conformal results for each fold
    """
    results = {}
    
    for fold in fold_indices:
        fold_id = fold['fold_id']
        
        # Get train/test data
        train_mask = df['game_id'].isin(fold['train_games'])
        test_mask = df['game_id'].isin(fold['test_games'])
        
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        # Use last calib_size games of training as calibration set
        df_calib = df_train.tail(calib_size)
        df_train_main = df_train.head(len(df_train) - calib_size)
        
        X_train = df_train_main[h1_features].values
        y_train = df_train_main['h2_total'].values
        
        X_calib = df_calib[h1_features].values
        y_calib = df_calib['h2_total'].values
        
        X_test = df_test[h1_features].values
        y_test = df_test['h2_total'].values
        
        # Train Ridge model on main training data
        ridge = Ridge(alpha=2.0, random_state=42)
        ridge.fit(X_train, y_train)
        
        # Generate predictions for calibration and test sets
        y_pred_calib = ridge.predict(X_calib)
        y_pred_test = ridge.predict(X_test)
        
        # Perform conformal calibration
        lower_bounds, upper_bounds, quantile = conformal_calibration_set(
            y_calib, y_pred_calib,
            y_test, y_pred_test,
            alpha=alpha,
        )
        
        # Evaluate coverage
        coverage_metrics = evaluate_coverage(y_test, lower_bounds, upper_bounds)
        
        # Store results
        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred_test,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'quantile': quantile,
            'coverage': coverage_metrics['coverage'],
            'mean_width': coverage_metrics['mean_width'],
            'std_width': coverage_metrics['std_width'],
        }
    
    return results


def generate_coverage_table(conformal_results: dict) -> pd.DataFrame:
    """
    Generate coverage table from conformal results.
    
    Args:
        conformal_results: Dictionary of conformal results per fold
    
    Returns:
        DataFrame with coverage and width metrics per fold
    """
    rows = []
    
    for fold_id, fold_results in conformal_results.items():
        rows.append({
            'fold_id': fold_id,
            'coverage': fold_results['coverage'],
            'target_coverage': 0.90,
            'mean_width': fold_results['mean_width'],
            'std_width': fold_results['std_width'],
            'quantile': fold_results['quantile'],
            'n_predictions': len(fold_results['y_true']),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('fold_id')
    
    # Calculate overall coverage
    all_in_interval = np.concatenate([
        evaluate_coverage(
            fold_results['y_true'],
            fold_results['lower_bounds'],
            fold_results['upper_bounds'],
        )['in_interval']
        for fold_results in conformal_results.values()
    ])
    overall_coverage = np.mean(all_in_interval)
    
    # Add overall row
    overall_row = {
        'fold_id': 'OVERALL',
        'coverage': overall_coverage,
        'target_coverage': 0.90,
        'mean_width': np.mean([r['mean_width'] for r in conformal_results.values()]),
        'std_width': np.mean([r['std_width'] for r in conformal_results.values()]),
        'quantile': np.mean([r['quantile'] for r in conformal_results.values()]),
        'n_predictions': sum([len(r['y_true']) for r in conformal_results.values()]),
    }
    
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    
    return df


def generate_interval_width_table(conformal_results: dict) -> pd.DataFrame:
    """
    Generate interval width statistics table.
    
    Args:
        conformal_results: Dictionary of conformal results per fold
    
    Returns:
        DataFrame with interval width statistics per fold
    """
    rows = []
    
    for fold_id, fold_results in conformal_results.items():
        widths = fold_results['upper_bounds'] - fold_results['lower_bounds']
        
        rows.append({
            'fold_id': fold_id,
            'min_width': np.min(widths),
            'max_width': np.max(widths),
            'mean_width': np.mean(widths),
            'median_width': np.median(widths),
            'std_width': np.std(widths),
            'p25_width': np.percentile(widths, 25),
            'p75_width': np.percentile(widths, 75),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('fold_id')
    
    # Calculate overall stats
    all_widths = np.concatenate([
        fold_results['upper_bounds'] - fold_results['lower_bounds']
        for fold_results in conformal_results.values()
    ])
    
    overall_row = {
        'fold_id': 'OVERALL',
        'min_width': np.min(all_widths),
        'max_width': np.max(all_widths),
        'mean_width': np.mean(all_widths),
        'median_width': np.median(all_widths),
        'std_width': np.std(all_widths),
        'p25_width': np.percentile(all_widths, 25),
        'p75_width': np.percentile(all_widths, 75),
    }
    
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    
    return df
