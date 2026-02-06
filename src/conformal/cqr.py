"""
Conformalized Quantile Regression (CQR) for uncertainty quantification.

Implements CQR to generate prediction intervals with valid coverage.
Uses split-conformal approach: train on calibration set,
then compute quantiles on holdout set.

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 6
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


def fit_quantile_regressor(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float,
    solver: str = "highs"
) -> object:
    """
    Fit quantile regression model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        quantile: Quantile to predict (0.0 to 1.0)
        solver: Solver for quantile regression
    
    Returns:
        Fitted quantile regression model
    """
    model = QuantileRegressor(quantile=quantile, solver=solver, alpha=0.0)
    model.fit(X, y)
    return model


def conformalized_quantile_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    random_state: Optional[int] = None,
    solver: str = "highs",
    test_size: float = 0.2
) -> dict:
    """
    Perform Conformalized Quantile Regression (CQR).
    
    CQR steps:
    1. Split data into train and calibration sets
    2. Fit lower and upper quantile regressors on train set
    3. Predict intervals on calibration set
    4. Compute non-conformity scores on calibration set
    5. Compute quantile of non-conformity scores
    6. Use quantile to adjust intervals for test set
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        alpha: Miscoverage rate (default: 0.1 for 90% coverage)
        random_state: Random seed for reproducibility
        solver: Solver for quantile regression
        test_size: Fraction of data for calibration set
    
    Returns:
        Dictionary with:
        - lower_model: Fitted lower quantile regressor
        - upper_model: Fitted upper quantile regressor
        - cal_q: Conformality quantile
        - train_idx: Indices of training set
        - cal_idx: Indices of calibration set
        - cal_lower: Lower quantile predictions on calibration set
        - cal_upper: Upper quantile predictions on calibration set
        - cal_scores: Non-conformity scores
        - interval_width: Average interval width on calibration set
    """
    n = len(X)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Step 1: Split data into train and calibration sets
    train_idx, cal_idx = train_test_split(
        np.arange(n), test_size=test_size, random_state=random_state
    )
    
    X_train, X_cal = X[train_idx], X[cal_idx]
    y_train, y_cal = y[train_idx], y[cal_idx]
    
    # Step 2: Fit lower and upper quantile regressors on train set
    lower_model = fit_quantile_regressor(
        X_train, y_train, quantile=alpha/2, solver=solver
    )
    upper_model = fit_quantile_regressor(
        X_train, y_train, quantile=1 - alpha/2, solver=solver
    )
    
    # Step 3: Predict intervals on calibration set
    cal_lower = lower_model.predict(X_cal)
    cal_upper = upper_model.predict(X_cal)
    
    # Step 4: Compute non-conformity scores on calibration set
    # Score E = max(undercoverage, overcoverage)
    # undercoverage = y_lower - y
    # overcoverage = y - y_upper
    cal_scores = np.maximum(
        cal_lower - y_cal,  # Undercoverage
        y_cal - cal_upper   # Overcoverage
    )
    
    # Step 5: Compute quantile of non-conformity scores
    # q_hat = quantile of scores with confidence 1 - alpha
    # Use (n_cal + 1) -th quantile for exact coverage
    n_cal = len(cal_scores)
    cal_q = np.quantile(cal_scores, 1 - alpha, method='higher')
    
    # Compute average interval width on calibration set
    interval_width = np.mean(cal_upper - cal_lower + 2 * cal_q)
    
    return {
        "lower_model": lower_model,
        "upper_model": upper_model,
        "cal_q": cal_q,
        "train_idx": train_idx,
        "cal_idx": cal_idx,
        "cal_lower": cal_lower,
        "cal_upper": cal_upper,
        "cal_scores": cal_scores,
        "interval_width": interval_width,
        "alpha": alpha,
        "coverage_target": 1 - alpha,
    }


def predict_intervals(
    cqr_results: dict,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate prediction intervals for new data using CQR.
    
    Args:
        cqr_results: Results from conformalized_quantile_regression()
        X_test: Feature matrix for test set (n_samples, n_features)
    
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    lower_model = cqr_results["lower_model"]
    upper_model = cqr_results["upper_model"]
    cal_q = cqr_results["cal_q"]
    
    # Predict quantiles
    lower_pred = lower_model.predict(X_test)
    upper_pred = upper_model.predict(X_test)
    
    # Adjust with conformality quantile
    lower_adj = lower_pred - cal_q
    upper_adj = upper_pred + cal_q
    
    return lower_adj, upper_adj


def evaluate_coverage(
    y_true: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray
) -> dict:
    """
    Evaluate coverage of prediction intervals.
    
    Args:
        y_true: True target values (n_samples,)
        lower_pred: Lower bound predictions (n_samples,)
        upper_pred: Upper bound predictions (n_samples,)
    
    Returns:
        Dictionary with coverage metrics
    """
    # Check if y_true is within interval
    covered = (y_true >= lower_pred) & (y_true <= upper_pred)
    coverage = np.mean(covered)
    
    # Compute average interval width
    interval_width = np.mean(upper_pred - lower_pred)
    
    # Compute sharpness (variance of interval widths)
    width_var = np.var(upper_pred - lower_pred)
    
    return {
        "coverage": float(coverage),
        "interval_width": float(interval_width),
        "width_std": float(np.sqrt(width_var)),
        "n_samples": len(y_true),
    }


if __name__ == "__main__":
    # Test CQR with current dataset
    print("Testing Conformalized Quantile Regression...")
    
    import sys
    sys.path.insert(0, "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3")
    
    df = pd.read_parquet(
        "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/data/processed/halftime_with_temporal_features_total.parquet"
    )
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Select features and target
    h1_features = [col for col in df.columns if col.startswith('h1_')]
    target = 'h2_total'
    
    print(f"\nUsing {len(h1_features)} h1_* features")
    print(f"Target: {target}")
    
    # Prepare data
    X = df[h1_features].values
    y = df[target].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Run CQR
    cqr_results = conformalized_quantile_regression(
        X, y, alpha=0.1, random_state=42, test_size=0.2
    )
    
    print(f"\nCQR Results:")
    print(f"  Alpha (miscoverage rate): {cqr_results['alpha']:.2f}")
    print(f"  Target coverage: {cqr_results['coverage_target']:.1%}")
    print(f"  Calibration quantile: {cqr_results['cal_q']:.4f}")
    print(f"  Average interval width: {cqr_results['interval_width']:.4f}")
    print(f"  Training samples: {len(cqr_results['train_idx'])}")
    print(f"  Calibration samples: {len(cqr_results['cal_idx'])}")
    
    # Generate intervals for calibration set
    X_cal = X[cqr_results['cal_idx']]
    y_cal = y[cqr_results['cal_idx']]
    
    lower_pred, upper_pred = predict_intervals(cqr_results, X_cal)
    
    # Evaluate coverage
    coverage_results = evaluate_coverage(y_cal, lower_pred, upper_pred)
    print(f"\nCoverage Evaluation (Calibration Set):")
    print(f"  Coverage: {coverage_results['coverage']:.1%}")
    print(f"  Interval width (mean): {coverage_results['interval_width']:.4f}")
    print(f"  Interval width (std): {coverage_results['width_std']:.4f}")
    
    if abs(coverage_results['coverage'] - cqr_results['coverage_target']) < 0.05:
        print(f"\n✅ Coverage close to target ({cqr_results['coverage_target']:.1%})")
    else:
        print(f"\n⚠️ Coverage deviates from target ({cqr_results['coverage_target']:.1%})")
