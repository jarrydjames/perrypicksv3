"""
Diebold-Mariano test for forecast accuracy.

Implements the DM test for comparing two competing forecast models.
Uses Newey-West variance estimate to handle autocorrelation in loss
differentials.

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 5.4
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm


def newey_west_variance(loss_differentials: np.ndarray, lags: int = 5) -> float:
    """
    Compute Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent)
    variance estimate.
    
    Newey-West corrects for autocorrelation in time series by applying
    a kernel that downweights distant lags.
    
    Args:
        loss_differentials: Array of loss differentials (L_new - L_baseline)
                            Shape: (n_games,)
        lags: Number of lags to consider (default: 5)
    
    Returns:
        Newey-West variance estimate
    """
    n = len(loss_differentials)
    
    if n <= lags:
        # Fall back to regular variance if too few observations
        return np.var(loss_differentials)
    
    # Compute sample variance
    sample_var = np.var(loss_differentials)
    
    # Compute autocovariances
    autocovs = np.zeros(lags)
    for lag in range(1, lags + 1):
        if lag < n:
            # Mean for current lag
            mean_first = np.mean(loss_differentials[lag:])
            mean_second = np.mean(loss_differentials[:n-lag])
            autocovs[lag-1] = np.sum(
                (loss_differentials[lag:] - mean_first) *
                (loss_differentials[:n-lag] - mean_second)
            ) / n
    
    # Newey-West estimator (Bartlett kernel)
    # NW = var + 2 * sum(w_k * autocov_k) where w_k = (1 - k/(m+1))
    # This is a simplified version
    nw_variance = sample_var
    for lag, autocov in enumerate(autocovs, start=1):
        # Bartlett kernel weight
        weight = 1.0 - (lag / (lags + 1))
        nw_variance += 2.0 * weight * autocov
    
    return nw_variance


def diebold_mariano_test(
    loss_baseline: np.ndarray,
    loss_new: np.ndarray,
    lags: int = 5,
    test_type: str = "two-sided"
) -> dict:
    """
    Perform Diebold-Mariano test for forecast accuracy.
    
    Tests null hypothesis that both models have equal forecast accuracy.
    Alternative hypothesis is that one model is significantly better.
    
    Args:
        loss_baseline: Array of baseline loss values (L_baseline)
                            Shape: (n_games,)
        loss_new: Array of new model loss values (L_new)
                            Shape: (n_games,)
        lags: Number of lags for Newey-West variance (default: 5)
        test_type: Type of test ('two-sided', 'less', 'greater')
    
    Returns:
        Dictionary with:
        - dm_statistic: DM test statistic
        - p_value: P-value for the test
        - significant: Whether p < 0.05 (significant at 5% level)
        - mean_diff: Mean of loss differentials (L_new - L_baseline)
        - nw_variance: Newey-West variance estimate
        - test_type: Test type used
    """
    # Check inputs
    n = len(loss_baseline)
    if n != len(loss_new):
        return {
            "dm_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "mean_diff": np.nan,
            "nw_variance": np.nan,
            "test_type": test_type,
            "error": "Loss arrays must have same length",
        }
    
    # Compute loss differentials: d_i = L_new - L_baseline
    loss_differentials = loss_new - loss_baseline
    mean_diff = np.mean(loss_differentials)
    
    # Compute Newey-West variance (HAC estimate)
    nw_variance = newey_west_variance(loss_differentials, lags=lags)
    
    # Avoid division by zero
    if nw_variance <= 0:
        return {
            "dm_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "mean_diff": float(mean_diff),
            "nw_variance": float(nw_variance),
            "test_type": test_type,
            "error": "Newey-West variance <= 0",
        }
    
    # Compute DM statistic (normalized difference of means)
    dm_statistic = mean_diff / np.sqrt(nw_variance / n)
    
    # Compute p-value using standard normal distribution
    if test_type == "two-sided":
        p_value = 2.0 * (1.0 - norm.cdf(abs(dm_statistic)))
    elif test_type == "less":
        p_value = norm.cdf(dm_statistic)  # H1: mean_diff < 0 (new model better)
    elif test_type == "greater":
        p_value = 1.0 - norm.cdf(dm_statistic)  # H1: mean_diff > 0 (baseline better)
    else:
        p_value = np.nan
    
    # Determine significance (p < 0.05)
    significant = p_value < 0.05
    
    return {
        "dm_statistic": float(dm_statistic),
        "p_value": float(p_value),
        "significant": significant,
        "mean_diff": float(mean_diff),
        "nw_variance": float(nw_variance),
        "test_type": test_type,
    }


def diebold_mariano_summary(dm_results: dict) -> str:
    """
    Generate human-readable summary of Diebold-Mariano test results.
    """
    lines = [
        "=" * 80,
        "DIEBOLD-MARIANO TEST RESULTS",
        "=" * 80,
        "",
        f"DM Statistic: {dm_results['dm_statistic']:.4f}",
        f"P-Value: {dm_results['p_value']:.4f}",
        f"Significance (p < 0.05): {'YES' if dm_results['significant'] else 'NO'}",
        f"",
        f"Mean Loss Differential (L_new - L_baseline): {dm_results['mean_diff']:.4f}",
        f"Newey-West Variance: {dm_results['nw_variance']:.4f}",
        f"Test Type: {dm_results['test_type']}",
        "",
    ]
    
    if 'error' in dm_results:
        lines.append(f"ERROR: {dm_results['error']}")
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test Diebold-Mariano
    np.random.seed(42)
    
    n = 100
    # Simulate losses: baseline has higher MAE (worse)
    loss_baseline = np.random.normal(loc=10.0, scale=1.0, size=n)
    # New model has lower MAE (better, mean_diff = -0.5)
    loss_new = np.random.normal(loc=9.5, scale=1.0, size=n)
    
    print("Testing Diebold-Mariano test...")
    print(f"Baseline MAE: {np.mean(loss_baseline):.4f}")
    print(f"New MAE: {np.mean(loss_new):.4f}")
    print(f"Mean Difference: {np.mean(loss_new - loss_baseline):.4f}")
    
    results = diebold_mariano_test(loss_baseline, loss_new)
    print(diebold_mariano_summary(results))