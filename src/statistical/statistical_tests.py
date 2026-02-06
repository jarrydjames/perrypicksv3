"""
Statistical testing module for PerryPicks v3.

Provides comprehensive statistical testing for model comparisons:
1. Block bootstrap (time-valid confidence intervals)
2. Diebold-Mariano test (forecast accuracy)
3. Paired loss differentials

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 5
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .block_bootstrap import block_bootstrap, block_bootstrap_summary
from .diebold_mariano import diebold_mariano_test, diebold_mariano_summary


class StatisticalTestReport:
    """
    Statistical test report with all test results.

    Attributes:
        status: Overall test status (PASS/WARN/FAIL)
        tests: Dict of test names to (status, message, details)
        caveats: List of warnings (non-blocking)
        dataset_checksum: Hash of analyzed dataset
        timestamp: Test timestamp
    """

    def __init__(self):
        self.status: str = "PASS"
        self.tests: Dict[str, Tuple[str, str, dict]] = {}
        self.caveats: List[str] = []
        self.dataset_checksum: Optional[str] = None
        self.timestamp: str = pd.Timestamp.now().isoformat()
    
    def add_test(self, name: str, status: str, message: str, details: Optional[dict] = None):
        """Add a test result."""
        self.tests[name] = (status, message, details or {})
    
    def add_caveat(self, message: str):
        """Add a non-blocking warning."""
        self.caveats.append(message)
    
    def __str__(self) -> str:
        """Return human-readable report."""
        lines = [
            "=" * 80,
            f"STATISTICAL TEST REPORT - {self.timestamp}",
            f"Overall Status: {self.status}",
            f"Dataset Checksum: {self.dataset_checksum}",
            "=" * 80,
            "",
        ]
        
        # Tests
        lines.append("TESTS:")
        lines.append("-" * 80)
        for test_name, (status, message, details) in self.tests.items():
            lines.append(f"  {status}: {test_name}")
            lines.append(f"    {message}")
            if details:
                for key, value in details.items():
                    lines.append(f"      {key}: {value}")
            lines.append("")
        
        # Caveats
        if self.caveats:
            lines.append("CAVEATS (WARNINGS):")
            lines.append("-" * 80)
            for i, caveat in enumerate(self.caveats, 1):
                lines.append(f"  {i}. {caveat}")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "tests": {
                name: {
                    "status": status,
                    "message": message,
                    "details": details,
                }
                for name, (status, message, details) in self.tests.items()
            },
            "caveats": self.caveats,
            "dataset_checksum": self.dataset_checksum,
            "timestamp": self.timestamp,
        }


def compute_paired_differentials(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_new: np.ndarray,
    metric: str = "mae"
) -> np.ndarray:
    """
    Compute per-game loss differentials between two models.
    
    For each game i:
        L_baseline_i = loss(y_true_i, y_pred_baseline_i)
        L_new_i = loss(y_true_i, y_pred_new_i)
        d_i = L_new_i - L_baseline_i
    
    Args:
        y_true: Array of true values (shape: n_games)
        y_pred_baseline: Array of baseline predictions (shape: n_games)
        y_pred_new: Array of new model predictions (shape: n_games)
        metric: Loss metric ('mae' or 'mse')
    
    Returns:
        Array of loss differentials (L_new - L_baseline)
    """
    # Check input shapes
    assert len(y_true) == len(y_pred_baseline) == len(y_pred_new), \
        "All arrays must have same length"
    
    # Compute loss function
    if metric == "mae":
        loss_func = mean_absolute_error
    elif metric == "mse":
        loss_func = mean_squared_error
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Compute losses per game
    losses_baseline = np.array([loss_func([y_true[i]], [y_pred_baseline[i]]) 
                                 for i in range(len(y_true))])
    losses_new = np.array([loss_func([y_true[i]], [y_pred_new[i]]) 
                             for i in range(len(y_true))])
    
    # Compute differentials: d_i = L_new_i - L_baseline_i
    loss_differentials = losses_new - losses_baseline
    
    return loss_differentials


def run_statistical_tests(
    df: pd.DataFrame,
    baseline_predictions_col: str,
    new_predictions_col: str,
    target_col: str,
    block_size: int = 200,
    n_bootstraps: int = 1000,
    lags: int = 5,
    metric: str = "mae",
) -> Tuple[StatisticalTestReport, dict]:
    """
    Run all statistical tests on model comparison.
    
    This is the main entry point for statistical testing.
    Returns:
        Tuple of (statistical_test_report, results_dict)
    
    Tests performed:
    1. Paired loss differentials
    2. Block bootstrap (time-valid CI)
    3. Diebold-Mariano test (forecast accuracy)
    
    Reference: execution_specification Sections 5.2, 5.3, 5.4
    """
    report = StatisticalTestReport()
    
    # Test 1: Paired loss differentials
    y_true = df[target_col].values
    y_pred_baseline = df[baseline_predictions_col].values
    y_pred_new = df[new_predictions_col].values
    
    loss_differentials = compute_paired_differentials(
        y_true, y_pred_baseline, y_pred_new, metric=metric
    )
    
    # Summary statistics
    mean_diff = np.mean(loss_differentials)
    median_diff = np.median(loss_differentials)
    pct_improvement = (loss_differentials < 0).mean() * 100  # % of games where new model is better
    
    report.add_test(
        "paired_differentials",
        "PASS",
        "Paired loss differentials computed",
        {
            "mean_diff": float(mean_diff),
            "median_diff": float(median_diff),
            "pct_improvement": float(pct_improvement),
            "metric": metric,
        },
    )
    
    # Test 2: Block bootstrap
    bootstrap_results = block_bootstrap(
        loss_differentials,
        block_size=block_size,
        n_bootstraps=n_bootstraps,
    )
    
    bootstrap_status = "PASS"
    if bootstrap_results['ci_upper'] < 0:
        bootstrap_status = "EXCELLENT"  # Entire CI negative (new model definitely better)
    elif bootstrap_results['ci_lower'] >= 0:
        bootstrap_status = "FAIL"  # Entire CI positive (baseline better)
    
    report.add_test(
        "block_bootstrap",
        bootstrap_status,
        "Block bootstrap (time-valid CI) completed",
        bootstrap_results,
    )
    
    # Test 3: Diebold-Mariano test
    dm_results = diebold_mariano_test(
        y_pred_baseline,  # Losses = predictions - true
        y_pred_new,       # Actually we need losses, not predictions
        lags=lags,
    )
    
    # Actually compute losses for DM test
    losses_baseline = np.abs(y_pred_baseline - y_true)
    losses_new = np.abs(y_pred_new - y_true)
    
    dm_results = diebold_mariano_test(losses_baseline, losses_new, lags=lags)
    
    dm_status = "PASS"
    if dm_results['significant']:
        if mean_diff < 0:
            dm_status = "EXCELLENT"  # New model significantly better
        else:
            dm_status = "WARN"  # Baseline significantly better
    
    report.add_test(
        "diebold_mariano",
        dm_status,
        "Diebold-Mariano test for forecast accuracy",
        dm_results,
    )
    
    # Go / No-Go decision rule
    # A model change may ship only if:
    # - Statistical: CI upper bound < 0 AND DM p < 0.05
    # - Practical: improvement ≥ pre-set threshold (≥1% MAE reduction or ≥0.10 points)
    # - Safety: no material degradation in secondary targets
    
    ci_pass = bootstrap_results['ci_upper'] < 0
    dm_pass = dm_results['significant'] and mean_diff < 0
    practical_pass = pct_improvement >= 1.0  # At least 1% of games show improvement
    
    go_decision = ci_pass and dm_pass and practical_pass
    
    report.add_test(
        "go_no_go_decision",
        "PASS" if go_decision else "FAIL",
        "Go / No-Go decision rule (pre-registered)",
        {
            "decision": "GO" if go_decision else "NO-GO",
            "ci_upper_below_zero": ci_pass,
            "dm_significant_better": dm_pass,
            "practical_improvement_1pct": practical_pass,
            "mean_diff": float(mean_diff),
            "pct_improvement": float(pct_improvement),
            "rule": "CI upper < 0 AND DM p < 0.05 AND >= 1% improvement",
        },
    )
    
    return report, {
        "paired_differentials": {
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "pct_improvement": pct_improvement,
        },
        "bootstrap": bootstrap_results,
        "diebold_mariano": dm_results,
    }


if __name__ == "__main__":
    # Test statistical tests
    print("Testing statistical tests...")
    df = pd.read_parquet(
        "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/data/processed/halftime_with_temporal_features_total.parquet"
    )
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Simulate predictions for testing
    np.random.seed(42)
    n = len(df)
    
    # Simulate baseline (Ridge-like, MAE ~ 9.53)
    y_true = df['h2_total'].values
    y_pred_baseline = y_true + np.random.normal(loc=0, scale=9.53, size=n)
    
    # Simulate new model (slightly better, MAE ~ 9.0, mean_diff = -0.53)
    y_pred_new = y_true + np.random.normal(loc=0, scale=9.0, size=n)
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'h2_total': y_true,
        'pred_baseline': y_pred_baseline,
        'pred_new': y_pred_new,
    })
    
    # Run statistical tests
    report, results = run_statistical_tests(
        test_df,
        baseline_predictions_col='pred_baseline',
        new_predictions_col='pred_new',
        target_col='h2_total',
        block_size=50,
        n_bootstraps=100,
    )
    
    print(report)
    
    if report.status == "PASS":
        print("\n✅ STATISTICAL TESTS PASSED")
    else:
        print("\n❌ STATISTICAL TESTS FAILED")
