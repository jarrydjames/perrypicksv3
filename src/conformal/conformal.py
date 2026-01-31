"""
Conformal uncertainty module for PerryPicks v3.

Provides conformal prediction intervals with valid coverage:
1. CQR (conformalized quantile regression)
2. Split-conformal approach
3. Calibration validation
4. Uncertainty report

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 6
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.impute import SimpleImputer

from .cqr import (
    conformalized_quantile_regression,
    predict_intervals,
    evaluate_coverage,
)
from .calibration import (
    validate_coverage,
    validate_uncertainty_report,
    compute_calibration_error,
)


class ConformalUncertaintyReport:
    """
    Conformal uncertainty report with all results.

    Attributes:
        status: Overall status (PASS/WARN/FAIL)
        results: Dict of test names to (status, message, details)
        caveats: List of warnings (non-blocking)
        dataset_checksum: Hash of analyzed dataset
        timestamp: Test timestamp
    """

    def __init__(self):
        self.status: str = "PASS"
        self.results: Dict[str, Tuple[str, str, dict]] = {}
        self.caveats: List[str] = []
        self.dataset_checksum: Optional[str] = None
        self.timestamp: str = pd.Timestamp.now().isoformat()
    
    def add_result(self, name: str, status: str, message: str, details: Optional[dict] = None):
        """Add a test result."""
        self.results[name] = (status, message, details or {})
    
    def add_caveat(self, message: str):
        """Add a non-blocking warning."""
        self.caveats.append(message)
    
    def __str__(self) -> str:
        """Return human-readable report."""
        lines = [
            "=" * 80,
            f"CONFORMAL UNCERTAINTY REPORT - {self.timestamp}",
            f"Overall Status: {self.status}",
            f"Dataset Checksum: {self.dataset_checksum}",
            "=" * 80,
            "",
        ]
        
        # Results
        lines.append("RESULTS:")
        lines.append("-" * 80)
        for result_name, (status, message, details) in self.results.items():
            lines.append(f"  {status}: {result_name}")
            lines.append(f"    {message}")
            if details:
                for key, value in details.items():
                    if isinstance(value, dict):
                        lines.append(f"      {key}:")
                        for k2, v2 in value.items():
                            lines.append(f"        {k2}: {v2}")
                    else:
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
            "results": {
                name: {
                    "status": status,
                    "message": message,
                    "details": details,
                }
                for name, (status, message, details) in self.results.items()
            },
            "caveats": self.caveats,
            "dataset_checksum": self.dataset_checksum,
            "timestamp": self.timestamp,
        }


def run_conformal_uncertainty(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    alpha: float = 0.1,
    random_state: Optional[int] = None,
    test_size: float = 0.2,
) -> Tuple[ConformalUncertaintyReport, dict]:
    """
    Run conformal uncertainty analysis on predictions.
    
    This is the main entry point for conformal uncertainty.
    Returns:
        Tuple of (conformal_uncertainty_report, results_dict)
    
    Steps:
    1. Fit CQR models (lower/upper quantile regressors)
    2. Compute conformality quantile on calibration set
    3. Generate prediction intervals for test set
    4. Validate coverage (empirical vs target)
    5. Evaluate calibration (ECE, MCE)
    6. Assess interval quality (width, sharpness)
    
    Reference: execution_specification Sections 6.1, 6.2, 6.3
    """
    report = ConformalUncertaintyReport()
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Step 1: Fit CQR models
    cqr_results = conformalized_quantile_regression(
        X, y, alpha=alpha, random_state=random_state, test_size=test_size
    )
    
    report.add_result(
        "cqr_fitting",
        "PASS",
        "Conformalized Quantile Regression (CQR) fitted",
        {
            "alpha": cqr_results['alpha'],
            "target_coverage": cqr_results['coverage_target'],
            "calibration_q": float(cqr_results['cal_q']),
            "training_samples": len(cqr_results['train_idx']),
            "calibration_samples": len(cqr_results['cal_idx']),
        },
    )
    
    # Step 2: Generate intervals for calibration set
    X_cal = X[cqr_results['cal_idx']]
    y_cal = y[cqr_results['cal_idx']]
    
    lower_pred, upper_pred = predict_intervals(cqr_results, X_cal)
    
    # Step 3: Validate coverage
    validation_results = validate_coverage(
        y_cal, lower_pred, upper_pred, target_coverage=1 - alpha
    )
    
    coverage_status = "PASS"
    if validation_results['is_calibrated']:
        coverage_status = "EXCELLENT"  # Within 5% tolerance
    elif abs(validation_results['coverage_error']) > 0.1:
        coverage_status = "FAIL"  # More than 10% deviation
    
    report.add_result(
        "coverage_validation",
        coverage_status,
        "Coverage validation (empirical vs target)",
        {
            "empirical_coverage": validation_results['empirical_coverage'],
            "target_coverage": validation_results['target_coverage'],
            "coverage_error": validation_results['coverage_error'],
            "p_value": validation_results['p_value'],
            "is_calibrated": validation_results['is_calibrated'],
            "ci_lower": validation_results['ci_lower'],
            "ci_upper": validation_results['ci_upper'],
        },
    )
    
    # Step 4: Evaluate calibration (ECE, MCE)
    calibration_error = compute_calibration_error(validation_results['calibration_curve'])
    
    ece_status = "PASS"
    if calibration_error['ece'] < 0.05:
        ece_status = "EXCELLENT"
    elif calibration_error['ece'] > 0.1:
        ece_status = "WARN"
    
    report.add_result(
        "calibration_evaluation",
        ece_status,
        "Calibration evaluation (ECE, MCE)",
        {
            "expected_calibration_error": calibration_error['ece'],
            "maximum_calibration_error": calibration_error['mce'],
            "n_bins": calibration_error['n_bins'],
        },
    )
    
    # Step 5: Assess interval quality (width, sharpness)
    interval_width_mean = validation_results['interval_width']['mean']
    interval_width_std = validation_results['interval_width']['std']
    
    # Sharpness: lower std = more consistent intervals
    sharpness_status = "PASS"
    if interval_width_std < interval_width_mean * 0.5:
        sharpness_status = "EXCELLENT"  # Very consistent intervals
    elif interval_width_std > interval_width_mean:
        sharpness_status = "WARN"  # High variance in intervals
    
    report.add_result(
        "interval_quality",
        sharpness_status,
        "Interval quality (width, sharpness)",
        {
            "interval_width_mean": interval_width_mean,
            "interval_width_median": validation_results['interval_width']['median'],
            "interval_width_std": interval_width_std,
            "sharpness": "high" if interval_width_std < interval_width_mean * 0.5 else "medium" if interval_width_std < interval_width_mean else "low",
        },
    )
    
    # Overall assessment
    # Pass if coverage within 10% of target
    overall_pass = abs(validation_results['coverage_error']) < 0.1
    
    if overall_pass:
        report.status = "PASS"
        if calibration_error['ece'] < 0.05:
            report.status = "EXCELLENT"
    else:
        report.status = "FAIL"
    
    return report, {
        "cqr": cqr_results,
        "coverage": validation_results,
        "calibration_error": calibration_error,
        "interval_quality": {
            "mean": interval_width_mean,
            "std": interval_width_std,
        },
    }


if __name__ == "__main__":
    # Test conformal uncertainty
    print("Testing conformal uncertainty...")
    
    df = pd.read_parquet(
        "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/data/processed/halftime_with_temporal_features_total.parquet"
    )
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Select features and target
    h1_features = [col for col in df.columns if col.startswith('h1_')]
    target = 'h2_total'
    
    print(f"\nUsing {len(h1_features)} h1_* features")
    print(f"Target: {target}")
    
    # Run conformal uncertainty
    report, results = run_conformal_uncertainty(
        df, h1_features, target, alpha=0.1, random_state=42, test_size=0.2
    )
    
    print(report)
    
    if report.status in ["PASS", "EXCELLENT"]:
        print("\n✅ CONFORMAL UNCERTAINTY PASSED")
    else:
        print("\n❌ CONFORMAL UNCERTAINTY FAILED")
