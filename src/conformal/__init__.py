"""
Conformal uncertainty module for PerryPicks v3.

Provides conformal prediction intervals with valid coverage:
1. CQR (conformalized quantile regression)
2. Split-conformal approach
3. Calibration validation
4. Uncertainty report

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 6
"""

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
from .conformal import (
    ConformalUncertaintyReport,
    run_conformal_uncertainty,
)

__all__ = [
    # CQR
    "conformalized_quantile_regression",
    "predict_intervals",
    "evaluate_coverage",
    # Calibration
    "validate_coverage",
    "validate_uncertainty_report",
    "compute_calibration_error",
    # Conformal
    "ConformalUncertaintyReport",
    "run_conformal_uncertainty",
]
