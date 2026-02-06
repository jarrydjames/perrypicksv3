"""
PerryPicks v3 - Main package initialization
"""

__version__ = "3.0.0"

from src.validation import validate_data, DataValidationReport, ValidationStatus
from src.leakage_detection import detect_leakage, LeakageDetectionReport, LeakageStatus
from src.statistical import (
    block_bootstrap,
    block_bootstrap_summary,
    diebold_mariano_test,
    diebold_mariano_summary,
    StatisticalTestReport,
    compute_paired_differentials,
    run_statistical_tests,
)
from src.conformal import (
    conformalized_quantile_regression,
    predict_intervals,
    evaluate_coverage,
    validate_coverage,
    validate_uncertainty_report,
    compute_calibration_error,
    ConformalUncertaintyReport,
    run_conformal_uncertainty,
)
from src.registry import (
    ModelMetadata,
    ModelRegistry,
    ModelRegistryExtended,
    ModelLineage,
    LineageGraph,
)

__all__ = [
    # Version
    "__version__",
    # Validation
    "validate_data",
    "DataValidationReport",
    "ValidationStatus",
    # Leakage Detection
    "detect_leakage",
    "LeakageDetectionReport",
    "LeakageStatus",
    # Statistical Testing
    "block_bootstrap",
    "block_bootstrap_summary",
    "diebold_mariano_test",
    "diebold_mariano_summary",
    "StatisticalTestReport",
    "compute_paired_differentials",
    "run_statistical_tests",
    # Conformal Uncertainty
    "conformalized_quantile_regression",
    "predict_intervals",
    "evaluate_coverage",
    "validate_coverage",
    "validate_uncertainty_report",
    "compute_calibration_error",
    "ConformalUncertaintyReport",
    "run_conformal_uncertainty",
    # Model Registry
    "ModelMetadata",
    "ModelRegistry",
    "ModelRegistryExtended",
    "ModelLineage",
    "LineageGraph",
]
