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
]
