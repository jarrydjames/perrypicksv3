"""
PerryPicks v3 - Main package initialization
"""

__version__ = "3.0.0"

from src.validation import validate_data, DataValidationReport, ValidationStatus
from src.leakage_detection import detect_leakage, LeakageDetectionReport, LeakageStatus

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
]
