"""
Statistical testing module for PerryPicks v3.

Provides statistical testing for model comparisons:
1. Block bootstrap (time-valid confidence intervals)
2. Diebold-Mariano test (forecast accuracy)
3. Paired loss differentials

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 5
"""

from .block_bootstrap import block_bootstrap, block_bootstrap_summary
from .diebold_mariano import diebold_mariano_test, diebold_mariano_summary
from .statistical_tests import (
    StatisticalTestReport,
    compute_paired_differentials,
    run_statistical_tests,
)

__all__ = [
    "block_bootstrap",
    "block_bootstrap_summary",
    "diebold_mariano_test",
    "diebold_mariano_summary",
    "StatisticalTestReport",
    "compute_paired_differentials",
    "run_statistical_tests",
]
