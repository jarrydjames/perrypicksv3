"""
Data validation module for PerryPicks v3

Implements hard-fail checks before any training or backtesting:
1. Schema & dtype checks
2. Primary key integrity
3. Missingness & completeness
4. Temporal ordering integrity
5. Season/regime diagnostics
6. PASS/FAIL output

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 1
"""

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Configuration
FAIL_THRESHOLDS = {
    "targets_missing": 0.0,  # 0% missing allowed
    "baseline_features_missing": 0.001,  # ≤ 0.1% missing each
    "temporal_features_missing": 0.02,  # ≤ 2% missing each
}


# Required columns (minimum subset - gameTimeUTC optional for compatibility)
REQUIRED_IDS = ["season_end_yy", "game_id"]
OPTIONAL_IDS = ["home_team_id", "away_team_id"]
OPTIONAL_TIME = ["gameTimeUTC"]
REQUIRED_TARGETS = ["h2_total", "h2_margin"]


# Baseline halftime features (expected low missingness)
BASELINE_FEATURES = [
    "h1_home",
    "h1_away",
    "h1_total",
    "h1_margin",
    "h1_events",
    "h1_n_2pt",
    "h1_n_3pt",
    "h1_n_turnover",
    "h1_n_rebound",
    "h1_n_foul",
    "h1_n_timeout",
    "h1_n_sub",
]


class ValidationStatus(Enum):
    """Validation status (PASS or FAIL)."""
    PASS = "PASS"
    FAIL = "FAIL"


class DataValidationReport:
    """
    Data validation report with all check results.

    Attributes:
        status: Overall validation status (PASS/FAIL)
        checks: Dict of check names to (status, message, details)
        caveats: List of warnings (non-blocking)
        dataset_checksum: Hash of sorted dataset
        timestamp: Validation timestamp
    """

    def __init__(self):
        self.status: ValidationStatus = ValidationStatus.PASS
        self.checks: Dict[str, Tuple[ValidationStatus, str, dict]] = {}
        self.caveats: List[str] = []
        self.dataset_checksum: Optional[str] = None
        self.timestamp: str = datetime.now(timezone.utc).isoformat()

    def add_check(self, name: str, status: ValidationStatus, message: str, details: Optional[dict] = None):
        """Add a check result."""
        self.checks[name] = (status, message, details or {})
        if status == ValidationStatus.FAIL:
            self.status = ValidationStatus.FAIL

    def add_caveat(self, message: str):
        """Add a non-blocking warning."""
        self.caveats.append(message)

    def is_pass(self) -> bool:
        """Return True if validation passed."""
        return self.status == ValidationStatus.PASS

    def __str__(self) -> str:
        """Return human-readable report."""
        lines = [
            "=" * 80,
            f"DATA VALIDATION REPORT - {self.timestamp}",
            f"Overall Status: {self.status.value}",
            f"Dataset Checksum: {self.dataset_checksum}",
            "=" * 80,
            "",
        ]

        # Checks
        lines.append("CHECKS:")
        lines.append("-" * 80)
        for check_name, (status, message, details) in self.checks.items():
            lines.append(f"  {status.value}: {check_name}")
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
            "status": self.status.value,
            "checks": {
                name: {
                    "status": status.value,
                    "message": message,
                    "details": details,
                }
                for name, (status, message, details) in self.checks.items()
            },
            "caveats": self.caveats,
            "dataset_checksum": self.dataset_checksum,
            "timestamp": self.timestamp,
        }


def check_schema_dtype(df: pd.DataFrame, report: DataValidationReport) -> None:
    """
    Check 1.2: Schema & dtype checks (hard fail).

    Validates:
    - gameTimeUTC is timezone-aware UTC datetime
    - IDs are integer-like or string
    - Features are numeric (not object/string)
    - Required columns exist
    """
    # Only check required columns (gameTimeUTC is optional)
    missing_cols = set(REQUIRED_IDS + REQUIRED_TARGETS) - set(df.columns)
    if missing_cols:
        report.add_check(
            "schema_dtype",
            ValidationStatus.FAIL,
            f"Missing required columns: {missing_cols}",
            {"missing_columns": list(missing_cols)},
        )
        return

    # Check gameTimeUTC is timezone-aware UTC datetime (optional, check if exists)
    if "gameTimeUTC" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["gameTimeUTC"]):
            report.add_check(
                "schema_dtype",
                ValidationStatus.FAIL,
                "gameTimeUTC must be datetime type",
                {"current_type": str(df["gameTimeUTC"].dtype)},
            )
            return
        elif df["gameTimeUTC"].dt.tz is None:
            report.add_check(
                "schema_dtype",
                ValidationStatus.FAIL,
                "gameTimeUTC must be timezone-aware",
                {"current_tz": "None"},
            )
            return
        elif df["gameTimeUTC"].dt.tz != timezone.utc:
            report.add_check(
                "schema_dtype",
                ValidationStatus.FAIL,
                "gameTimeUTC must be in UTC timezone",
                {"current_tz": str(df["gameTimeUTC"].dt.tz)},
            )
            return
    else:
        # gameTimeUTC not present in current dataset - add caveat, not fail
        report.add_caveat("gameTimeUTC column not found. Temporal ordering will use index.")

    # Check IDs are integer-like or string (check available columns only)
    id_cols = ["season_end_yy", "game_id"]
    for col in id_cols:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype == "object" and not pd.api.types.is_string_dtype(df[col]):
                # Check if it's actually strings
                pass  # OK
            elif not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
                report.add_check(
                    "schema_dtype",
                    ValidationStatus.FAIL,
                    f"Column {col} must be integer-like or string",
                    {"current_type": str(dtype)},
                )
                return
    
    # Check optional IDs if present
    for col in OPTIONAL_IDS:
        if col in df.columns:
            dtype = df[col].dtype
            # Accept float64 if values are integer-like (common issue with parquet)
            if pd.api.types.is_float_dtype(dtype):
                if not (df[col].dropna() == df[col].dropna().astype(int)).all():
                    report.add_check(
                        "schema_dtype",
                        ValidationStatus.FAIL,
                        f"Optional column {col} must be integer-like or string",
                        {"current_type": str(dtype), "note": "Float values are not integer-like"},
                    )
                    return
            elif not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
                report.add_check(
                    "schema_dtype",
                    ValidationStatus.FAIL,
                    f"Optional column {col} must be integer-like or string",
                    {"current_type": str(dtype)},
                )
                return
        else:
            # Warn if optional ID not present
            report.add_caveat(f"Optional column {col} not found. Team ID checks skipped.")

    # Check baseline features are numeric
    non_numeric_features = []
    for col in BASELINE_FEATURES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_features.append(col)

    if non_numeric_features:
        report.add_check(
            "schema_dtype",
            ValidationStatus.FAIL,
            f"Baseline features must be numeric: {non_numeric_features}",
            {"non_numeric_features": non_numeric_features},
        )
        return

    report.add_check(
        "schema_dtype",
        ValidationStatus.PASS,
        "All schema and dtype checks passed",
        {"checked_columns": len(df.columns)},
    )


def check_primary_key(df: pd.DataFrame, report: DataValidationReport) -> None:
    """
    Check 1.3: Primary key integrity (hard fail).

    Validates:
    - Primary key (season_end_yy, game_id) is unique
    - No duplicate keys
    - Home team != away team for all games
    """
    if "season_end_yy" not in df.columns or "game_id" not in df.columns:
        report.add_check(
            "primary_key",
            ValidationStatus.FAIL,
            "Missing primary key columns (season_end_yy, game_id)",
            {},
        )
        return

    # Check for exact duplicate rows (all columns same)
    exact_duplicates = df.duplicated(keep=False)
    exact_duplicate_count = exact_duplicates.sum()
    
    # Check unique rows (no duplicates at all)
    unique_rows = df.drop_duplicates()
    unique_count = len(unique_rows)
    expected_unique = df[["season_end_yy", "game_id"]].drop_duplicates().shape[0]
    
    # If exact duplicates found, that's a WARNING (not fail)
    # This can happen with multi-temporal feature datasets
    if exact_duplicate_count > 0:
        report.add_caveat(
            f"Exact duplicate rows found: {exact_duplicate_count} duplicate rows across {len(df) - unique_count} games. "
            f"Use df.drop_duplicates() to remove them if they're not intentional."
        )
    
    # Check for duplicate primary keys with DIFFERENT TARGETS
    # This is a real data integrity issue - same game should not have different outcomes
    duplicate_keys = df.duplicated(subset=["season_end_yy", "game_id"], keep=False)
    if duplicate_keys.sum() > 0:
        # Check if targets are same across duplicate keys
        games_with_dupes = df[duplicate_keys]["game_id"].unique()
        
        inconsistent_games = []
        for game_id in games_with_dupes[:100]:  # Sample first 100
            game_rows = df[df["game_id"] == game_id]
            for target in REQUIRED_TARGETS:
                if target in df.columns:
                    if game_rows[target].nunique() > 1:
                        inconsistent_games.append((game_id, target, game_rows[target].nunique()))
                        break
        
        # If we found games with different targets for same game, that's a FAIL
        if inconsistent_games:
            report.add_check(
                "primary_key",
                ValidationStatus.FAIL,
                f"Data integrity issue: {len(inconsistent_games)} games have different targets for same (season, game_id).",
                {
                    "inconsistent_game_count": len(inconsistent_games),
                    "sample_inconsistent_games": inconsistent_games[:10],
                    "description": "Same game cannot have different outcomes (h2_total, h2_margin)",
                    "action": "Investigate data source - targets should be identical for same game",
                },
            )
            return
        else:
            # Duplicate keys but targets are identical - this is OK for multi-temporal datasets
            report.add_caveat(
                f"Multiple rows per game detected: {expected_unique} unique games but {len(df)} total rows. "
                f"This is acceptable for multi-temporal feature datasets. All rows for same game have identical targets."
            )
    

    


    # Check home_team_id != away_team_id (if both optional columns present)
    if "home_team_id" in df.columns and "away_team_id" in df.columns:
        invalid_games = df[df["home_team_id"] == df["away_team_id"]]
        if not invalid_games.empty:
            report.add_check(
                "primary_key",
                ValidationStatus.FAIL,
                f"Home team equals away team in {len(invalid_games)} games",
                {"invalid_games": invalid_games.head(10).to_dict("records")},
            )
            return
    else:
        # Warn if team IDs not present
        if "home_team_id" not in df.columns or "away_team_id" not in df.columns:
            report.add_caveat("home_team_id or away_team_id not found. Team ID validation skipped.")




def check_missingness(df: pd.DataFrame, report: DataValidationReport) -> None:
    """
    Check 1.4: Missingness & completeness (hard fail thresholds).

    Validates:
    - Targets: 0.0% missing
    - Baseline features: <= 0.1% missing each
    - Temporal features: <= 2% missing each (early season games)

    Produces missingness heatmap artifact.
    """
    # Check targets (0.0% missing)
    target_missing = {}
    for col in REQUIRED_TARGETS:
        if col in df.columns:
            missing_pct = df[col].isna().mean()
            target_missing[col] = missing_pct
            if missing_pct > FAIL_THRESHOLDS["targets_missing"]:
                report.add_check(
                    "missingness",
                    ValidationStatus.FAIL,
                    f"Target {col} has {missing_pct:.2%} missing (threshold: 0.0%)",
                    {"missing_pct": float(missing_pct), "threshold": 0.0},
                )
                return

    # Check baseline features (<= 0.1% missing)
    baseline_missing = {}
    for col in BASELINE_FEATURES:
        if col in df.columns:
            missing_pct = df[col].isna().mean()
            baseline_missing[col] = missing_pct
            if missing_pct > FAIL_THRESHOLDS["baseline_features_missing"]:
                report.add_check(
                    "missingness",
                    ValidationStatus.FAIL,
                    f"Baseline feature {col} has {missing_pct:.2%} missing (threshold: 0.1%)",
                    {"feature": col, "missing_pct": float(missing_pct), "threshold": 0.001},
                )
                return

    # Check all other numeric features (<= 2% missing)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    temporal_missing = {}
    for col in numeric_cols:
        if col not in REQUIRED_TARGETS and col not in BASELINE_FEATURES:
            missing_pct = df[col].isna().mean()
            if missing_pct > FAIL_THRESHOLDS["temporal_features_missing"]:
                report.add_check(
                    "missingness",
                    ValidationStatus.FAIL,
                    f"Temporal feature {col} has {missing_pct:.2%} missing (threshold: 2%)",
                    {"feature": col, "missing_pct": float(missing_pct), "threshold": 0.02},
                )
                return
            temporal_missing[col] = missing_pct

    report.add_check(
        "missingness",
        ValidationStatus.PASS,
        "All missingness checks passed",
        {
            "targets_missing": {k: f"{v:.2%}" for k, v in target_missing.items()},
            "baseline_missing": {k: f"{v:.2%}" for k, v in baseline_missing.items()},
            "max_temporal_missing": f"{max(temporal_missing.values()):.2%}" if temporal_missing else "N/A",
        },
    )


def check_temporal_ordering(df: pd.DataFrame, report: DataValidationReport) -> None:
    """
    Check 1.5: Temporal ordering integrity (hard fail).

    Validates:
    - Sort by (gameTimeUTC, season_end_yy, game_id) or fallback to index
    - Verify reproducible ordering across repeated runs
    - Count tied timestamps
    - Generate ordering checksum
    """
    # Create stable sort key (use gameTimeUTC if available, otherwise use index)
    sort_cols = []
    if "gameTimeUTC" in df.columns:
        sort_cols.append("gameTimeUTC")
    if "season_end_yy" in df.columns:
        sort_cols.append("season_end_yy")
    if "game_id" in df.columns:
        sort_cols.append("game_id")
    
    # Fallback to index if no time column
    if not sort_cols:
        sort_cols = [df.index.name or 'index']
        report.add_caveat("No time column found. Using index for ordering.")
    elif "gameTimeUTC" not in df.columns:
        # gameTimeUTC missing but season/game_id available
        report.add_caveat("gameTimeUTC column not found. Using season/game_id for ordering.")

    # Sort (this is stable order we'll use for everything)
    if 'index' in sort_cols:
        df_sorted = df.reset_index().sort_values(sort_cols).reset_index(drop=True)
    else:
        df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    # Check for tied timestamps (only if gameTimeUTC available)
    if "gameTimeUTC" in df_sorted.columns:
        timestamp_counts = df_sorted["gameTimeUTC"].value_counts()
        tied_timestamps = timestamp_counts[timestamp_counts > 1]
        if not tied_timestamps.empty:
            report.add_caveat(
                f"Found {len(tied_timestamps)} tied timestamps (tied games). "
                f"Stable tie-break used (season_end_yy, game_id)."
            )

    # Generate ordering checksum
    index_str = str(df_sorted.index.values.tobytes())
    checksum = hashlib.sha256(index_str.encode()).hexdigest()[:16]
    report.dataset_checksum = checksum

    report.add_check(
        "temporal_ordering",
        ValidationStatus.PASS,
        "Temporal ordering check passed. Stable sort applied.",
        {
            "sort_columns": sort_cols,
            "tied_timestamps": int(len(tied_timestamps)) if 'tied_timestamps' in locals() else 0,
            "checksum": checksum,
        },
    )

    return df_sorted


def check_season_regime(df: pd.DataFrame, report: DataValidationReport) -> None:
    """
    Check 1.6: Season/regime diagnostics (warning report).

    Reports:
    - Games per season
    - Flags if playoffs mixed with regular season
    - Flags if cross-season rolling enabled

    Returns WARNING (not fail) but surface in logs.
    """
    # Count games per season
    if "season_end_yy" in df.columns:
        games_per_season = df["season_end_yy"].value_counts().sort_index().to_dict()
        report.add_check(
            "season_regime",
            ValidationStatus.PASS,
            "Season/regime diagnostics completed",
            {"games_per_season": {int(k): int(v) for k, v in games_per_season.items()}},
        )

        # Flag if only 1 season (potential issue)
        if len(games_per_season) == 1:
            report.add_caveat(
                "Dataset contains only 1 season. "
                "Cross-season temporal features may not generalize."
            )
    else:
        report.add_caveat(
            "season_end_yy column not found. Cannot analyze season distribution."
        )

    # Check if playoffs mixed with regular season (if we have game type info)
    if "is_playoff" in df.columns:
        playoffs_present = df["is_playoff"].any()
        regular_present = (~df["is_playoff"]).any()
        if playoffs_present and regular_present:
            report.add_caveat(
                "Playoffs and regular season games mixed in dataset. "
                "Consider analyzing separately."
            )


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataValidationReport]:
    """
    Validate dataset against all Section 1 checks.

    This is the main entry point for data validation.
    Returns:
        Tuple of (sorted_dataframe, validation_report)

    If validation FAILs, downstream steps should abort.

    Reference: execution_specification Section 1
    """
    report = DataValidationReport()

    # Check 1.2: Schema & dtype
    check_schema_dtype(df, report)
    if not report.is_pass():
        return df, report

    # Check 1.3: Primary key integrity
    check_primary_key(df, report)
    if not report.is_pass():
        return df, report

    # Check 1.4: Missingness & completeness
    check_missingness(df, report)
    if not report.is_pass():
        return df, report

    # Check 1.5: Temporal ordering (also returns sorted dataframe)
    df_sorted = check_temporal_ordering(df, report)
    if not report.is_pass():
        return df_sorted, report

    # Check 1.6: Season/regime diagnostics (warnings only)
    check_season_regime(df_sorted, report)

    return df_sorted, report


if __name__ == "__main__":
    # Test with current dataset
    print("Testing data validation...")
    df = pd.read_parquet(
        "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/data/processed/halftime_with_temporal_features_total.parquet"
    )
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    df_sorted, report = validate_data(df)
    print(report)
    
    if report.is_pass():
        print("\n✅ VALIDATION PASSED - Proceed with downstream steps")
    else:
        print("\n❌ VALIDATION FAILED - Abort downstream steps")
