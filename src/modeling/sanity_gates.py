"""Sanity gates for nested backtest - prevents invalid evaluations.

This module implements comprehensive sanity checks to catch:
- Target scale issues (scaled vs raw)
- Leakage (target-derived features)
- Ill-conditioning (constant/duplicate features)
- Pathological folds

If any gate fails, the entire backtest stops with a clear error.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class SanityGateResult:
    """Result of sanity gate checks."""
    passed: bool
    gate_name: str
    message: str
    details: Dict[str, any] | None = None


# Banned tokens that suggest leakage or target-derived features
BANNED_FEATURE_TOKENS = {
    "final", "result", "win", "outcome", "target", "label",
    "score", "points", "end"
}

# Whitelist for allowed tokens (e.g., "points_allowed" might be valid)
ALLOWED_TOKENS = {
    # Add any legitimate feature names here that contain banned tokens
    # e.g., "opponent_points_allowed", "team_points_scored_avg"
}


def check_target_scale_gate(
    y_total_train: np.ndarray,
    y_margin_train: np.ndarray,
    state: str = "halftime"
) -> SanityGateResult:
    """Check that targets are in reasonable scale.
    
    Prevents scaled-y vs raw-y confusion.
    
    Rules:
    - For NBA totals: mean should be 120-320
    - For margins: std should be > 1
    """
    y_total_mean = float(np.mean(y_total_train))
    y_margin_std = float(np.std(y_margin_train))
    
    # Total mean should be in reasonable range
    if state == "pregame":
        # Full game totals typically 180-280
        if not (150 <= y_total_mean <= 350):
            return SanityGateResult(
                passed=False,
                gate_name="target_scale_gate",
                message=f"Total mean {y_total_mean:.1f} outside expected range [150, 350] for pregame",
                details={"y_total_mean": y_total_mean, "state": state}
            )
    else:
        # Half/game totals typically 90-140
        if not (60 <= y_total_mean <= 200):
            return SanityGateResult(
                passed=False,
                gate_name="target_scale_gate",
                message=f"Total mean {y_total_mean:.1f} outside expected range [60, 200] for {state}",
                details={"y_total_mean": y_total_mean, "state": state}
            )
    
    # Margin std should be > 1 (not scaled)
    if y_margin_std < 1.0:
        return SanityGateResult(
            passed=False,
            gate_name="target_scale_gate",
            message=f"Margin std {y_margin_std:.3f} too low - likely scaled data",
            details={"y_margin_std": y_margin_std}
        )
    
    return SanityGateResult(
        passed=True,
        gate_name="target_scale_gate",
        message=f"Target scales OK: total_mean={y_total_mean:.1f}, margin_std={y_margin_std:.1f}"
    )


def check_feature_name_gate(feature_names: List[str]) -> SanityGateResult:
    """Check for banned feature names that suggest leakage.
    
    Rules:
    - No feature name may contain: final, result, win, outcome, target, label
    """
    banned_features = []
    
    for feat in feature_names:
        feat_lower = feat.lower()
        
        # Check if any banned token is in the feature name
        for token in BANNED_FEATURE_TOKENS:
            if token in feat_lower:
                # Check if it's whitelisted
                if feat not in ALLOWED_TOKENS:
                    banned_features.append((feat, token))
    
    if banned_features:
        return SanityGateResult(
            passed=False,
            gate_name="feature_name_gate",
            message=f"Found {len(banned_features)} features with banned tokens",
            details={"banned_features": banned_features}
        )
    
    return SanityGateResult(
        passed=True,
        gate_name="feature_name_gate",
        message="All feature names OK"
    )


def check_constant_feature_gate(X_train: np.ndarray, feature_names: List[str]) -> SanityGateResult:
    """Check for constant features that cause ill-conditioning.
    
    Rules:
    - No feature may be all-constant in training slice
    """
    constant_features = []
    
    for i, feat_name in enumerate(feature_names):
        col = X_train[:, i]
        if np.all(col == col[0]):
            constant_features.append(feat_name)
    
    if constant_features:
        return SanityGateResult(
            passed=False,
            gate_name="constant_feature_gate",
            message=f"Found {len(constant_features)} constant features",
            details={"constant_features": constant_features}
        )
    
    return SanityGateResult(
        passed=True,
        gate_name="constant_feature_gate",
        message="No constant features found"
    )


def check_duplicate_feature_gate(X_train: np.ndarray, feature_names: List[str]) -> SanityGateResult:
    """Check for duplicate features by name or by value.
    
    Rules:
    - No duplicate columns by name
    - No duplicate columns by value
    """
    # Check for duplicate names
    name_counts = pd.Series(feature_names).value_counts()
    duplicate_names = name_counts[name_counts > 1].index.tolist()
    
    if duplicate_names:
        return SanityGateResult(
            passed=False,
            gate_name="duplicate_feature_gate",
            message=f"Found {len(duplicate_names)} duplicate feature names",
            details={"duplicate_names": duplicate_names}
        )
    
    # Check for duplicate values (exact matches)
    duplicate_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if np.array_equal(X_train[:, i], X_train[:, j]):
                duplicate_pairs.append((feature_names[i], feature_names[j]))
    
    if duplicate_pairs:
        return SanityGateResult(
            passed=False,
            gate_name="duplicate_feature_gate",
            message=f"Found {len(duplicate_pairs)} duplicate feature pairs by value",
            details={"duplicate_pairs": duplicate_pairs[:10]}  # Show first 10
        )
    
    return SanityGateResult(
        passed=True,
        gate_name="duplicate_feature_gate",
        message="No duplicate features found"
    )


def check_leakage_gate(
    X_train: np.ndarray,
    y_total_train: np.ndarray,
    y_margin_train: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.995
) -> SanityGateResult:
    """Check for features highly correlated with targets (leakage).
    
    Rules:
    - Block any feature with |corr(feature, target)| > threshold
    """
    leaking_features_total = []
    leaking_features_margin = []
    
    for i, feat_name in enumerate(feature_names):
        col = X_train[:, i]
        
        # Skip constant columns (handled by other gate)
        if np.all(col == col[0]):
            continue
        
        # Check correlation with total
        if np.std(y_total_train) > 0:
            corr_total = abs(np.corrcoef(col, y_total_train)[0, 1])
            if not np.isnan(corr_total) and corr_total > threshold:
                leaking_features_total.append((feat_name, corr_total))
        
        # Check correlation with margin
        if np.std(y_margin_train) > 0:
            corr_margin = abs(np.corrcoef(col, y_margin_train)[0, 1])
            if not np.isnan(corr_margin) and corr_margin > threshold:
                leaking_features_margin.append((feat_name, corr_margin))
    
    if leaking_features_total or leaking_features_margin:
        return SanityGateResult(
            passed=False,
            gate_name="leakage_gate",
            message=f"Found {len(leaking_features_total) + len(leaking_features_margin)} features with high target correlation",
            details={
                "leaking_with_total": leaking_features_total[:10],
                "leaking_with_margin": leaking_features_margin[:10]
            }
        )
    
    return SanityGateResult(
        passed=True,
        gate_name="leakage_gate",
        message="No leakage features detected"
    )


def run_all_sanity_gates(
    X_train: np.ndarray,
    y_total_train: np.ndarray,
    y_margin_train: np.ndarray,
    feature_names: List[str],
    state: str = "halftime",
    fold_i: int = 0
) -> List[SanityGateResult]:
    """Run all sanity gates and return results.
    
    Raises:
        RuntimeError: If any gate fails
    """
    results = []
    
    # Gate 1: Target scale
    result = check_target_scale_gate(y_total_train, y_margin_train, state)
    results.append(result)
    print(f"[fold {fold_i}] {result.gate_name}: {result.message}", flush=True)
    if not result.passed:
        raise RuntimeError(f"Sanity gate failed: {result.gate_name}\n{result.message}\nDetails: {result.details}")
    
    # Gate 2: Feature names
    result = check_feature_name_gate(feature_names)
    results.append(result)
    print(f"[fold {fold_i}] {result.gate_name}: {result.message}", flush=True)
    if not result.passed:
        raise RuntimeError(f"Sanity gate failed: {result.gate_name}\n{result.message}\nDetails: {result.details}")
    
    # Gate 3: Constant features
    result = check_constant_feature_gate(X_train, feature_names)
    results.append(result)
    print(f"[fold {fold_i}] {result.gate_name}: {result.message}", flush=True)
    if not result.passed:
        raise RuntimeError(f"Sanity gate failed: {result.gate_name}\n{result.message}\nDetails: {result.details}")
    
    # Gate 4: Duplicate features
    result = check_duplicate_feature_gate(X_train, feature_names)
    results.append(result)
    print(f"[fold {fold_i}] {result.gate_name}: {result.message}", flush=True)
    if not result.passed:
        raise RuntimeError(f"Sanity gate failed: {result.gate_name}\n{result.message}\nDetails: {result.details}")
    
    # Gate 5: Leakage
    result = check_leakage_gate(X_train, y_total_train, y_margin_train, feature_names)
    results.append(result)
    print(f"[fold {fold_i}] {result.gate_name}: {result.message}", flush=True)
    if not result.passed:
        raise RuntimeError(f"Sanity gate failed: {result.gate_name}\n{result.message}\nDetails: {result.details}")
    
    return results


def compute_fold_diagnostics(
    X_train: np.ndarray,
    y_total_train: np.ndarray,
    y_margin_train: np.ndarray,
    feature_names: List[str],
    fold_i: int
) -> Dict[str, any]:
    """Compute fold diagnostics for logging.
    
    Returns:
        Dict with diagnostic information
    """
    diagnostics = {
        "fold": fold_i,
        "n_train": int(len(X_train)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
    }
    
    # Zero-variance features
    zero_var_count = 0
    for i in range(len(feature_names)):
        if np.all(X_train[:, i] == X_train[0, i]):
            zero_var_count += 1
    diagnostics["zero_variance_features"] = zero_var_count
    
    # Near-duplicate features (by correlation)
    high_corr_pairs = []
    for i in range(min(len(feature_names), 100)):  # Limit to first 100 features for speed
        for j in range(i + 1, min(len(feature_names), 100)):
            if np.std(X_train[:, i]) > 0 and np.std(X_train[:, j]) > 0:
                corr = abs(np.corrcoef(X_train[:, i], X_train[:, j])[0, 1])
                if not np.isnan(corr) and corr > 0.999:
                    high_corr_pairs.append((feature_names[i], feature_names[j], float(corr)))
    diagnostics["near_duplicate_pairs"] = high_corr_pairs[:10]
    
    # Top correlated with targets
    total_corrs = []
    margin_corrs = []
    for i, feat_name in enumerate(feature_names):
        col = X_train[:, i]
        if np.std(col) > 0:
            if np.std(y_total_train) > 0:
                corr_t = abs(np.corrcoef(col, y_total_train)[0, 1])
                if not np.isnan(corr_t):
                    total_corrs.append((feat_name, float(corr_t)))
            if np.std(y_margin_train) > 0:
                corr_m = abs(np.corrcoef(col, y_margin_train)[0, 1])
                if not np.isnan(corr_m):
                    margin_corrs.append((feat_name, float(corr_m)))
    
    total_corrs.sort(key=lambda x: x[1], reverse=True)
    margin_corrs.sort(key=lambda x: x[1], reverse=True)
    
    diagnostics["top_total_correlations"] = total_corrs[:10]
    diagnostics["top_margin_correlations"] = margin_corrs[:10]
    
    # Condition number estimate (smallest singular value)
    try:
        # Sample first 1000 rows for speed
        X_sample = X_train[:1000] if len(X_train) > 1000 else X_train
        U, s, Vt = np.linalg.svd(X_sample, full_matrices=False)
        diagnostics["condition_number"] = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        diagnostics["min_singular_value"] = float(s[-1])
    except Exception:
        diagnostics["condition_number"] = None
        diagnostics["min_singular_value"] = None
    
    # Target statistics
    diagnostics["y_total_mean"] = float(np.mean(y_total_train))
    diagnostics["y_total_std"] = float(np.std(y_total_train))
    diagnostics["y_margin_mean"] = float(np.mean(y_margin_train))
    diagnostics["y_margin_std"] = float(np.std(y_margin_train))
    
    return diagnostics


def treat_warnings_as_errors():
    """Configure warnings to be treated as errors during training.
    
    Specifically:
    - LinAlgWarning -> error
    - ConvergenceWarning -> error
    """
    warnings.filterwarnings("error", category=Warning, module="sklearn")
    
    # Specifically catch ill-conditioned matrix warnings
    warnings.filterwarnings(
        "error",
        message=".*ill-conditioned.*",
        category=Warning
    )
    
    # Catch convergence warnings
    warnings.filterwarnings(
        "error",
        message=".*Objective did not converge.*",
        category=Warning
    )
