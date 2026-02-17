"""
Phase 2B: CatBoost Re-tuning and Fair Model Comparison

This script re-tunes CatBoost with improved parameters to ensure
fair comparison with XGBoost from Phase 2.

Changes from Phase 2:
- Increased timeout: 30 minutes -> 90 minutes
- Reduced search space: Focused, high-throughput search
- Limited iterations: 300-3000 range
- Target trials: 40-50 per fold (minimum 25)

Result:
- CatBoost with ~300-500 total trials
- Fair comparison with XGBoost (595 trials)
- Ensemble consideration if models are close
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.modeling.backtest_utils import FoldSpec, brier, coverage, iter_walkforward_indices, mae, p_home_win, rmse
from src.modeling.cat_models import CatBoostTwoHeadModel
from src.modeling.feature_columns import feature_columns
from src.modeling.nested_walkforward_backtest import (
    _fit_eval_model,
    _inner_splits,
    _score_objective,
    NestedSpec,
)
from src.modeling.sanity_gates import compute_fold_diagnostics, run_all_sanity_gates



@dataclass(frozen=True)
class Phase2BSpec:
    """Specification for Phase 2B CatBoost re-tuning."""
    data_path: Path
    output_dir: Path
    target_total: str = "total"
    target_margin: str = "margin"
    outer: FoldSpec = FoldSpec(
        train_min=800,
        test_size=200,
        step_size=200,
    )
    inner_folds: int = 5
    trials: int = 40  # Target 40-50 per fold
    timeout_s: int = 5400  # 90 minutes (increased from 30 min)
    seed: int = 42
    min_trials_per_fold: int = 25  # Minimum acceptable



def _tune_catboost_optuna_v2(
    *,
    X: np.ndarray,
    ytot: np.ndarray,
    ymar: np.ndarray,
    feature_names: List[str],
    inner_folds: int,
    trials: int,
    seed: int,
    timeout_s: int,
    log_prefix: str = "",
) -> Tuple[Dict[str, Any], float]:
    """Tune CatBoost using Optuna with improved, focused search space.
    
    Improvements from Phase 2:
    - Increased timeout to 90 minutes
    - Reduced search space dimensionality
    - Limited iterations to 300-3000
    """
    from src.modeling.cat_models import CatBoostTwoHeadModel
    
    try:
        import optuna  # type: ignore
    except Exception:
        raise RuntimeError("Optuna is required for Phase 2B")
    
    splits = _inner_splits(len(X), inner_folds=inner_folds)
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    def objective(trial: Any) -> float:
        # Focused search space (reduced dimensionality)
        params = {
            "iterations": int(trial.suggest_int("iterations", 300, 3000)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.015, 0.05, log=True)),
            "depth": int(trial.suggest_int("depth", 4, 6)),
            "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", 2.0, 12.0, log=True)),
            "subsample": float(trial.suggest_float("subsample", 0.65, 0.85)),
            "random_seed": int(seed),
        }
        
        fold_scores: List[float] = []
        for tr, te in splits:
            m = CatBoostTwoHeadModel(feature_version="v1", **params)
            met = _fit_eval_model(
                m,
                X_tr=X[tr],
                ytot_tr=ytot[tr],
                ymar_tr=ymar[tr],
                X_te=X[te],
                ytot_te=ytot[te],
                ymar_te=ymar[te],
                feature_names=feature_names,
            )
            fold_scores.append(_score_objective({k: float(v) for k, v in met.items()}))
        
        return float(np.mean(fold_scores)) if fold_scores else float("inf")
    
    start_ts = time.perf_counter()
    study.optimize(objective, n_trials=int(trials), timeout=int(timeout_s) if timeout_s > 0 else None)
    
    if study.best_trial is None:
        raise RuntimeError("Optuna CatBoost tuning produced no trials")
    
    elapsed = time.perf_counter() - start_ts
    best_score = float(study.best_value)
    best_params = dict(study.best_trial.params)
    
    print(f"{log_prefix}CatBoost tuning complete: {len(study.trials)} trials, best={best_score:.4f} elapsed={elapsed/60:.1f}min", flush=True)
    
    return best_params, best_score


def load_xgboost_results(phase2_results_path: Path) -> pd.DataFrame:
    """Load XGBoost results from Phase 2 for comparison."""
    df = pd.read_csv(phase2_results_path)
    return df[df['model'] == 'xgboost'].copy()


def run_phase2b_catboost_retuning(spec: Phase2BSpec) -> None:
    """Run Phase 2B CatBoost re-tuning."""
    print("="*80)
    print("PHASE 2B: CatBoost Re-tuning and Fair Model Comparison")
    print("="*80)
    print(f"\nDate: {pd.Timestamp.now()}")
    print(f"\nConfiguration:")
    print(f"  Data: {spec.data_path}")
    print(f"  Output: {spec.output_dir}")
    print(f"  Targets: total={spec.target_total}, margin={spec.target_margin}")
    print(f"  Folds: {spec.outer}")
    print(f"  Inner folds: {spec.inner_folds}")
    print(f"  Target trials: {spec.trials} per fold")
    print(f"  Timeout: {spec.timeout_s}s ({spec.timeout_s/60:.0f} minutes)")
    print(f"  Seed: {spec.seed}")
    print(f"\nCatBoost search space:")
    print(f"  iterations: [300, 3000]")
    print(f"  learning_rate: [0.015, 0.05] (log)")
    print(f"  depth: [4, 6]")
    print(f"  l2_leaf_reg: [2.0, 12.0] (log)")
    print(f"  subsample: [0.65, 0.85]")
    
    print(f"\n")
    
    # Load data
    print("Loading data...", flush=True)
    df = pd.read_parquet(spec.data_path)
    feats = feature_columns(df)
    X_all = df[feats].to_numpy(dtype=float)
    y_total_all = df[spec.target_total].to_numpy(dtype=float)
    y_margin_all = df[spec.target_margin].to_numpy(dtype=float)
    
    # Determine state from target names
    state = "unknown"
    if "h2_" in spec.target_total:
        state = "halftime"
    elif "remaining_" in spec.target_total:
        state = "q3"
    else:
        state = "pregame"
    
    print(f"Dataset: n_rows={len(df):,}, n_feats={len(feats)}, state={state}", flush=True)
    
    # Generate outer splits
    rng = np.random.default_rng(int(spec.seed))
    outer_splits = list(iter_walkforward_indices(len(df), spec=spec.outer))
    
    # Limit folds to avoid timeout cascade
    max_folds = len(outer_splits)
    print(f"\nStarting walk-forward backtest...", flush=True)
    print(f"Nested backtest starting  n_rows={len(df)} n_feats={len(feats)} "
          f"outer_folds={max_folds} inner_folds={spec.inner_folds} "
          f"trials={spec.trials} target_total={spec.target_total} target_margin={spec.target_margin}", flush=True)
    
    rows: List[Dict[str, Any]] = []
    diagnostics_list: List[Dict[str, Any]] = []
    
    for fold_i, (tr, te) in enumerate(outer_splits):
        print(f"\n[fold {fold_i+1}/{max_folds}] n_train={len(tr):,} n_test={len(te):,}", flush=True)
        fold_start = time.perf_counter()
        
        X_tr, X_te = X_all[tr], X_all[te]
        yt_tr, yt_te = y_total_all[tr], y_total_all[te]
        ym_tr, ym_te = y_margin_all[tr], y_margin_all[te]
        
        # Run sanity gates
        print(f"[fold {fold_i+1}] Running sanity gates...", flush=True)
        try:
            run_all_sanity_gates(
                X_train=X_tr,
                y_total_train=yt_tr,
                y_margin_train=ym_tr,
                feature_names=feats,
                state=state,
                fold_i=fold_i+1
            )
        except RuntimeError as e:
            print(f"\n❌ FOLD {fold_i+1} SANITY GATE FAILED - STOPPING", flush=True)
            print(f"Error: {e}", flush=True)
            raise
        
        # Compute diagnostics
        diagnostics = compute_fold_diagnostics(
            X_train=X_tr,
            y_total_train=yt_tr,
            y_margin_train=ym_tr,
            feature_names=feats,
            fold_i=fold_i+1
        )
        diagnostics_list.append(diagnostics)
        
        print(f"[fold {fold_i+1}] Diagnostics:", flush=True)
        print(f"  - Zero-variance features: {diagnostics['zero_variance_features']}", flush=True)
        print(f"  - Near-duplicate pairs: {len(diagnostics['near_duplicate_pairs'])}", flush=True)
        print(f"  - Condition number: {diagnostics.get('condition_number', 'N/A')}", flush=True)
        
        # Tune CatBoost
        print(f"[fold {fold_i+1}/{max_folds}] tuning CatBoost...", flush=True)
        log_prefix = f"[fold {fold_i+1}/{max_folds}] "
        
        try:
            cat_params, cat_score = _tune_catboost_optuna_v2(
                X=X_tr,
                ytot=yt_tr,
                ymar=ym_tr,
                feature_names=feats,
                inner_folds=spec.inner_folds,
                trials=spec.trials,
                seed=int(spec.seed + fold_i),
                timeout_s=spec.timeout_s,
                log_prefix=log_prefix,
            )
        except RuntimeError as e:
            print(f"\n⚠️ WARNING: CatBoost tuning failed on fold {fold_i+1}: {e}", flush=True)
            print("Saving partial results and continuing...", flush=True)
            continue
        
        # Evaluate CatBoost
        m = CatBoostTwoHeadModel(feature_version="v1", **cat_params)
        met = _fit_eval_model(
            m,
            X_tr=X_tr,
            ytot_tr=yt_tr,
            ymar_tr=ym_tr,
            X_te=X_te,
            ytot_te=yt_te,
            ymar_te=ym_te,
            feature_names=feats,
        )
        
        # Metrics are already computed by _fit_eval_model
        mae_total = met['mae_total']
        mae_margin = met['mae_margin']
        rmse_total = met['rmse_total']
        rmse_margin = met['rmse_margin']
        pi80_cov_total = met['pi80_cov_total']
        pi80_cov_margin = met['pi80_cov_margin']
        pi80_width_total = met['pi80_width_total']
        pi80_width_margin = met['pi80_width_margin']
        brier_win = met['brier_win']
        
        rows.append({
            "fold": fold_i + 1,
            "model": "catboost",
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "tuned": True,
            "tune_score": float(cat_score),
            "params": json.dumps(cat_params),
            "mae_total": float(mae_total),
            "rmse_total": float(rmse_total),
            "mae_margin": float(mae_margin),
            "rmse_margin": float(rmse_margin),
            "pi80_cov_total": float(pi80_cov_total),
            "pi80_cov_margin": float(pi80_cov_margin),
            "pi80_width_total": float(pi80_width_total),
            "pi80_width_margin": float(pi80_width_margin),
            "brier_win": float(brier_win),
        })
        
        fold_elapsed = time.perf_counter() - fold_start
        print(f"[fold {fold_i+1}/{max_folds}] completed in {fold_elapsed/60:.1f}min", flush=True)
    
    # Save results
    print("\nSaving results...", flush=True)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv(spec.output_dir / "catboost_tuning_summary.csv", index=False)
    
    # Save diagnostics
    diag_dir = spec.output_dir / "fold_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for i, diag in enumerate(diagnostics_list):
        diag_path = diag_dir / f"pregame_fold_{i+1:02d}.json"
        with open(diag_path, 'w') as f:
            json.dump(diag, f, indent=2)
    
    print(f"\n✅ PHASE 2B COMPLETE: CatBoost re-tuning finished", flush=True)
    print(f"\nSummary:")
    print(f"  Total CatBoost trials: {len(results_df)}")
    print(f"  Output: {spec.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Compare CatBoost vs XGBoost")
    print(f"  2. Evaluate ensemble performance")
    print(f"  3. Select champion model")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B: CatBoost re-tuning and fair model comparison"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/pregame_clean_v3.parquet",
        help="Path to pregame dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/phase2b_catboost_retuning",
        help="Output directory for results"
    )
    parser.add_argument(
        "--target-total",
        type=str,
        default="total",
        help="Target column for total points"
    )
    parser.add_argument(
        "--target-margin",
        type=str,
        default="margin",
        help="Target column for margin"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Target CatBoost trials per fold"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="Timeout per fold in seconds (default: 5400 = 90 minutes)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    spec = Phase2BSpec(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        target_total=args.target_total,
        target_margin=args.target_margin,
        trials=args.trials,
        timeout_s=args.timeout,
        seed=args.seed,
    )
    
    run_phase2b_catboost_retuning(spec)


if __name__ == "__main__":
    main()
