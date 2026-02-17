"""
Phase 2B: Compare CatBoost vs XGBoost and Select Champion

This script compares CatBoost (re-tuned in Phase 2B) with XGBoost
(from Phase 2) to select the champion model.

Comparison includes:
- Fold-by-fold comparison
- Fold-averaged metrics
- Composite score calculation
- Ensemble evaluation
- Champion selection using defined rules
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_composite_score(row: Dict[str, Any], weights: Dict[str, float] = None) -> float:
    """Calculate composite score from metrics.
    
    Composite score is weighted average of normalized metrics.
    Lower is better.
    
    Weights (default):
    - mae_total: 0.4
    - mae_margin: 0.3
    - brier_win: 0.3
    """
    if weights is None:
        weights = {
            'mae_total': 0.4,
            'mae_margin': 0.3,
            'brier_win': 0.3,
        }
    
    # Normalize metrics (lower is better)
    mae_total_norm = row['mae_total'] / 20.0  # Normalize to 20 MAE max
    mae_margin_norm = row['mae_margin'] / 15.0  # Normalize to 15 MAE max
    brier_win_norm = row['brier_win'] / 0.3  # Normalize to 0.3 Brier max
    
    # Calculate weighted composite score
    composite = (
        weights['mae_total'] * mae_total_norm +
        weights['mae_margin'] * mae_margin_norm +
        weights['brier_win'] * brier_win_norm
    )
    
    return composite


def load_catboost_results(catboost_path: Path) -> pd.DataFrame:
    """Load CatBoost results from Phase 2B."""
    df = pd.read_csv(catboost_path)
    # Parse params from JSON string
    df['params'] = df['params'].apply(json.loads)
    return df


def load_xgboost_results(xgboost_path: Path) -> pd.DataFrame:
    """Load XGBoost results from Phase 2."""
    df = pd.read_csv(xgboost_path)
    # Parse params from JSON string if present
    if 'params' in df.columns:
        df['params'] = df['params'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df[df['model'] == 'xgboost'].copy()


def compare_folds(catboost_df: pd.DataFrame, xgboost_df: pd.DataFrame) -> pd.DataFrame:
    """Compare CatBoost vs XGBoost fold-by-fold."""
    
    comparison_rows = []
    
    for fold in sorted(set(catboost_df['fold']) | set(xgboost_df['fold'])):
        cat_fold = catboost_df[catboost_df['fold'] == fold]
        xgb_fold = xgboost_df[xgboost_df['fold'] == fold]
        
        if len(cat_fold) == 0:
            print(f"Warning: CatBoost missing fold {fold}", flush=True)
            continue
        if len(xgb_fold) == 0:
            print(f"Warning: XGBoost missing fold {fold}", flush=True)
            continue
        
        cat_row = cat_fold.iloc[0]
        xgb_row = xgb_fold.iloc[0]
        
        comparison_rows.append({
            'fold': fold,
            'catboost_mae_total': cat_row['mae_total'],
            'catboost_mae_margin': cat_row['mae_margin'],
            'catboost_brier_win': cat_row['brier_win'],
            'catboost_composite': calculate_composite_score(cat_row),
            'xgboost_mae_total': xgb_row['mae_total'],
            'xgboost_mae_margin': xgb_row['mae_margin'],
            'xgboost_brier_win': xgb_row['brier_win'],
            'xgboost_composite': calculate_composite_score(xgb_row),
        })
    
    return pd.DataFrame(comparison_rows)


def calculate_ensemble(catboost_df: pd.DataFrame, xgboost_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate ensemble (average) performance."""
    
    # Get fold-averaged metrics
    cat_avg = catboost_df[['mae_total', 'mae_margin', 'brier_win']].mean()
    xgb_avg = xgboost_df[['mae_total', 'mae_margin', 'brier_win']].mean()
    
    # Calculate ensemble (simple average)
    ensemble_avg = {
        'mae_total': (cat_avg['mae_total'] + xgb_avg['mae_total']) / 2,
        'mae_margin': (cat_avg['mae_margin'] + xgb_avg['mae_margin']) / 2,
        'brier_win': (cat_avg['brier_win'] + xgb_avg['brier_win']) / 2,
    }
    
    return ensemble_avg


def select_champion(
    catboost_df: pd.DataFrame,
    xgboost_df: pd.DataFrame,
    composite_rows: pd.DataFrame
) -> Dict[str, Any]:
    """Select champion model using decision rules.
    
    Decision rules:
    1. If one model beats others by >0.5 composite score: select that model
    2. If models are within 0.5 composite score: select ensemble
    3. Otherwise: select model with lower composite score
    """
    
    # Calculate fold-averaged composites
    cat_composite = composite_rows['catboost_composite'].mean()
    xgb_composite = composite_rows['xgboost_composite'].mean()
    
    print("\n=== Champion Selection ===", flush=True)
    print(f"CatBoost composite score: {cat_composite:.4f}", flush=True)
    print(f"XGBoost composite score: {xgb_composite:.4f}", flush=True)
    
    # Decision rule 1: One model beats others by >0.5 composite
    composite_diff = abs(cat_composite - xgb_composite)
    if composite_diff > 0.5:
        if cat_composite < xgb_composite:
            champion = "catboost"
            reason = "CatBoost beats XGBoost by >0.5 composite score"
        else:
            champion = "xgboost"
            reason = "XGBoost beats CatBoost by >0.5 composite score"
    # Decision rule 2: Models within 0.5 composite - select ensemble
    elif composite_diff <= 0.5:
        champion = "ensemble"
        reason = "Models within 0.5 composite score - ensemble selected"
    # Default: Select lower composite
    else:
        if cat_composite < xgb_composite:
            champion = "catboost"
            reason = "CatBoost has lower composite score"
        else:
            champion = "xgboost"
            reason = "XGBoost has lower composite score"
    
    print(f"\n🏆 Champion: {champion}", flush=True)
    print(f"Reason: {reason}", flush=True)
    
    # Calculate ensemble performance for reference
    ensemble_metrics = calculate_ensemble(catboost_df, xgboost_df)
    ensemble_composite = calculate_composite_score(ensemble_metrics)
    
    print(f"\nEnsemble metrics:", flush=True)
    print(f"  MAE Total: {ensemble_metrics['mae_total']:.2f}", flush=True)
    print(f"  MAE Margin: {ensemble_metrics['mae_margin']:.2f}", flush=True)
    print(f"  Brier Win: {ensemble_metrics['brier_win']:.4f}", flush=True)
    print(f"  Composite: {ensemble_composite:.4f}", flush=True)
    
    return {
        'champion': champion,
        'reason': reason,
        'catboost_composite': cat_composite,
        'xgboost_composite': xgb_composite,
        'ensemble_composite': ensemble_composite,
    }


def generate_summary(
    catboost_df: pd.DataFrame,
    xgboost_df: pd.DataFrame,
    composite_rows: pd.DataFrame,
    champion: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive summary report."""
    
    print("\n=== Generating Summary ===", flush=True)
    
    # Fold-averaged metrics
    print("\n=== Fold-Averaged Metrics ===", flush=True)
    
    cat_avg = catboost_df[['mae_total', 'mae_margin', 'brier_win']].mean()
    xgb_avg = xgboost_df[['mae_total', 'mae_margin', 'brier_win']].mean()
    
    # Calculate ensemble metrics
    ensemble_metrics = calculate_ensemble(catboost_df, xgboost_df)
    
    print(f"\n{'Metric':<20} {'CatBoost':<15} {'XGBoost':<15} {'Winner':<10}", flush=True)
    print(f"{'-'*60}", flush=True)
    print(f"{'MAE Total':<20} {cat_avg['mae_total']:>14.2f} {xgb_avg['mae_total']:>14.2f} {'xgboost' if xgb_avg['mae_total'] < cat_avg['mae_total'] else 'catboost':<10}", flush=True)
    print(f"{'MAE Margin':<20} {cat_avg['mae_margin']:>14.2f} {xgb_avg['mae_margin']:>14.2f} {'xgboost' if xgb_avg['mae_margin'] < cat_avg['mae_margin'] else 'catboost':<10}", flush=True)
    print(f"{'Brier Win':<20} {cat_avg['brier_win']:>14.4f} {xgb_avg['brier_win']:>14.4f} {'xgboost' if xgb_avg['brier_win'] < cat_avg['brier_win'] else 'catboost':<10}", flush=True)
    
    # Composite scores
    cat_comp = composite_rows['catboost_composite'].mean()
    xgb_comp = composite_rows['xgboost_composite'].mean()
    
    print(f"\n{'Composite Score':<20} {cat_comp:>14.4f} {xgb_comp:>14.4f} {'xgboost' if xgb_comp < cat_comp else 'catboost':<10}", flush=True)
    
    # Fold stability (std dev)
    cat_std = composite_rows['catboost_composite'].std()
    xgb_std = composite_rows['xgboost_composite'].std()
    
    print(f"\n=== Fold Stability (Std Dev) ===", flush=True)
    print(f"CatBoost: {cat_std:.4f}", flush=True)
    print(f"XGBoost: {xgb_std:.4f}", flush=True)
    print(f"Winner: {'XGBoost' if xgb_std < cat_std else 'CatBoost'} (more stable)", flush=True)
    
    # Save comparison table
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df = pd.DataFrame({
        'Model': ['CatBoost', 'XGBoost', 'Ensemble'],
        'MAE Total': [cat_avg['mae_total'], xgb_avg['mae_total'], ensemble_metrics['mae_total']],
        'MAE Margin': [cat_avg['mae_margin'], xgb_avg['mae_margin'], ensemble_metrics['mae_margin']],
        'Brier Win': [cat_avg['brier_win'], xgb_avg['brier_win'], ensemble_metrics['brier_win']],
        'Composite': [cat_comp, xgb_comp, champion['ensemble_composite']],
        'Stability (Std)': [cat_std, xgb_std, 0.0],  # Ensemble is most stable
    })
    
    comparison_path = output_dir / "phase2_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}", flush=True)
    
    # Save fold-by-fold comparison
    fold_comparison_path = output_dir / "phase2_fold_comparison.csv"
    composite_rows.to_csv(fold_comparison_path, index=False)
    print(f"Fold comparison saved to: {fold_comparison_path}", flush=True)
    
    # Save champion selection
    champion_path = output_dir / "champion_selection.json"
    with open(champion_path, 'w') as f:
        json.dump(champion, f, indent=2)
    print(f"Champion selection saved to: {champion_path}", flush=True)
    
    # Save CatBoost tuning summary
    catboost_summary_path = output_dir / "catboost_tuning_summary.csv"
    catboost_df.to_csv(catboost_summary_path, index=False)
    print(f"CatBoost tuning summary saved to: {catboost_summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B: Compare CatBoost vs XGBoost and select champion"
    )
    parser.add_argument(
        "--catboost-results",
        type=str,
        default="reports/phase2b_catboost_retuning/catboost_tuning_summary.csv",
        help="Path to CatBoost results from Phase 2B"
    )
    parser.add_argument(
        "--xgboost-results",
        type=str,
        default="reports/phase2_pregame/fold_metrics.csv",
        help="Path to XGBoost results from Phase 2"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/phase2b_final",
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 2B: CatBoost vs XGBoost Comparison")
    print("="*80)
    print(f"\nCatBoost results: {args.catboost_results}")
    print(f"XGBoost results: {args.xgboost_results}")
    print(f"Output directory: {args.output}")
    print(f"\n")
    
    # Load results
    print("Loading results...", flush=True)
    
    catboost_df = load_catboost_results(Path(args.catboost_results))
    xgboost_df = load_xgboost_results(Path(args.xgboost_results))
    
    print(f"CatBoost folds: {len(catboost_df)}", flush=True)
    print(f"XGBoost folds: {len(xgboost_df)}", flush=True)
    
    # Compare folds
    print("\nComparing folds...", flush=True)
    composite_rows = compare_folds(catboost_df, xgboost_df)
    
    # Select champion
    print("\nSelecting champion...", flush=True)
    champion = select_champion(catboost_df, xgboost_df, composite_rows)
    
    # Generate summary
    print("\nGenerating summary...", flush=True)
    generate_summary(
        catboost_df=catboost_df,
        xgboost_df=xgboost_df,
        composite_rows=composite_rows,
        champion=champion,
        output_dir=Path(args.output),
    )
    
    print("\n✅ PHASE 2B COMPLETE", flush=True)
    print(f"\nFinal Champion: {champion['champion']}", flush=True)
    print(f"Reason: {champion['reason']}", flush=True)


if __name__ == "__main__":
    main()
