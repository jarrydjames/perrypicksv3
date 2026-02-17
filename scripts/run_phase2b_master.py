"""
Phase 2B Master: CatBoost Re-tuning and Fair Model Comparison

This script runs the complete Phase 2B workflow:
1. Re-tune CatBoost with improved parameters
2. Compare CatBoost vs XGBoost
3. Select champion model
4. Generate comprehensive reports

Usage:
    python scripts/run_phase2b_master.py

The script will:
- Run CatBoost tuning with 90-minute timeout
- Target 40-50 trials per fold (minimum 25)
- Compare with XGBoost from Phase 2
- Select champion using decision rules
- Save all results to reports/phase2b_final/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)


def run_command(cmd: list[str], cwd: Path, venv_name: str = ".venv_catboost") -> bool:
    """Run a command and return success status."""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(cwd)
    
    # Use specified venv (default: .venv_catboost for CatBoost support)
    if cmd[0].endswith("python"):
        cmd[0] = str(cwd / venv_name / "bin" / "python")
    
    print(f"\nRunning: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=False,
        text=True,
        env=env
    )
    success = result.returncode == 0
    
    if not success:
        print(f"\n❌ Command failed with exit code {result.returncode}", flush=True)
    
    return success


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
        "--phase2-results",
        type=str,
        default="reports/phase2_pregame/fold_metrics.csv",
        help="Path to Phase 2 results (XGBoost)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Target CatBoost trials per fold (default: 40)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="Timeout per fold in seconds (default: 5400 = 90 minutes)"
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip CatBoost tuning and only run comparison (if already done)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path.cwd()
    
    print("="*80)
    print("PHASE 2B: CatBoost Re-tuning and Fair Model Comparison")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Phase 2 results: {args.phase2_results}")
    print(f"  CatBoost trials: {args.trials}")
    print(f"  Timeout: {args.timeout}s ({args.timeout/60:.0f} minutes)")
    print(f"  Skip tuning: {args.skip_tuning}")
    print(f"\n")
    
    # Step 1: Run CatBoost re-tuning
    if not args.skip_tuning:
        print("\n=== STEP 1: CatBoost Re-tuning ===", flush=True)
        
        tuning_cmd = [
            "python",  # Will be replaced with .venv_catboost/bin/python by run_command
            "scripts/run_pregame_phase2b_catboost_retuning.py",
            "--data", args.data,
            "--trials", str(args.trials),
            "--timeout", str(args.timeout),
        ]
        
        success = run_command(tuning_cmd, base_dir)
        
        if not success:
            print("\n❌ CatBoost re-tuning failed", flush=True)
            sys.exit(1)
        
        print("\n✅ CatBoost re-tuning complete", flush=True)
    else:
        print("\n=== SKIP: CatBoost re-tuning (already done) ===", flush=True)
    
    # Step 2: Compare models and select champion
    print("\n=== STEP 2: Model Comparison and Champion Selection ===", flush=True)
    
    comparison_cmd = [
        "python",  # Will be replaced with .venv_catboost/bin/python by run_command
        "scripts/compare_phase2b_models.py",
        "--catboost-results", "reports/phase2b_catboost_retuning/catboost_tuning_summary.csv",
        "--xgboost-results", args.phase2_results,
        "--output", "reports/phase2b_final",
    ]
    
    success = run_command(comparison_cmd, base_dir)
    
    if not success:
        print("\n❌ Model comparison failed", flush=True)
        sys.exit(1)
    
    print("\n✅ Phase 2B complete", flush=True)
    
    # Print summary
    print("\n" + "="*80, flush=True)
    print("PHASE 2B SUMMARY", flush=True)
    print("="*80, flush=True)
    print("\nOutputs:")
    print("  reports/phase2b_catboost_retuning/")
    print("    - CatBoost tuning summary")
    print("    - Fold diagnostics")
    print("\n  reports/phase2b_final/")
    print("    - Model comparison table")
    print("    - Fold-by-fold comparison")
    print("    - Champion selection")
    print("\nNext steps:")
    print("  1. Review champion selection in reports/phase2b_final/champion_selection.json")
    print("  2. Train final champion model on full dataset")
    print("  3. Deploy to production")
    print("\n" + "="*80, flush=True)

if __name__ == "__main__":
    main()
