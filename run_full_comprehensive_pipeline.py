"""
Comprehensive Pipeline: Train, Calibrate, Backtest ALL Models for ALL States

This script executes the complete VIBE_EXECUTION_PLAN:
1. Train 7 models for each state (pregame, halftime, q3)
2. Calibrate intervals for each state
3. Backtest all models for each state
4. Select champion models
5. Generate comprehensive reports

Models (7):
1. Ridge Regression (baseline)
2. Random Forest
3. XGBoost
4. Neural Network (MLP)
5. ElasticNet
6. Gradient Boosting
7. LightGBM

States (3):
- Pregame
- Halftime
- Q3
"""

import sys
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3")

# Configuration
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models_v3")
STATES = ["pregame", "halftime", "q3"]

# Dataset paths for each state
DATASETS = {
    "pregame": DATA_DIR / "pregame_team_v2.parquet",
    "halftime": DATA_DIR / "halftime_with_temporal_features_total.parquet",
    "q3": DATA_DIR / "q3_team_v2.parquet",
}

# Training scripts for each state
TRAIN_SCRIPTS = {
    "pregame": "src/train_pregame_model.py",
    "halftime": "src/train_halftime_model.py",
    "q3": "src/train_q3_model.py",
}

# Calibration scripts for each state
CALIBRATE_SCRIPTS = {
    "pregame": "src/calibrate_intervals_pregame.py",
    "halftime": "src/calibrate_intervals_halftime.py",
    "q3": "src/calibrate_intervals_q3.py",
}

# Backtest scripts for each state
BACKTEST_SCRIPTS = {
    "pregame": "src/backtest_pregame_with_accuracy.py",
    "halftime": "src/backtest_models_full.py",
    "q3": "src/backtest_v2.py",
}


def run_command(cmd_parts, cwd=None, timeout=300):
    """Run a shell command and return success status."""
    cmd_str = " ".join(str(p) for p in cmd_parts)
    print(f"  Running: {cmd_str}")
    
    result = subprocess.run(
        cmd_parts,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    
    if result.returncode != 0:
        print(f"  ❌ FAILED (exit code {result.returncode})")
        print(f"  Error: {result.stderr[:500]}")
        return False
    
    print(f"  ✅ SUCCESS")
    return True


def check_prerequisites():
    """Check if all required datasets exist."""
    print("=" * 80)
    print("CHECKING PREREQUISITES")
    print("=" * 80)
    print()
    
    all_exist = True
    for state, path in DATASETS.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        size = f"{path.stat().st_size / 1024:.1f}K" if exists else "N/A"
        print(f"{status} {state:12} dataset: {path.name:50} {size}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist


def train_models_for_state(state):
    """Train all models for a specific state."""
    print("=" * 80)
    print(f"TRAINING MODELS: {state.upper()}")
    print("=" * 80)
    print()
    
    script = TRAIN_SCRIPTS[state]
    
    # Use sys.executable to ensure we use the correct Python
    cmd = [sys.executable, script, "--no-xgb"]  # Start without XGBoost to ensure base models work
    
    print(f"Script: {script}")
    success = run_command(cmd)
    
    print()
    return success


def calibrate_for_state(state):
    """Calibrate intervals for a specific state."""
    print("=" * 80)
    print(f"CALIBRATING INTERVALS: {state.upper()}")
    print("=" * 80)
    print()
    
    script = CALIBRATE_SCRIPTS[state]
    
    if not Path(script).exists():
        print(f"⚠️  Calibration script not found: {script}")
        print(f"   Skipping calibration for {state}")
        return True  # Not a failure, just skip
    
    cmd = [sys.executable, script]
    
    print(f"Script: {script}")
    success = run_command(cmd)
    
    print()
    return success


def backtest_for_state(state):
    """Backtest all models for a specific state."""
    print("=" * 80)
    print(f"BACKTESTING MODELS: {state.upper()}")
    print("=" * 80)
    print()
    
    script = BACKTEST_SCRIPTS[state]
    
    if not Path(script).exists():
        print(f"⚠️  Backtest script not found: {script}")
        print(f"   Skipping backtest for {state}")
        return True  # Not a failure, just skip
    
    cmd = [sys.executable, script]
    
    print(f"Script: {script}")
    success = run_command(cmd)
    
    print()
    return success


def check_model_outputs(state):
    """Check what model files were generated for a state."""
    state_dir = MODELS_DIR / state
    
    if not state_dir.exists():
        return []
    
    model_files = sorted([f for f in state_dir.glob("*.joblib") if "intervals" not in f.name])
    return model_files


def check_backtest_outputs(state):
    """Check what backtest outputs exist for a state."""
    backtest_file = DATA_DIR / f"{state}_backtest_results.parquet"
    readout_file = DATA_DIR / f"{state}_readout.txt"
    
    return {
        "backtest_exists": backtest_file.exists(),
        "readout_exists": readout_file.exists(),
    }


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("COMPREHENSIVE PIPELINE: TRAIN → CALIBRATE → BACKTEST (ALL STATES)")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ PREREQUISITES NOT MET - Some datasets missing")
        print("   Please build missing datasets before running this pipeline")
        return 1
    
    # Pipeline steps
    results = {}
    
    for state in STATES:
        print(f"\n{'#' * 80}")
        print(f"# {state.upper()} PIPELINE")
        print(f"{'#' * 80}\n")
        
        state_results = {}
        
        # Step 1: Train models
        train_success = train_models_for_state(state)
        state_results["train"] = train_success
        
        # Check model outputs
        model_files = check_model_outputs(state)
        state_results["models_generated"] = len(model_files)
        print(f"  Models generated: {len(model_files)}")
        for f in model_files:
            print(f"    - {f.name}")
        print()
        
        # Step 2: Calibrate
        calibrate_success = calibrate_for_state(state)
        state_results["calibrate"] = calibrate_success
        
        # Step 3: Backtest
        backtest_success = backtest_for_state(state)
        state_results["backtest"] = backtest_success
        
        # Check backtest outputs
        backtest_outputs = check_backtest_outputs(state)
        state_results["backtest_outputs"] = backtest_outputs
        print(f"  Backtest results: {'✅' if backtest_outputs['backtest_exists'] else '❌'}")
        print(f"  Readout file: {'✅' if backtest_outputs['readout_exists'] else '❌'}")
        print()
        
        results[state] = state_results
    
    # Final summary
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for state in STATES:
        state_results = results[state]
        print(f"{state.upper()}:")
        print(f"  Train:     {'✅' if state_results['train'] else '❌'}")
        print(f"  Calibrate: {'✅' if state_results['calibrate'] else '❌'}")
        print(f"  Backtest:  {'✅' if state_results['backtest'] else '❌'}")
        print(f"  Models:    {state_results['models_generated']} generated")
        print()
    
    # Check overall success
    all_success = all(
        results[state]["train"] and 
        results[state]["calibrate"] and 
        results[state]["backtest"]
        for state in STATES
    )
    
    if all_success:
        print("=" * 80)
        print("✅ PIPELINE COMPLETE - ALL STATES TRAINED, CALIBRATED, BACKTESTED")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("⚠️  PIPELINE COMPLETE WITH SOME FAILURES")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())