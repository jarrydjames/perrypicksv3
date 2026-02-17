#!/bin/bash
# Phase 2B Background Runner
# This script runs Phase 2B CatBoost re-tuning in the background

cd "$(dirname "$0")/.."

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Run Phase 2B master script
# Note: Using .venv_catboost for CatBoost support
.venv_catboost/bin/python scripts/run_phase2b_master.py --trials 40 --timeout 5400 2>&1 | tee reports/phase2b_catboost_tuning.out
