#!/bin/bash
# Pregame Stages A-C Only - Stop Before Production Run
# Follows ROBUST_TUNING_PLAYBOOK.md methodology

set -e

echo "========================================"
echo "PREGAME STAGES A-C (PRE-PRODUCTION)"
echo "========================================"
echo ""

# Configuration
STATE="pregame"
DATA_FILE="data/processed/pregame_team_v2.parquet"
OUT_FILE="reports/champion_runs/latest/${STATE}_fold_metrics.csv"
TARGET_TOTAL="total"
TARGET_MARGIN="margin"

# Activate environment
echo "Activating environment..."
source .venv_catboost/bin/activate
export PYTHONPATH="$(pwd)"

# Clean up old results
echo "Cleaning old pregame results..."
rm -f "$OUT_FILE"
rm -rf reports/champion_runs/latest/fold_diagnostics/${STATE}_*.json

# ============================================
# STAGE A: Wiring Smoke (Dry-Run)
# ============================================
echo ""
echo "========================================"
echo "STAGE A: WIRING SMOKE (DRY-RUN)"
echo "========================================"
echo "Purpose: Verify orchestration and output locations"
echo "Duration: Seconds"
echo ""

python src/pipelines/champion_e2e.py \
  --config config/champion_testing_v1.json \
  --dry-run \
  --skip-checks

if [ $? -ne 0 ]; then
  echo "❌ STAGE A FAILED: Dry-run returned non-zero exit code"
  exit 1
fi

echo ""
echo "✅ STAGE A PASSED: Dry-run completed successfully"
echo ""

# ============================================
# STAGE B: Baseline Random Search (Small Scale)
# ============================================
echo ""
echo "========================================"
echo "STAGE B: BASELINE RANDOM SEARCH"
echo "========================================"
echo "Purpose: Produce reliable baseline with minimal compute"
echo "Config: 5 outer folds, 5 inner folds, 30 trials"
echo "Duration: 20-40 minutes"
echo ""

python src/modeling/nested_walkforward_backtest.py \
  --data "$DATA_FILE" \
  --out "$OUT_FILE" \
  --include-xgb --include-cat \
  --target-total "$TARGET_TOTAL" \
  --target-margin "$TARGET_MARGIN" \
  --tuner random \
  --max-folds 5 \
  --inner-folds 5 \
  --trials 30 \
  --seed 42 \
  --train-min 800 \
  --test-size 200 \
  --step-size 200

if [ $? -ne 0 ]; then
  echo "❌ STAGE B FAILED: Training returned non-zero exit code"
  exit 1
fi

# Check output file
if [ ! -f "$OUT_FILE" ]; then
  echo "❌ STAGE B FAILED: Output file not created: $OUT_FILE"
  exit 1
fi

# Check file is non-empty
if [ ! -s "$OUT_FILE" ]; then
  echo "❌ STAGE B FAILED: Output file is empty: $OUT_FILE"
  exit 1
fi

# Check number of rows (should have 5 folds × 7 models = 35 rows minimum)
ROW_COUNT=$(wc -l < "$OUT_FILE" | tr -d ' ')
if [ "$ROW_COUNT" -lt 35 ]; then
  echo "❌ STAGE B FAILED: Expected at least 35 rows, got $ROW_COUNT"
  exit 1
fi

echo ""
echo "✅ STAGE B PASSED: Baseline training completed successfully"
echo "   Output: $OUT_FILE ($ROW_COUNT rows)"
echo ""

# ============================================
# STAGE C: Manual Verification
# ============================================
echo ""
echo "========================================"
echo "STAGE C: MANUAL VERIFICATION"
echo "========================================"
echo "Please verify the following:"
echo ""

echo "1. Check output file structure:"
echo "---"
head -5 "$OUT_FILE"
echo "..."
echo ""

echo "2. Check all models are present:"
echo "---"
cut -d',' -f2 "$OUT_FILE" | tail -n +2 | sort | uniq -c
echo ""

echo "3. Check fold distribution:"
echo "---"
cut -d',' -f1 "$OUT_FILE" | tail -n +2 | sort | uniq -c
echo ""

echo "4. Sample metrics (first 10 rows):"
echo "---"
head -11 "$OUT_FILE" | column -t -s','
echo ""

echo "5. Check fold diagnostics:"
echo "---"
ls -lh reports/champion_runs/latest/fold_diagnostics/${STATE}_*.json 2>&1 | head -10
echo ""

echo "========================================"
echo "STAGES A-C COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Baseline results are ready for review."
echo ""
echo "Next step: MANUAL CONFIRMATION"
echo ""
echo "If everything looks correct, you can proceed to Stage D (production run):"
echo ""
echo "  ./scripts/run_pregame_stage_d_production.sh"
echo ""
echo "This will start a 6-8 hour production run with full Optuna tuning."
echo ""
