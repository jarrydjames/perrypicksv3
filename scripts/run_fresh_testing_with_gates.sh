#!/bin/bash
# Fresh Champion Testing with Validation Gates
# Follows ROBUST_TUNING_PLAYBOOK.md methodology

set -e

echo "========================================"
echo "FRESH CHAMPION TESTING WITH GATES"
echo "========================================"
echo ""

# Configuration
STATE="halftime"
DATA_FILE="data/processed/halftime_with_temporal_features_total.parquet"
OUT_FILE="reports/champion_runs/latest/${STATE}_fold_metrics.csv"
TARGET_TOTAL="h2_total"
TARGET_MARGIN="h2_margin"

# Activate environment
echo "Activating environment..."
source .venv_catboost/bin/activate
export PYTHONPATH="$(pwd)"

# Clean up old results
echo "Cleaning old results..."
rm -rf reports/champion_runs/latest/*

# ============================================
# STAGE A: Wiring Smoke (Dry-Run)
# ============================================
echo ""
echo "========================================"
echo "STAGE A: WIRING SMOKE (DRY-RUN)"
echo "========================================"
echo "Purpose: Verify orchestration and output locations"
echo ""

python src/pipelines/champion_e2e.py \
  --config config/champion_testing_v1.json \
  --dry-run \
  --skip-checks

if [ $? -ne 0 ]; then
  echo "❌ STAGE A FAILED: Dry-run returned non-zero exit code"
  exit 1
fi

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
echo "Config: 5 outer folds, 3 inner folds, 10 trials"
echo ""

python src/modeling/nested_walkforward_backtest.py \
  --data "$DATA_FILE" \
  --out "$OUT_FILE" \
  --include-xgb --include-cat \
  --target-total "$TARGET_TOTAL" \
  --target-margin "$TARGET_MARGIN" \
  --tuner random \
  --inner-folds 3 \
  --trials 10 \
  --seed 42 \
  --train-min 500 \
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
head -5 "$OUT_FILE"
echo ""
echo "2. Check all models are present:"
cut -d',' -f2 "$OUT_FILE" | sort | uniq -c
echo ""
echo "3. Check fold distribution:"
cut -d',' -f1 "$OUT_FILE" | sort | uniq -c
echo ""

read -p "Does everything look correct? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "❌ STAGE C FAILED: Manual verification rejected"
  exit 1
fi

echo "✅ STAGE C PASSED: Manual verification approved"
echo ""

# ============================================
# STAGE D: Full Production Run (Optuna)
# ============================================
echo ""
echo "========================================"
echo "STAGE D: FULL PRODUCTION RUN"
echo "========================================"
echo "Purpose: Production-grade Bayesian tuning"
echo "Config: 11 outer folds, 5 inner folds, 50 trials"
echo "Tuner: Optuna with 30-minute timeout per model"
echo ""

read -p "Ready to start 6-8 hour production run? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "❌ Aborted by user"
  exit 1
fi

# Remove baseline output
rm -f "$OUT_FILE"

# Run full production test
python src/modeling/nested_walkforward_backtest.py \
  --data "$DATA_FILE" \
  --out "$OUT_FILE" \
  --include-xgb --include-cat --include-lgbm \
  --target-total "$TARGET_TOTAL" \
  --target-margin "$TARGET_MARGIN" \
  --tuner optuna \
  --optuna-timeout-s 1800 \
  --inner-folds 5 \
  --trials 50 \
  --seed 42 \
  --train-min 500 \
  --test-size 200 \
  --step-size 200

if [ $? -ne 0 ]; then
  echo "❌ STAGE D FAILED: Production run returned non-zero exit code"
  exit 1
fi

echo "✅ STAGE D PASSED: Production run completed successfully"
echo ""

# ============================================
# STAGE E: Leaderboard Generation
# ============================================
echo ""
echo "========================================"
echo "STAGE E: LEADERBOARD GENERATION"
echo "========================================"
echo ""

python src/pipelines/build_champion_leaderboard.py \
  --input "$OUT_FILE" \
  --output "reports/champion_runs/latest/${STATE}_leaderboard.csv" \
  --state "$STATE"

if [ $? -ne 0 ]; then
  echo "❌ STAGE E FAILED: Leaderboard generation failed"
  exit 1
fi

echo "✅ STAGE E PASSED: Leaderboard generated successfully"
echo ""
echo "========================================"
echo "ALL STAGES COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Fold metrics: $OUT_FILE"
echo "  - Leaderboard: reports/champion_runs/latest/${STATE}_leaderboard.csv"
echo ""
echo "Next steps:"
echo "  1. Review leaderboard for champion selection"
echo "  2. Run pregame testing (same process)"
echo "  3. Run Q3 testing (same process)"
echo ""