#!/bin/bash
# Pregame Stage D: Production Run (Requires Manual Confirmation)
# This script should ONLY be run after Stages A-C pass and you've reviewed results

set -e

echo "========================================"
echo "PREGAME STAGE D: PRODUCTION RUN"
echo "========================================"
echo ""
echo "⚠️ WARNING: This will take 6-8 hours to complete!"
echo ""
echo "Configuration:"
echo "  - 11 outer folds"
echo "  - 5 inner folds"
echo "  - 50 Optuna trials per model"
echo "  - 30-minute timeout per model"
echo "  - 8 models (5 baseline + 3 tuned)"
echo "  - Targets: total, margin"
echo ""

# Configuration
STATE="pregame"
DATA_FILE="data/processed/pregame_team_v2.parquet"
OUT_FILE="reports/champion_runs/latest/${STATE}_fold_metrics.csv"
TARGET_TOTAL="total"
TARGET_MARGIN="margin"

# Check if baseline exists
if [ ! -f "$OUT_FILE" ]; then
  echo "❌ ERROR: Baseline file not found. Run Stages A-C first."
  exit 1
fi

echo "Current baseline file: $OUT_FILE"
echo "Rows: $(wc -l < "$OUT_FILE")"
echo ""

read -p "Have you reviewed the baseline results from Stages A-C? (yes/no): " CONFIRM1
if [ "$CONFIRM1" != "yes" ]; then
  echo "❌ Aborted. Please review baseline results first."
  exit 1
fi

read -p "Ready to start 6-8 hour production run? This will replace the baseline file. (yes/no): " CONFIRM2
if [ "$CONFIRM2" != "yes" ]; then
  echo "❌ Aborted by user."
  exit 1
fi

echo ""
echo "Starting production run..."
echo "Start time: $(date)"
echo ""

# Activate environment
source .venv_catboost/bin/activate
export PYTHONPATH="$(pwd)"

# Backup baseline
cp "$OUT_FILE" "${OUT_FILE}.baseline_backup"
echo "✅ Baseline backed up to: ${OUT_FILE}.baseline_backup"
echo ""

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
  --train-min 800 \
  --test-size 200 \
  --step-size 200

if [ $? -ne 0 ]; then
  echo ""
  echo "❌ STAGE D FAILED: Production run returned non-zero exit code"
  echo "Baseline backup available at: ${OUT_FILE}.baseline_backup"
  exit 1
fi

echo ""
echo "✅ STAGE D PASSED: Production run completed successfully"
echo "End time: $(date)"
echo ""

# Check output
ROW_COUNT=$(wc -l < "$OUT_FILE" | tr -d ' ')
echo "Output file: $OUT_FILE"
echo "Total rows: $ROW_COUNT"
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

echo ""
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
echo "Champion Rankings:"
cat "reports/champion_runs/latest/${STATE}_leaderboard.csv" | column -t -s','
echo ""
echo "Next steps:"
echo "  1. Review leaderboard for champion selection"
echo "  2. Compare all 3 states (halftime, Q3, pregame)"
echo "  3. Select final champions for deployment"
echo ""
