#!/bin/bash
# Halftime Production Run - No Confirmation (Baseline Already Reviewed)
# This runs the full 51-fold production test

set -e

echo "========================================"
echo "HALFTIME PRODUCTION RUN (51 FOLDS)"
echo "========================================"
echo ""
echo "⚠️ This will take 6-8 hours to complete!"
echo ""
echo "Configuration:"
echo "  - 51 outer folds (full dataset)"
echo "  - 5 inner folds"
echo "  - 50 Optuna trials per model"
echo "  - 30-minute timeout per model"
echo "  - 2 models (XGBoost + CatBoost for ensemble)"
echo "  - Targets: h2_total, h2_margin"
echo ""

# Configuration
STATE="halftime"
DATA_FILE="data/processed/halftime_with_temporal_features_total.parquet"
OUT_FILE="reports/champion_runs/latest/${STATE}_fold_metrics.csv"
TARGET_TOTAL="h2_total"
TARGET_MARGIN="h2_margin"

echo "Start time: $(date)"
echo ""

# Activate environment
source .venv_catboost/bin/activate
export PYTHONPATH="$(pwd)"

# Backup baseline
if [ -f "$OUT_FILE" ]; then
  cp "$OUT_FILE" "${OUT_FILE}.baseline_backup"
  echo "✅ Baseline backed up to: ${OUT_FILE}.baseline_backup"
  echo ""
fi

# Remove baseline output
rm -f "$OUT_FILE"

echo "Starting production run..."
echo ""

# Run full production test
python src/modeling/nested_walkforward_backtest.py \
  --data "$DATA_FILE" \
  --out "$OUT_FILE" \
  --include-xgb --include-cat \
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
  echo "❌ PRODUCTION RUN FAILED"
  echo "Baseline backup available at: ${OUT_FILE}.baseline_backup"
  exit 1
fi

echo ""
echo "✅ PRODUCTION RUN COMPLETED SUCCESSFULLY"
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
echo "  2. Analyze results for potential ensemble"
echo "  3. Compare XGBoost vs CatBoost performance"
echo ""
