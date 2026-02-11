#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROMOTE_FLAG=""
DRY_RUN_FLAG=""
SKIP_CHECKS_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --promote)
      PROMOTE_FLAG="--promote"
      ;;
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      ;;
    --skip-checks)
      SKIP_CHECKS_FLAG="--skip-checks"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: scripts/vibe_run_champion_pipeline.sh [--dry-run] [--skip-checks] [--promote]"
      exit 64
      ;;
  esac
done

echo "[1/2] Running vibe preflight checks..."
PREFLIGHT_ALLOW_FAIL=""
if [[ -n "$DRY_RUN_FLAG" && -n "$SKIP_CHECKS_FLAG" ]]; then
  PREFLIGHT_ALLOW_FAIL="--allow-fail"
fi
python scripts/vibe_preflight.py $PREFLIGHT_ALLOW_FAIL

echo "[2/2] Running champion ops cycle..."
python src/pipelines/run_champion_ops_cycle.py $DRY_RUN_FLAG $SKIP_CHECKS_FLAG $PROMOTE_FLAG

echo "Done. Inspect reports/champion_runs/latest/run_report.json"
