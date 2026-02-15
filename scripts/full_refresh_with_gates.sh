#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN=false
SKIP_PREGAME=false
SKIP_HALFTIME=false
SKIP_Q3=false
SKIP_AUDITS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --skip-pregame)
      SKIP_PREGAME=true
      shift
      ;;
    --skip-halftime)
      SKIP_HALFTIME=true
      shift
      ;;
    --skip-q3)
      SKIP_Q3=true
      shift
      ;;
    --skip-audits)
      SKIP_AUDITS=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

run_step() {
  local name="$1"
  local cmd="$2"
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] $name: $cmd"
  else
    echo "[RUN] $name"
    (cd "$PROJECT_DIR" && eval "$cmd")
  fi
}

if [ "$SKIP_PREGAME" = false ]; then
  run_step "Pregame pipeline" "python scripts/run_pipeline.py pregame"
fi

if [ "$SKIP_HALFTIME" = false ]; then
  run_step "Halftime pipeline" "python scripts/run_pipeline.py halftime"
fi

if [ "$SKIP_Q3" = false ]; then
  run_step "Q3 pipeline" "python scripts/run_pipeline.py q3"
fi

if [ "$SKIP_AUDITS" = false ]; then
  run_step "Data freshness audit" "python src/data/data_freshness_audit.py --policy config/data_freshness_policy_v1.json --out reports/champion_runs/data_freshness_audit.json --strict"
  run_step "Refresh readiness" "python src/pipelines/champion_refresh_cycle.py --policy config/champion_refresh_policy_v1.json --data-policy config/data_freshness_policy_v1.json --out reports/champion_runs/refresh_readiness.json --data-audit-out reports/champion_runs/data_freshness_audit.json"
fi

echo "Full refresh with gates completed."
