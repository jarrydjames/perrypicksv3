#!/usr/bin/env bash
set -euo pipefail

DB_PATH="${AUTOMATION_DB_PATH:-data/automation.db}"
REPORT_DATE="${REPORT_DATE:-$(date +%F)}"
DELIVER_REPORTS="${DELIVER_REPORTS:-0}"

python scripts/clv_report.py --db "$DB_PATH" --days 7 --out reports/clv_report.md
python scripts/experiment_report.py --db "$DB_PATH" --out reports/experiment_report.md
python scripts/publish_nightly_snapshot.py --db "$DB_PATH" --date "$REPORT_DATE" --out reports/nightly_snapshot.md

if [[ "$DELIVER_REPORTS" == "1" ]]; then
  python scripts/deliver_reports.py --report-dir reports --date "$REPORT_DATE"
fi

echo "Nightly reports generated for ${REPORT_DATE}"
