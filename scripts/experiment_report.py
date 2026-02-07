"""Generate experiment registry and performance coverage report."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def build_report(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute(
        """
        SELECT e.experiment_id,
               e.model_version,
               e.calibration_version,
               e.bet_policy_version,
               e.output_template_version,
               COUNT(p.id) AS picks
        FROM experiments e
        LEFT JOIN picks p ON p.experiment_id = e.experiment_id
        GROUP BY e.experiment_id, e.model_version, e.calibration_version, e.bet_policy_version, e.output_template_version
        ORDER BY picks DESC, e.experiment_id ASC
        """
    )
    except sqlite3.OperationalError:
        rows = []
    else:
        rows = cur.fetchall()
    conn.close()

    lines = ["# Experiment Coverage Report", "", "| Experiment | Model | Calibration | Bet Policy | Template | Picks |", "|---|---|---|---|---|---:|"]
    if not rows:
        lines.append("| none | - | - | - | - | 0 |")
    for r in rows:
        lines.append(
            f"| {r['experiment_id']} | {r['model_version'] or '-'} | {r['calibration_version'] or '-'} | {r['bet_policy_version'] or '-'} | {r['output_template_version'] or '-'} | {int(r['picks'] or 0)} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/automation.db")
    parser.add_argument("--out", default="reports/experiment_report.md")
    args = parser.parse_args()

    report = build_report(Path(args.db))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
