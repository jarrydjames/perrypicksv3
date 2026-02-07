"""Generate CLV report segmented by trigger window and period."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def build_report(db_path: Path, days: int = 7) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute(
        """
        SELECT trigger_type,
               COUNT(*) AS picks,
               AVG(clv_points) AS avg_clv,
               SUM(CASE WHEN clv_points > 0 THEN 1 ELSE 0 END) AS positive_clv
        FROM clv_tracking
        WHERE created_at_utc >= datetime('now', ?)
        GROUP BY trigger_type
        ORDER BY picks DESC
        """,
        (f"-{int(days)} days",),
    )
    except sqlite3.OperationalError:
        rows = []
    else:
        rows = cur.fetchall()
    conn.close()

    lines = [f"# CLV Report (last {days} days)", "", "| Trigger | Picks | Avg CLV | Positive CLV % |", "|---|---:|---:|---:|"]
    if not rows:
        lines.append("| none | 0 | 0.00 | 0.0% |")
    for r in rows:
        picks = int(r["picks"] or 0)
        pos = int(r["positive_clv"] or 0)
        pct = (100.0 * pos / picks) if picks else 0.0
        lines.append(f"| {r['trigger_type']} | {picks} | {(r['avg_clv'] or 0):.3f} | {pct:.1f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/automation.db")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", default="reports/clv_report.md")
    args = parser.parse_args()

    report = build_report(Path(args.db), args.days)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
