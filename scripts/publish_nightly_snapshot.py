"""Publish nightly what-failed-today snapshot with miss explanations."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def build_snapshot(db_path: Path, date_str: str) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute(
        """
        SELECT g.game_id, g.away_team, g.home_team, g.score_away, g.score_home,
               m.trigger_type, m.explanation_bullets_json, m.created_at_utc
        FROM miss_explanations m
        JOIN games g ON g.game_id = m.game_id
        WHERE date(m.created_at_utc) = date(?)
        ORDER BY m.created_at_utc DESC
        """,
        (date_str,),
    )
    except sqlite3.OperationalError:
        rows = []
    else:
        rows = cur.fetchall()
    conn.close()

    lines = [f"# Nightly Snapshot: {date_str}", "", "## What failed today", ""]
    if not rows:
        lines.append("No miss explanations were recorded for this date.")
        return "\n".join(lines) + "\n"

    import json

    for r in rows:
        lines.append(f"### {r['away_team']} @ {r['home_team']} ({r['game_id']}) — {r['trigger_type']}")
        lines.append(f"Final: {r['away_team']} {r['score_away']} - {r['home_team']} {r['score_home']}")
        bullets = json.loads(r["explanation_bullets_json"])
        for b in bullets[:3]:
            lines.append(f"- {b}")
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/automation.db")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="reports/nightly_snapshot.md")
    args = parser.parse_args()

    report = build_snapshot(Path(args.db), args.date)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
