from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv

from src.predict_api import predict_game
from src.automation import build_message, format_prediction, post_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Post predictions to Discord")
    parser.add_argument("game_ids", nargs="+", help="Game IDs to post")
    parser.add_argument("--mode", default="auto", choices=["auto", "pregame", "halftime", "q3"])
    parser.add_argument("--out", default=None, help="Optional CSV path to save predictions")
    args = parser.parse_args()

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise SystemExit("DISCORD_WEBHOOK_URL is required")

    lines: List[str] = []
    rows = []
    for gid in args.game_ids:
        pred = predict_game(gid, mode=args.mode)
        lines.append(format_prediction(gid, pred))
        if pred.get("status") == "success":
            rows.append(
                {
                    "game_id": gid,
                    "total": pred.get("total"),
                    "margin": pred.get("margin"),
                    "winner": pred.get("winner"),
                    "model": pred.get("model_used") or pred.get("model"),
                }
            )

    message = build_message(lines)
    post_message(webhook_url, content=message, username="PerryPicks")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["game_id", "total", "margin", "winner", "model"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
