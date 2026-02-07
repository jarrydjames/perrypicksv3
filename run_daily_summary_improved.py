#!/usr/bin/env python3
"""Improved daily summary runner with data-source transparency."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.env import load_environment
from core.timezone import now_utc
from core.discord_client import DiscordWebhookClient
from src.predict_api import predict_game


load_environment(search_from=Path(__file__).resolve())

TARGET_DATE = os.getenv("SUMMARY_DATE", "2026-02-05")
DB_PATH = os.getenv("AUTOMATION_DB_PATH", "data/automation.db")


def load_games(target_date: str) -> list[dict[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT game_id, home_team, away_team, start_time_utc
        FROM games
        WHERE game_date = ?
          AND NOT game_id LIKE 'test_%'
        ORDER BY start_time_utc
        """,
        (target_date,),
    )
    games = [
        {
            "game_id": row[0],
            "home_team": row[1],
            "away_team": row[2],
            "start_time_utc": row[3],
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return games


def main() -> int:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        print("ERROR: DISCORD_WEBHOOK_URL environment variable not set")
        return 1

    games = load_games(TARGET_DATE)
    print(f"Found {len(games)} games for {TARGET_DATE}")

    predictions: list[dict[str, object]] = []
    source_counter: Counter[str] = Counter()
    warnings: list[str] = []
    stale_games = 0

    for i, game in enumerate(games):
        if i > 0:
            time.sleep(1.0)

        result = predict_game(game_input=game["game_id"], mode="pregame", fetch_odds=False)
        if result.get("status") not in {"success", "warning"}:
            print(f"✗ Failed {game['away_team']} @ {game['home_team']}: {result.get('error', 'unknown')}" )
            continue

        total = float(result.get("total", 0.0))
        margin = float(result.get("margin", 0.0))
        pred_home = (total - margin) / 2
        pred_away = (total + margin) / 2
        pred_winner = result.get("home_name") if margin < 0 else result.get("away_name")

        data_source = result.get("data_source", {})
        home_source = str(data_source.get("home_stats_season", "UNKNOWN"))
        away_source = str(data_source.get("away_stats_season", "UNKNOWN"))
        source_counter.update([home_source, away_source])

        if result.get("data_warning"):
            warnings.append(f"{game['away_team']} @ {game['home_team']}: {result['data_warning']}")

        freshness = result.get("data_freshness", {})
        if isinstance(freshness, dict) and freshness.get("is_stale"):
            stale_games += 1

        predictions.append(
            {
                "game_id": game["game_id"],
                "away_name": result.get("away_name", game["away_team"]),
                "home_name": result.get("home_name", game["home_team"]),
                "predicted_away_score": pred_away,
                "predicted_home_score": pred_home,
                "predicted_total": total,
                "predicted_margin": margin,
                "predicted_winner": pred_winner,
                "model_used": result.get("model_used", "UNKNOWN"),
                "data_source": f"{home_source}/{away_source}",
            }
        )

        badge = "⚠" if result.get("status") == "warning" else "✓"
        print(
            f"{badge} {game['away_team']} @ {game['home_team']} -> "
            f"{pred_away:.1f} @ {pred_home:.1f} | src={home_source}/{away_source}"
        )

    discord = DiscordWebhookClient(webhook_url=webhook_url)
    message = discord.format_daily_summary_post(predictions=predictions, timestamp=now_utc(), date=TARGET_DATE)

    if source_counter:
        source_lines = ["", "**Data Sources**"]
        for source, count in sorted(source_counter.items()):
            source_lines.append(f"• {source}: {count} team lookups")
        message += "\n" + "\n".join(source_lines)

    if stale_games:
        message += f"\n\n**Data Freshness**\n• Stale-data flagged games: {stale_games}"

    if warnings:
        message += "\n\n**Warnings**\n" + "\n".join(f"• {w}" for w in warnings)

    message_id = discord.post_message(message)
    if message_id:
        print(f"✅ Posted to Discord. Message ID: {message_id}")
    else:
        print("⚠️ Posted to Discord (no message ID returned)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
