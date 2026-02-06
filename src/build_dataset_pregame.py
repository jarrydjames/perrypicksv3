"""Build pregame training dataset - follows same methodology as halftime/Q3 models.

This dataset creates features from pregame information only (team stats, form, etc.)
and trains models to predict final game outcomes (total, margin).

Key differences from halftime/Q3:
- No game state features (no h1_home, q3_home, etc.)
- Only pregame features: team stats, recent form, schedule factors
- Predicts same targets: final total, final margin

This allows predictions BEFORE the game starts, complementing halftime and Q3 models.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

from datetime import datetime, timezone


# Reuse feature builders and API endpoints from existing code

# Same endpoints
CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"


def extract_pregame_row(gid: str) -> dict:
    """
    Extract a single pregame training row for a game.
    
    Pregame features only (no game state):
    - Team statistics from box score
    - Rate features (points per possession, etc.)
    - Team identifiers
    
    Targets: final game total and margin
    """
    from src.build_dataset_team_v2 import final_score_from_box
    from src.build_dataset_v2 import add_rate_features, fetch_box, team_totals_from_box_team

    game = fetch_box(gid)
    
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    home_tri = home.get("teamTricode", "HOME")
    away_tri = away.get("teamTricode", "AWAY")
    
    # Get team stats (full game stats, not game-state specific)
    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)

    game_date = _parse_game_date(game)

    # Build row with pregame features only
    # Note: No game state (h1_home, q3_home, etc.) - this is pregame!
    row = {
        "game_id": gid,
        "game_date": game_date.isoformat() if game_date else None,
        "home_tri": home_tri,
        "away_tri": away_tri,
        # Pregame features: team stats and rates
        **add_rate_features("home", ht, at),
        **add_rate_features("away", at, ht),
    }
    
    # Add team totals as priors (same as halftime/Q3)
    for k, v in ht.items():
        row[f"home_{k}"] = v
    for k, v in at.items():
        row[f"away_{k}"] = v
    
    # Get final scores (targets - same as halftime/Q3)
    fin = final_score_from_box(game)
    if fin is None:
        raise ValueError(f"Missing final score for game {gid}")
    final_home, final_away = fin
    
    row["home_score"] = final_home
    row["away_score"] = final_away
    row["total"] = final_home + final_away
    row["margin"] = final_home - final_away
    
    return row


def _parse_game_date(game: dict) -> Optional[datetime]:
    """Parse a game date from boxscore payload."""
    for key in ("gameDateUTC", "gameTimeUTC", "gameDate", "gameTimeLocal"):
        value = game.get(key)
        if not value:
            continue
        try:
            if isinstance(value, str):
                v = value.replace("Z", "+00:00")
                return datetime.fromisoformat(v).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def add_temporal_features(rows: List[dict]) -> List[dict]:
    """Add rest days, recent form, and head-to-head features using prior games only."""
    rows_with_dates = [r for r in rows if r.get("game_date")]
    rows_missing_dates = [r for r in rows if not r.get("game_date")]

    rows_with_dates.sort(key=lambda r: r["game_date"])

    team_history: Dict[str, List[dict]] = {}
    matchup_history: Dict[tuple, List[str]] = {}

    def recent_stats(team: str, n: int = 5) -> dict:
        history = team_history.get(team, [])
        recent = history[-n:]
        if not recent:
            return {
                "recent_points": 0.0,
                "recent_allowed": 0.0,
                "recent_margin": 0.0,
                "recent_wins": 0.0,
                "recent_opp_margin": 0.0,
            }
        points = [g["points_for"] for g in recent]
        allowed = [g["points_against"] for g in recent]
        margins = [g["margin"] for g in recent]
        opp_margins = [g["opp_margin"] for g in recent]
        wins = [1.0 if m > 0 else 0.0 for m in margins]
        return {
            "recent_points": sum(points) / len(points),
            "recent_allowed": sum(allowed) / len(allowed),
            "recent_margin": sum(margins) / len(margins),
            "recent_wins": sum(wins) / len(wins),
            "recent_opp_margin": sum(opp_margins) / len(opp_margins),
        }

    for row in rows_with_dates:
        game_date = datetime.fromisoformat(row["game_date"]).date()
        home = row["home_tri"]
        away = row["away_tri"]

        def rest_days(team: str) -> float:
            history = team_history.get(team, [])
            if not history:
                return 7.0
            last_date = history[-1]["date"]
            return float((game_date - last_date).days)

        home_rest = rest_days(home)
        away_rest = rest_days(away)

        row["home_rest_days"] = home_rest
        row["away_rest_days"] = away_rest
        row["rest_days_diff"] = home_rest - away_rest
        row["home_is_b2b"] = 1.0 if home_rest == 1.0 else 0.0
        row["away_is_b2b"] = 1.0 if away_rest == 1.0 else 0.0
        row["b2b_diff"] = row["home_is_b2b"] - row["away_is_b2b"]

        home_recent = recent_stats(home)
        away_recent = recent_stats(away)
        row["home_recent_points"] = home_recent["recent_points"]
        row["home_recent_allowed"] = home_recent["recent_allowed"]
        row["home_recent_margin"] = home_recent["recent_margin"]
        row["home_recent_wins"] = home_recent["recent_wins"]
        row["home_recent_opp_margin"] = home_recent["recent_opp_margin"]
        row["away_recent_points"] = away_recent["recent_points"]
        row["away_recent_allowed"] = away_recent["recent_allowed"]
        row["away_recent_margin"] = away_recent["recent_margin"]
        row["away_recent_wins"] = away_recent["recent_wins"]
        row["away_recent_opp_margin"] = away_recent["recent_opp_margin"]
        row["recent_points_diff"] = row["home_recent_points"] - row["away_recent_points"]
        row["recent_margin_diff"] = row["home_recent_margin"] - row["away_recent_margin"]
        row["recent_wins_diff"] = row["home_recent_wins"] - row["away_recent_wins"]
        row["sos_diff"] = row["home_recent_opp_margin"] - row["away_recent_opp_margin"]

        matchup_key = tuple(sorted([home, away]))
        history = matchup_history.get(matchup_key, [])
        if history:
            home_wins = sum(1 for winner in history if winner == home)
            away_wins = sum(1 for winner in history if winner == away)
            total = len(history)
            row["h2h_games"] = float(total)
            row["h2h_home_wins"] = float(home_wins)
            row["h2h_away_wins"] = float(away_wins)
            row["h2h_home_win_pct"] = float(home_wins) / float(total)
        else:
            row["h2h_games"] = 0.0
            row["h2h_home_wins"] = 0.0
            row["h2h_away_wins"] = 0.0
            row["h2h_home_win_pct"] = 0.5

        # Update histories after feature creation
        winner = home if row["margin"] > 0 else away
        matchup_history.setdefault(matchup_key, []).append(winner)

        team_history.setdefault(home, []).append(
            {
                "date": game_date,
                "points_for": row["home_score"],
                "points_against": row["away_score"],
                "margin": row["home_score"] - row["away_score"],
                "opp_margin": row["away_score"] - row["home_score"],
            }
        )
        team_history.setdefault(away, []).append(
            {
                "date": game_date,
                "points_for": row["away_score"],
                "points_against": row["home_score"],
                "margin": row["away_score"] - row["home_score"],
                "opp_margin": row["home_score"] - row["away_score"],
            }
        )

    for row in rows_missing_dates:
        row["home_rest_days"] = 7.0
        row["away_rest_days"] = 7.0
        row["rest_days_diff"] = 0.0
        row["home_is_b2b"] = 0.0
        row["away_is_b2b"] = 0.0
        row["b2b_diff"] = 0.0
        row["home_recent_points"] = 0.0
        row["home_recent_allowed"] = 0.0
        row["home_recent_margin"] = 0.0
        row["home_recent_wins"] = 0.0
        row["home_recent_opp_margin"] = 0.0
        row["away_recent_points"] = 0.0
        row["away_recent_allowed"] = 0.0
        row["away_recent_margin"] = 0.0
        row["away_recent_wins"] = 0.0
        row["away_recent_opp_margin"] = 0.0
        row["recent_points_diff"] = 0.0
        row["recent_margin_diff"] = 0.0
        row["recent_wins_diff"] = 0.0
        row["sos_diff"] = 0.0
        row["h2h_games"] = 0.0
        row["h2h_home_wins"] = 0.0
        row["h2h_away_wins"] = 0.0
        row["h2h_home_win_pct"] = 0.5

    return rows_with_dates + rows_missing_dates


def build_pregame_dataset(
    game_ids: List[str],
    out_parquet: Path,
) -> None:
    """
    Build pregame training dataset from a list of game IDs.
    
    Args:
        game_ids: List of GAME_IDs to process
        out_parquet: Output parquet file path
    """
    rows = []
    errors = []
    
    from tqdm import tqdm

    for gid in tqdm(game_ids, desc="Building pregame dataset"):
        try:
            row = extract_pregame_row(gid)
            rows.append(row)
        except Exception as e:
            errors.append((gid, str(e)))
    
    if errors:
        print(f"Errors in {len(errors)} games:")
        for gid, err in errors[:5]:
            print(f"  {gid}: {err}")
    
    rows = add_temporal_features(rows)
    import pandas as pd

    df = pd.DataFrame(rows)
    
    # Same output format as halftime/Q3 datasets
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Saved pregame dataset: {out_parquet} ({len(df)} rows)")
    
    # Save error log
    if errors:
        err_path = out_parquet.with_suffix(".errors.jsonl")
        with open(err_path, "w") as f:
            for gid, err in errors:
                f.write(json.dumps({"game_id": gid, "error": err}) + "\\n")
        print(f"Saved error log: {err_path}")


def main() -> None:
    """CLI entry point for building pregame dataset."""
    # SCHED_PATH = "data/processed/game_ids_3_seasons.json"  # 3 seasons combined
    
    with open("data/processed/game_ids_3_seasons.json", "r") as f:
        sched = json.load(f)
    
    # Only completed games (need final scores for training)
    game_ids = [g["gameId"] for g in sched if int(g.get("gameStatus", 0)) == 3]
    game_ids = list(dict.fromkeys(game_ids))
    
    out_parquet = Path("data/processed/pregame_team_v2.parquet")
    
    print(f"Building pregame dataset from {len(game_ids)} completed games...")
    build_pregame_dataset(game_ids, out_parquet)


if __name__ == "__main__":
    main()
