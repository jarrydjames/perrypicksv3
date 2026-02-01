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
from typing import Dict, List

import pandas as pd
import requests
from tqdm import tqdm

# Reuse feature builders and API endpoints from existing code
from src.build_dataset_v2 import (
    add_rate_features,
    fetch_box,
    fetch_json,
    fetch_pbp_df,
    team_totals_from_box_team,
)
from src.build_dataset_team_v2 import final_score_from_box

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
    game = fetch_box(gid)
    
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    home_tri = home.get("teamTricode", "HOME")
    away_tri = away.get("teamTricode", "AWAY")
    
    # Get team stats (full game stats, not game-state specific)
    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)
    
    # Build row with pregame features only
    # Note: No game state (h1_home, q3_home, etc.) - this is pregame!
    row = {
        "game_id": gid,
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
    
    row["total"] = final_home + final_away
    row["margin"] = final_home - final_away
    
    return row


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
    from src.build_dataset_v2 import SCHED_PATH
    
    with open(SCHED_PATH, "r") as f:
        sched = json.load(f)
    
    # Only completed games (need final scores for training)
    game_ids = [g["gameId"] for g in sched if int(g.get("gameStatus", 0)) == 3]
    game_ids = list(dict.fromkeys(game_ids))
    
    out_parquet = Path("data/processed/pregame_team_v2.parquet")
    
    print(f"Building pregame dataset from {len(game_ids)} completed games...")
    build_pregame_dataset(game_ids, out_parquet)


if __name__ == "__main__":
    main()
