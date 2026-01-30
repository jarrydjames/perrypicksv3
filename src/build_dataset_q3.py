"""Build Q3 training dataset - follows same methodology as halftime dataset builder.

This dataset filters to end-of-Q3 game state (periods 1-3) and creates
features for training a Q3 model that can evaluate at any game clock.

Key differences from halftime dataset:
- Filters to periods 1-3 (instead of 1-2)
- Creates cumulative features up to end of Q3
- Labels are still game outcomes (total, margin) - same as halftime
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from tqdm import tqdm

# Reuse feature builders from halftime dataset
from src.build_dataset_v2 import (
    add_rate_features,
    behavior_counts_1h,
    fetch_box,
    fetch_json,
    fetch_pbp_df,
    first_half_score,
    sum_first2,
    team_totals_from_box_team,
)

# Same endpoints as halftime
CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"


def sum_first3(periods):
    """Sum scores from periods 1-3."""
    s = 0
    for p in (periods or []):
        period_num = int(p.get("period", 0))
        if 1 <= period_num <= 3:
            for key in ("score", "points", "pts"):
                if key in p and p[key] is not None:
                    s += int(p[key])
                    break
    return s


def third_quarter_score(game):
    """Extract home and away scores after Q3."""
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    return sum_first3(home.get("periods")), sum_first3(away.get("periods"))


def behavior_counts_q3(pbp: pd.DataFrame) -> dict:
    """
    Count action types in first 3 quarters.
    
    Same structure as behavior_counts_1h, but filters to periods 1-3.
    """
    q3 = pbp[pbp["period"].astype(int) <= 3].copy()
    at = q3.get("actionType", pd.Series([""] * len(q3))).astype(str).fillna("")
    
    def c(prefix):
        return int(at.str.startswith(prefix).sum())
    
    return {
        "q3_events": int(len(q3)),
        "q3_n_2pt": c("2pt"),
        "q3_n_3pt": c("3pt"),
        "q3_n_turnover": c("turnover"),
        "q3_n_rebound": c("rebound"),
        "q3_n_foul": c("foul"),
        "q3_n_timeout": c("timeout"),
        "q3_n_sub": c("substitution"),
    }


def extract_q3_row(gid: str) -> dict:
    """
    Extract a single Q3 training row for a game.
    
    Returns a dict with features and labels (same structure as halftime row).
    """
    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)
    
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    home_tri = home.get("teamTricode", "HOME")
    away_tri = away.get("teamTricode", "AWAY")
    
    # Get scores after Q3
    q3_home, q3_away = third_quarter_score(game)
    
    # Get behavior counts for Q3
    beh = behavior_counts_q3(pbp)
    
    # Get team stats (same priors as halftime)
    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)
    
    # Start with same features as halftime
    row = {
        "game_id": gid,
        "home_tri": home_tri,
        "away_tri": away_tri,
        "q3_home": q3_home,
        "q3_away": q3_away,
        "q3_total": q3_home + q3_away,
        "q3_margin": q3_home - q3_away,
        **beh,
        **add_rate_features("home", ht, at),
        **add_rate_features("away", at, ht),
    }
    
    # Add team totals as priors (same as halftime)
    for k, v in ht.items():
        row[f"home_{k}"] = v
    for k, v in at.items():
        row[f"away_{k}"] = v
    
    # Add final labels (game outcomes - same as halftime)
    final_home = sum_first2(home.get("periods"))
    final_away = sum_first2(away.get("periods"))
    
    row["total"] = final_home + final_away
    row["margin"] = final_home - final_away
    
    return row


def build_q3_dataset(
    game_ids: List[str],
    out_parquet: Path,
) -> None:
    """
    Build Q3 training dataset from a list of game IDs.
    
    Args:
        game_ids: List of GAME_IDs to process
        out_parquet: Output parquet file path
    """
    rows = []
    errors = []
    
    for gid in tqdm(game_ids, desc="Building Q3 dataset"):
        try:
            row = extract_q3_row(gid)
            rows.append(row)
        except Exception as e:
            errors.append((gid, str(e)))
    
    if errors:
        print(f"Errors in {len(errors)} games:")
        for gid, err in errors[:5]:
            print(f"  {gid}: {err}")
    
    df = pd.DataFrame(rows)
    
    # Same output format as halftime dataset
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Saved Q3 dataset: {out_parquet} ({len(df)} rows)")
    
    # Save error log
    if errors:
        err_path = out_parquet.with_suffix(".errors.jsonl")
        with open(err_path, "w") as f:
            for gid, err in errors:
                f.write(json.dumps({"game_id": gid, "error": err}) + "\n")
        print(f"Saved error log: {err_path}")


def main() -> None:
    """CLI entry point for building Q3 dataset."""
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--game-ids-file",
        type=Path,
        default=Path("data/processed/game_ids_2023_2024.json"),
        help="Path to JSON file with list of GAME_IDs",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("data/processed/q3_team_v2.parquet"),
        help="Output parquet file path",
    )
    args = ap.parse_args()
    
    # Load game IDs
    with open(args.game_ids_file) as f:
        game_ids = json.load(f)
    
    if isinstance(game_ids, dict):
        # Handle both list and dict formats
        game_ids = list(game_ids.values()) if "game_ids" not in game_ids else game_ids.get("game_ids", [])
    
    build_q3_dataset(game_ids, args.out_parquet)


if __name__ == "__main__":
    main()
