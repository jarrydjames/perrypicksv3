"""Build Q3 training dataset for the *5:00 remaining in Q3* decision point.

The Q3 model should run mid-quarter (with ~5:00 left in Q3) and predict the
balance of the game (rest of Q3 + Q4). This dataset therefore:

- Creates features from the game state at the first event with <=5:00 in Q3.
- Stores snapshot score (`q3_5m_*`) as context.
- Uses labels for remaining game outcome from the snapshot, not raw Q3 points.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _clock_to_seconds_remaining(clock_value: object) -> Optional[float]:
    """Convert clock value (ISO PT string or MM:SS) into seconds remaining."""
    if clock_value is None:
        return None

    text = str(clock_value).strip()
    if not text:
        return None

    if text.startswith("PT") and "M" in text:
        try:
            body = text.removeprefix("PT").removesuffix("S")
            mins, secs = body.split("M", 1)
            return float(mins) * 60.0 + float(secs)
        except Exception:
            return None

    if ":" in text:
        try:
            mins, secs = text.split(":", 1)
            return float(mins) * 60.0 + float(secs)
        except Exception:
            return None

    return None


def _score_from_pbp_row(row: pd.Series, side: str) -> Optional[float]:
    """Extract cumulative score from a PBP row for one side (home/away)."""
    candidates = [
        f"score{side.title()}",
        f"{side}Score",
        f"{side}_score",
    ]
    for key in candidates:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                continue
    return None


def _q3_five_minute_snapshot(pbp: pd.DataFrame) -> Tuple[float, float]:
    """Return (home_score, away_score) at the first event with <=5:00 left in Q3."""
    if pbp is None or pbp.empty:
        raise ValueError("Missing play-by-play data")

    q3 = pbp[pbp["period"].astype(int) == 3].copy()
    if q3.empty:
        raise ValueError("No Q3 rows in play-by-play")

    clock_col = "clock" if "clock" in q3.columns else ("gameClock" if "gameClock" in q3.columns else None)
    if clock_col is None:
        raise ValueError("PBP missing clock column")

    q3["clock_seconds_remaining"] = q3[clock_col].map(_clock_to_seconds_remaining)
    trigger_rows = q3[q3["clock_seconds_remaining"].notna() & (q3["clock_seconds_remaining"] <= 300.0)]
    if trigger_rows.empty:
        raise ValueError("No Q3 event at or below 5:00 remaining")

    snapshot = trigger_rows.iloc[0]
    home_score = _score_from_pbp_row(snapshot, "home")
    away_score = _score_from_pbp_row(snapshot, "away")

    if home_score is None or away_score is None:
        raise ValueError("Q3 snapshot row missing score fields")

    return home_score, away_score


def behavior_counts_q3(pbp: pd.DataFrame) -> dict:
    """
    Count action types in first 3 quarters.
    
    Same structure as behavior_counts_1h, but filters to periods 1-3.
    """
    q3 = pbp[pbp["period"].astype(int) <= 3].copy()
    clock_col = "clock" if "clock" in q3.columns else ("gameClock" if "gameClock" in q3.columns else None)
    if clock_col:
        q3["clock_seconds_remaining"] = q3[clock_col].map(_clock_to_seconds_remaining)
        q3 = q3[
            (q3["period"].astype(int) < 3)
            | (
                (q3["period"].astype(int) == 3)
                & q3["clock_seconds_remaining"].notna()
                & (q3["clock_seconds_remaining"] >= 300.0)
            )
        ]
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
    
    # Snapshot at 5:00 remaining in Q3
    q3_5m_home, q3_5m_away = _q3_five_minute_snapshot(pbp)
    
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
        "q3_5m_home": q3_5m_home,
        "q3_5m_away": q3_5m_away,
        "q3_5m_total": q3_5m_home + q3_5m_away,
        "q3_5m_margin": q3_5m_home - q3_5m_away,
        **beh,
        **add_rate_features("home", ht, at),
        **add_rate_features("away", at, ht),
    }
    
    # Add team totals as priors (same as halftime)
    for k, v in ht.items():
        row[f"home_{k}"] = v
    for k, v in at.items():
        row[f"away_{k}"] = v
    
    # Add final labels (game outcomes - FINAL GAME SCORE, not halftime)
    # Import final_score_from_box to get actual final game scores
    from src.build_dataset_team_v2 import final_score_from_box
    
    fin = final_score_from_box(game)
    if fin is None:
        raise ValueError(f"Missing final score for game {gid}")
    final_home, final_away = fin
    
    final_total = final_home + final_away
    final_margin = final_home - final_away

    snapshot_total = q3_5m_home + q3_5m_away
    snapshot_margin = q3_5m_home - q3_5m_away

    # Primary targets for Q3-5m model = remaining game from snapshot.
    row["remaining_total"] = final_total - snapshot_total
    row["remaining_margin"] = final_margin - snapshot_margin

    # Keep final labels for diagnostics and optional downstream consumers.
    row["total"] = final_total
    row["margin"] = final_margin
    
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
        default=Path("data/processed/game_ids_3_seasons.json"),
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
        # Handle dict format
        if "game_ids" in game_ids:
            game_ids = game_ids.get("game_ids", [])
        else:
            # Extract gameId from each dict in values
            game_ids = [g.get("gameId", g) if isinstance(g, dict) else g for g in game_ids.values()]
    elif isinstance(game_ids, list):
        # Extract gameId from each dict if present
        game_ids = [g.get("gameId", g) if isinstance(g, dict) else g for g in game_ids]
    
    build_q3_dataset(game_ids, args.out_parquet)


if __name__ == "__main__":
    main()
