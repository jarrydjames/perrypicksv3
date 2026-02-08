from __future__ import annotations

from typing import Any, Dict, Tuple
import requests
import math

import pandas as pd  # required by predict_from_gameid_v2 helpers


Z80 = 1.2815515655446004  # central 80% interval for Normal


def _norm_q10_q90(mean: float, sd: float) -> Tuple[float, float]:
    # 10th/90th percentiles correspond to +/- Z80 * sd
    return (mean - Z80 * sd, mean + Z80 * sd)


def _clamp_sd(x: float, lo: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    if not math.isfinite(x) or x <= 0:
        return lo
    return max(x, lo)


def _safe_team_name(team_block: dict, fallback: str) -> str:
    # NBA endpoints vary; be defensive
    # PREFER tricodes for consistency across the system
    for k in ("teamTricode", "teamName", "teamCity", "name"):
        v = team_block.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return fallback


def _extract_status(game: dict) -> str:
    # Return a simple string status for post generator compatibility.
    # 'success' - game completed or in progress (valid for predictions)
    # 'warning' - game has minor issues but still valid
    # 'error' - game failed or invalid data
    
    game_status = game.get("gameStatus")
    
    if game_status == 3:  # Final
        return "success"
    elif game_status in (1, 2):  # In progress (Q1, Q2, Q3, Q4, or OT)
        return "success"
    elif game_status == 0:  # Not started
        return "success"
    else:
        # Unknown or other status
        return "warning"


def _elapsed_since_halftime_seconds(status: dict) -> int:
    # Minimal: if game is final or clock 0:00, assume full 2H elapsed (24 min = 1440 sec).
    # Otherwise, leave as 0 (your UI can still function).
    gc = status.get("gameClock")
    gs = status.get("gameStatus")
    try:
        if gc == "PT00M00.00S" or int(gs) == 3:
            return 1440
    except Exception:
        pass
    return 0


def predict_from_game_id(gid_or_url: str, use_binned_intervals: bool = True, fetch_odds: bool = True) -> Dict[str, Any]:
    """
    Cloud-safe 'v2_ci' entrypoint that RETURNS a dict the Streamlit app expects:
      - bands80: central 80% bands for 2H total/margin and final total/margin + team finals
      - normal: q10/q90 for same main metrics (kept for backward compatibility)
      - labels: human-readable notes about how bands were produced
      - odds: current odds from API (if fetch_odds=True)
    NOTE: This implementation is intentionally conservative and self-contained.
    It uses your existing halftime model outputs and layers normal-based uncertainty bands.
    """
    from src.predict_from_gameid_v2 import (
        extract_game_id,
        fetch_box,
        fetch_pbp_df,
        first_half_score,
        behavior_counts_1h,
    )
    from src.predict_from_gameid import predict_from_halftime

    gid = extract_game_id(gid_or_url)

    # Fetch game data with error handling
    try:
        game = fetch_box(gid)
    except requests.HTTPError as e:
        raise ValueError(f"Failed to fetch game data from NBA.com API: {e}")
    
    # Fetch play-by-play data with error handling
    try:
        pbp = fetch_pbp_df(gid)
    except requests.HTTPError as e:
        # If PBP fails, we can still make a basic prediction with box score only
        # Use empty behavior counts as fallback
        import logging
        logging.warning(f"PBP API failed for {gid}: {e}")
        logging.warning("Using empty behavior counts as fallback")
        pbp = pd.DataFrame()  # Empty DataFrame

    h1_home, h1_away = first_half_score(game)
    beh = behavior_counts_1h(pbp)

    _, pred = predict_from_halftime(h1_home, h1_away, beh)

    # Means (2H)
    h2_total_mu = float(pred.get("pred_2h_total"))
    h2_margin_mu = float(pred.get("pred_2h_margin"))

    # Means (final)
    final_home_mu = float(pred.get("pred_final_home"))
    final_away_mu = float(pred.get("pred_final_away"))
    final_total_mu = final_home_mu + final_away_mu
    # home margin = home - away
    final_margin_mu = final_home_mu - final_away_mu

    # Conservative base SDs (tuned to avoid 0/100% probability explosions)
    # These match your typical tuning defaults in the app: total ~12, margin ~8.
    sd_h2_total = 12.0
    sd_h2_margin = 8.0

    # If requested, modestly widen when using "binned" mode (kept tiny, but non-zero)
    if use_binned_intervals:
        sd_h2_total *= 1.03
        sd_h2_margin *= 1.03

    # Final SDs: same as 2H for totals, and margin slightly wider because team-score adds noise
    sd_final_total = sd_h2_total
    sd_final_margin = sd_h2_margin

    # Team SDs: split total uncertainty across teams (very rough but stable)
    sd_final_team = sd_final_total / 2.0

    # Build intervals
    h2_total_q10, h2_total_q90 = _norm_q10_q90(h2_total_mu, sd_h2_total)
    h2_margin_q10, h2_margin_q90 = _norm_q10_q90(h2_margin_mu, sd_h2_margin)

    final_total_q10, final_total_q90 = _norm_q10_q90(final_total_mu, sd_final_total)
    final_margin_q10, final_margin_q90 = _norm_q10_q90(final_margin_mu, sd_final_margin)

    final_home_q10, final_home_q90 = _norm_q10_q90(final_home_mu, sd_final_team)
    final_away_q10, final_away_q90 = _norm_q10_q90(final_away_mu, sd_final_team)

    # Status + names
    home_team = game.get("homeTeam") or {}
    away_team = game.get("awayTeam") or {}
    
    # Validate that team data exists
    if not home_team or not away_team:
        import logging
        logging.error(f"Game data missing team information. homeTeam={home_team}, awayTeam={away_team}")
        raise ValueError(f"Invalid game data: Missing team information for game {gid}")
    
    home_name = _safe_team_name(home_team, "Home")
    away_name = _safe_team_name(away_team, "Away")
    status = _extract_status(game)
    
    # Don't need elapsed_since_halftime_seconds for post generator
    elapsed_since_halftime = 0
    
    # Fetch odds if requested (same as Q3 model)
    odds_data = {}
    if fetch_odds:
        try:
            from src.odds.odds_api import OddsAPIMarketSnapshot, OddsAPIError, fetch_nba_odds_snapshot
            from src.odds.persistent_cache import PersistentOddsCache
            
            cache = PersistentOddsCache()
            odds_snapshot = cache.get_or_fetch(home_name, away_name)
            
            if odds_snapshot:
                odds_data = {
                    "odds_home_ml": odds_snapshot.moneyline_home,
                    "odds_away_ml": odds_snapshot.moneyline_away,
                    "odds_total_line": odds_snapshot.total_points,
                    "odds_total_over": odds_snapshot.total_over_odds,
                    "odds_total_under": odds_snapshot.total_under_odds,
                    "odds_spread_home_line": odds_snapshot.spread_home,
                    "odds_spread_home": odds_snapshot.spread_home_odds,
                    "odds_spread_away": odds_snapshot.spread_away_odds,
                }
        except Exception as e:
            import logging
            logging.warning(f"Odds API error for {gid}: {e}")
            odds_data["odds_error"] = str(e)

    # Match your app's previous rich keys
    out: Dict[str, Any] = {
        "game_id": gid,
        "home_name": home_name,
        "away_name": away_name,
        "h1_home": int(h1_home),
        "h1_away": int(h1_away),
        "pred_final_home": float(final_home_mu),
        "pred_final_away": float(final_away_mu),
        "total": float(final_total_mu),
        "margin": float(final_margin_mu),
        "status": status,
        "elapsed_since_halftime_seconds": elapsed_since_halftime,
        "current_home": None,
        "current_away": None,
        "clock_adjustment": None,
        # A compact text block for UI
        "text": (
            f"GAME_ID: {gid}\n"
            f"Matchup: {away_name} @ {home_name}\n"
            f"Halftime: {home_name} {h1_home} – {h1_away} {away_name}\n\n"
            f"2H Projection:\n"
            f"  Total:  {h2_total_mu:.2f}  (80% CI: {h2_total_q10:.1f} – {h2_total_q90:.1f})\n"
            f"  Margin: {h2_margin_mu:.2f} (80% CI: {h2_margin_q10:.1f} – {h2_margin_q90:.1f})  [home={home_name}]\n\n"
            f"Final Projection:\n"
            f"  {home_name}: {final_home_mu:.1f}  (80% CI: {final_home_q10:.1f} – {final_home_q90:.1f})\n"
            f"  {away_name}: {final_away_mu:.1f}  (80% CI: {final_away_q10:.1f} – {final_away_q90:.1f})\n"
            f"  Total: {final_total_mu:.2f}  (80% CI: {final_total_q10:.1f} – {final_total_q90:.1f})\n"
            f"  Margin ({home_name}): {final_margin_mu:.2f}  (80% CI: {final_margin_q10:.1f} – {final_margin_q90:.1f})"
        ),
        # For compatibility with your app’s bet-prob logic
        "normal": {
            "h2_total": [h2_total_q10, h2_total_q90],
            "h2_margin": [h2_margin_q10, h2_margin_q90],
            "final_total": [final_total_q10, final_total_q90],
            "final_margin": [final_margin_q10, final_margin_q90],
        },
        "bands80": {
            "h2_total": [h2_total_q10, h2_total_q90],
            "h2_margin": [h2_margin_q10, h2_margin_q90],
            "final_total": [final_total_q10, final_total_q90],
            "final_margin": [final_margin_q10, final_margin_q90],
            "final_home": [final_home_q10, final_home_q90],
            "final_away": [final_away_q10, final_away_q90],
        },
        "labels": {
            "total": f"normal(sd={sd_final_total:.2f})" + ("+binned" if use_binned_intervals else ""),
            "margin": f"normal(sd={sd_final_margin:.2f})" + ("+binned" if use_binned_intervals else ""),
        },
        # Keep original pred block too (handy for debugging)
        "pred": pred,
    }
    
    # Add odds data to the result (if fetched)
    out.update(odds_data)

    return out


# Optional CLI convenience
def main():
    import sys, json
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.predict_from_gameid_v2_ci <GAME_ID or NBA.com game URL>")
    gid = sys.argv[1]
    out = predict_from_game_id(gid, use_binned_intervals=True)
    print(json.dumps(out, indent=2)[:2500])


if __name__ == "__main__":
    main()