import sys
import re
import os
import requests
import pandas as pd
import joblib
import time
from pathlib import Path

CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

# Cache settings
CACHE_DIR = Path(".cache/nba_cdn")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 300  # 5 minutes

def extract_game_id(arg: str) -> str:
    m = re.search(r"(00\d{8,10})", arg)
    if not m:
        raise ValueError(f"Could not find a GAME_ID in: {arg}")
    return m.group(1)

# User-Agent headers to avoid NBA.com blocking
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
}

def _get_cache_path(url: str) -> Path:
    """Get cache file path for a URL."""
    # Use URL hash as filename
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{url_hash}.json"

def _load_from_cache(cache_path: Path) -> dict:
    """Load data from cache if not expired."""
    if not cache_path.exists():
        return None
    
    # Check if expired
    cache_age = time.time() - cache_path.stat().st_mtime
    if cache_age > CACHE_TTL_SECONDS:
        return None
    
    # Load from cache
    import json
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception:
        # Cache file corrupted, ignore
        return None

def _save_to_cache(cache_path: Path, data: dict) -> None:
    """Save data to cache."""
    import json
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception:
        # Failed to write to cache, ignore
        pass

def fetch_json(url: str, max_retries: int = 5) -> dict:
    """Fetch JSON from NBA.com CDN with proper headers, retry logic, and caching.
    
    Args:
        url: URL to fetch
        max_retries: Number of retries on 403/429 errors (default 5)
        
    Returns:
        JSON response as dict
        
    Raises:
        requests.HTTPError: If all retries fail
    """
    import logging
    
    # Check cache first
    cache_path = _get_cache_path(url)
    cached_data = _load_from_cache(cache_path)
    if cached_data is not None:
        logging.debug(f"Using cached data for {url}")
        return cached_data
    
    # Fetch from CDN with retry logic
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=25, headers=NBA_HEADERS)
            r.raise_for_status()
            data = r.json()
            # Save to cache
            _save_to_cache(cache_path, data)
            return data
        except requests.HTTPError as e:
            # Retry on rate limiting (429) or forbidden (403) errors
            if e.response.status_code in (403, 429) and attempt < max_retries - 1:
                # Longer backoff for CDN endpoint: 2s, 4s, 8s, 16s
                wait_time = 2 ** (attempt + 1)
                logging.warning(f"NBA.com CDN API returned {e.response.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            # For other errors or final retry, re-raise
            raise

def fetch_box(gid: str) -> dict:
    """Fetch game box score from NBA.com CDN.
    
    Falls back to scoreboard endpoint if boxscore endpoint is blocked.
    
    Args:
        gid: Game ID
        
    Returns:
        Game data dict
    """
    try:
        # Try boxscore endpoint first (has more detailed data)
        return fetch_json(CDN_BOX.format(gid=gid))["game"]
    except requests.HTTPError as e:
        if e.response.status_code == 403:
            # Boxscore endpoint is blocked, try scoreboard fallback
            import logging
            logging.warning(f"Boxscore endpoint blocked for {gid}, using scoreboard fallback")
            return fetch_box_from_scoreboard(gid)
        # For other errors, re-raise
        raise

def fetch_box_from_scoreboard(gid: str) -> dict:
    """Fetch minimal game data from scoreboard endpoint.
    
    Used as fallback when boxscore endpoint is blocked.
    
    Args:
        gid: Game ID
        
    Returns:
        Game data dict with compatible structure
    """
    import logging
    
    # Try today's scoreboard first
    scoreboard_urls = [
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_01.json",
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_02.json",
    ]
    
    for url in scoreboard_urls:
        try:
            data = fetch_json(url)
            games = data.get("scoreboard", {}).get("games", [])
            
            # Find the game by ID
            for game in games:
                if game.get("gameId") == gid:
                    # Transform scoreboard data to boxscore-like structure
                    return _transform_scoreboard_to_boxscore(game)
        except Exception as e:
            logging.debug(f"Failed to fetch from {url}: {e}")
            continue
    
    # If we get here, game not found in any scoreboard
    raise ValueError(f"Game {gid} not found in any scoreboard endpoint")

def _transform_scoreboard_to_boxscore(game: dict) -> dict:
    """Transform scoreboard data to boxscore-like structure.
    
    Maps scoreboard fields to match the structure expected by first_half_score.
    """
    # The scoreboard data already has the structure we need for first_half_score
    # It has homeTeam.periods and awayTeam.periods with per-period scores
    return game

def fetch_pbp_df(gid: str) -> pd.DataFrame:
    data = fetch_json(CDN_PBP.format(gid=gid))
    return pd.DataFrame(data["game"]["actions"])

def sum_first2(periods):
    """Sum scores from periods 1-2."""
    s = 0
    for p in (periods or []):
        # Skip if p is not a dict (handles string periods, etc.)
        if not isinstance(p, dict):
            continue
        try:
            period_num = int(float(p.get("period", 0)))
        except (ValueError, TypeError):
            period_num = 0
        if period_num in (1, 2):
            for key in ("score", "points", "pts"):
                if key in p and p[key] is not None:
                    try:
                        s += float(p[key])
                    except (ValueError, TypeError):
                        s += 0
                    break
    return s

def first_half_score(game):
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    return sum_first2(home.get("periods")), sum_first2(away.get("periods"))

def behavior_counts_1h(pbp: pd.DataFrame) -> dict:
    fh = pbp[pbp["period"].astype(int) <= 2].copy()
    at = fh.get("actionType", pd.Series([""] * len(fh))).astype(str).fillna("")
    def c(prefix): return int(at.str.startswith(prefix).sum())
    return {
        "h1_events": int(len(fh)),
        "h1_n_2pt": c("2pt"),
        "h1_n_3pt": c("3pt"),
        "h1_n_turnover": c("turnover"),
        "h1_n_rebound": c("rebound"),
        "h1_n_foul": c("foul"),
        "h1_n_timeout": c("timeout"),
        "h1_n_sub": c("substitution"),
    }

def team_totals_from_box_team(team: dict) -> dict:
    stats = (team.get("statistics") or {})
    def gi(k, default=0):
        v = stats.get(k, default)
        try: return int(v)
        except: return default
    return {
        "fga": gi("fieldGoalsAttempted"),
        "fgm": gi("fieldGoalsMade"),
        "tpa": gi("threePointersAttempted"),
        "tpm": gi("threePointersMade"),
        "fta": gi("freeThrowsAttempted"),
        "ftm": gi("freeThrowsMade"),
        "oreb": gi("reboundsOffensive"),
        "dreb": gi("reboundsDefensive"),
        "reb": gi("reboundsTotal"),
        "ast": gi("assists"),
        "stl": gi("steals"),
        "blk": gi("blocks"),
        "to": gi("turnovers"),
        "pf": gi("foulsPersonal"),
        "pts": gi("points"),
    }

def add_rate_features(prefix: str, t: dict, opp: dict) -> dict:
    poss = t["fga"] + 0.44 * t["fta"] + t["to"] - t["oreb"]
    poss = max(poss, 1.0)
    efg = (t["fgm"] + 0.5 * t["tpm"]) / max(t["fga"], 1)
    ftr = t["fta"] / max(t["fga"], 1)
    tpar = t["tpa"] / max(t["fga"], 1)
    tor = t["to"] / poss
    orbp = t["oreb"] / max(t["oreb"] + opp["dreb"], 1)
    return {
        f"{prefix}_efg": efg,
        f"{prefix}_ftr": ftr,
        f"{prefix}_tpar": tpar,
        f"{prefix}_tor": tor,
        f"{prefix}_orbp": orbp,
    }

def load_model(path):
    obj = joblib.load(path)
    return obj["features"], obj["model"]

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/predict_from_gameid_v2.py <GAME_ID or NBA.com URL>")

    gid = extract_game_id(sys.argv[1])

    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    home_tri = home.get("teamTricode", "HOME")
    away_tri = away.get("teamTricode", "AWAY")

    h1_home, h1_away = first_half_score(game)
    if h1_home == 0 and h1_away == 0:
        raise ValueError("Missing 1H period scoring in boxscore JSON.")

    beh = behavior_counts_1h(pbp)
    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)

    row = {
        "h1_home": h1_home,
        "h1_away": h1_away,
        "h1_total": h1_home + h1_away,
        "h1_margin": h1_home - h1_away,
    }
    row.update(beh)
    row.update(add_rate_features("home", ht, at))
    row.update(add_rate_features("away", at, ht))

    X = pd.DataFrame([row])

    f_total, m_total = load_model("models/team_v2_2h_total.joblib")
    f_margin, m_margin = load_model("models/team_v2_2h_margin.joblib")

    pred_2h_total = float(m_total.predict(X[f_total])[0])
    pred_2h_margin = float(m_margin.predict(X[f_margin])[0])

    h2_home = (pred_2h_total + pred_2h_margin) / 2.0
    h2_away = (pred_2h_total - pred_2h_margin) / 2.0

    final_home = h1_home + h2_home
    final_away = h1_away + h2_away

    print(f"GAME_ID: {gid}")
    print(f"Teams: {away_tri} @ {home_tri} (home={home_tri})")
    print(f"1H score: {home_tri} {h1_home} - {h1_away} {away_tri}")
    print("1H behavior:", beh)
    print("Rate features:", {k: round(row[k], 4) for k in row if k.startswith(("home_","away_"))})

    print("\nPrediction:")
    print(f"  pred_2h_total: {pred_2h_total:.2f}")
    print(f"  pred_2h_margin: {pred_2h_margin:.2f}")
    print(f"  pred_2h_home: {h2_home:.2f}")
    print(f"  pred_2h_away: {h2_away:.2f}")
    print(f"  pred_final_home: {final_home:.2f}")
    print(f"  pred_final_away: {final_away:.2f}")
    print(f"  pred_final_total: {final_home + final_away:.2f}")
    print(f"  pred_final_margin: {final_home - final_away:.2f}")

if __name__ == "__main__":
    main()
