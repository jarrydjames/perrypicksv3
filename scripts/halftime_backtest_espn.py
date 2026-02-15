#!/usr/bin/env python3
"""
Halftime Backtest using ESPN Schedule + NBA CDN Mapping

This script performs a strict halftime-only backtest using:
1. ESPN API for schedule (no rate limiting)
2. NBA CDN for ID mapping (no rate limiting)
3. Production CatBoost model for predictions
4. Strict halftime-only features (no second-half data)
"""

from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the schedule fetching function
from fetch_game_schedule import main_with_output
from src.modeling.cat_models import CatBoostTwoHeadModel
from src.modeling.feature_columns import feature_columns


# Configuration
CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
METRICS_PATH = Path("reports/champion_runs/latest/halftime_fold_metrics.csv")
OUTPUT_DIR = Path("reports/backtest")
TARGET_FOLD = 51  # Latest production fold

RECENCY_BASE_FEATURES = [
    "pts_scored_avg_5",
    "pts_allowed_avg_5",
    "margin_avg_5",
    "current_streak_5",
    "days_since_last",
    "is_back_to_back",
    "efg",
    "tor",
    "tpar",
    "ftr",
    "orbp",
]

CRITICAL_FEATURES = [
    "home_team_id",
    "away_team_id",
    "home_pts_scored_avg_5",
    "away_pts_scored_avg_5",
    "home_efg",
    "away_efg",
]


def _safe_default(hist_df: pd.DataFrame, feature_name: str, fallback: float) -> float:
    """Get a robust default value from historical data when possible."""
    if feature_name not in hist_df.columns:
        return float(fallback)

    series = pd.to_numeric(hist_df[feature_name], errors="coerce").dropna()
    if series.empty:
        return float(fallback)
    return float(series.median())


def _build_team_id_maps(hist_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Build team_tricode -> team_id maps from historical dataframe."""
    tri_to_id: Dict[str, float] = {}
    city_name_to_tri: Dict[str, str] = {}

    # Load the custom ID mapping (triCode -> 0-29)
    import json
    from pathlib import Path
    custom_id_path = Path("data/processed/team_tricode_to_custom_id.json")
    if custom_id_path.exists():
        with open(custom_id_path, 'r') as f:
            tri_to_custom_id = json.load(f)
        # Convert to float for consistency
        tri_to_id = {k: float(v) for k, v in tri_to_custom_id.items()}
        print(f"  Loaded custom ID mapping for {len(tri_to_id)} teams")
    else:
        # Fallback: build from historical data (may not match refined temporal dataset)
        for side in ["home", "away"]:
            tri_col = f"{side}_tri"
            id_col = f"{side}_team_id"

            if tri_col in hist_df.columns and id_col in hist_df.columns:
                pairs = hist_df[[tri_col, id_col]].dropna().drop_duplicates()
                for _, row in pairs.iterrows():
                    tri = str(row[tri_col]).upper().strip()
                    team_id = pd.to_numeric(pd.Series([row[id_col]]), errors="coerce").iloc[0]
                    if tri and pd.notna(team_id):
                        tri_to_id[tri] = float(team_id)

    for side in ["home", "away"]:
        tri_col = f"{side}_tri"
        city_col = f"{side}_city"
        name_col = f"{side}_name"

        if tri_col in hist_df.columns and city_col in hist_df.columns and name_col in hist_df.columns:
            triples = hist_df[[tri_col, city_col, name_col]].dropna().drop_duplicates()
            for _, row in triples.iterrows():
                tri = str(row[tri_col]).upper().strip()
                city = str(row[city_col]).strip().upper()
                name = str(row[name_col]).strip().upper()
                if tri and city and name:
                    city_name_to_tri[f"{city} {name}"] = tri

    return tri_to_id, city_name_to_tri


def _extract_team_id(
    team_payload: Dict[str, Any],
    *,
    tri_to_id: Dict[str, float],
    city_name_to_tri: Dict[str, str],
) -> float:
    """Extract robust team id from potentially variant CDN payloads.
    
    Priority:
    1. Use triCode mapping (to get custom IDs 0-29)
    2. Fallback to city+name mapping
    3. Return 0 if not found
    """
    # First, try triCode (most reliable for custom ID mapping)
    tri = str(team_payload.get("teamTricode", "")).upper().strip()
    if tri and tri in tri_to_id:
        return float(tri_to_id[tri])

    # Second, try city+name -> triCode -> ID
    city = str(team_payload.get("teamCity", "")).upper().strip()
    name = str(team_payload.get("teamName", "")).upper().strip()
    city_name = f"{city} {name}".strip()
    if city_name in city_name_to_tri:
        tri_guess = city_name_to_tri[city_name]
        if tri_guess in tri_to_id:
            return float(tri_to_id[tri_guess])

    return 0.0


def _fit_sigma_scalers(
    model: CatBoostTwoHeadModel,
    *,
    X_train: np.ndarray,
    y_total_train: np.ndarray,
    y_margin_train: np.ndarray,
    feature_names: List[str],
    calib_frac: float,
) -> Tuple[float, float]:
    """Fit nested-style sigma calibration factors on tail training split."""
    n_tr = int(len(X_train))
    n_cal = int(max(50, round(n_tr * float(calib_frac))))
    n_cal = min(n_cal, max(0, n_tr - 50))
    if n_cal <= 0:
        return 1.0, 1.0

    X_fit, X_cal = X_train[:-n_cal], X_train[-n_cal:]
    yt_fit, yt_cal = y_total_train[:-n_cal], y_total_train[-n_cal:]
    ym_fit, ym_cal = y_margin_train[:-n_cal], y_margin_train[-n_cal:]

    model.fit(X_fit, feature_names, yt_fit, ym_fit)
    mu_t_cal, mu_m_cal = model.predict_heads(X_cal)

    heads_fit = model.trained_heads()
    sig_t_raw = float(heads_fit.total.residual_sigma)
    sig_m_raw = float(heads_fit.margin.residual_sigma)

    z = 1.2815515655446004
    q_t = float(np.quantile(np.abs(yt_cal - mu_t_cal), 0.80))
    q_m = float(np.quantile(np.abs(ym_cal - mu_m_cal), 0.80))
    k_t = q_t / max(1e-6, (z * sig_t_raw))
    k_m = q_m / max(1e-6, (z * sig_m_raw))
    k_t = float(max(0.5, min(3.0, k_t)))
    k_m = float(max(0.5, min(3.0, k_m)))
    return k_t, k_m


def _feature_health_report(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    *,
    critical_features: List[str],
) -> Dict[str, Any]:
    """Build feature health diagnostics for train vs inference parity."""
    report: Dict[str, Any] = {"features": {}, "issues": []}

    for feat in critical_features:
        stats: Dict[str, Any] = {
            "exists_in_train": feat in train_df.columns,
            "exists_in_inference": feat in infer_df.columns,
        }
        if feat in train_df.columns:
            tvals = pd.to_numeric(train_df[feat], errors="coerce")
            stats["train_median"] = float(tvals.median()) if tvals.notna().any() else None
            stats["train_std"] = float(tvals.std()) if tvals.notna().any() else None
        if feat in infer_df.columns:
            ivals = pd.to_numeric(infer_df[feat], errors="coerce")
            stats["infer_zero_rate"] = float((ivals.fillna(0.0) == 0.0).mean())
            stats["infer_unique"] = int(ivals.nunique(dropna=True))
            stats["infer_min"] = float(ivals.min()) if ivals.notna().any() else None
            stats["infer_max"] = float(ivals.max()) if ivals.notna().any() else None

            if stats["infer_unique"] <= 1 and len(infer_df) > 1:
                report["issues"].append(f"{feat} is constant across inference slate")
            if feat.endswith("team_id") and stats["infer_zero_rate"] > 0.0:
                report["issues"].append(f"{feat} contains zero values")

        report["features"][feat] = stats

    const_numeric = []
    for col in infer_df.columns:
        if pd.api.types.is_numeric_dtype(infer_df[col]) and infer_df[col].nunique(dropna=True) <= 1:
            const_numeric.append(col)
    report["constant_numeric_features"] = const_numeric
    n_numeric = max(1, infer_df.select_dtypes(include=[np.number]).shape[1])
    report["constant_numeric_rate"] = float(len(const_numeric) / n_numeric)
    if report["constant_numeric_rate"] > 0.50 and len(infer_df) > 1:
        report["issues"].append("more than 50% of numeric inference features are constant")

    report["ok"] = len(report["issues"]) == 0
    report["n_inference_rows"] = int(len(infer_df))
    return report


def _extract_team_recency_features(
    hist_df: pd.DataFrame,
    team_id: float,
    target_dt: pd.Timestamp,
    out_prefix: str,
) -> Dict[str, float]:
    """Build team recency features from latest leakage-safe historical row.
    
    For refined temporal dataset, features are already calculated.
    We just need to find the most recent game for this team and extract the features.
    """
    
    # Get default values from historical medians
    defaults = {}
    for metric in RECENCY_BASE_FEATURES:
        col_with_prefix = f"{out_prefix}_{metric}"
        if col_with_prefix in hist_df.columns:
            defaults[col_with_prefix] = _safe_default(hist_df, col_with_prefix, 0.0)
        else:
            defaults[col_with_prefix] = 0.0
    
    defaults[f"{out_prefix}_team_id"] = float(team_id)

    if team_id <= 0:
        return defaults

    game_date_col = "game_date" if "game_date" in hist_df.columns else None
    if game_date_col is None:
        return defaults

    latest_row = None
    latest_date = None

    # Find the most recent game for this team before target date
    for side in ["home", "away"]:
        id_col = f"{side}_team_id"
        if id_col not in hist_df.columns:
            continue

        subset = hist_df[hist_df[id_col] == team_id].copy()
        if subset.empty:
            continue

        subset[game_date_col] = pd.to_datetime(subset[game_date_col], errors="coerce", utc=True)
        subset = subset[subset[game_date_col] < target_dt]
        if subset.empty:
            continue

        idx = subset[game_date_col].idxmax()
        row = subset.loc[idx]
        row_date = row[game_date_col]

        if latest_date is None or row_date > latest_date:
            latest_date = row_date
            latest_row = row

    if latest_row is None:
        return defaults

    # Extract features from the latest row
    features = defaults.copy()
    
    # Get all columns that start with out_prefix (e.g., "home_efg", "home_pts_scored_avg_5")
    for col in hist_df.columns:
        if col.startswith(f"{out_prefix}_"):
            if col in latest_row:
                val = pd.to_numeric(pd.Series([latest_row[col]]), errors="coerce").iloc[0]
                if pd.notna(val):
                    features[col] = float(val)

    return features


def fetch_game_data(game_id: str) -> Dict[str, Any]:
    """Fetch play-by-play and boxscore data from NBA CDN."""
    
    # Fetch play-by-play
    pbp_url = CDN_PBP.format(gid=game_id)
    try:
        pbp_resp = requests.get(pbp_url, timeout=30)
        pbp_resp.raise_for_status()
        pbp_data = pbp_resp.json()
    except Exception as e:
        print(f"    ❌ Error fetching PBP: {e}")
        pbp_data = {}
    
    # Fetch boxscore
    box_url = CDN_BOX.format(gid=game_id)
    try:
        box_resp = requests.get(box_url, timeout=30)
        box_resp.raise_for_status()
        box_data = box_resp.json()
    except Exception as e:
        print(f"    ❌ Error fetching boxscore: {e}")
        box_data = {}
    
    return {
        "pbp": pbp_data,
        "box": box_data,
    }


def extract_halftime_features(
    game_data: Dict,
    hist_df: pd.DataFrame,
    target_dt: pd.Timestamp,
    *,
    tri_to_id: Dict[str, float],
    city_name_to_tri: Dict[str, str],
) -> Dict[str, float]:
    """Extract STRICT halftime-only features.
    
    ENFORCED RULES:
    ✅ ALLOWED: First-half stats, pregame features
    ❌ FORBIDDEN: Second-half stats, final scores, post-halftime data
    """
    
    pbp = game_data.get("pbp", {})
    box = game_data.get("box", {})
    
    features = {}
    
    # Initialize all halftime stats
    h1_total = 0.0
    h1_home = 0.0
    h1_away = 0.0
    h1_margin = 0.0
    h1_events = 0.0
    h1_n_2pt = 0.0
    h1_n_3pt = 0.0
    h1_n_foul = 0.0
    h1_n_rebound = 0.0
    h1_n_sub = 0.0
    h1_n_timeout = 0.0
    h1_n_turnover = 0.0
    
    # Parse play-by-play for FIRST HALF ONLY (Q1 and Q2)
    actions = pbp.get("game", {}).get("actions", [])
    for action in actions:
        period = action.get("period")
        period_num = period if isinstance(period, int) else (period.get("number", 0) if isinstance(period, dict) else 0)
        
        if period_num <= 2:  # STRICT: First half only
            h1_events += 1
            
            action_type = action.get("actionType", "").lower()
            if "2pt" in action_type or "field goal made" in action_type:
                h1_n_2pt += 1
            if "3pt" in action_type:
                h1_n_3pt += 1
            if "foul" in action_type:
                h1_n_foul += 1
            if "rebound" in action_type:
                h1_n_rebound += 1
            if "sub" in action_type:
                h1_n_sub += 1
            if "timeout" in action_type:
                h1_n_timeout += 1
            if "turnover" in action_type:
                h1_n_turnover += 1
    
    # Extract STRICT halftime score from boxscore (Q1 + Q2 only)
    if box:
        home_team = box.get("game", {}).get("homeTeam", {})
        away_team = box.get("game", {}).get("awayTeam", {})
        
        home_periods = home_team.get("periods", [])
        away_periods = away_team.get("periods", [])
        
        if len(home_periods) >= 2 and len(away_periods) >= 2:
            h1_home = sum([p.get("score", 0) for p in home_periods[:2]])
            h1_away = sum([p.get("score", 0) for p in away_periods[:2]])
            h1_total = h1_home + h1_away
            h1_margin = h1_home - h1_away
    
    features["h1_total"] = float(h1_total)
    features["h1_home"] = float(h1_home)
    features["h1_away"] = float(h1_away)
    features["h1_margin"] = float(h1_margin)
    features["h1_events"] = float(h1_events)
    features["h1_n_2pt"] = float(h1_n_2pt)
    features["h1_n_3pt"] = float(h1_n_3pt)
    features["h1_n_foul"] = float(h1_n_foul)
    features["h1_n_rebound"] = float(h1_n_rebound)
    features["h1_n_sub"] = float(h1_n_sub)
    features["h1_n_timeout"] = float(h1_n_timeout)
    features["h1_n_turnover"] = float(h1_n_turnover)
    
    home_team_payload = box.get("game", {}).get("homeTeam", {})
    away_team_payload = box.get("game", {}).get("awayTeam", {})
    home_team_id = _extract_team_id(
        home_team_payload,
        tri_to_id=tri_to_id,
        city_name_to_tri=city_name_to_tri,
    )
    away_team_id = _extract_team_id(
        away_team_payload,
        tri_to_id=tri_to_id,
        city_name_to_tri=city_name_to_tri,
    )

    features.update(_extract_team_recency_features(hist_df, home_team_id, target_dt, "home"))
    features.update(_extract_team_recency_features(hist_df, away_team_id, target_dt, "away"))
    
    features["season"] = float(target_dt.year % 100)
    features["game_date"] = float(target_dt.strftime("%Y%m%d"))
    
    return features


def get_final_results(game_data: Dict) -> Dict[str, float]:
    """Extract final game results (for evaluation only, NOT used in features)."""
    
    box = game_data.get("box", {})
    
    if not box:
        return {}
    
    home_team = box.get("game", {}).get("homeTeam", {})
    away_team = box.get("game", {}).get("awayTeam", {})
    
    home_score = home_team.get("score", 0)
    away_score = away_team.get("score", 0)
    
    final_total = home_score + away_score
    final_margin = home_score - away_score
    home_won = 1.0 if home_score > away_score else 0.0
    
    return {
        "final_total": float(final_total),
        "final_margin": float(final_margin),
        "home_won": home_won,
    }




def _robust_topk_params(fold_metrics: pd.DataFrame) -> Dict[str, Any]:
    """Build a robust CatBoost param set from top-k tuned folds.

    - numeric params use median
    - categorical/discrete params use most frequent value
    """
    parsed = []
    for _, row in fold_metrics.iterrows():
        try:
            parsed.append(json.loads(row["params"]))
        except Exception:
            continue

    if not parsed:
        raise ValueError("Unable to parse CatBoost params for top-k selection")

    keys = sorted({k for item in parsed for k in item.keys()})
    out: Dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in parsed if key in item]
        if not values:
            continue

        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            med = float(np.median(np.asarray(values, dtype=float)))
            if key in {"iterations", "depth", "random_seed"}:
                out[key] = int(round(med))
            else:
                out[key] = med
        else:
            counts: Dict[str, int] = {}
            for v in values:
                token = json.dumps(v, sort_keys=True)
                counts[token] = counts.get(token, 0) + 1
            best_token = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            out[key] = json.loads(best_token)

    if "random_seed" not in out:
        out["random_seed"] = int(TARGET_FOLD)

    return out
def load_production_model_params(selection: str = "topk", topk: int = 5) -> Tuple[Dict[str, Any], List[int]]:
    """Load production CatBoost hyperparameters."""

    metrics_df = pd.read_csv(METRICS_PATH)
    fold_metrics = metrics_df[metrics_df["model"] == "catboost"].copy()

    if selection == "fold":
        fold_metrics = fold_metrics[fold_metrics["fold"] == TARGET_FOLD]
    else:
        fold_metrics = fold_metrics.sort_values("tune_score").head(max(1, int(topk)))

    if len(fold_metrics) == 0:
        raise ValueError("No CatBoost metrics found for selection")

    selected_folds = [int(x) for x in fold_metrics["fold"].tolist()]

    if selection == "topk" and len(fold_metrics) > 1:
        params = _robust_topk_params(fold_metrics)
    else:
        params_str = fold_metrics.iloc[0]["params"]
        params = json.loads(params_str)

    return params, selected_folds


def main(
    target_date: str = "2026-02-11",
    *,
    param_selection: str = "topk",
    param_topk: int = 5,
    sigma_calib_frac: float = 0.15,
    fail_on_feature_issues: bool = True,
):
    """Main entry point."""
    
    print("="*80)
    print("HALFTIME BACKTEST - ESPN SCHEDULE + NBA CDN MAPPING")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Target date
    target_dt = pd.to_datetime(target_date, utc=True)
    
    print(f"\n📅 Target Date: {target_date}")
    print(f"\n{'='*80}")
    print("STEP 1: FETCHING SCHEDULE (ESPN + NBA CDN)")
    print(f"{'='*80}")
    
    # Fetch schedule using ESPN + NBA CDN approach
    schedule_data = main_with_output(target_date)
    
    games = schedule_data.get('games', [])
    
    if not games:
        print(f"\n❌ No games found for {target_date}")
        return
    
    print(f"\n✅ Found {len(games)} games")
    for i, game in enumerate(games, 1):
        print(f"  {i}. {game['away_team']} @ {game['home_team']}")
        print(f"     ESPN ID: {game['espn_id']}")
        print(f"     NBA ID: {game['nba_id']}")
    
    print(f"\n{'='*80}")
    print("STEP 2: LOADING PRODUCTION MODEL")
    print(f"{'='*80}")

    print("\nLoading historical feature store...")
    data_path = Path("data/processed/halftime_with_refined_temporal.parquet")
    hist_df = pd.read_parquet(data_path)
    hist_df["game_date"] = pd.to_datetime(hist_df["game_date"], errors="coerce", utc=True)
    tri_to_id, city_name_to_tri = _build_team_id_maps(hist_df)

    params, selected_param_folds = load_production_model_params(selection=param_selection, topk=param_topk)
    print(f"\nProduction parameters (selection={param_selection}, folds={selected_param_folds}):")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("STEP 3: FETCHING GAME DATA & EXTRACTING HALFTIME FEATURES")
    print(f"{'='*80}")
    
    results = []
    
    for i, game in enumerate(games, 1):
        print(f"\n[{i}/{len(games)}] {game['away_team']} @ {game['home_team']}")
        print(f"  NBA Game ID: {game['nba_id']}")
        
        if not game['nba_id']:
            print(f"  ⚠️  No NBA ID mapped, skipping...")
            continue
        
        # Fetch game data from NBA CDN
        print(f"  Fetching play-by-play and boxscore...")
        game_data = fetch_game_data(game['nba_id'])
        
        if not game_data['pbp'] or not game_data['box']:
            print(f"  ❌ Incomplete game data, skipping...")
            continue
        
        # Extract halftime features
        print(f"  Extracting halftime features...")
        h1_features = extract_halftime_features(
            game_data,
            hist_df,
            target_dt,
            tri_to_id=tri_to_id,
            city_name_to_tri=city_name_to_tri,
        )
        
        # Get final results (for evaluation)
        final_results = get_final_results(game_data)
        
        if final_results:
            results.append({
                "game_id": game['nba_id'],
                "away_team": game['away_team'],
                "home_team": game['home_team'],
                **h1_features,
                **final_results,
            })
            
            print(f"  ✅ H1: {h1_features['h1_away']:.0f}-{h1_features['h1_home']:.0f} ({h1_features['h1_total']:.0f} total)")
            print(f"     Final: {final_results['final_margin']:.0f} margin, {final_results['final_total']:.0f} total")
        else:
            print(f"  ❌ Missing final results")
    if not results:
        print(f"\n❌ No valid game data could be extracted")
        return

    # Fail-fast feature health gate (before any model training)
    results_df = pd.DataFrame(results)
    train_df = hist_df[hist_df['game_date'] < target_dt].copy()
    health_report = _feature_health_report(train_df, results_df, critical_features=CRITICAL_FEATURES)
    health_path = OUTPUT_DIR / f"feature_health_{target_date}.json"
    with open(health_path, "w") as f:
        json.dump(health_report, f, indent=2)
    print(f"\nFeature health report saved to {health_path}")
    if health_report.get("issues"):
        print("⚠️  Feature health issues detected:")
        for issue in health_report["issues"]:
            print(f"   - {issue}")
        if fail_on_feature_issues:
            raise RuntimeError("Feature health gate failed. Re-run with --allow-feature-issues to continue.")

    print(f"\n{'='*80}")
    print("STEP 4: TRAINING PRODUCTION MODEL")
    print(f"{'='*80}")

    # Load historical data for training
    print(f"\nLoading historical data...")
    
    print(f"  Training on {len(train_df)} historical games")
    print(f"  Date range: {train_df['game_date'].min()} to {train_df['game_date'].max()}")
    
    # Prepare training features
    feat_cols = feature_columns(train_df)
    numeric_feats = []
    for col in feat_cols:
        if col in train_df.columns:
            if train_df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
                numeric_feats.append(col)
    
    X_train = train_df[numeric_feats].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_total_train = train_df['h2_total'].values
    y_margin_train = train_df['h2_margin'].values
    
    # Train production model
    print(f"\nTraining CatBoost model with production parameters...")
    model = CatBoostTwoHeadModel(feature_version="v1", **params)

    print("Calibrating sigma using nested-style tail split...")
    sigma_k_total, sigma_k_margin = _fit_sigma_scalers(
        model,
        X_train=X_train,
        y_total_train=y_total_train,
        y_margin_train=y_margin_train,
        feature_names=numeric_feats,
        calib_frac=sigma_calib_frac,
    )
    print(f"  sigma_k_total={sigma_k_total:.3f}, sigma_k_margin={sigma_k_margin:.3f}")

    model.fit(X_train, numeric_feats, y_total_train, y_margin_train)
    print(f"✅ Model trained")
    
    print(f"\n{'='*80}")
    print("STEP 5: GENERATING PREDICTIONS")
    print(f"{'='*80}")
    # Prepare test features

    # Ensure feature alignment
    test_feats = [col for col in numeric_feats if col in results_df.columns]
    missing_feats = [col for col in numeric_feats if col not in results_df.columns]
    
    if missing_feats:
        print(f"\n⚠️  Warning: {len(missing_feats)} missing features (will use defaults)")
        for feat in missing_feats:
            results_df[feat] = 0.0
    
    X_test = results_df[numeric_feats].values
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Generate predictions
    print(f"\nGenerating predictions for {len(results_df)} games...")
    mu_total, mu_margin = model.predict_heads(X_test)
    
    # Get win probability
    # Model predicts H2 (second half) margin
    # Full game margin = H1_margin (known) + H2_margin (predicted with uncertainty)
    # Win prob = P(H1_margin + H2_margin > 0)
    #          = P(H2_margin > -H1_margin)
    trained_heads = model.trained_heads()
    sig_margin = trained_heads.margin.residual_sigma * sigma_k_margin
    
    # Calculate probability that home wins the full game
    h1_margin = results_df['h1_margin'].values
    p_win = 1 - norm.cdf(-h1_margin, loc=mu_margin, scale=sig_margin)
    
    # Add predictions to results
    # Model predicts h2 (second half), so add h1 to get full game prediction
    results_df['pred_h2_total'] = mu_total
    results_df['pred_h2_margin'] = mu_margin
    results_df['pred_total'] = results_df['h1_total'] + mu_total  # Full game = h1 + h2
    results_df['pred_margin'] = results_df['h1_margin'] + mu_margin  # Full game margin
    results_df['pred_win_prob'] = p_win
    results_df['pred_winner'] = (results_df['pred_margin'] > 0).astype(int)
    results_df['actual_winner'] = (results_df['final_margin'] > 0).astype(int)
    results_df['correct_winner'] = (results_df['pred_winner'] == results_df['actual_winner']).astype(int)
    results_df['total_error'] = results_df['pred_total'] - results_df['final_total']
    results_df['margin_error'] = results_df['pred_margin'] - results_df['final_margin']
    
    print(f"✅ Predictions generated")
    
    print(f"\n{'='*80}")
    print("STEP 6: COMPUTING METRICS")
    print(f"{'='*80}")
    
    # Compute metrics
    mae_total = mean_absolute_error(results_df['final_total'], results_df['pred_total'])
    rmse_total = np.sqrt(mean_squared_error(results_df['final_total'], results_df['pred_total']))
    mae_margin = mean_absolute_error(results_df['final_margin'], results_df['pred_margin'])
    rmse_margin = np.sqrt(mean_squared_error(results_df['final_margin'], results_df['pred_margin']))
    win_accuracy = results_df['correct_winner'].mean()
    brier = brier_score_loss(results_df['actual_winner'], results_df['pred_win_prob'])
    
    print(f"\n✅ Metrics computed")
    
    print(f"\n{'='*80}")
    print("PER-GAME RESULTS")
    print(f"{'='*80}")
    
    # Print per-game table
    for idx, row in results_df.iterrows():
        print(f"\n{row['away_team']} @ {row['home_team']}")
        print(f"  Pred Total:  {row['pred_total']:.1f}  |  Actual: {row['final_total']:.1f}  |  Error: {row['total_error']:+.1f}")
        print(f"  Pred Margin: {row['pred_margin']:+.1f}  |  Actual: {row['final_margin']:+.1f}  |  Error: {row['margin_error']:+.1f}")
        print(f"  Pred Winner: {row['pred_winner']}  |  Actual: {row['actual_winner']}  |  {'✅' if row['correct_winner'] else '❌'}")
    
    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}")
    print(f"\nNumber of Games: {len(results_df)}")
    print(f"\nTOTAL POINTS:")
    print(f"  MAE: {mae_total:.2f}")
    print(f"  RMSE: {rmse_total:.2f}")
    print(f"\nMARGIN:")
    print(f"  MAE: {mae_margin:.2f}")
    print(f"  RMSE: {rmse_margin:.2f}")
    print(f"\nWINNER PREDICTION:")
    print(f"  Accuracy: {win_accuracy*100:.1f}%")
    print(f"  Brier Score: {brier:.4f}")
    
    # Interpretation
    print(f"\n{'='*80}")
    print("PERFORMANCE INTERPRETATION")
    print(f"{'='*80}")
    
    print(f"\nTOTAL POINTS:")
    if mae_total <= 8.0:
        print(f"  ✅ STRONG - MAE {mae_total:.2f} ≤ 8.0")
    elif mae_total <= 10.0:
        print(f"  ⚠️  ACCEPTABLE - MAE {mae_total:.2f} in [8, 10]")
    else:
        print(f"  ❌ NEEDS INVESTIGATION - MAE {mae_total:.2f} > 10.0")
    
    print(f"\nWINNER PREDICTION:")
    if win_accuracy >= 0.60:
        print(f"  ✅ STRONG - Accuracy {win_accuracy*100:.1f}% ≥ 60%")
    elif win_accuracy >= 0.55:
        print(f"  ⚠️  ACCEPTABLE - Accuracy {win_accuracy*100:.1f}% in [55%, 60%)")
    else:
        print(f"  ❌ NEEDS INVESTIGATION - Accuracy {win_accuracy*100:.1f}% < 55%")
    
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    
    if mae_total <= 8.0 and win_accuracy >= 0.60:
        print(f"\n✅ PERFORMANCE MATCHES EXPECTATIONS")
        print(f"Model is performing well on out-of-sample data.")
    elif mae_total > 10.0 or win_accuracy < 0.55:
        print(f"\n❌ PERFORMANCE BELOW EXPECTATIONS")
        print(f"Model may need retraining or feature updates.")
    else:
        print(f"\n⚠️  PERFORMANCE ACCEPTABLE")
        print(f"Model is performing adequately but could be improved.")
    
    # Save detailed results
    output_path = OUTPUT_DIR / f"halftime_backtest_{target_date}_detailed.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Detailed results saved to {output_path}")
    
    # Save metrics
    metrics_path = OUTPUT_DIR / f"metrics_{target_date}.json"
    metrics = {
        "test_date": target_date,
        "n_games": len(results_df),
        "mae_total": float(mae_total),
        "rmse_total": float(rmse_total),
        "mae_margin": float(mae_margin),
        "rmse_margin": float(rmse_margin),
        "win_accuracy": float(win_accuracy),
        "brier_score": float(brier),
        "sigma_k_total": float(sigma_k_total),
        "sigma_k_margin": float(sigma_k_margin),
        "params_selection": param_selection,
        "params_topk": int(param_topk),
        "selected_param_folds": selected_param_folds,
        "feature_health_ok": bool(health_report.get("ok", False)),
        "feature_health_path": str(health_path),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")
    
    # Save results
    output_path = OUTPUT_DIR / f"halftime_backtest_{target_date}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    
    print(f"\n{'='*80}")
    print("✅ BACKTEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESPN+NBA-CDN halftime backtest")
    parser.add_argument("--date", default="2026-02-11", help="Target date in YYYY-MM-DD")
    parser.add_argument("--param-selection", choices=["fold", "topk"], default="topk")
    parser.add_argument("--param-topk", type=int, default=5, help="Top-k folds by tune_score when --param-selection topk")
    parser.add_argument("--sigma-calib-frac", type=float, default=0.15, help="Tail fraction for sigma calibration")
    parser.add_argument("--allow-feature-issues", action="store_true", help="Continue even when feature-health gate fails")
    args = parser.parse_args()
    main(
        target_date=args.date,
        param_selection=args.param_selection,
        param_topk=args.param_topk,
        sigma_calib_frac=args.sigma_calib_frac,
        fail_on_feature_issues=(not args.allow_feature_issues),
    )
