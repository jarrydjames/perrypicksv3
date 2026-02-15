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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


def extract_halftime_features(game_data: Dict) -> Dict[str, float]:
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
    
    # Placeholder for rolling stats (would come from database in production)
    features["home_pts_scored_avg_5"] = 115.0
    features["home_pts_allowed_avg_5"] = 112.0
    features["home_margin_avg_5"] = 3.0
    features["home_current_streak_5"] = 0.0
    features["home_days_since_last"] = 2.0
    features["home_is_back_to_back"] = 0.0
    features["home_efg"] = 0.52
    features["home_tor"] = 0.12
    features["home_tpar"] = 0.35
    features["home_ftr"] = 0.25
    features["home_orbp"] = 0.25
    features["home_team_id"] = 0.0
    
    features["away_pts_scored_avg_5"] = 113.0
    features["away_pts_allowed_avg_5"] = 114.0
    features["away_margin_avg_5"] = -1.0
    features["away_current_streak_5"] = 0.0
    features["away_days_since_last"] = 2.0
    features["away_is_back_to_back"] = 0.0
    features["away_efg"] = 0.51
    features["away_tor"] = 0.13
    features["away_tpar"] = 0.34
    features["away_ftr"] = 0.24
    features["away_orbp"] = 0.24
    features["away_team_id"] = 0.0
    
    features["season"] = 26.0
    features["game_date"] = 20260211.0
    
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


def load_production_model_params() -> Dict:
    """Load production CatBoost hyperparameters."""
    
    metrics_df = pd.read_csv(METRICS_PATH)
    
    fold_metrics = metrics_df[
        (metrics_df["fold"] == TARGET_FOLD) & 
        (metrics_df["model"] == "catboost")
    ]
    
    if len(fold_metrics) == 0:
        raise ValueError(f"No CatBoost metrics found for fold {TARGET_FOLD}")
    
    params_str = fold_metrics.iloc[0]["params"]
    params = json.loads(params_str)
    
    return params


def main():
    """Main entry point."""
    
    print("="*80)
    print("HALFTIME BACKTEST - ESPN SCHEDULE + NBA CDN MAPPING")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Target date
    target_date = "2026-02-11"
    
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
    
    params = load_production_model_params()
    print(f"\nProduction parameters (fold {TARGET_FOLD}):")
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
        h1_features = extract_halftime_features(game_data)
        
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
    
    print(f"\n{'='*80}")
    print("STEP 4: TRAINING PRODUCTION MODEL")
    print(f"{'='*80}")
    
    # Load historical data for training
    print(f"\nLoading historical data...")
    DATA_PATH = Path("data/processed/halftime_with_temporal_features_total.parquet")
    hist_df = pd.read_parquet(DATA_PATH)
    hist_df['game_date'] = pd.to_datetime(hist_df['game_date'])
    
    # Filter to before target date
    target_dt = pd.to_datetime(target_date).tz_localize('UTC')
    train_df = hist_df[hist_df['game_date'] < target_dt].copy()
    
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
    model.fit(X_train, numeric_feats, y_total_train, y_margin_train)
    print(f"✅ Model trained")
    
    print(f"\n{'='*80}")
    print("STEP 5: GENERATING PREDICTIONS")
    print(f"{'='*80}")
    
    # Prepare test features
    results_df = pd.DataFrame(results)
    
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
    trained_heads = model.trained_heads()
    sig_margin = trained_heads.margin.residual_sigma
    p_win = 1 - norm.cdf(0, loc=mu_margin, scale=sig_margin)
    
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
    main()
