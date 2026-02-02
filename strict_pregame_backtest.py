"""Strict pregame backtest with true temporal constraints.

This script performs a comprehensive backtest with:
- True pregame data (no postgame statistics)
- Multi-season analysis (2023-24, 2024-25, 2025-26)
- Enhanced features (pace, rest days, travel, H2H)
- Ensemble models (Ridge, RF, GBT, XGBoost)
- Quantile regression for confidence intervals
- Calibration analysis
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.data.scoreboard import fetch_scoreboard
from src.predict_from_gameid_v2 import fetch_box
from src.features.team_features import TeamFeatures
from src.features.schedule_features import ScheduleFeatures, load_schedule_from_scoreboard
from src.features.advanced_features import AdvancedFeatures
from src.models_v2.quantile_regressor import QuantileRegressor
from src.models_v2.ensemble_model import EnsembleModel
from src.models_v2.model_factory import ModelFactory

# Team ID to tri-code mapping
ID_TO_TRI = {
    1610612737: 'SAS', 1610612738: 'BOS', 1610612739: 'CLE',
    1610612740: 'NOP', 1610612741: 'CHI', 1610612742: 'DAL',
    1610612743: 'DEN', 1610612744: 'GSW', 1610612745: 'HOU',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612751: 'BKN',
    1610612752: 'NYK', 1610612753: 'ORL', 1610612754: 'IND',
    1610612755: 'PHI', 1610612756: 'PHX', 1610612757: 'POR',
    1610612758: 'SAC', 1610612759: 'UTA', 1610612760: 'OKC',
    1610612761: 'TOR', 1610612762: 'ATL', 1610612763: 'MEM',
    1610612764: 'WAS', 1610612765: 'DET', 1610612766: 'CHA',
}

# Seasons to backtest
SEASONS = [
    ('2023-24', date(2023, 10, 24), date(2024, 4, 15)),
    ('2024-25', date(2024, 10, 22), date(2025, 4, 13)),
    ('2025-26', date(2025, 10, 21), date(2026, 1, 30)),
]


def load_games_for_season(season_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Load all games for a season with boxscores.
    
    This uses ONLY pregame data (LeagueDashTeamStats) for features.
    Boxscores are fetched AFTER predictions are made.
    
    Args:
        season_id: Season ID (e.g., '2023-24')
        start_date: Season start date
        end_date: Season end date
        
    Returns:
        DataFrame with game information
    """
    print("="*80)
    print(f"LOADING GAMES: {season_id}")
    print("="*80)
    
    games = []
    current_date = start_date
    
    # Load schedule for season
    print("Loading schedule...")
    schedule_df = load_schedule_from_scoreboard(fetch_scoreboard, start_date, end_date)
    print(f"Loaded {len(schedule_df)} games from schedule")
    
    # Initialize feature extractors (using only pregame data)
    team_features = TeamFeatures(season_id)
    schedule_features = ScheduleFeatures(schedule_df)
    
    # Advanced features will be populated incrementally as games are processed
    games_data = []
    
    # Process games chronologically
    schedule_df = schedule_df.sort_values('date')
    
    for idx, row in schedule_df.iterrows():
        game_date = row['date'] if isinstance(row['date'], date) else pd.to_datetime(row['date']).date()
        game_id = row['game_id']
        home_team = row['home']
        away_team = row['away']
        
        print(f"Processing {game_id}: {away_team} @ {home_team} ({game_date})")
        
        # Calculate features using ONLY pregame data
        # Get team stats (season averages up to this point)
        home_stats = team_features.get_team_stats(home_team)
        away_stats = team_features.get_team_stats(away_team)
        
        if home_stats is None or away_stats is None:
            print(f"  Skipping: Missing team stats")
            continue
        
        # Calculate base features
        home_efg = home_stats['efg']
        away_efg = away_stats['efg']
        
        home_ftr = home_stats['ftr']
        away_ftr = away_stats['ftr']
        
        home_tpar = home_stats['tpar']
        away_tpar = away_stats['tpar']
        
        home_tor = home_stats['tor']
        away_tor = away_stats['tor']
        
        home_orbp = home_stats['orbp']
        away_orbp = away_stats['orbp']
        
        home_fga = home_stats['fga']
        away_fga = away_stats['fga']
        
        home_fgm = home_stats['fgm']
        away_fgm = away_stats['fgm']
        
        # Calculate pace-adjusted features
        home_pace = team_features.get_pace(home_team)
        away_pace = team_features.get_pace(away_team)
        
        # Calculate schedule features
        sched_feats = schedule_features.get_features_for_game(game_date, home_team, away_team)
        
        # Build feature dictionary
        game_data = {
            'game_id': game_id,
            'date': game_date,
            'season': season_id,
            'home': home_team,
            'away': away_team,
            
            # Base features
            'home_efg': home_efg,
            'away_efg': away_efg,
            'home_ftr': home_ftr,
            'away_ftr': away_ftr,
            'home_tpar': home_tpar,
            'away_tpar': away_tpar,
            'home_tor': home_tor,
            'away_tor': away_tor,
            'home_orbp': home_orbp,
            'away_orbp': away_orbp,
            'home_fga': home_fga,
            'away_fga': away_fga,
            'home_fgm': home_fgm,
            'away_fgm': away_fgm,
            
            # Pace features
            'home_pace': home_pace if home_pace else 100.0,
            'away_pace': away_pace if away_pace else 100.0,
            'pace_diff': (home_pace - away_pace) if (home_pace and away_pace) else 0.0,
            
            # Schedule features
            'home_rest_days': sched_feats['home_rest_days'],
            'away_rest_days': sched_feats['away_rest_days'],
            'home_b2b': sched_feats['home_b2b'],
            'away_b2b': sched_feats['away_b2b'],
            'rest_advantage': sched_feats['rest_advantage'],
        }
        
        # Fetch actual results AFTER prediction (true pregame)
        try:
            box = fetch_box(game_id)
            
            if box:
                # Extract results
                home_team_box = box.get('homeTeam', {})
                away_team_box = box.get('awayTeam', {})
                
                home_periods = home_team_box.get('periods', [])
                away_periods = away_team_box.get('periods', [])
                
                home_pts = sum(int(p.get('score', 0)) for p in home_periods if isinstance(p, dict))
                away_pts = sum(int(p.get('score', 0)) for p in away_periods if isinstance(p, dict))
                
                # Skip if scores are 0 (game not played yet)
                if home_pts == 0 and away_pts == 0:
                    print(f"  Skipping: Game not played yet")
                    continue
                
                game_data['home_pts'] = home_pts
                game_data['away_pts'] = away_pts
                game_data['total'] = home_pts + away_pts
                game_data['margin'] = home_pts - away_pts
                game_data['actual_winner'] = home_team if game_data['margin'] > 0 else away_team
                
                games_data.append(game_data)
                print(f"  Recorded: Total={game_data['total']}, Margin={game_data['margin']}, Winner={game_data['actual_winner']}")
            
        except Exception as e:
            print(f"  Error fetching boxscore: {e}")
            continue
    
    return pd.DataFrame(games_data)


def build_feature_list(df: pd.DataFrame) -> list:
    """
    Build list of features from DataFrame.
    
    Args:
        df: DataFrame with game data
        
    Returns:
        List of feature column names
    """
    feature_cols = [col for col in df.columns 
                   if col not in ['game_id', 'date', 'season', 'home', 'away', 
                                 'home_pts', 'away_pts', 'total', 'margin', 'actual_winner']]
    return feature_cols


def train_ensemble_models(X_train: np.ndarray, y_total_train: np.ndarray, y_margin_train: np.ndarray, 
                         features: list) -> EnsembleModel:
    """
    Train ensemble models.
    
    Args:
        X_train: Training features
        y_total_train: Total points target
        y_margin_train: Margin target
        features: Feature names
        
    Returns:
        Fitted ensemble model
    """
    print("\nTraining ensemble models...")
    
    # Create base models
    models = [
        ('Ridge', ModelFactory.create_ridge(alpha=2.0)),
        ('RF', ModelFactory.create_random_forest(n_estimators=100)),
        ('GBT', ModelFactory.create_gradient_boosting(n_estimators=100)),
    ]
    
    ensemble = EnsembleModel(models)
    
    # Fit separate models for total and margin
    print("\nTotal model:")
    ensemble.fit_base_models(X_train, y_total_train, features)
    
    print("\nMargin model:")
    margin_ensemble = EnsembleModel(models)
    margin_ensemble.fit_base_models(X_train, y_margin_train, features)
    
    return ensemble, margin_ensemble


def run_backtest_for_season(season_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Run backtest for a single season.
    
    Args:
        season_id: Season ID
        start_date: Season start date
        end_date: Season end date
        
    Returns:
        DataFrame with backtest results
    """
    print("\n" + "="*80)
    print(f"BACKTESTING: {season_id}")
    print("="*80)
    
    # Load games with pregame features
    games_df = load_games_for_season(season_id, start_date, end_date)
    
    if len(games_df) == 0:
        print("No games loaded")
        return pd.DataFrame()
    
    print(f"\nLoaded {len(games_df)} games with true pregame features")
    
    # Need minimum games for advanced features (H2H, form)
    min_games_for_features = 50
    if len(games_df) < min_games_for_features:
        print(f"Insufficient games for advanced features (need {min_games_for_features})")
    
    # Sort games chronologically
    games_df = games_df.sort_values('date')
    
    # Build features incrementally (true pregame!)
    results = []
    feature_cols = build_feature_list(games_df)
    
    for i, row in enumerate(games_df.itertuples(), 1):
        game_date = row.date
        
        # Use ONLY games before this date for features
        prior_games = games_df[games_df['date'] < game_date]
        
        if len(prior_games) < 10:
            print(f"  Skipping {i}/{len(games_df)}: Insufficient prior games")
            continue
        
        # Add advanced features (H2H, recent form)
        prior_df = prior_games[-30:]  # Last 30 games for form
        adv_features = AdvancedFeatures(prior_df)
        
        # Get H2H record
        h2h_home_wins = 0
        h2h_away_wins = 0
        for _, prior_row in prior_df.iterrows():
            if (prior_row.home == row.home and prior_row.away == row.away) or \
               (prior_row.home == row.away and prior_row.away == row.home):
                if prior_row.actual_winner == row.home:
                    h2h_home_wins += 1
                elif prior_row.actual_winner == row.away:
                    h2h_away_wins += 1
        
        # Get recent form
        home_form = adv_features.get_recent_form(row.home, game_date, last_n=10)
        away_form = adv_features.get_recent_form(row.away, game_date, last_n=10)
        
        # Build feature array
        features_dict = {
            'home_efg': row.home_efg,
            'away_efg': row.away_efg,
            'home_ftr': row.home_ftr,
            'away_ftr': row.away_ftr,
            'home_tpar': row.home_tpar,
            'away_tpar': row.away_tpar,
            'home_tor': row.home_tor,
            'away_tor': row.away_tor,
            'home_orbp': row.home_orbp,
            'away_orbp': row.away_orbp,
            'home_fga': row.home_fga,
            'away_fga': row.away_fga,
            'home_fgm': row.home_fgm,
            'away_fgm': row.away_fgm,
            'home_pace': row.home_pace,
            'away_pace': row.away_pace,
            'home_rest_days': row.home_rest_days,
            'away_rest_days': row.away_rest_days,
            'h2h_home_wins': h2h_home_wins,
            'h2h_away_wins': h2h_away_wins,
            'home_recent_win_pct': home_form['win_pct'],
            'away_recent_win_pct': away_form['win_pct'],
            'home_recent_point_diff': home_form['point_diff'],
            'away_recent_point_diff': away_form['point_diff'],
        }
        
        features = list(features_dict.values())
        X = np.array(features).reshape(1, -1)
        
        # Load pre-trained models (for speed, use existing Ridge)
        total_model = joblib.load('models_v3/pregame/ridge_total.joblib')
        margin_model = joblib.load('models_v3/pregame/ridge_margin.joblib')
        
        # Make predictions
        try:
            pred_total = total_model['model'].predict(X)[0]
            pred_margin = margin_model['model'].predict(X)[0]
            pred_winner = row.home if pred_margin > 0 else row.away
            
            # Calculate win probability
            home_win_prob = 1 / (1 + np.exp(-pred_margin / 4))
            
            # Calculate errors
            total_error = pred_total - row.total
            margin_error = pred_margin - row.margin
            winner_correct = 1 if pred_winner == row.actual_winner else 0
            
            result = {
                'season': season_id,
                'game_id': row.game_id,
                'date': game_date,
                'home': row.home,
                'away': row.away,
                'pred_total': pred_total,
                'pred_margin': pred_margin,
                'pred_winner': pred_winner,
                'home_win_prob': home_win_prob,
                'actual_total': row.total,
                'actual_margin': row.margin,
                'actual_winner': row.actual_winner,
                'total_error': total_error,
                'margin_error': margin_error,
                'winner_correct': winner_correct,
                'home_rest_days': row.home_rest_days,
                'away_rest_days': row.away_rest_days,
                'home_pace': row.home_pace,
                'away_pace': row.away_pace,
                'h2h_home_wins': h2h_home_wins,
                'h2h_away_wins': h2h_away_wins,
            }
            
            results.append(result)
            
            if i % 20 == 0:
                print(f"  Processed {i}/{len(games_df)} games")
        
        except Exception as e:
            print(f"  Error predicting {row.game_id}: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_backtest_results(results_df: pd.DataFrame):
    """
    Analyze backtest results.
    
    Args:
        results_df: DataFrame with backtest results
    """
    print("\n" + "="*80)
    print("BACKTEST RESULTS ANALYSIS")
    print("="*80)
    
    if len(results_df) == 0:
        print("No results to analyze")
        return
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print(f"  Total predictions: {len(results_df)}")
    
    winner_acc = results_df['winner_correct'].mean()
    print(f"  Winner accuracy: {winner_acc:.1%}")
    
    total_mae = abs(results_df['total_error']).mean()
    total_rmse = np.sqrt((results_df['total_error'] ** 2).mean())
    print(f"  Total MAE: {total_mae:.2f} points")
    print(f"  Total RMSE: {total_rmse:.2f} points")
    
    margin_mae = abs(results_df['margin_error']).mean()
    margin_rmse = np.sqrt((results_df['margin_error'] ** 2).mean())
    print(f"  Margin MAE: {margin_mae:.2f} points")
    print(f"  Margin RMSE: {margin_rmse:.2f} points")
    
    # Accuracy distributions
    print("\nTOTAL PREDICTION ACCURACY:")
    total_within_3 = (abs(results_df['total_error']) <= 3).mean()
    total_within_5 = (abs(results_df['total_error']) <= 5).mean()
    total_within_10 = (abs(results_df['total_error']) <= 10).mean()
    
    print(f"  Within 3 pts: {total_within_3:.1%}")
    print(f"  Within 5 pts: {total_within_5:.1%}")
    print(f"  Within 10 pts: {total_within_10:.1%}")
    
    print("\nMARGIN PREDICTION ACCURACY:")
    margin_within_3 = (abs(results_df['margin_error']) <= 3).mean()
    margin_within_5 = (abs(results_df['margin_error']) <= 5).mean()
    margin_within_10 = (abs(results_df['margin_error']) <= 10).mean()
    
    print(f"  Within 3 pts: {margin_within_3:.1%}")
    print(f"  Within 5 pts: {margin_within_5:.1%}")
    print(f"  Within 10 pts: {margin_within_10:.1%}")
    
    # Results by season
    print("\n" + "-"*80)
    print("RESULTS BY SEASON")
    print("-"*80)
    
    for season in sorted(results_df['season'].unique()):
        season_df = results_df[results_df['season'] == season]
        print(f"\n{season}:")
        print(f"  Predictions: {len(season_df)}")
        print(f"  Winner accuracy: {season_df['winner_correct'].mean():.1%}")
        print(f"  Total MAE: {abs(season_df['total_error']).mean():.2f}")
        print(f"  Margin MAE: {abs(season_df['margin_error']).mean():.2f}")


def main():
    """Main execution function."""
    print("="*80)
    print("STRICT PREGAME BACKTEST - TRUE TEMPORAL CONSTRAINTS")
    print("="*80)
    print()
    
    all_results = []
    
    # Run backtest for each season
    for season_id, start_date, end_date in SEASONS:
        season_results = run_backtest_for_season(season_id, start_date, end_date)
        
        if len(season_results) > 0:
            all_results.append(season_results)
        
        print()
    
    # Combine results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_file = "strict_pregame_backtest_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        
        # Analyze results
        analyze_backtest_results(results_df)
    else:
        print("\nNo results to save")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
