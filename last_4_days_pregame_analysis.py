"""Last 4 days pregame prediction analysis using season averages.

This script:
1. Fetches games for the last 4 days
2. Uses LeagueDashTeamStats season averages (true pregame data)
3. Makes predictions before games start
4. Compares predictions to actual results
5. Generates comprehensive accuracy report

True pregame: Only data available before game start is used.
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

# Load pregame models
print("Loading pregame models...")
total_model = joblib.load('models_v3/pregame/ridge_total.joblib')
margin_model = joblib.load('models_v3/pregame/ridge_margin.joblib')
feature_list = total_model['features']
print("Models loaded with " + str(len(feature_list)) + " features")
print()

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

def fetch_season_team_stats():
    """Fetch team season averages using LeagueDashTeamStats."""
    print("Fetching team stats for 2025-26 season...")
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        stats_obj = leaguedashteamstats.LeagueDashTeamStats(
            season='2025-26',
            per_mode_detailed='PerGame',
        )
        df = stats_obj.get_data_frames()[0]
        print("Fetched stats for " + str(len(df)) + " teams")
        
        # Build tri-code mapping
        df['tri_code'] = df['TEAM_ID'].map(ID_TO_TRI)
        return df
    except Exception as e:
        print("Error fetching team stats: " + str(e))
        import traceback
        traceback.print_exc()
        return None

def calculate_features(home_tri, away_tri, season_stats_df):
    """Calculate prediction features from season averages."""
    
    # Get home team stats
    home_row = season_stats_df[season_stats_df['tri_code'] == home_tri]
    if len(home_row) == 0:
        return None
    home_row = home_row.iloc[0]
    
    # Get away team stats
    away_row = season_stats_df[season_stats_df['tri_code'] == away_tri]
    if len(away_row) == 0:
        return None
    away_row = away_row.iloc[0]
    
    # Calculate EFG
    home_efg = (home_row['FGM'] + 0.5 * home_row['FG3M']) / home_row['FGA'] if home_row['FGA'] > 0 else 0.5
    away_efg = (away_row['FGM'] + 0.5 * away_row['FG3M']) / away_row['FGA'] if away_row['FGA'] > 0 else 0.5
    
    # Home features
    home_fga = home_row['FGA']
    home_fta = home_row['FTA']
    home_tpa = home_row['FG3A']
    home_tov = home_row['TOV']
    home_orb = home_row['OREB']
    home_drb = home_row['DREB']
    
    home_ftr = home_fta / max(home_fga, 1)
    home_tpar = home_tpa / max(home_fga, 1)
    home_possessions = max(home_fga + 0.44 * home_fta + home_tov, 1)
    home_tor = home_tov / home_possessions
    home_orbp = (home_orb / max(home_orb + home_drb, 1)) if (home_orb + home_drb) > 0 else 0
    
    # Away features
    away_fga = away_row['FGA']
    away_fta = away_row['FTA']
    away_tpa = away_row['FG3A']
    away_tov = away_row['TOV']
    away_orb = away_row['OREB']
    away_drb = away_row['DREB']
    
    away_ftr = away_fta / max(away_fga, 1)
    away_tpar = away_tpa / max(away_fga, 1)
    away_possessions = max(away_fga + 0.44 * away_fta + away_tov, 1)
    away_tor = away_tov / away_possessions
    away_orbp = (away_orb / max(away_orb + away_drb, 1)) if (away_orb + away_drb) > 0 else 0
    
    features_dict = {
        'home_efg': home_efg,
        'home_ftr': home_ftr,
        'home_tpar': home_tpar,
        'home_tor': home_tor,
        'home_orbp': home_orbp,
        'away_efg': away_efg,
        'away_ftr': away_ftr,
        'away_tpar': away_tpar,
        'away_tor': away_tor,
        'away_orbp': away_orbp,
        'home_fga': home_fga,
        'home_fgm': home_row['FGM'],
        'away_fga': away_fga,
        'away_fgm': away_row['FGM'],
    }
    
    return features_dict

# Dates for last 4 days
test_dates = [
    date(2026, 1, 26),
    date(2026, 1, 27),
    date(2026, 1, 28),
    date(2026, 1, 29),
]

print("="*80)
print("LAST 4 DAYS PREGAME PREDICTION ANALYSIS")
print("="*80)
print()

# Fetch season averages (one call)
season_stats_df = fetch_season_team_stats()

if season_stats_df is None:
    print("Could not fetch season stats")
    sys.exit(1)

print()

# Fetch games for each date and make predictions
all_results = []

for test_date in test_dates:
    print("-"*80)
    print("Processing: " + test_date.strftime('%Y-%m-%d'))
    print("-"*80)
    
    try:
        games = fetch_scoreboard(test_date, include_live=False)
    except Exception as e:
        print("Error fetching games: " + str(e))
        continue
    
    if len(games) == 0:
        print("No games found")
        continue
    
    print("Found " + str(len(games)) + " games")
    print()
    
    for game in games:
        game_id = game.game_id
        home_tri = game.home
        away_tri = game.away
        
        # Calculate features from season averages
        features = calculate_features(home_tri, away_tri, season_stats_df)
        
        if features is None:
            print("  Skipping " + str(game_id) + ": Missing team stats")
            continue
        
        try:
            # Build feature array
            X = np.array([features.get(f, 0.0) for f in feature_list]).reshape(1, -1)
        except Exception as e:
            print("  Skipping " + str(game_id) + ": Feature error: " + str(e))
            continue
        
        # Make prediction
        try:
            pred_total = total_model['model'].predict(X)[0]
            pred_margin = margin_model['model'].predict(X)[0]
            pred_winner = home_tri if pred_margin > 0 else away_tri
            
            # Calculate win probability
            home_win_prob = 1 / (1 + np.exp(-pred_margin / 4))
            
        except Exception as e:
            print("  Skipping " + str(game_id) + ": Prediction error: " + str(e))
            continue
        
        # Fetch actual results from boxscore
        try:
            box_data = fetch_box(game_id)
            
            if box_data:
                # Extract scores from boxscore
                home_team = box_data.get('homeTeam', {})
                away_team = box_data.get('awayTeam', {})
                
                home_periods = home_team.get('periods', [])
                away_periods = away_team.get('periods', [])
                
                # Calculate total points
                home_pts = sum(int(p.get('score', 0)) for p in home_periods if isinstance(p, dict) and p.get('score'))
                away_pts = sum(int(p.get('score', 0)) for p in away_periods if isinstance(p, dict) and p.get('score'))
                
                # If scores are 0, game might not have been played
                if home_pts == 0 and away_pts == 0:
                    print("  Skipping " + str(game_id) + ": Game not played (no scores)")
                    continue
                
                actual_total = home_pts + away_pts
                actual_margin = home_pts - away_pts
                actual_winner = home_tri if actual_margin > 0 else away_tri
                
                total_error = pred_total - actual_total
                margin_error = pred_margin - actual_margin
                winner_correct = 1 if pred_winner == actual_winner else 0
                
                result = {
                    'date': test_date,
                    'game_id': game_id,
                    'home': home_tri,
                    'away': away_tri,
                    'pred_total': round(pred_total, 1),
                    'pred_margin': round(pred_margin, 1),
                    'pred_winner': pred_winner,
                    'home_win_prob': round(home_win_prob, 3),
                    'actual_total': actual_total,
                    'actual_margin': actual_margin,
                    'actual_winner': actual_winner,
                    'total_error': round(total_error, 1),
                    'margin_error': round(margin_error, 1),
                    'winner_correct': winner_correct,
                }
                
                all_results.append(result)
                
                print("  " + str(game_id) + ": " + away_tri + " @ " + home_tri + 
                      " | Pred: Total=" + str(round(pred_total, 1)) + 
                      ", Margin=" + str(round(pred_margin, 1)) + 
                      ", Winner=" + pred_winner + 
                      " | Actual: Total=" + str(actual_total) + 
                      ", Margin=" + str(actual_margin) + 
                      ", Winner=" + actual_winner + 
                      " | Correct: " + ("✓" if winner_correct else "✗"))
        except Exception as e:
            print("  Skipping " + str(game_id) + ": Boxscore error: " + str(e))
            import traceback
            traceback.print_exc()
            continue

print()
print("="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print()

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    
    print("Total predictions: " + str(len(results_df)))
    print("Dates analyzed: " + str(len(test_dates)))
    print()
    
    # Winner accuracy
    winner_acc = results_df['winner_correct'].mean()
    print("Winner Accuracy: " + str(round(winner_acc * 100, 1)) + "%")
    print()
    
    # Total MAE and RMSE
    total_mae = abs(results_df['total_error']).mean()
    total_rmse = np.sqrt((results_df['total_error'] ** 2).mean())
    print("Total MAE: " + str(round(total_mae, 2)) + " points")
    print("Total RMSE: " + str(round(total_rmse, 2)) + " points")
    print()
    
    # Margin MAE and RMSE
    margin_mae = abs(results_df['margin_error']).mean()
    margin_rmse = np.sqrt((results_df['margin_error'] ** 2).mean())
    print("Margin MAE: " + str(round(margin_mae, 2)) + " points")
    print("Margin RMSE: " + str(round(margin_rmse, 2)) + " points")
    print()
    
    # Error distribution
    total_within_3 = (abs(results_df['total_error']) <= 3).mean()
    total_within_5 = (abs(results_df['total_error']) <= 5).mean()
    total_within_10 = (abs(results_df['total_error']) <= 10).mean()
    
    margin_within_3 = (abs(results_df['margin_error']) <= 3).mean()
    margin_within_5 = (abs(results_df['margin_error']) <= 5).mean()
    margin_within_10 = (abs(results_df['margin_error']) <= 10).mean()
    
    print("Total Prediction Accuracy:")
    print("  Within 3 pts: " + str(round(total_within_3 * 100, 1)) + "% (" + str(int(total_within_3 * len(results_df))) + "/" + str(len(results_df)) + ")")
    print("  Within 5 pts: " + str(round(total_within_5 * 100, 1)) + "%")
    print("  Within 10 pts: " + str(round(total_within_10 * 100, 1)) + "%")
    print()
    
    print("Margin Prediction Accuracy:")
    print("  Within 3 pts: " + str(round(margin_within_3 * 100, 1)) + "% (" + str(int(margin_within_3 * len(results_df))) + "/" + str(len(results_df)) + ")")
    print("  Within 5 pts: " + str(round(margin_within_5 * 100, 1)) + "%")
    print("  Within 10 pts: " + str(round(margin_within_10 * 100, 1)) + "%")
    print()
    
    # Results by date
    print("="*80)
    print("RESULTS BY DATE")
    print("="*80)
    
    for test_date in test_dates:
        date_df = results_df[results_df['date'] == test_date]
        if len(date_df) > 0:
            print("\n" + test_date.strftime('%Y-%m-%d') + ":")
            print("  Games: " + str(len(date_df)))
            print("  Winner Accuracy: " + str(round(date_df['winner_correct'].mean() * 100, 1)) + "%")
            print("  Total MAE: " + str(round(abs(date_df['total_error']).mean(), 2)))
            print("  Margin MAE: " + str(round(abs(date_df['margin_error']).mean(), 2)))
    
    # Save results
    output_file = "pregame_last_4_days_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print()
    print("Saved results to: " + output_file)
    
else:
    print("No results to analyze")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
