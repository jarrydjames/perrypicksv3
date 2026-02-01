"""True pregame predictions using team season averages from NBA API.

This script:
1. Fetches team season averages for 25-26 season
2. Maps team tri-codes to team IDs
3. Extracts features from season averages (not boxscore)
4. Makes predictions for scheduled games
5. Saves results to CSV

This allows TRUE pregame predictions before games start.
"""
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.data.scoreboard import fetch_scoreboard

print("="*80)
print("TRUE PREGAME PREDICTIONS USING SEASON AVERAGES")
print("="*80)
print()

# Load pregame models
print("Loading pregame models...")
try:
    total_model = joblib.load('models_v3/pregame/ridge_total.joblib')
    margin_model = joblib.load('models_v3/pregame/ridge_margin.joblib')
    feature_list = total_model['features']
    print(f"✅ Models loaded with features: {feature_list}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

print()

# Team ID to tri-code mapping (NBA 25-26 season)
ID_TO_TRI = {
    1610612737: 'SAS',
    1610612738: 'BOS',
    1610612739: 'CLE',
    1610612740: 'NOP',
    1610612741: 'CHI',
    1610612742: 'DAL',
    1610612743: 'DEN',
    1610612744: 'GSW',
    1610612745: 'HOU',
    1610612746: 'LAC',
    1610612747: 'LAL',
    1610612748: 'MIA',
    1610612749: 'MIL',
    1610612750: 'MIN',
    1610612751: 'BKN',
    1610612752: 'NYK',
    1610612753: 'ORL',
    1610612754: 'IND',
    1610612755: 'PHI',
    1610612756: 'PHX',
    1610612757: 'POR',
    1610612758: 'SAC',
    1610612759: 'UTA',
    1610612760: 'OKC',
    1610612761: 'TOR',
    1610612762: 'ATL',
    1610612763: 'MEM',
    1610612764: 'WAS',
    1610612765: 'DET',
    1610612766: 'CHA',
}

print(f"Team ID map loaded with {len(ID_TO_TRI)} teams")
print()

# Fetch games for Feb 1, 2026
game_date = date(2026, 2, 1)

print(f"Fetching games for {game_date}...")
try:
    games = fetch_scoreboard(game_date, include_live=False)
except Exception as e:
    print(f"❌ Error fetching games: {e}")
    sys.exit(1)

if len(games) == 0:
    print(f"No games found for {game_date}")
    sys.exit(0)

print(f"Found {len(games)} games")
print()

# Get unique teams for this date
unique_teams = set()
for game in games:
    unique_teams.add(game.home)
    unique_teams.add(game.away)

print(f"Unique teams needed: {sorted(unique_teams)} ({len(unique_teams)} teams)")
print()

# Import nba_api to fetch season averages
print("Fetching team season averages (LeagueDashTeamStats - single API call)...")
try:
    from nba_api.stats.endpoints import leaguedashteamstats
    
    # Fetch season averages for all teams in one call
    stats_obj = leaguedashteamstats.LeagueDashTeamStats(
        season='2025-26',
        per_mode_detailed='PerGame',
    )
    stats_df = stats_obj.get_data_frames()[0]
    
    print(f"✅ Fetched stats for {len(stats_df)} teams")
    print()
    
    # Build tri-code to stats mapping
    season_stats = {}
    
    for _, row in stats_df.iterrows():
        team_id = row['TEAM_ID']
        tri_code = ID_TO_TRI.get(team_id)
        
        if tri_code is None:
            continue
        
        # Calculate EFG_PCT: (FGM + 0.5 * FG3M) / FGA
        fgm = row['FGM']
        fg3m = row['FG3M']
        fga = row['FGA']
        
        if fga > 0:
            efg_pct = (fgm + 0.5 * fg3m) / fga
        else:
            efg_pct = 0.5
        
        # Calculate other metrics
        fta = row['FTA']
        ftm = row['FTM']
        tpa = row['FG3A']
        tpm = row['FG3M']
        tov = row['TOV']
        orb = row['OREB']
        drb = row['DREB']
        
        season_stats[tri_code] = {
            'efg_pct': efg_pct,
            'fga': fga,
            'fgm': fgm,
            'fta': fta,
            'ftm': ftm,
            'tpa': tpa,
            'tpm': tpm,
            'tov': tov,
            'orb': orb,
            'drb': drb,
        }
        
        if tri_code in unique_teams:
            print(f"  ✅ {tri_code}: EFG={efg_pct:.3f}, FGA={fga:.1f}")
            
except ImportError as e:
    print(f"❌ nba_api not installed: {e}")
    print("Please install with: pip install nba_api")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error fetching season stats: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print(f"Successfully fetched stats for {len([s for s in season_stats.values() if s is not None])} teams")
print()

# Extract features and make predictions
print("="*80)
print("MAKING TRUE PREGAME PREDICTIONS")
print("="*80)
print()

results = []

for i, game in enumerate(games, 1):
    game_id = game.game_id
    home_tri = game.home
    away_tri = game.away
    
    print(f"{i}. Game ID: {game_id}")
    print(f"   {away_tri} @ {home_tri}")
    print(f"   Status: {game.status_text}")
    
    # Get season averages for both teams
    home_avgs = season_stats.get(home_tri)
    away_avgs = season_stats.get(away_tri)
    
    if home_avgs is None:
        print(f"  ❌ No season averages for {home_tri}")
        continue
    if away_avgs is None:
        print(f"  ❌ No season averages for {away_tri}")
        continue
    
    # Calculate features from season averages
    try:
        # Home features
        home_efg = home_avgs['efg_pct']
        home_fga = home_avgs['fga']
        home_fta = home_avgs['fta']
        home_tpa = home_avgs['tpa']
        home_tov = home_avgs['tov']
        home_orb = home_avgs['orb']
        home_drb = home_avgs['drb']
        
        home_ftr = home_fta / max(home_fga, 1)
        home_tpar = home_tpa / max(home_fga, 1)
        home_possessions = max(home_fga + 0.44 * home_fta + home_tov, 1)
        home_tor = home_tov / home_possessions
        home_orbp = (home_orb / max(home_orb + home_drb, 1)) if (home_orb + home_drb) > 0 else 0
        
        # Away features
        away_efg = away_avgs['efg_pct']
        away_fga = away_avgs['fga']
        away_fta = away_avgs['fta']
        away_tpa = away_avgs['tpa']
        away_tov = away_avgs['tov']
        away_orb = away_avgs['orb']
        away_drb = away_avgs['drb']
        
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
            'home_fgm': home_avgs['fgm'],
            'away_fga': away_fga,
            'away_fgm': away_avgs['fgm'],
        }
        
        # Build feature array in correct order AND RESHAPE TO 2D!
        X = np.array([features_dict.get(f, 0.0) for f in feature_list]).reshape(1, -1)
        
    except Exception as e:
        print(f"  ❌ Feature extraction failed: {e}")
        continue
    
    # Make prediction
    try:
        pred_total = total_model['model'].predict(X)[0]
        pred_margin = margin_model['model'].predict(X)[0]
        
        total_sigma = total_model.get('residual_sigma', 8.0)
        margin_sigma = margin_model.get('residual_sigma', 5.0)
        
        total_q10 = pred_total - 1.28 * total_sigma
        total_q90 = pred_total + 1.28 * total_sigma
        margin_q10 = pred_margin - 1.28 * margin_sigma
        margin_q90 = pred_margin + 1.28 * margin_sigma
        
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        continue
    
    # Determine predicted winner
    pred_winner = home_tri if pred_margin > 0 else away_tri
    
    # Calculate win probability
    home_win_prob = 1 / (1 + np.exp(-pred_margin / 4))
    
    print(f"  Prediction: Total={pred_total:.1f}, Margin={pred_margin:.1f}, Winner={pred_winner}")
    print(f"  Win Prob: Home={home_win_prob:.1%}, Away={1-home_win_prob:.1%}")
    print()
    
    results.append({
        'game_id': game_id,
        'home_team': home_tri,
        'away_team': away_tri,
        'status': game.status_text,
        'pred_total': round(pred_total, 1),
        'pred_margin': round(pred_margin, 1),
        'pred_winner': pred_winner,
        'home_win_prob': round(home_win_prob, 3),
        'away_win_prob': round(1 - home_win_prob, 3),
        'total_q10': round(total_q10, 1),
        'total_q90': round(total_q90, 1),
        'margin_q10': round(margin_q10, 1),
        'margin_q90': round(margin_q90, 1),
        'error': None,
    })

# Create DataFrame and save
results_df = pd.DataFrame(results)

print("="*80)
print("PREDICTION SUMMARY")
print("="*80)
print()

if len(results_df) > 0:
    print(f"Total games: {len(results_df)}")
    print(f"Successful predictions: {len(results_df)}")
    print()
    
    if len(results_df) > 0:
        # Summary statistics
        print(f"Predicted Totals: Min={results_df['pred_total'].min():.1f}, Max={results_df['pred_total'].max():.1f}, Mean={results_df['pred_total'].mean():.1f}")
        print(f"Predicted Margins: Min={results_df['pred_margin'].min():.1f}, Max={results_df['pred_margin'].max():.1f}, Mean={results_df['pred_margin'].mean():.1f}")
        print()
        
        # Winner predictions
        home_wins = (results_df['pred_winner'] == results_df['home_team']).sum()
        print(f"Predicted home winners: {home_wins}/{len(results_df)} ({home_wins/len(results_df)*100:.1f}%)")
        print(f"Predicted away winners: {len(results_df)-home_wins}/{len(results_df)} ({(len(results_df)-home_wins)/len(results_df)*100:.1f}%)")
    
    # Save to CSV
    output_file = "pregame_predictions_FEB1_2026_SEASON_AVGS.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ Saved results to: {output_file}")
    print()
    
    print("="*80)
    print("FULL PREDICTIONS - FEBRUARY 1, 2026 (TRUE PREGAME)")
    print("="*80)
    print()
    
    # Display all predictions
    for idx, row in results_df.iterrows():
        print(f"{row['game_id']}: {row['away_team']} @ {row['home_team']}")
        print(f"  Prediction: Total={row['pred_total']}, Margin={row['pred_margin']}, Winner={row['pred_winner']}")
        print(f"  Win Prob: Home={row['home_win_prob']*100:.1f}%, Away={row['away_win_prob']*100:.1f}%")
        print(f"  Total 80% CI: [{row['total_q10']}, {row['total_q90']}]")
        print(f"  Margin 80% CI: [{row['margin_q10']}, {row['margin_q90']}]")
        print()
else:
    print("No results to save")

print()
print("="*80)
print("PREDICTION COMPLETE - TRUE PREGAME USING SEASON AVERAGES")
print("="*80)
