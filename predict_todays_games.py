"""
Predict all of today's games using pregame data
Uses 2025-26 season data (CURRENT SEASON IN PROGRESS)
"""
import logging
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.endpoints import leaguegamefinder
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Season to use - CLEARLY DEFINED FOR 2025-26
SEASON = '2025-26'

# Team ID mappings
TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
    'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
    'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
    'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
    'TOR': 1610612761, 'WAS': 1610612762, 'MEM': 1610612763, 'UTA': 1610612764,
    'DET': 1610612765, 'CHA': 1610612766
}


def parse_matchup(matchup):
    """
    Parse matchup string to extract home and away teams.
    
    Formats:
    - "CHI @ MIA" → Away=CHI, Home=MIA
    - "MIA vs. CHI" → Home=MIA, Away=CHI
    """
    matchup = matchup.strip()
    
    # Check for " @ " format (Away @ Home)
    if ' @ ' in matchup:
        parts = matchup.split(' @ ')
        if len(parts) == 2:
            away = parts[0].strip()
            home = parts[1].strip()
            return home, away
    
    # Check for " vs. " format (Home vs Away)
    if ' vs. ' in matchup:
        parts = matchup.split(' vs. ')
        if len(parts) == 2:
            home = parts[0].strip()
            away = parts[1].strip()
            return home, away
    
    # Check for " vs " format (no dot)
    if ' vs ' in matchup:
        parts = matchup.split(' vs ')
        if len(parts) == 2:
            home = parts[0].strip()
            away = parts[1].strip()
            return home, away
    
    logger.warning(f"Could not parse matchup: {matchup}")
    return 'Unknown', 'Unknown'


def get_todays_games():
    """Get all games for today (Feb 1, 2026)."""
    logger.info(f"Fetching today's games (Season {SEASON})...")
    
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00',
            season_nullable=SEASON,
            season_type_nullable='Regular Season'
        )
        df = gamefinder.get_data_frames()[0]
        
        # Filter for today's games
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        todays_games = df[df['GAME_DATE'].dt.date == pd.Timestamp('2026-02-01').date()]
        
        if len(todays_games) == 0:
            logger.warning("No games found for today")
            return []
        
        # Deduplicate by game_id (keep first occurrence)
        todays_games = todays_games.drop_duplicates(subset=['GAME_ID'], keep='first')
        
        logger.info(f"Found {len(todays_games)} unique games for today")
        
        game_list = []
        for _, game in todays_games.iterrows():
            matchup = game.get('MATCHUP', '')
            
            # Parse home/away teams
            home_team, away_team = parse_matchup(matchup)
            
            # Get team IDs
            home_id = TEAM_IDS.get(home_team)
            away_id = TEAM_IDS.get(away_team)
            
            # Determine game status
            wl = game.get('WL', '')
            if isinstance(wl, str) and ('W' in wl or 'L' in wl):
                status = 3  # Final
            else:
                status = 2  # In Progress or Upcoming
            
            game_list.append({
                'game_id': game.get('GAME_ID'),
                'home_team': home_team,
                'away_team': away_team,
                'home_id': home_id,
                'away_id': away_id,
                'status': status
            })
        
        return game_list
        
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_team_stats(team_id, team_name):
    """Fetch current season stats for a team (Advanced mode)."""
    if team_id is None:
        logger.warning(f"No team ID for {team_name}")
        return None
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season=SEASON,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        
        if len(df) == 0:
            logger.warning(f"No stats found for {team_name}")
            return None
        
        return df.iloc[0]
    except Exception as e:
        logger.error(f"Error fetching {team_name} stats: {e}")
        return None


def fetch_recent_games(team_id, team_name, n=10):
    """Fetch recent games for a team."""
    if team_id is None:
        logger.warning(f"No team ID for {team_name}")
        return None
    
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=SEASON
        )
        df = gamelog.get_data_frames()[0]
        
        if len(df) == 0:
            logger.warning(f"No games found for {team_name}")
            return None
        
        return df.head(n)
    except Exception as e:
        logger.error(f"Error fetching {team_name} games: {e}")
        return None


def calculate_features(home_stats, away_stats, home_recent, away_recent):
    """Calculate pregame features."""
    features = {}
    
    # Home team features
    if home_stats is not None:
        features['home_pace'] = home_stats.get('PACE', 100.0)
        features['home_off_rating'] = home_stats.get('OFF_RATING', 110.0)
        features['home_def_rating'] = home_stats.get('DEF_RATING', 110.0)
        features['home_efg'] = home_stats.get('EFG_PCT', 0.50)
        features['home_win_pct'] = home_stats.get('W', 0.5) / home_stats.get('GP', 1.0)
    
    # Away team features
    if away_stats is not None:
        features['away_pace'] = away_stats.get('PACE', 100.0)
        features['away_off_rating'] = away_stats.get('OFF_RATING', 110.0)
        features['away_def_rating'] = away_stats.get('DEF_RATING', 110.0)
        features['away_efg'] = away_stats.get('EFG_PCT', 0.50)
        features['away_win_pct'] = away_stats.get('W', 0.5) / away_stats.get('GP', 1.0)
    
    # Recent games
    if home_recent is not None and len(home_recent) > 0:
        features['home_recent_points'] = home_recent['PTS'].mean()
        features['home_road_win_pct'] = (home_recent['WL'] == 'W').mean()
    
    if away_recent is not None and len(away_recent) > 0:
        features['away_recent_points'] = away_recent['PTS'].mean()
        features['away_road_win_pct'] = (away_recent['WL'] == 'W').mean()
    
    # Differentials
    if 'home_pace' in features and 'away_pace' in features:
        features['pace_diff'] = features['home_pace'] - features['away_pace']
    
    if 'home_win_pct' in features and 'away_win_pct' in features:
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
    
    return features


def make_prediction(features, home_team, away_team):
    """Make prediction based on features."""
    if 'home_pace' not in features or 'away_pace' not in features:
        return None
    
    predicted_pace = (features['home_pace'] + features['away_pace']) / 2
    
    # Estimate total points
    avg_off_rating = (features['home_off_rating'] + features['away_off_rating']) / 2
    predicted_total = (avg_off_rating / 100) * predicted_pace * 2
    
    # Estimate margin
    scoring_diff = 0.0
    win_diff = 0.0
    home_court = 3.0
    
    if 'home_recent_points' in features and 'away_recent_points' in features:
        scoring_diff = (features['home_recent_points'] - features['away_recent_points']) * 0.5
    
    if 'win_pct_diff' in features:
        win_diff = features['win_pct_diff'] * 30
    
    predicted_margin = scoring_diff + win_diff + home_court
    
    # Winner and confidence
    predicted_winner = home_team if predicted_margin > 0 else away_team
    confidence = min(0.85, 0.55 + abs(predicted_margin) / 20)
    
    return {
        'total': predicted_total,
        'home_score': (predicted_total + predicted_margin) / 2,
        'away_score': (predicted_total - predicted_margin) / 2,
        'margin': predicted_margin,
        'winner': predicted_winner,
        'confidence': confidence
    }


def main():
    logger.info("=" * 70)
    logger.info("PREGAME PREDICTIONS FOR FEB 1, 2026")
    logger.info(f"Season: {SEASON} (IN PROGRESS)")
    logger.info("=" * 70)
    
    # Get today's games
    games = get_todays_games()
    
    if not games:
        logger.error("No games to predict")
        return
    
    # Process each game
    predictions = []
    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        home_id = game['home_id']
        away_id = game['away_id']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Game: {away_team} @ {home_team} ({game_id})")
        logger.info(f"{'='*70}")
        
        # Fetch team data
        home_stats = fetch_team_stats(home_id, home_team)
        away_stats = fetch_team_stats(away_id, away_team)
        
        home_recent = fetch_recent_games(home_id, home_team, n=10)
        away_recent = fetch_recent_games(away_id, away_team, n=10)
        
        # Calculate features
        features = calculate_features(home_stats, away_stats, home_recent, away_recent)
        
        # Make prediction
        prediction = make_prediction(features, home_team, away_team)
        
        if prediction is not None:
            pred_data = {
                'game_id': game_id,
                'matchup': f"{away_team} @ {home_team}",
                'predicted_total': prediction['total'],
                'predicted_home_score': prediction['home_score'],
                'predicted_away_score': prediction['away_score'],
                'predicted_margin': prediction['margin'],
                'predicted_winner': prediction['winner'],
                'confidence': prediction['confidence']
            }
            predictions.append(pred_data)
            
            logger.info(f"\nPREDICTION:")
            logger.info(f"  Total: {prediction['total']:.1f} ± 15.6")
            logger.info(f"  {home_team}: {prediction['home_score']:.1f} ± 7.8")
            logger.info(f"  {away_team}: {prediction['away_score']:.1f} ± 7.8")
            logger.info(f"  Margin: {prediction['margin']:.1f} ± 11.2")
            logger.info(f"  Winner: {prediction['winner']}")
            logger.info(f"  Confidence: {prediction['confidence']:.2f}")
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("PREDICTIONS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"\n{'Matchup':<30} {'Total':<10} {'Home':<8} {'Away':<8} {'Winner':<10} {'Conf':<6}")
    logger.info("-" * 80)
    
    for pred in predictions:
        matchup = pred['matchup']
        total = f"{pred['predicted_total']:.0f}"
        home = f"{pred['predicted_home_score']:.0f}"
        away = f"{pred['predicted_away_score']:.0f}"
        winner = pred['predicted_winner']
        conf = f"{pred['confidence']:.2f}"
        
        logger.info(f"{matchup:<30} {total:<10} {home:<8} {away:<8} {winner:<10} {conf:<6}")
    
    # Save predictions to file
    output_df = pd.DataFrame(predictions)
    output_path = Path('data/predictions/todays_predictions_2026-02-01.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"\nPredictions saved to: {output_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info("PREDICTIONS COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
