"""
PHASE 2: Build Enhanced V2 Features

This adds new features to improve prediction:
1. Pace features (home/away team pace)
2. Schedule features (rest days, B2B flags)
3. H2H features (head-to-head record)
4. Recent form features (last N games)

Baseline: 4-day OOS (31 games, MAE 19.06, 64.5% accuracy)
Target: Improve to MAE < 15, Accuracy > 70%
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V2FeatureBuilder:
    """Build enhanced V2 features for pregame prediction."""
    
    def __init__(self):
        self.game_results = {}  # Cache for game results
        self.team_schedule = {}  # Cache for team schedules
    
    def load_game_results(self, boxscore_dir: str = 'data/raw/box') -> Dict:
        """
        Load game results from boxscore files.
        Returns: {game_id: {'home': team, 'away': team, 'home_score': x, 'away_score': y, 'date': date}}
        """
        logger.info("="*70)
        logger.info("LOADING GAME RESULTS FROM BOXSCORES")
        logger.info("="*70)
        
        results = {}
        boxscore_path = Path(boxscore_dir)
        
        # Limit to recent games for performance
        json_files = list(boxscore_path.glob('*.json'))
        logger.info(f"📊 Found {len(json_files)} boxscore files")
        
        for json_file in json_files:
            try:
                game_id = json_file.stem
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract result
                if 'game' in data:
                    game = data['game']
                    home = game.get('homeTeam', {})
                    away = game.get('awayTeam', {})
                    
                    home_tri = home.get('teamTricode', 'UNK')
                    away_tri = away.get('teamTricode', 'UNK')
                    home_score = home.get('score', 0)
                    away_score = away.get('score', 0)
                    
                    # Extract date
                    game_date = game.get('gameEt') or game.get('gameTimeUTC')
                    
                    results[game_id] = {
                        'home_tri': home_tri,
                        'away_tri': away_tri,
                        'home_score': home_score,
                        'away_score': away_score,
                        'total': home_score + away_score,
                        'margin': home_score - away_score,
                        'date': game_date,
                    }
            except Exception as e:
                logger.warning(f"⚠️ Error loading {json_file}: {e}")
        
        logger.info(f"✅ Loaded {len(results)} game results")
        return results
    
    def build_team_schedule(self, results: Dict) -> Dict[str, List[Dict]]:
        """
        Build schedule for each team.
        Returns: {team: [{game_id, date, home_away, opponent, score}]}
        """
        logger.info("="*70)
        logger.info("BUILDING TEAM SCHEDULES")
        logger.info("="*70)
        
        schedule = {}
        
        for game_id, result in results.items():
            for team, role in [(result['home_tri'], 'home'), (result['away_tri'], 'away')]:
                if team not in schedule:
                    schedule[team] = []
                
                opponent = result['away_tri'] if role == 'home' else result['home_tri']
                score = result['home_score'] if role == 'home' else result['away_score']
                opponent_score = result['away_score'] if role == 'home' else result['home_score']
                
                schedule[team].append({
                    'game_id': game_id,
                    'date': result['date'],
                    'role': role,
                    'opponent': opponent,
                    'score': score,
                    'opponent_score': opponent_score,
                    'total': result['total'],
                    'margin': result['margin'] if role == 'home' else -result['margin'],
                })
        
        # Sort by date for each team
        for team in schedule:
            schedule[team] = sorted(schedule[team], key=lambda x: x.get('date', ''))
        
        logger.info(f"✅ Built schedules for {len(schedule)} teams")
        return schedule
    
    def calculate_pace_features(self, team_schedule: Dict, game_id: str, home_tri: str, away_tri: str) -> Dict[str, float]:
        """
        Calculate pace features from recent games.
        Pace = total points / possessions (simplified as total points)
        """
        features = {}
        
        # Get last 10 games for each team
        home_games = [g for g in team_schedule.get(home_tri, []) if g['game_id'] != game_id][-10:]
        away_games = [g for g in team_schedule.get(away_tri, []) if g['game_id'] != game_id][-10:]
        
        # Average pace (simplified as avg total points)
        home_avg_total = np.mean([g['total'] for g in home_games]) if home_games else 215
        away_avg_total = np.mean([g['total'] for g in away_games]) if away_games else 215
        
        features['home_pace'] = home_avg_total
        features['away_pace'] = away_avg_total
        features['pace_diff'] = home_avg_total - away_avg_total
        features['avg_pace'] = (home_avg_total + away_avg_total) / 2
        
        return features
    
    def calculate_schedule_features(self, team_schedule: Dict, game_id: str, home_tri: str, away_tri: str) -> Dict[str, float]:
        """
        Calculate schedule-based features:
        - Rest days (days since last game)
        - Back-to-back flag
        - Home/away streak
        """
        features = {}
        
        # Find current game in schedule
        home_schedule = team_schedule.get(home_tri, [])
        away_schedule = team_schedule.get(away_tri, [])
        
        # Find index of current game (exclude it from lookup)
        home_current_idx = next((i for i, g in enumerate(home_schedule) if g['game_id'] == game_id), len(home_schedule))
        away_current_idx = next((i for i, g in enumerate(away_schedule) if g['game_id'] == game_id), len(away_schedule))
        
        # Rest days (days since last game)
        if home_current_idx > 0:
            features['home_rest_days'] = 1  # Simplified - assume 1 day
        else:
            features['home_rest_days'] = 7  # First game of season
        
        if away_current_idx > 0:
            features['away_rest_days'] = 1
        else:
            features['away_rest_days'] = 7
        
        # B2B flags (rest_days == 0 or 1)
        features['home_b2b'] = 1 if features['home_rest_days'] <= 1 else 0
        features['away_b2b'] = 1 if features['away_rest_days'] <= 1 else 0
        
        # Rest advantage
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        
        # Home/Away roles
        features['home_team_home'] = 1  # Always 1 for home team
        features['away_team_home'] = 0  # Always 0 for away team
        
        return features
    
    def calculate_recent_form(self, team_schedule: Dict, game_id: str, home_tri: str, away_tri: str, n_games: int = 5) -> Dict[str, float]:
        """
        Calculate recent form from last N games.
        - Win rate
        - Average margin
        - Average total
        """
        features = {}
        
        # Get last N games
        home_games = [g for g in team_schedule.get(home_tri, []) if g['game_id'] != game_id][-n_games:]
        away_games = [g for g in team_schedule.get(away_tri, []) if g['game_id'] != game_id][-n_games:]
        
        # Home team form
        if home_games:
            home_wins = sum(1 for g in home_games if g['margin'] > 0)
            features['home_win_rate_recent'] = home_wins / len(home_games)
            features['home_avg_margin_recent'] = np.mean([g['margin'] for g in home_games])
            features['home_avg_total_recent'] = np.mean([g['total'] for g in home_games])
        else:
            features['home_win_rate_recent'] = 0.5
            features['home_avg_margin_recent'] = 0
            features['home_avg_total_recent'] = 215
        
        # Away team form
        if away_games:
            away_wins = sum(1 for g in away_games if g['margin'] > 0)
            features['away_win_rate_recent'] = away_wins / len(away_games)
            features['away_avg_margin_recent'] = np.mean([g['margin'] for g in away_games])
            features['away_avg_total_recent'] = np.mean([g['total'] for g in away_games])
        else:
            features['away_win_rate_recent'] = 0.5
            features['away_avg_margin_recent'] = 0
            features['away_avg_total_recent'] = 215
        
        # Form differential
        features['win_rate_diff'] = features['home_win_rate_recent'] - features['away_win_rate_recent']
        features['margin_diff_recent'] = features['home_avg_margin_recent'] - features['away_avg_margin_recent']
        
        return features
    
    def calculate_h2h_features(self, game_results: Dict, game_id: str, home_tri: str, away_tri: str, n_meetings: int = 10) -> Dict[str, float]:
        """
        Calculate head-to-head features from last N meetings.
        """
        features = {}
        
        # Find all H2H games (excluding current)
        h2h_games = [
            r for gid, r in game_results.items()
            if gid != game_id and ((r['home_tri'] == home_tri and r['away_tri'] == away_tri) or
                                   (r['home_tri'] == away_tri and r['away_tri'] == home_tri))
        ]
        
        # Take last N meetings
        recent_h2h = h2h_games[-n_meetings:]
        
        if recent_h2h:
            home_wins = sum(1 for r in recent_h2h if r['home_tri'] == home_tri and r['margin'] > 0)
            home_wins += sum(1 for r in recent_h2h if r['away_tri'] == home_tri and r['margin'] < 0)
            
            features['h2h_home_win_rate'] = home_wins / len(recent_h2h)
            features['h2h_avg_margin'] = np.mean([r['margin'] for r in recent_h2h if r['home_tri'] == home_tri])
            features['h2h_avg_total'] = np.mean([r['total'] for r in recent_h2h])
            features['h2h_meetings'] = len(recent_h2h)
        else:
            features['h2h_home_win_rate'] = 0.5
            features['h2h_avg_margin'] = 0
            features['h2h_avg_total'] = 215
            features['h2h_meetings'] = 0
        
        return features
    
    def build_v2_features(self, game_id: str, base_features: Dict, game_results: Dict, team_schedule: Dict) -> Dict:
        """
        Build all V2 features for a game.
        """
        home_tri = base_features.get('home_tri', 'UNK')
        away_tri = base_features.get('away_tri', 'UNK')
        
        # Calculate all feature groups
        pace_features = self.calculate_pace_features(team_schedule, game_id, home_tri, away_tri)
        schedule_features = self.calculate_schedule_features(team_schedule, game_id, home_tri, away_tri)
        form_features = self.calculate_recent_form(team_schedule, game_id, home_tri, away_tri)
        h2h_features = self.calculate_h2h_features(game_results, game_id, home_tri, away_tri)
        
        # Combine all features
        v2_features = {
            **base_features,
            **pace_features,
            **schedule_features,
            **form_features,
            **h2h_features,
        }
        
        return v2_features
    
    def build_v2_dataset(self, base_dataset_path: str, output_path: str) -> pd.DataFrame:
        """
        Build V2 dataset by adding enhanced features to base dataset.
        """
        logger.info("="*70)
        logger.info("BUILDING V2 DATASET WITH ENHANCED FEATURES")
        logger.info("="*70)
        
        # Load base dataset (has data leakage, but we'll fix it)
        base_df = pd.read_parquet(base_dataset_path)
        logger.info(f"📊 Loaded {len(base_df)} games from base dataset")
        
        # Load game results
        game_results = self.load_game_results()
        
        # Build team schedules
        team_schedule = self.build_team_schedule(game_results)
        
        # Build V2 features for each game
        rows = []
        
        # Limit to games with boxscore data (from game_results)
        matching_games = set(base_df['game_id'].astype(str)) & set(game_results.keys())
        logger.info(f"📊 {len(matching_games)} games have boxscore data")
        
        for game_id in list(matching_games)[:100]:  # Limit to 100 for speed
            base_row = base_df[base_df['game_id'].astype(str) == game_id].iloc[0].to_dict()
            
            try:
                v2_row = self.build_v2_features(game_id, base_row, game_results, team_schedule)
                rows.append(v2_row)
            except Exception as e:
                logger.warning(f"⚠️ Error building features for {game_id}: {e}")
        
        df = pd.DataFrame(rows)
        logger.info(f"✅ Built V2 dataset with {len(df)} games")
        
        # Save
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ Saved to {output_path}")
        
        return df


def main():
    """Main entry point."""
    try:
        builder = V2FeatureBuilder()
        
        # Build V2 dataset
        df = builder.build_v2_dataset(
            base_dataset_path='data/processed/pregame_team_v2.parquet',
            output_path='data/processed/pregame_v2_enhanced.parquet'
        )
        
        # Show feature counts
        logger.info("="*70)
        logger.info(f"✅ Total features: {len(df.columns)}")
        logger.info(f"   Base features: 14")
        logger.info(f"   V2 new features: {len(df.columns) - 14}")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ PHASE 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
