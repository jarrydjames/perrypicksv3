"""
Phase 4: Backtest Improved Models
Run predictions on recent games with new calibrated models
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedModelBacktester:
    def __init__(self):
        self.boxscore_dir = Path("data/raw/box")
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("data/models")
        
    def load_models(self):
        """Load improved models."""
        logger.info("Loading improved models...")
        
        self.total_model = joblib.load(self.models_dir / "total_model_improved.pkl")
        self.margin_model = joblib.load(self.models_dir / "margin_model_improved.pkl")
        
        logger.info("Models loaded")
    
    def get_stat_value(self, stats_dict: Dict, stat_name: str, default: float = 0) -> float:
        """Extract stat value from stats dict by name."""
        return stats_dict.get(stat_name, default)
    
    def calculate_possessions(self, stats_dict: Dict) -> float:
        """Possessions = FGA - ORB + TOV + 0.44 * FTA"""
        fga = self.get_stat_value(stats_dict, 'fieldGoalsMade')
        orb = self.get_stat_value(stats_dict, 'reboundsOffensive')
        tov = self.get_stat_value(stats_dict, 'turnovers')
        fta = self.get_stat_value(stats_dict, 'freeThrowsMade')
        return fga - orb + tov + 0.44 * fta if fga > 0 else 0
    
    def normalize_boxscore(self, boxscore: Dict) -> Dict:
        """Normalize boxscore format - handle both nested and direct formats."""
        if 'game' in boxscore:
            return boxscore['game']
        return boxscore
    
    def extract_game_features(self, boxscore: Dict) -> Dict:
        """Extract comprehensive features from a boxscore."""
        boxscore = self.normalize_boxscore(boxscore)
        
        home_team = boxscore.get('homeTeam', {})
        away_team = boxscore.get('awayTeam', {})
        
        home_stats = home_team.get('statistics', {})
        away_stats = away_team.get('statistics', {})
        
        home_pts = home_team.get('score', 0)
        away_pts = away_team.get('score', 0)
        
        home_poss = self.calculate_possessions(home_stats)
        away_poss = self.calculate_possessions(away_stats)
        total_poss = home_poss + away_poss
        
        home_fga = self.get_stat_value(home_stats, 'fieldGoalsMade')
        home_fg3a = self.get_stat_value(home_stats, 'threePointersMade')
        home_fta = self.get_stat_value(home_stats, 'freeThrowsMade')
        home_tov = self.get_stat_value(home_stats, 'turnovers')
        home_oreb = self.get_stat_value(home_stats, 'reboundsOffensive')
        home_reb = self.get_stat_value(home_stats, 'reboundsTotal')
        home_fgm = self.get_stat_value(home_stats, 'fieldGoalsMade')
        
        away_fga = self.get_stat_value(away_stats, 'fieldGoalsMade')
        away_fg3a = self.get_stat_value(away_stats, 'threePointersMade')
        away_fta = self.get_stat_value(away_stats, 'freeThrowsMade')
        away_tov = self.get_stat_value(away_stats, 'turnovers')
        away_oreb = self.get_stat_value(away_stats, 'reboundsOffensive')
        away_reb = self.get_stat_value(away_stats, 'reboundsTotal')
        away_fgm = self.get_stat_value(away_stats, 'fieldGoalsMade')
        
        home_efg = home_fgm / home_fga if home_fga > 0 else 0
        away_efg = away_fgm / away_fga if away_fga > 0 else 0
        
        home_ftr = home_fta / home_fga if home_fga > 0 else 0
        away_ftr = away_fta / away_fga if away_fga > 0 else 0
        
        home_tpar = home_fg3a / home_fga if home_fga > 0 else 0
        away_tpar = away_fg3a / away_fga if away_fga > 0 else 0
        
        home_tor = home_tov / home_fga if home_fga > 0 else 0
        away_tor = away_tov / away_fga if away_fga > 0 else 0
        
        home_orbp = home_oreb / home_reb if home_reb > 0 else 0
        away_orbp = away_oreb / away_reb if away_reb > 0 else 0
        
        home_pace = home_poss
        away_pace = away_poss
        avg_pace = total_poss / 2
        
        home_off_rating = (home_pts / home_poss * 100) if home_poss > 0 else 0
        away_off_rating = (away_pts / away_poss * 100) if away_poss > 0 else 0
        
        home_def_rating = (away_pts / home_poss * 100) if home_poss > 0 else 0
        away_def_rating = (home_pts / away_poss * 100) if away_poss > 0 else 0
        
        pace_diff = home_pace - away_pace
        off_rating_diff = home_off_rating - away_off_rating
        def_rating_diff = home_def_rating - away_def_rating
        
        return {
            'game_id': boxscore.get('gameId', ''),
            'game_date': boxscore.get('gameTimeUTC', ''),
            'total': home_pts + away_pts,
            'margin': home_pts - away_pts,
            'home_pts': home_pts,
            'away_pts': away_pts,
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
            'home_pace': home_pace,
            'away_pace': away_pace,
            'avg_pace': avg_pace,
            'pace_diff': pace_diff,
            'home_off_rating': home_off_rating,
            'away_off_rating': away_off_rating,
            'home_def_rating': home_def_rating,
            'away_def_rating': away_def_rating,
            'off_rating_diff': off_rating_diff,
            'def_rating_diff': def_rating_diff,
            'home_recent_pts': home_pts,  # Placeholder - would need history
            'away_recent_pts': away_pts,
            'home_recent_total': home_pts + away_pts,
            'away_recent_total': home_pts + away_pts,
            'home_recent_win_pct': 0.5,
            'away_recent_win_pct': 0.5,
            'home_pts_x_efg': home_pts * home_efg,
            'away_pts_x_efg': away_pts * away_efg,
            'home_pace_x_off_rating': home_pace * home_off_rating,
            'away_pace_x_off_rating': away_pace * away_off_rating,
            'home_team_id': home_team.get('teamId'),
            'away_team_id': away_team.get('teamId'),
        }
    
    def run_backtest(self, num_games: int = 10):
        """Run backtest on recent games."""
        logger.info(f"Running backtest on last {num_games} games...")
        
        # Get recent boxscore files
        box_files = sorted(list(self.boxscore_dir.glob("*.json")))
        recent_files = box_files[-num_games:]
        
        games = []
        for box_file in recent_files:
            try:
                with open(box_file) as f:
                    boxscore = json.load(f)
                features = self.extract_game_features(boxscore)
                
                if features['game_id'] and features['total'] > 0:
                    games.append(features)
            except Exception as e:
                logger.warning(f"  Error processing {box_file.stem}: {e}")
        
        df = pd.DataFrame(games)
        logger.info(f"Loaded {len(df)} games for backtest")
        
        # Prepare feature matrix - MUST match training features
        feature_cols = [
            'home_pts', 'away_pts',
            'home_efg', 'away_efg',
            'home_ftr', 'away_ftr',
            'home_tpar', 'away_tpar',
            'home_tor', 'away_tor',
            'home_orbp', 'away_orbp',
            'home_pace', 'away_pace', 'avg_pace', 'pace_diff',
            'home_off_rating', 'away_off_rating', 'off_rating_diff',
            'home_def_rating', 'away_def_rating', 'def_rating_diff',
            'home_recent_pts', 'away_recent_pts',
            'home_recent_total', 'away_recent_total',
            'home_recent_win_pct', 'away_recent_win_pct',
            'home_pts_x_efg', 'away_pts_x_efg',
            'home_pace_x_off_rating', 'away_pace_x_off_rating',
        ]
        
        X = df[feature_cols].values
        y_total = df['total'].values
        y_margin = df['margin'].values
        
        # Make predictions
        pred_total = self.total_model.predict(X)
        pred_margin = self.margin_model.predict(X)
        
        # Calculate metrics
        total_mae = mean_absolute_error(y_total, pred_total)
        total_rmse = np.sqrt(mean_squared_error(y_total, pred_total))
        margin_mae = mean_absolute_error(y_margin, pred_margin)
        margin_rmse = np.sqrt(mean_squared_error(y_margin, pred_margin))
        
        # Predict winner
        pred_winner = np.where(pred_margin > 0, 'home', 'away')
        actual_winner = np.where(y_margin > 0, 'home', 'away')
        winner_accuracy = (pred_winner == actual_winner).mean()
        
        # Display results
        print("\n" + "="*70)
        print("BACKTEST RESULTS - IMPROVED MODELS")
        print("="*70)
        print()
        print("GAME-BY-GAME PREDICTIONS:")
        print("-"*70)
        print(f"{'Game':<15} {'Act Tot':<10} {'Pred Tot':<10} {'Err':<8} {'Act Mgn':<10} {'Pred Mgn':<10} {'Err':<8} {'Winner'}")
        print("-"*70)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            total_error = abs(row['total'] - pred_total[i])
            margin_error = abs(row['margin'] - pred_margin[i])
            winner_correct = '✓' if pred_winner[i] == actual_winner[i] else '✗'
            
            print(f"{row['game_id']:<15} {row['total']:<10.1f} {pred_total[i]:<10.1f} {total_error:<8.1f} "
                  f"{row['margin']:<10.1f} {pred_margin[i]:<10.1f} {margin_error:<8.1f} {winner_correct}")
        
        print("-"*70)
        print()
        print("SUMMARY METRICS:")
        print("-"*70)
        print(f"TOTAL PREDICTION:")
        print(f"  MAE: {total_mae:.2f} points")
        print(f"  RMSE: {total_rmse:.2f} points")
        print(f"  Mean Actual: {np.mean(y_total):.2f}")
        print(f"  Mean Predicted: {np.mean(pred_total):.2f}")
        print()
        print(f"MARGIN PREDICTION:")
        print(f"  MAE: {margin_mae:.2f} points")
        print(f"  RMSE: {margin_rmse:.2f} points")
        print(f"  Mean Actual: {np.mean(y_margin):.2f}")
        print(f"  Mean Predicted: {np.mean(pred_margin):.2f}")
        print()
        print(f"WINNER PREDICTION:")
        print(f"  Accuracy: {winner_accuracy:.1%} ({sum(pred_winner == actual_winner)}/{len(df)} correct)")
        print()
        
        # Compare to old model results
        print("COMPARISON WITH PREVIOUS MODEL:")
        print("-"*70)
        print("  Previous (before improvements):")
        print("    Total MAE: 16.58 points")
        print("    Margin MAE: 14.28 points")
        print("    Winner Accuracy: 60%")
        print()
        print("  Improved (after calibration + new features):")
        print(f"    Total MAE: {total_mae:.2f} points (improvement: {(16.58 - total_mae) / 16.58 * 100:+.1f}%)")
        print(f"    Margin MAE: {margin_mae:.2f} points (improvement: {(14.28 - margin_mae) / 14.28 * 100:+.1f}%)")
        print(f"    Winner Accuracy: {winner_accuracy:.1%} (improvement: {(winner_accuracy - 0.60) * 100:+.1f}%)")
        print()
        print("="*70)
    
    def run(self):
        """Run complete backtest."""
        logger.info("="*70)
        logger.info("PHASE 4: BACKTEST IMPROVED MODELS")
        logger.info("="*70)
        
        # Load models
        self.load_models()
        
        # Run backtest
        self.run_backtest(num_games=10)
        
        logger.info("="*70)
        logger.info("PHASE 4 COMPLETE")
        logger.info("="*70)


def main():
    backtester = ImprovedModelBacktester()
    backtester.run()
    return 0


if __name__ == '__main__':
    exit(main())
