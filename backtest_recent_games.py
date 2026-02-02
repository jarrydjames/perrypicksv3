"""
Backtest on the most recent games in our dataset.
Using latest available data and running predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecentGamesBacktester:
    def __init__(self):
        self.boxscore_dir = Path("data/raw/box")
        self.season_avgs_dir = Path("data/season_averages")
        self.games = []
        
    def get_season_from_date(self, date_str: str) -> str:
        """Determine season from game date."""
        # Format: 2025-04-17T02:00:00Z
        year = int(date_str[:4])
        month = int(date_str[5:7])
        
        # NBA season: Oct-May, season labeled by starting year
        # If month >= 10, season is year-(year+1)
        # If month <= 5, season is (year-1)-year
        if month >= 10:
            season = f"{year}-{str(year+1)[2:]}"
        else:
            season = f"{year-1}-{str(year)[2:]}"
        
        return season
    
    def get_recent_games(self, num_games: int = 10):
        """Get the most recent games from boxscores."""
        logger.info(f"Finding {num_games} most recent games...")
        
        box_files = sorted(list(self.boxscore_dir.glob("*.json")))
        recent_files = box_files[-num_games:]
        
        for f in recent_files:
            game_id = f.stem
            with open(f) as f:
                data = json.load(f)
            
            game_date = data.get('gameTimeUTC', '')
            home_team_id = data.get('homeTeam', {}).get('teamId')
            away_team_id = data.get('awayTeam', {}).get('teamId')
            home_team_name = data.get('homeTeam', {}).get('teamName')
            away_team_name = data.get('awayTeam', {}).get('teamName')
            home_score = data.get('homeTeam', {}).get('score')
            away_score = data.get('awayTeam', {}).get('score')
            
            self.games.append({
                'game_id': game_id,
                'game_date': game_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_name': home_team_name,
                'away_team_name': away_team_name,
                'total': (home_score + away_score) if (home_score and away_score) else None,
                'margin': (home_score - away_score) if (home_score and away_score) else None,
            })
        
        logger.info(f"Found {len(self.games)} games")
        return self.games
    
    def get_season_averages(self, season: str) -> Dict[int, Dict]:
        """Load season averages from cache."""
        avgs_file = self.season_avgs_dir / f"season_avgs_{season}.parquet"
        
        if not avgs_file.exists():
            logger.warning(f"Season averages not found for {season}")
            return {}
        
        df = pd.read_parquet(avgs_file)
        team_lookup = {}
        for _, row in df.iterrows():
            team_lookup[int(row['TEAM_ID'])] = row.to_dict()
        
        logger.info(f"Loaded {len(team_lookup)} teams for {season}")
        return team_lookup
    
    def build_features(self):
        """Build feature set for all games."""
        logger.info("Building features...")
        
        features = []
        for game in self.games:
            game_id = game['game_id']
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            game_date = game['game_date']
            
            if not game_date:
                logger.warning(f"  No date for {game_id}")
                continue
            
            # Get season from date
            season = self.get_season_from_date(game_date)
            
            # Get season averages
            season_avgs = self.get_season_averages(season)
            
            home_avgs = season_avgs.get(home_team_id)
            away_avgs = season_avgs.get(away_team_id)
            
            if not home_avgs or not away_avgs:
                logger.warning(f"  Missing averages for {game_id} (season: {season})")
                continue
            
            # Build features
            feat = {
                'game_id': game_id,
                'game_date': game_date,
                'season': season,
                'home_team_name': game['home_team_name'],
                'away_team_name': game['away_team_name'],
                
                # Targets
                'total': game['total'],
                'margin': game['margin'],
                
                # Season averages (pregame features)
                'home_pts': home_avgs.get('PTS'),
                'away_pts': away_avgs.get('PTS'),
                'home_efg': home_avgs.get('FG_PCT'),
                'home_ftr': home_avgs.get('FTA') / home_avgs.get('FGA') if home_avgs.get('FGA', 0) > 0 else None,
                'home_tpar': home_avgs.get('FG3A') / home_avgs.get('FGA') if home_avgs.get('FGA', 0) > 0 else None,
                'home_tor': home_avgs.get('TOV') / home_avgs.get('FGA') if home_avgs.get('FGA', 0) > 0 else None,
                'home_orbp': home_avgs.get('OREB') / home_avgs.get('REB') if home_avgs.get('REB', 0) > 0 else None,
                'away_efg': away_avgs.get('FG_PCT'),
                'away_ftr': away_avgs.get('FTA') / away_avgs.get('FGA') if away_avgs.get('FGA', 0) > 0 else None,
                'away_tpar': away_avgs.get('FG3A') / away_avgs.get('FGA') if away_avgs.get('FGA', 0) > 0 else None,
                'away_tor': away_avgs.get('TOV') / away_avgs.get('FGA') if away_avgs.get('FGA', 0) > 0 else None,
                'away_orbp': away_avgs.get('OREB') / away_avgs.get('REB') if away_avgs.get('REB', 0) > 0 else None,
            }
            
            features.append(feat)
        
        self.features_df = pd.DataFrame(features)
        logger.info(f"Built features for {len(features)} games")
        return self.features_df
    
    def load_models(self):
        """Load trained models."""
        logger.info("Loading models...")
        
        self.total_model = joblib.load('data/models/total_model.pkl')
        self.margin_model = joblib.load('data/models/margin_model.pkl')
        
        logger.info("Models loaded")
    
    def make_predictions(self):
        """Make predictions for all games."""
        logger.info("Making predictions...")
        
        feature_cols = [
            'home_pts', 'away_pts',
            'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
            'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp'
        ]
        
        X = self.features_df[feature_cols].values
        
        self.features_df['pred_total'] = self.total_model.predict(X)
        self.features_df['pred_margin'] = self.margin_model.predict(X)
        
        # Predict winner
        self.features_df['pred_winner'] = np.where(
            self.features_df['pred_margin'] > 0, 'home', 'away'
        )
        self.features_df['actual_winner'] = np.where(
            self.features_df['margin'] > 0, 'home', 'away'
        )
        
        logger.info("Predictions made")
    
    def evaluate(self):
        """Evaluate predictions vs actuals."""
        logger.info("="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        
        # Total prediction metrics
        total_mae = mean_absolute_error(
            self.features_df['total'], 
            self.features_df['pred_total']
        )
        total_rmse = np.sqrt(mean_squared_error(
            self.features_df['total'], 
            self.features_df['pred_total']
        ))
        
        # Margin prediction metrics
        margin_mae = mean_absolute_error(
            self.features_df['margin'], 
            self.features_df['pred_margin']
        )
        margin_rmse = np.sqrt(mean_squared_error(
            self.features_df['margin'], 
            self.features_df['pred_margin']
        ))
        
        # Winner accuracy
        winner_acc = (
            self.features_df['pred_winner'] == self.features_df['actual_winner']
        ).mean()
        
        # Display
        print()
        print("GAME-BY-GAME RESULTS:")
        print("-"*70)
        print(f"{'Game':<15} {'Date':<20} {'Matchup':<35} {'Act Tot':<10} {'Pred Tot':<10} {'Err':<8} {'Act Mgn':<10} {'Pred Mgn':<10} {'Err':<8} {'Win'}")
        print("-"*70)
        
        for _, row in self.features_df.iterrows():
            matchup = f"{row['home_team_name']} vs {row['away_team_name']}"
            total_error = abs(row['total'] - row['pred_total'])
            margin_error = abs(row['margin'] - row['pred_margin'])
            winner_correct = '✓' if row['pred_winner'] == row['actual_winner'] else '✗'
            
            print(f"{row['game_id']:<15} {row['game_date'][:20]:<20} {matchup:<35} "
                  f"{row['total']:<10.1f} {row['pred_total']:<10.1f} {total_error:<8.1f} "
                  f"{row['margin']:<10.1f} {row['pred_margin']:<10.1f} {margin_error:<8.1f} {winner_correct}")
        
        print("-"*70)
        print()
        print("SUMMARY METRICS:")
        print("-"*70)
        print(f"TOTAL PREDICTION:")
        print(f"  MAE: {total_mae:.2f} points")
        print(f"  RMSE: {total_rmse:.2f} points")
        print(f"  Mean Actual: {self.features_df['total'].mean():.2f}")
        print(f"  Mean Predicted: {self.features_df['pred_total'].mean():.2f}")
        print()
        print(f"MARGIN PREDICTION:")
        print(f"  MAE: {margin_mae:.2f} points")
        print(f"  RMSE: {margin_rmse:.2f} points")
        print(f"  Mean Actual: {self.features_df['margin'].mean():.2f}")
        print(f"  Mean Predicted: {self.features_df['pred_margin'].mean():.2f}")
        print()
        print(f"WINNER PREDICTION:")
        print(f"  Accuracy: {winner_acc:.1%} ({self.features_df['pred_winner'].eq(self.features_df['actual_winner']).sum()}/{len(self.features_df)} correct)")
        print()
        
        # Diagnosis
        print("="*70)
        print("DIAGNOSIS & RECOMMENDATIONS")
        print("="*70)
        
        # Check if we're systematically over/under predicting
        total_bias = (self.features_df['pred_total'] - self.features_df['total']).mean()
        margin_bias = (self.features_df['pred_margin'] - self.features_df['margin']).mean()
        
        print(f"BIAS ANALYSIS:")
        print(f"  Total bias (pred - actual): {total_bias:+.2f} points")
        if abs(total_bias) > 10:
            print(f"  ⚠️  Systematic over/under prediction detected!")
            print(f"  → RECOMMENDATION: Calibrate model intercept by {-total_bias:.1f} points")
        print(f"  Margin bias (pred - actual): {margin_bias:+.2f} points")
        if abs(margin_bias) > 5:
            print(f"  ⚠️  Margin prediction biased!")
            print(f"  → RECOMMENDATION: Calibrate margin model by {-margin_bias:.1f} points")
        else:
            print(f"  ✓ Margin predictions well-calibrated")
        print()
        
        # Check prediction variance
        total_pred_std = self.features_df['pred_total'].std()
        total_actual_std = self.features_df['total'].std()
        
        print(f"VARIANCE ANALYSIS:")
        print(f"  Predicted total std: {total_pred_std:.2f}")
        print(f"  Actual total std: {total_actual_std:.2f}")
        if total_pred_std < total_actual_std * 0.5:
            print(f"  ⚠️  Predictions lack variance - too conservative!")
            print(f"  → RECOMMENDATION: Add more variance to predictions (use smaller regularization)")
        print()
        
        # Check for patterns
        print("PATTERN ANALYSIS:")
        
        # Are we overpredicting on high-scoring games?
        if len(self.features_df) >= 4:
            high_scoring = self.features_df[self.features_df['total'] > self.features_df['total'].quantile(0.75)]
            low_scoring = self.features_df[self.features_df['total'] <= self.features_df['total'].quantile(0.25)]
            
            if len(high_scoring) > 0 and len(low_scoring) > 0:
                high_bias = (high_scoring['pred_total'] - high_scoring['total']).mean()
                low_bias = (low_scoring['pred_total'] - low_scoring['total']).mean()
                
                print(f"  High-scoring games bias: {high_bias:+.2f} points")
                print(f"  Low-scoring games bias: {low_bias:+.2f} points")
                
                if abs(high_bias) > abs(low_bias) * 1.5:
                    print(f"  ⚠️  Worse on high-scoring games - model may be too linear")
                    print(f"  → RECOMMENDATION: Consider interaction features or nonlinear models")
        
        print()
        
        # Overall recommendations
        print("IMPROVEMENT RECOMMENDATIONS:")
        if winner_acc < 0.6:
            print("  1. Winner prediction below 60% - need better discriminative features")
            print("     → Add: rest days, recent form (last 5-10 games), injuries")
            print("     → Try: probabilistic models with confidence intervals")
        else:
            print("  1. ✓ Winner predictions reasonable")
        
        if abs(total_bias) < 5 and abs(margin_bias) < 3:
            print("  2. ✓ Models well-calibrated")
        else:
            print("  2. Apply intercept corrections to models")
            if abs(total_bias) > 5:
                print(f"     → Total: Adjust intercept by {-total_bias:.1f} points")
            if abs(margin_bias) > 3:
                print(f"     → Margin: Adjust intercept by {-margin_bias:.1f} points")
        
        # Additional suggestions based on MAE
        if total_mae > 20:
            print("  3. Total MAE > 20 - Consider:")
            print("     → Adding more recent form features (last 5-10 games)")
            print("     → Adding matchup-specific features (team vs team history)")
            print("     → Feature engineering: pace, possession-based metrics")
        
        print()
    
    def run(self, num_games: int = 10):
        """Run complete backtest."""
        logger.info("="*70)
        logger.info(f"RECENT GAMES BACKTEST (last {num_games} games)")
        logger.info("="*70)
        
        # Step 1: Get recent games
        self.get_recent_games(num_games)
        
        if not self.games:
            logger.error("No games found - cannot continue")
            return
        
        # Step 2: Build features
        self.build_features()
        
        if len(self.features_df) == 0:
            logger.error("No features built - cannot continue")
            return
        
        # Step 3: Load models
        self.load_models()
        
        # Step 4: Make predictions
        self.make_predictions()
        
        # Step 5: Evaluate
        self.evaluate()
        
        logger.info("="*70)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*70)


def main():
    backtester = RecentGamesBacktester()
    backtester.run(num_games=10)
    return 0


if __name__ == '__main__':
    exit(main())
