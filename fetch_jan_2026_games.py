"""
Fetch games for Jan 28-31, 2026 and run predictions.

Steps:
1. Fetch schedule for those dates
2. Fetch boxscores (actuals)
3. Fetch season averages (pregame features)
4. Run predictions
5. Compare to actuals
6. Diagnose
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Jan2026Backtester:
    def __init__(self):
        self.boxscore_dir = Path("data/raw/box")
        self.season_avgs_dir = Path("data/season_averages")
        self.games = []
        self.boxscores = {}
        
    def fetch_schedule(self, dates: List[str]):
        """Fetch NBA games for given dates using balldontlie API."""
        logger.info(f"Fetching schedule for dates: {dates}")
        
        base_url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00"
        
        for date in dates:
            # Format: YYYYMMDD
            date_str = date.replace('-', '')
            url = f"{base_url}{date_str}.json"
            
            try:
                logger.info(f"  Fetching {date}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Parse games
                games_data = data.get('scoreboard', {}).get('games', [])
                
                for game in games_data:
                    self.games.append({
                        'game_id': game.get('gameId'),
                        'game_date': date,
                        'game_code': game.get('gameCode'),
                        'home_team_id': game.get('homeTeam', {}).get('teamId'),
                        'away_team_id': game.get('awayTeam', {}).get('teamId'),
                        'home_team_name': game.get('homeTeam', {}).get('teamName'),
                        'away_team_name': game.get('awayTeam', {}).get('teamName'),
                    })
                
                logger.info(f"    Found {len(games_data)} games")
            
            except Exception as e:
                logger.error(f"    Failed to fetch {date}: {e}")
        
        logger.info(f"Total games found: {len(self.games)}")
        return self.games
    
    def fetch_boxscores(self):
        """Fetch boxscores using NBA API."""
        logger.info("Fetching boxscores...")
        
        base_url = "https://cdn.nba.com/static/json/liveData/boxscore/"
        
        for game in self.games:
            game_id = game['game_id']
            url = f"{base_url}{game_id}_full.json"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                boxscore = response.json()
                self.boxscores[game_id] = boxscore
                
                # Save to file
                output_path = self.boxscore_dir / f"{game_id}.json"
                with open(output_path, 'w') as f:
                    json.dump(boxscore, f, indent=2)
                
                logger.info(f"  {game_id}: Saved")
            
            except Exception as e:
                logger.error(f"  {game_id}: Failed - {e}")
        
        logger.info(f"Fetched {len(self.boxscores)} boxscores")
        return self.boxscores
    
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
    
    def extract_features(self, game: Dict) -> Optional[Dict]:
        """Extract pregame features for a game."""
        game_id = game['game_id']
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        
        # Determine season from game_id (format: XXYYMMDDDD)
        season_code = game_id[1:3]
        season = f"20{season_code[:2]}-{season_code[2:]}"
        
        # Get season averages
        season_avgs = self.get_season_averages(season)
        
        home_avgs = season_avgs.get(home_team_id)
        away_avgs = season_avgs.get(away_team_id)
        
        if not home_avgs or not away_avgs:
            logger.warning(f"  Missing averages for {game_id}")
            return None
        
        # Extract boxscore for actuals
        boxscore = self.boxscores.get(game_id, {})
        
        home_score = boxscore.get('homeTeam', {}).get('score')
        away_score = boxscore.get('awayTeam', {}).get('score')
        
        if not home_score or not away_score:
            logger.warning(f"  Missing scores for {game_id}")
            return None
        
        # Build features
        features = {
            'game_id': game_id,
            'game_date': game['game_date'],
            'season': season,
            
            # Targets (actuals)
            'total': home_score + away_score,
            'margin': home_score - away_score,
            
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
            
            # Team info
            'home_team_name': game['home_team_name'],
            'away_team_name': game['away_team_name'],
        }
        
        return features
    
    def build_features(self):
        """Build feature set for all games."""
        logger.info("Building features...")
        
        features = []
        for game in self.games:
            feat = self.extract_features(game)
            if feat:
                features.append(feat)
        
        self.features_df = pd.DataFrame(features)
        logger.info(f"Built features for {len(features)} games")
        return self.features_df
    
    def load_models(self):
        """Load trained models."""
        import joblib
        
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
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
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
        print(f"{'Game':<10} {'Date':<12} {'Matchup':<35} {'Actual':<12} {'Pred':<12} {'Err Tot':<10} {'Err Mgn':<10} {'Winner'}")
        print("-"*70)
        
        for _, row in self.features_df.iterrows():
            matchup = f"{row['home_team_name']} vs {row['away_team_name']}"
            total_error = abs(row['total'] - row['pred_total'])
            margin_error = abs(row['margin'] - row['pred_margin'])
            winner_correct = '✓' if row['pred_winner'] == row['actual_winner'] else '✗'
            
            print(f"{row['game_id']:<10} {row['game_date']:<12} {matchup:<35} "
                  f"{row['total']:<12.1f} {row['pred_total']:<12.1f} {total_error:<10.1f} "
                  f"{margin_error:<10.1f} {winner_correct}")
        
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
        print(f"  Accuracy: {winner_acc:.1%} ({self.features_df['pred_winner'].eq(self.features_df['actual_winner']).sum()}/{len(self.features_df)})")
        print()
        
        # Diagnosis
        print("="*70)
        print("DIAGNOSIS & RECOMMENDATIONS")
        print("="*70)
        
        # Check if we're systematically over/under predicting
        total_bias = (self.features_df['pred_total'] - self.features_df['total']).mean()
        margin_bias = (self.features_df['pred_margin'] - self.features_df['margin']).mean()
        
        print(f"Bias Analysis:")
        print(f"  Total bias (pred - actual): {total_bias:+.2f} points")
        if abs(total_bias) > 10:
            print(f"  ⚠️  Systematic over/under prediction detected!")
        print(f"  Margin bias (pred - actual): {margin_bias:+.2f} points")
        print()
        
        # Check prediction variance
        total_pred_std = self.features_df['pred_total'].std()
        total_actual_std = self.features_df['total'].std()
        
        print(f"Variance Analysis:")
        print(f"  Predicted total std: {total_pred_std:.2f}")
        print(f"  Actual total std: {total_actual_std:.2f}")
        if total_pred_std < total_actual_std * 0.5:
            print(f"  ⚠️  Predictions lack variance - too conservative!")
        print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        if total_bias > 10:
            print("  1. Apply negative intercept correction to total model")
        elif total_bias < -10:
            print("  1. Apply positive intercept correction to total model")
        else:
            print("  1. Total predictions well-calibrated ✓")
        
        if abs(margin_bias) > 5:
            print("  2. Calibrate margin model intercept")
        
        if winner_acc < 0.5:
            print("  3. Winner predictions worse than random - consider:")
            print("     a. More features (rest days, injuries, recent form)")
            print("     b. Different model (probabilistic instead of deterministic)")
        else:
            print("  2. Winner predictions beating random ✓")
        
        print()
    
    def run(self):
        """Run complete backtest."""
        logger.info("="*70)
        logger.info("JAN 2026 BACKTEST")
        logger.info("="*70)
        
        # Dates to backtest
        dates = ['2026-01-28', '2026-01-29', '2026-01-30', '2026-01-31']
        
        # Step 1: Fetch schedule
        self.fetch_schedule(dates)
        
        if not self.games:
            logger.error("No games found - cannot continue")
            return
        
        # Step 2: Fetch boxscores
        self.fetch_boxscores()
        
        # Step 3: Build features
        self.build_features()
        
        # Step 4: Load models
        self.load_models()
        
        # Step 5: Make predictions
        self.make_predictions()
        
        # Step 6: Evaluate
        self.evaluate()
        
        logger.info("="*70)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*70)


def main():
    backtester = Jan2026Backtester()
    backtester.run()
    return 0


if __name__ == '__main__':
    exit(main())
