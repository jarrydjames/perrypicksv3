"""Baseline backtest using existing boxscore data.

This script:
1. Loads existing boxscore games from data/raw/box/
2. Calculates pregame features using LeagueDashTeamStats
3. Predicts using existing models in models_v3/pregame/
4. Evaluates performance metrics
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
import json
import glob
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.data.scoreboard import fetch_scoreboard
from src.predict_from_gameid_v2 import fetch_box
from src.features.team_features import TeamFeatures

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


def load_boxscore_games(limit: int = 100) -> list:
    """
    Load games from boxscore JSON files.
    
    Args:
        limit: Maximum number of games to load
        
    Returns:
        List of game dictionaries
    """
    box_dir = Path('data/raw/box')
    games = []
    
    # Get sorted boxscore files
    box_files = sorted(box_dir.glob('*.json'))
    
    print(f"Loading games from {len(box_files)} boxscore files...")
    
    for box_file in box_files[:limit]:
        try:
            with open(box_file, 'r') as f:
                box_data = json.load(f)
            
            # Extract game info
            game_id = box_file.stem
            
            # Get team info from boxscore
            home_team = box_data.get('homeTeam', {}).get('triCode')
            away_team = box_data.get('awayTeam', {}).get('triCode')
            
            # Get scores from periods
            home_periods = box_data.get('homeTeam', {}).get('periods', [])
            away_periods = box_data.get('awayTeam', {}).get('periods', [])
            
            home_pts = sum(int(p.get('score', 0)) for p in home_periods if isinstance(p, dict))
            away_pts = sum(int(p.get('score', 0)) for p in away_periods if isinstance(p, dict))
            
            # Skip if scores are 0 (game not played yet)
            if home_pts == 0 and away_pts == 0:
                continue
            
            game = {
                'game_id': game_id,
                'home': home_team,
                'away': away_team,
                'home_pts': home_pts,
                'away_pts': away_pts,
                'total': home_pts + away_pts,
                'margin': home_pts - away_pts,
                'actual_winner': home_team if home_pts > away_pts else away_team,
            }
            
            games.append(game)
            
        except Exception as e:
            print(f"  Error loading {box_file}: {e}")
            continue
    
    print(f"Loaded {len(games)} games")
    return games


def extract_features(game: dict, team_features: TeamFeatures, season_id: str = '2024-25') -> dict:
    """
    Extract pregame features for a game.
    
    Args:
        game: Game dictionary
        team_features: TeamFeatures instance
        season_id: Season ID
        
    Returns:
        Feature dictionary
    """
    home_team = game['home']
    away_team = game['away']
    
    # Get team stats (pregame season averages)
    home_stats = team_features.get_team_stats(home_team)
    away_stats = team_features.get_team_stats(away_team)
    
    if home_stats is None or away_stats is None:
        return None
    
    # Calculate features
    features = {
        'home_efg': home_stats['efg'],
        'away_efg': away_stats['efg'],
        'home_ftr': home_stats['ftr'],
        'away_ftr': away_stats['ftr'],
        'home_tpar': home_stats['tpar'],
        'away_tpar': away_stats['tpar'],
        'home_tor': home_stats['tor'],
        'away_tor': away_stats['tor'],
        'home_orbp': home_stats['orbp'],
        'away_orbp': away_stats['orbp'],
        'home_fga': home_stats['fga'],
        'away_fga': away_stats['fga'],
        'home_fgm': home_stats['fgm'],
        'away_fgm': away_stats['fgm'],
    }
    
    return features


def predict(features: dict, total_model, margin_model) -> dict:
    """
    Make predictions using models.
    
    Args:
        features: Feature dictionary
        total_model: Total points model
        margin_model: Margin model
        
    Returns:
        Prediction dictionary
    """
    # Build feature array
    feature_order = [
        'home_efg', 'away_efg',
        'home_ftr', 'away_ftr',
        'home_tpar', 'away_tpar',
        'home_tor', 'away_tor',
        'home_orbp', 'away_orbp',
        'home_fga', 'away_fga',
        'home_fgm', 'away_fgm',
    ]
    
    X = np.array([features[f] for f in feature_order]).reshape(1, -1)
    
    # Make predictions
    try:
        pred_total = total_model.predict(X)[0]
        pred_margin = margin_model.predict(X)[0]
        pred_winner = features['home_efg'] > features['away_efg']  # Simple heuristic if margin fails
        pred_winner = 'home' if pred_margin > 0 else 'away'
        
        # Calculate win probability
        home_win_prob = 1 / (1 + np.exp(-pred_margin / 4))
        
        return {
            'pred_total': pred_total,
            'pred_margin': pred_margin,
            'pred_winner': 'home' if pred_margin > 0 else 'away',
            'home_win_prob': home_win_prob,
        }
    except Exception as e:
        print(f"  Prediction error: {e}")
        return None


def evaluate_results(results: list) -> dict:
    """
    Evaluate prediction results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Metrics dictionary
    """
    if not results:
        return {}
    
    # Calculate errors
    total_errors = [r['pred_total'] - r['actual_total'] for r in results]
    margin_errors = [r['pred_margin'] - r['actual_margin'] for r in results]
    winner_correct = [r['pred_winner'] == r['actual_winner'] for r in results]
    
    # Calculate metrics
    metrics = {
        'total_predictions': len(results),
        'winner_accuracy': np.mean(winner_correct),
        'total_mae': np.mean(np.abs(total_errors)),
        'total_rmse': np.sqrt(np.mean(np.array(total_errors) ** 2)),
        'margin_mae': np.mean(np.abs(margin_errors)),
        'margin_rmse': np.sqrt(np.mean(np.array(margin_errors) ** 2)),
        
        # Accuracy distributions
        'total_within_3': np.mean(np.abs(total_errors) <= 3),
        'total_within_5': np.mean(np.abs(total_errors) <= 5),
        'total_within_10': np.mean(np.abs(total_errors) <= 10),
        'margin_within_3': np.mean(np.abs(margin_errors) <= 3),
        'margin_within_5': np.mean(np.abs(margin_errors) <= 5),
        'margin_within_10': np.mean(np.abs(margin_errors) <= 10),
    }
    
    return metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics."""
    print("\n" + "="*80)
    print("BASELINE BACKTEST RESULTS")
    print("="*80)
    
    print("\nOVERALL METRICS:")
    print(f"  Total predictions: {metrics['total_predictions']}")
    print(f"  Winner accuracy: {metrics['winner_accuracy']:.1%}")
    
    print("\nTOTAL PREDICTION:")
    print(f"  MAE: {metrics['total_mae']:.2f} points")
    print(f"  RMSE: {metrics['total_rmse']:.2f} points")
    print(f"  Within 3 pts: {metrics['total_within_3']:.1%}")
    print(f"  Within 5 pts: {metrics['total_within_5']:.1%}")
    print(f"  Within 10 pts: {metrics['total_within_10']:.1%}")
    
    print("\nMARGIN PREDICTION:")
    print(f"  MAE: {metrics['margin_mae']:.2f} points")
    print(f"  RMSE: {metrics['margin_rmse']:.2f} points")
    print(f"  Within 3 pts: {metrics['margin_within_3']:.1%}")
    print(f"  Within 5 pts: {metrics['margin_within_5']:.1%}")
    print(f"  Within 10 pts: {metrics['margin_within_10']:.1%}")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("BASELINE BACKTEST - Using Existing Boxscore Data")
    print("="*80)
    
    # Load games from boxscore files
    games = load_boxscore_games(limit=200)
    
    if len(games) == 0:
        print("No games loaded. Exiting.")
        return
    
    # Initialize team features (pregame data only)
    season_id = '2024-25'
    team_features = TeamFeatures(season_id)
    
    # Load existing models
    try:
        total_model = joblib.load('models_v3/pregame/ridge_total.joblib')
        margin_model = joblib.load('models_v3/pregame/ridge_margin.joblib')
        print(f"\nLoaded models from models_v3/pregame/")
    except Exception as e:
        print(f"\nError loading models: {e}")
        return
    
    # Extract features and make predictions
    results = []
    
    for i, game in enumerate(games, 1):
        print(f"\nProcessing game {i}/{len(games)}: {game['game_id']} - {game['away']} @ {game['home']}")
        
        # Extract features
        features = extract_features(game, team_features, season_id)
        
        if features is None:
            print(f"  Skipping: Missing features")
            continue
        
        # Make predictions
        predictions = predict(features, total_model['model'], margin_model['model'])
        
        if predictions is None:
            print(f"  Skipping: Prediction failed")
            continue
        
        # Calculate errors
        total_error = predictions['pred_total'] - game['total']
        margin_error = predictions['pred_margin'] - game['margin']
        winner_correct = (predictions['pred_winner'] == 'home' and game['margin'] > 0) or \
                       (predictions['pred_winner'] == 'away' and game['margin'] < 0)
        
        # Build result
        result = {
            'game_id': game['game_id'],
            'home': game['home'],
            'away': game['away'],
            'pred_total': predictions['pred_total'],
            'pred_margin': predictions['pred_margin'],
            'pred_winner': predictions['pred_winner'],
            'home_win_prob': predictions['home_win_prob'],
            'actual_total': game['total'],
            'actual_margin': game['margin'],
            'actual_winner': game['actual_winner'],
            'total_error': total_error,
            'margin_error': margin_error,
            'winner_correct': winner_correct,
        }
        
        results.append(result)
        
        # Print quick stats
        print(f"  Pred: Total={predictions['pred_total']:.1f}, Margin={predictions['pred_margin']:.1f}, Winner={predictions['pred_winner']}")
        print(f"  Actual: Total={game['total']}, Margin={game['margin']}, Winner={game['actual_winner']}")
        print(f"  Errors: Total={total_error:.1f}, Margin={margin_error:.1f}, Correct={winner_correct}")
    
    # Evaluate results
    if results:
        metrics = evaluate_results(results)
        print_metrics(metrics)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('baseline_backtest_results.csv', index=False)
        print(f"\nSaved results to: baseline_backtest_results.csv")
    else:
        print("\nNo results to evaluate.")
    
    print("\n" + "="*80)
    print("BASELINE BACKTEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
