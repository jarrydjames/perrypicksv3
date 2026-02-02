"""
Phase 8: Realistic Backtest
Evaluate pre-game model performance on recent games (no data leakage!).
"""

import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PregameBacktester:
    """Run realistic backtest using pre-game model."""
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("data/models")
    
    def load_data_and_models(self):
        """Load test data and trained models."""
        logger.info("Loading data and models...")
        
        # Load pre-game features
        df = pd.read_parquet(self.processed_dir / "pregame_features.parquet")
        
        # Load feature list
        with open(self.processed_dir / "pregame_feature_list.txt") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        
        # Load models
        total_model = joblib.load(self.models_dir / "total_model_pregame.pkl")
        margin_model = joblib.load(self.models_dir / "margin_model_pregame.pkl")
        
        logger.info(f"  Loaded {len(df)} games")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info("  Models loaded")
        
        return df, feature_cols, total_model, margin_model
    
    def run_backtest(self, num_games: int = 50):
        """Run backtest on recent games."""
        logger.info(f"\nRunning backtest on last {num_games} games...")
        
        # Load data and models
        df, feature_cols, total_model, margin_model = self.load_data_and_models()
        
        # Get recent games (sorted by date)
        df_sorted = df.sort_values('game_date')
        test_df = df_sorted.iloc[-num_games:]
        
        logger.info(f"  Testing on {len(test_df)} games from {test_df['game_date'].min()} to {test_df['game_date'].max()}")
        
        # Prepare features
        X_test = test_df[feature_cols].values
        y_test_total = test_df['total'].values
        y_test_margin = test_df['margin'].values
        y_test_home_score = test_df['home_score'].values
        y_test_away_score = test_df['away_score'].values
        
        # Make predictions
        pred_total = total_model.predict(X_test)
        pred_margin = margin_model.predict(X_test)
        
        # Predict scores
        pred_home_score = (pred_total + pred_margin) / 2
        pred_away_score = (pred_total - pred_margin) / 2
        
        # Calculate metrics
        total_mae = mean_absolute_error(y_test_total, pred_total)
        total_rmse = np.sqrt(mean_squared_error(y_test_total, pred_total))
        total_r2 = r2_score(y_test_total, pred_total)
        
        margin_mae = mean_absolute_error(y_test_margin, pred_margin)
        margin_rmse = np.sqrt(mean_squared_error(y_test_margin, pred_margin))
        margin_r2 = r2_score(y_test_margin, pred_margin)
        
        # Bias
        total_bias = (pred_total - y_test_total).mean()
        margin_bias = (pred_margin - y_test_margin).mean()
        
        # Winner prediction
        pred_winner = np.where(pred_margin > 0, 'home', 'away')
        actual_winner = np.where(y_test_margin > 0, 'home', 'away')
        winner_accuracy = (pred_winner == actual_winner).mean()
        
        # Over/Under prediction (bet line = predicted total)
        pred_line = pred_total
        over_correct = (y_test_total > pred_line).sum()
        under_correct = (y_test_total < pred_line).sum()
        push = (y_test_total == pred_line).sum()
        
        # Display detailed results
        print("\n" + "="*70)
        print("PREGAME MODEL BACKTEST RESULTS")
        print("="*70)
        print("\nGAME-BY-GAME PREDICTIONS (Last 20 of {} games):".format(num_games))
        print("-"*70)
        print(f"{'Date':<12} {'Game':<10} {'Home':<6} {'Away':<6} {'Act Tot':<8} {'Pred Tot':<8} {'Err':<6} {'Act Mgn':<8} {'Pred Mgn':<8} {'Err':<6} {'Win'}")
        print("-"*70)
        
        for i in range(min(20, len(test_df))):
            row = test_df.iloc[i]
            idx = len(test_df) - num_games + i
            
            total_error = abs(y_test_total[i] - pred_total[i])
            margin_error = abs(y_test_margin[i] - pred_margin[i])
            winner_correct = '✓' if pred_winner[i] == actual_winner[i] else '✗'
            
            date_str = str(row['game_date'].date()) if hasattr(row['game_date'], 'date') else row['game_date'][:10]
            
            print(f"{date_str:<12} {row['game_id'][-6:]:<10} {int(row['home_score']):<6} {int(row['away_score']):<6} "
                  f"{y_test_total[i]:<8.1f} {pred_total[i]:<8.1f} {total_error:<6.1f} "
                  f"{y_test_margin[i]:<8.1f} {pred_margin[i]:<8.1f} {margin_error:<6.1f} {winner_correct}")
        
        print("-"*70)
        print("\nOVERALL METRICS:")
        print("-"*70)
        print(f"\nTOTAL POINTS PREDICTION:")
        print(f"  MAE: {total_mae:.2f} points")
        print(f"  RMSE: {total_rmse:.2f} points")
        print(f"  R²: {total_r2:.3f}")
        print(f"  Bias: {total_bias:.2f} points")
        print(f"  Mean Actual: {np.mean(y_test_total):.2f}")
        print(f"  Mean Predicted: {np.mean(pred_total):.2f}")
        
        print(f"\nMARGIN PREDICTION:")
        print(f"  MAE: {margin_mae:.2f} points")
        print(f"  RMSE: {margin_rmse:.2f} points")
        print(f"  R²: {margin_r2:.3f}")
        print(f"  Bias: {margin_bias:.2f} points")
        print(f"  Mean Actual: {np.mean(y_test_margin):.2f}")
        print(f"  Mean Predicted: {np.mean(pred_margin):.2f}")
        
        print(f"\nWINNER PREDICTION:")
        print(f"  Accuracy: {winner_accuracy:.1%} ({sum(pred_winner == actual_winner)}/{len(test_df)} correct)")
        print(f"  Home win rate: {(actual_winner == 'home').mean():.1%}")
        print(f"  Model picks home: {(pred_winner == 'home').mean():.1%}")
        
        print(f"\nOVER/UNDER PREDICTION:")
        print(f"  Over correct: {over_correct}/{len(test_df)} ({over_correct/len(test_df):.1%})")
        print(f"  Under correct: {under_correct}/{len(test_df)} ({under_correct/len(test_df):.1%})")
        print(f"  Push: {push}/{len(test_df)}")
        
        # Compare to betting benchmarks
        print("\n" + "="*70)
        print("COMPARISON TO BETTING BENCHMARKS")
        print("="*70)
        print("\nWhat you'd need to beat (typical):")
        print("  Total prediction: ~11-14 points MAE (professional handicappers)")
        print("  Margin prediction: ~10-12 points MAE")
        print("  Winner accuracy: ~52-54% (break-even vs -110 odds)")
        print(f"  Winner accuracy: ~57-58% (profitable vs -110 odds)")
        print()
        print("Your model:")
        print(f"  Total MAE: {total_mae:.2f} points " + ("✓ Good!" if total_mae < 14 else "≈ Needs improvement"))
        print(f"  Margin MAE: {margin_mae:.2f} points " + ("✓ Good!" if margin_mae < 12 else "≈ Needs improvement"))
        print(f"  Winner accuracy: {winner_accuracy:.1%} " + ("✓ Profitable!" if winner_accuracy > 0.525 else "≈ Needs improvement"))
        print()
        
        # Feature importance (if available)
        if hasattr(total_model, 'coef_'):
            print("="*70)
            print("TOP FEATURES (Total Model Coefficients)")
            print("="*70)
            importance = list(zip(feature_cols, total_model.coef_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            print("\n".join([f"  {feat:30s} {coef: .3f}" for feat, coef in importance[:10]]))
        
        print("="*70)
    
    def run(self):
        """Run complete backtest."""
        logger.info("="*70)
        logger.info("PHASE 8: REALISTIC BACKTEST")
        logger.info("="*70)
        
        self.run_backtest(num_games=100)
        
        logger.info("="*70)
        logger.info("PHASE 8 COMPLETE")
        logger.info("="*70)


def main():
    backtester = PregameBacktester()
    backtester.run()
    return 0


if __name__ == '__main__':
    exit(main())
