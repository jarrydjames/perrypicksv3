"""
Backtest predictions for last 4 days using calibrated models.
Predicts: Total, Margin, Winner (home/away)
"""

import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("="*70)
print("LAST 4 DAYS BACKTEST")
print("="*70)
print()

# Load dataset
df = pd.read_parquet('data/processed/pregame_leakage_free.parquet')

# Get latest date and filter for last 4 days
df['game_date_dt'] = pd.to_datetime(df['game_date'])
latest_date = df['game_date_dt'].max()
four_days_ago = latest_date - timedelta(days=4)
backtest_df = df[df['game_date_dt'] >= four_days_ago].copy()

print(f"Latest date in dataset: {latest_date}")
print(f"Backtest period: {four_days_ago} to {latest_date}")
print(f"Games to predict: {len(backtest_df)}")
print()

# Load models
total_model = joblib.load('data/models/total_model.pkl')
margin_model = joblib.load('data/models/margin_model.pkl')

# Features (same as used in training)
feature_cols = [
    'home_pts', 'away_pts',
    'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
    'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp'
]

# Make predictions
X_backtest = backtest_df[feature_cols].values
backtest_df['pred_total'] = total_model.predict(X_backtest)
backtest_df['pred_margin'] = margin_model.predict(X_backtest)

# Predict winner (home if margin > 0, else away)
backtest_df['pred_winner'] = np.where(backtest_df['pred_margin'] > 0, 'home', 'away')
backtest_df['actual_winner'] = np.where(backtest_df['margin'] > 0, 'home', 'away')

# Calculate accuracy
total_mae = mean_absolute_error(backtest_df['total'], backtest_df['pred_total'])
total_rmse = np.sqrt(mean_squared_error(backtest_df['total'], backtest_df['pred_total']))
margin_mae = mean_absolute_error(backtest_df['margin'], backtest_df['pred_margin'])
margin_rmse = np.sqrt(mean_squared_error(backtest_df['margin'], backtest_df['pred_margin']))
winner_accuracy = (backtest_df['pred_winner'] == backtest_df['actual_winner']).mean()

# Display results
print("="*70)
print("PREDICTION RESULTS")
print("="*70)
print()

# Detailed game-by-game results
print("GAME-BY-GAME PREDICTIONS:")
print("-"*70)
print(f"{'Game ID':<15} {'Date':<20} {'Actual Tot':<12} {'Pred Tot':<12} {'Err':<8} {'Actual Mgn':<12} {'Pred Mgn':<12} {'Err':<8} {'Actual Wn':<12} {'Pred Wn':<12} {'Correct'}")
print("-"*70)

for _, row in backtest_df.iterrows():
    total_error = abs(row['total'] - row['pred_total'])
    margin_error = abs(row['margin'] - row['pred_margin'])
    winner_correct = '✓' if row['pred_winner'] == row['actual_winner'] else '✗'
    
    print(f"{row['game_id']:<15} {row['game_date'][:20]:<20} "
          f"{row['total']:<12.1f} {row['pred_total']:<12.1f} {total_error:<8.1f} "
          f"{row['margin']:<12.1f} {row['pred_margin']:<12.1f} {margin_error:<8.1f} "
          f"{row['actual_winner']:<12} {row['pred_winner']:<12} {winner_correct}")

print("-"*70)
print()

# Summary metrics
print("="*70)
print("SUMMARY METRICS")
print("="*70)
print()

print(f"TOTAL POINTS PREDICTION:")
print(f"  MAE: {total_mae:.2f} points")
print(f"  RMSE: {total_rmse:.2f} points")
print(f"  Mean Actual: {backtest_df['total'].mean():.2f}")
print(f"  Mean Predicted: {backtest_df['pred_total'].mean():.2f}")
print()

print(f"MARGIN PREDICTION:")
print(f"  MAE: {margin_mae:.2f} points")
print(f"  RMSE: {margin_rmse:.2f} points")
print(f"  Mean Actual: {backtest_df['margin'].mean():.2f}")
print(f"  Mean Predicted: {backtest_df['pred_margin'].mean():.2f}")
print()

print(f"WINNER PREDICTION:")
print(f"  Accuracy: {winner_accuracy:.1%} ({backtest_df['pred_winner'].eq(backtest_df['actual_winner']).sum()}/{len(backtest_df)} correct)")
print(f"  Home Wins: {(backtest_df['actual_winner'] == 'home').sum()}/{len(backtest_df)} actual")
print(f"  Predicted Home Wins: {(backtest_df['pred_winner'] == 'home').sum()}/{len(backtest_df)} predicted")
print()

# Directional accuracy (did we correctly predict winner direction?)
print(f"DIRECTIONAL ANALYSIS:")
correct_direction = ((backtest_df['pred_margin'] > 0) == (backtest_df['margin'] > 0)).sum()
directional_accuracy = correct_direction / len(backtest_df)
print(f"  Directional Accuracy: {directional_accuracy:.1%}")
print()

# Betting context (if margin > 0, home covers spread; else away)
print("BETTING CONTEXT:")
home_covered = (backtest_df['margin'] > 0).sum()
away_covered = (backtest_df['margin'] <= 0).sum()
print(f"  Actual Home Cover: {home_covered}/{len(backtest_df)}")
print(f"  Actual Away Cover: {away_covered}/{len(backtest_df)}")

pred_home_cover = (backtest_df['pred_margin'] > 0).sum()
pred_away_cover = (backtest_df['pred_margin'] <= 0).sum()
print(f"  Predicted Home Cover: {pred_home_cover}/{len(backtest_df)}")
print(f"  Predicted Away Cover: {pred_away_cover}/{len(backtest_df)}")
print()

print("="*70)
print("BACKTEST COMPLETE")
print("="*70)
