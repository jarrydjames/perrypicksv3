"""Test pregame model predictions vs actual results.

Use the training dataset to simulate predictions on unseen games
and compare to actual outcomes.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load models (use ridge models - they're true pregame models)
print("="*80)
print("LOADING PREGAME MODELS")
print("="*80)
print()

# Load the ridge models (trained without margin/total as features)
total_model = joblib.load('models_v3/pregame/ridge_total.joblib')
margin_model = joblib.load('models_v3/pregame/ridge_margin.joblib')

# Get feature list from model
feature_list = total_model['features']
print(f"✅ Models loaded successfully!")
print(f"   Features expected: {len(feature_list)}")
print(f"   Features: {feature_list}")
print()

# Load training data
print("="*80)
print("LOADING TRAINING DATA")
print("="*80)
print()

df = pd.read_parquet('data/processed/pregame_team_v2.parquet')
print(f"✅ Loaded {len(df)} games from training data")
print()

# Simulate "out-of-sample" testing
# Use the last 15 games as our "recent" test set
# (In production, you'd fetch actual recent games via API)
print("="*80)
print("TESTING ON LAST 15 GAMES (OUT-OF-SAMPLE)")
print("="*80)
print()

# Get the last 15 games as test set
test_games = df.tail(15).copy()

results = []

for i, (_, row) in enumerate(test_games.iterrows(), 1):
    game_id = row['game_id']
    home_tri = row['home_tri']
    away_tri = row['away_tri']
    actual_total = row['total']
    actual_margin = row['margin']
    
    # Determine actual winner
    if actual_margin > 0:
        actual_winner = home_tri
    elif actual_margin < 0:
        actual_winner = away_tri
    else:
        actual_winner = "TIE"
    
    print(f"{i}. Game ID: {game_id}")
    print(f"   {home_tri} vs {away_tri}")
    print(f"   Actual: Total={actual_total}, Margin={actual_margin}, Winner={actual_winner}")
    
    # Extract features (use the feature list from the model)
    try:
        X = np.array([row[feature_list].values])
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        continue
    
    # Make prediction using models directly
    try:
        # Total prediction
        total_model_obj = total_model['model']
        pred_total = total_model_obj.predict(X)[0]
        
        # Margin prediction
        margin_model_obj = margin_model['model']
        pred_margin = margin_model_obj.predict(X)[0]
        
        # Calculate residual sigma for intervals
        total_sigma = total_model.get('residual_sigma', 8.0)
        margin_sigma = margin_model.get('residual_sigma', 5.0)
        
        # Create approximate 80% confidence intervals
        total_q10 = pred_total - 1.28 * total_sigma
        total_q90 = pred_total + 1.28 * total_sigma
        margin_q10 = pred_margin - 1.28 * margin_sigma
        margin_q90 = pred_margin + 1.28 * margin_sigma
            
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # Determine predicted winner
    if pred_margin > 0:
        pred_winner = home_tri
    elif pred_margin < 0:
        pred_winner = away_tri
    else:
        pred_winner = "TIE"
    
    # Calculate errors
    total_error = abs(pred_total - actual_total)
    margin_error = abs(pred_margin - actual_margin)
    total_error_pct = (total_error / actual_total) * 100 if actual_total != 0 else 0
    
    # Check if winner prediction is correct
    winner_correct = (pred_winner == actual_winner)
    
    # Calculate win probability from margin (simple probit-like)
    home_win_prob = 1 / (1 + np.exp(-pred_margin / 4))
    
    print(f"   Prediction: Total={pred_total:.1f}, Margin={pred_margin:.1f}, Winner={pred_winner}")
    print(f"   Error: Total={total_error:.1f} ({total_error_pct:.1f}%), Margin={margin_error:.1f}")
    print(f"   Winner Correct: {winner_correct}")
    print()
    
    results.append({
        'game_id': game_id,
        'home_team': home_tri,
        'away_team': away_tri,
        'pred_total': round(pred_total, 1),
        'pred_margin': round(pred_margin, 1),
        'pred_winner': pred_winner,
        'actual_total': actual_total,
        'actual_margin': actual_margin,
        'actual_winner': actual_winner,
        'total_error': round(total_error, 1),
        'total_error_pct': round(total_error_pct, 1),
        'margin_error': round(margin_error, 1),
        'winner_correct': winner_correct,
        'home_win_prob': round(home_win_prob, 3),
        'total_q10': round(total_q10, 1),
        'total_q90': round(total_q90, 1),
        'margin_q10': round(margin_q10, 1),
        'margin_q90': round(margin_q90, 1),
    })

# Create DataFrame
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("❌ No predictions were generated!")
    sys.exit(1)

print("="*80)
print("PREDICTION SUMMARY")
print("="*80)
print()

# Calculate aggregate metrics
total_mae = results_df['total_error'].mean()
margin_mae = results_df['margin_error'].mean()
winner_acc = results_df['winner_correct'].sum() / len(results_df) * 100

# Calculate accuracy percentages
acc_3 = (results_df['total_error'] <= 3).sum() / len(results_df) * 100
acc_5 = (results_df['total_error'] <= 5).sum() / len(results_df) * 100
acc_10 = (results_df['total_error'] <= 10).sum() / len(results_df) * 100

print(f"Games tested: {len(results_df)}")
print()
print(f"Total MAE: {total_mae:.2f} points")
print(f"Margin MAE: {margin_mae:.2f} points")
print(f"Winner Accuracy: {winner_acc:.1f}%")
print()
print(f"Total Accuracy:")
print(f"  Within 3 pts: {acc_3:.1f}%")
print(f"  Within 5 pts: {acc_5:.1f}%")
print(f"  Within 10 pts: {acc_10:.1f}%")
print()

# Save to CSV
output_file = "pregame_predictions_vs_actual.csv"
results_df.to_csv(output_file, index=False)
print(f"✅ Saved results to: {output_file}")
print()

# Display full CSV
print("="*80)
print("FULL PREDICTIONS TABLE")
print("="*80)
print(results_df.to_string())
