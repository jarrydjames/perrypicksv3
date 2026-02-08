#!/usr/bin/env python3
"""Test Q3 model to see what it's actually predicting"""
import sys
sys.path.insert(0, '.')

from src.predict_api import predict_game
from src.automation.post_generator import PostGenerator

game_id = '0022500753'

# Q3 model prediction
print('=== Q3 MODEL PREDICTION (raw) ===')
result = predict_game(game_id, mode='q3', fetch_odds=False)

print(f'Game ID: {result.get("game_id")}')
print(f'Home Team: {result.get("home_name")}')
print(f'Away Team: {result.get("away_name")}')
print(f'Period: {result.get("period")}')
print(f'Clock: {result.get("clock")}')
print()
print(f'Q3 Cumulative Scores:')
print(f'  Home: {result.get("home_score")}')
print(f'  Away: {result.get("away_score")}')
print(f'  Total: {result.get("home_score") + result.get("away_score")}')
print()
print(f'Model Predicted FINAL (WRONG - model is broken):')
print(f'  Total: {result.get("total"):.2f}')
print(f'  Margin: {result.get("margin"):.2f}')
print(f'  Home: {(result.get("total") + result.get("margin")) / 2:.2f}')
print(f'  Away: {(result.get("total") - result.get("margin")) / 2:.2f}')
print()

# Calculate what Q4 would need to be if using model predictions
q3_cumulative = result.get('home_score', 0) + result.get('away_score', 0)
pred_final = result.get('total', 0)
needed_q4 = pred_final - q3_cumulative

print(f'Q3 Cumulative: {q3_cumulative}')
print(f'Predicted Final (model): {pred_final:.2f}')
print(f'Q4 needed (model): {needed_q4:.2f} points total ❌ TOO LOW!')
print(f'Q4 per team (model): {needed_q4/2:.2f} points each ❌ IMPOSSIBLE!')
print()

# Test post generator with quarter progression heuristic
print('=== POST GENERATOR WITH QUARTER PROGRESSION HEURISTIC (CORRECT) ===')
gen = PostGenerator()
post = gen.generate_q3_post(result, 'discord')
print(post)
print()

# Show the calculation
print('=== CALCULATION ===')
q4_estimate_total = q3_cumulative * 0.32
q4_home_base = q4_estimate_total / 2
q4_away_base = q4_estimate_total / 2
q3_margin = result.get('home_score', 0) - result.get('away_score', 0)
margin_adjustment = q3_margin * 0.2
q4_home = max(20, min(35, q4_home_base + margin_adjustment))
q4_away = max(20, min(35, q4_away_base - margin_adjustment))

print(f'Q4 estimate (quarter progression): {q4_estimate_total:.2f} total')
print(f'Q4 home adjustment: +{margin_adjustment:.2f} (momentum)')
print(f'Q4 home: {q4_home:.1f}')
print(f'Q4 away: {q4_away:.1f}')
print(f'Projected final: {result.get("away_score") + q4_away:.1f} - {result.get("home_score") + q4_home:.1f}')
