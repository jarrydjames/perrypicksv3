#!/usr/bin/env python3
"""Test Q3 prediction and see what fields it has"""
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from src.predict_api import predict_game

print("------------------------------------------------------------")
print("Testing Q3 prediction for game 0022500753")
print("------------------------------------------------------------")
print()

# Run Q3 prediction
result = predict_game('0022500753', mode='q3', fetch_odds=False)

print()
print("------------------------------------------------------------")
print("PREDICTION RESULT KEYS:")
print("------------------------------------------------------------")
for key in result.keys():
    value = result[key]
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {str(value)[:100]}" if len(str(value)) > 100 else f"  {key}: {value}")
print()
print("------------------------------------------------------------")