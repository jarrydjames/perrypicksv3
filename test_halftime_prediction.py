#!/usr/bin/env python3
"""Test halftime prediction and see what fields it has"""
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from src.predict_api import predict_game

print("------------------------------------------------------------")
print("Testing halftime prediction for game 0022500751")
print("------------------------------------------------------------")
print()

# Run halftime prediction
result = predict_game('0022500751', mode='halftime', fetch_odds=False)

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
