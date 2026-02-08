#!/usr/bin/env python3
"""Test pregame prediction for debugging"""
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from src.predict_api import predict_game

print("------------------------------------------------------------")
print("Testing pregame prediction for game 0022500751")
print("------------------------------------------------------------")
print()

result = predict_game('0022500751', mode='pregame', fetch_odds=False)

print()
print("------------------------------------------------------------")
print("PREDICTION RESULT:")
print("------------------------------------------------------------")
for key, value in result.items():
    print(f"{key}: {value}")
print()
print("------------------------------------------------------------")
