import logging
logging.basicConfig(level=logging.DEBUG)

from src.predict_api import predict_game

pred = predict_game('0022500771', mode='q3', fetch_odds=True)
print('Status:', pred.get('status'))
print('Error:', pred.get('error'))
print('Odds error:', pred.get('odds_error'))
print()

odds_keys = [k for k in pred.keys() if 'odd' in k.lower()]
print('Odds keys:', odds_keys)
print('Has odds?', len(odds_keys) > 0)
