import logging
logging.basicConfig(level=logging.DEBUG)

from src.odds.odds_api import fetch_nba_odds_snapshot, OddsAPIError

try:
    odds = fetch_nba_odds_snapshot(home_name='NYK', away_name='IND')
    print('Odds:', odds)
    print('Has odds?', odds is not None)
except OddsAPIError as e:
    print('OddsAPIError:', e)
    print('Error message:', str(e))
except Exception as e:
    print('Exception:', type(e).__name__)
    print('Error:', e)
