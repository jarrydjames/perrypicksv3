import logging
logging.basicConfig(level=logging.DEBUG)

from src.odds.persistent_cache import PersistentOddsCache

cache = PersistentOddsCache()
odds = cache.get_or_fetch('NYK', 'IND')
print('Odds snapshot:', odds)
print('Has odds?', odds is not None)
if odds:
    print('Total line:', odds.total_points)
