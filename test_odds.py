from src.predict_api import predict_game
from src.automation.post_generator_helpers import _generate_best_bets

pred = predict_game('0022500771', mode='q3', fetch_odds=True)
print('Status:', pred.get('status'))
print('Has odds?', 'odds_total_line' in pred)
print()

if 'odds_total_line' in pred:
    print('Total line:', pred.get('odds_total_line'))
    print('Total over:', pred.get('odds_total_over'))
    print('Spread line:', pred.get('odds_spread_home_line'))
    print()
    
    bets = _generate_best_bets(pred, 'q3', max_bets=3, min_edge=0.06)
    print(f'Bets (6%): {len(bets)}')
    if not bets:
        print('No bets at 6% threshold, trying 2%...')
        bets_low = _generate_best_bets(pred, 'q3', max_bets=3, min_edge=0.02)
        print(f'Bets (2%): {len(bets_low)}')
        for bet in bets_low:
            print(f"  {bet.get('type')} {bet.get('side')} @ {bet.get('odds')} (edge {bet.get('edge')*100:.1f}%)")
    else:
        for bet in bets:
            print(f"  {bet.get('type')} {bet.get('side')} @ {bet.get('odds')} (edge {bet.get('edge')*100:.1f}%)")
else:
    print('No odds!')
