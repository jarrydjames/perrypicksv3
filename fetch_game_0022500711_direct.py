"""
Fetch game 0022500711 directly using BoxScoreV3
"""
from nba_api.stats.endpoints.boxscoretraditionalv3 import BoxScoreTraditionalV3

game_id = '0022500711'

print("=" * 70)
print(f"Fetching game {game_id} directly")
print("=" * 70)

try:
    boxscore = BoxScoreTraditionalV3(game_id=game_id)
    data = boxscore.get_dict()
    
    # Check if game exists
    if 'resultSets' not in data:
        print("Game not found or not played yet")
    else:
        # Get team stats
        result_sets = data['resultSets']
        for rs in result_sets:
            if rs['name'] == 'TeamStats':
                print("\nTeam Stats:")
                print("=" * 70)
                for row in rs['rowSet']:
                    team_id = row[1] if len(row) > 1 else None
                    team_abbrev = row[4] if len(row) > 4 else None
                    team_name = row[2] if len(row) > 2 else None
                    pts = row[18] if len(row) > 18 else None
                    print(f"Team: {team_name} ({team_abbrev}) - PTS: {pts}")
                    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
