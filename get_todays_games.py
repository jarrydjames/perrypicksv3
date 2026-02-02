"""
Get all games for today (Feb 1, 2026) using LeagueGameFinder
"""
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

def get_todays_games():
    """Get all games for today."""
    print("Fetching today's games (Season 2025-26)...")
    
    try:
        # Get all games - we'll filter client-side
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00',
            season_nullable='2025-26',
            season_type_nullable='Regular Season'
        )
        df = gamefinder.get_data_frames()[0]
        
        if len(df) == 0:
            print("No games found")
            return []
        
        # Filter for today's games (Feb 1, 2026)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        todays_games = df[df['GAME_DATE'].dt.date == pd.Timestamp('2026-02-01').date()]
        
        if len(todays_games) == 0:
            print("No games found for today (Feb 1, 2026)")
            # Show latest games for debugging
            print(f"\nLatest 5 games:")
            print(df[['GAME_DATE', 'MATCHUP', 'WL']].head(5))
            return []
        
        print(f"Found {len(todays_games)} games for today")
        
        game_list = []
        for _, game in todays_games.iterrows():
            matchup = game.get('MATCHUP', '')
            
            # Parse home/away teams
            if ' vs ' in matchup:
                teams = matchup.split(' vs ')
                home = teams[0].strip()
                away = teams[1].strip()
            else:
                # Try to parse other formats
                parts = matchup.split()
                if len(parts) >= 3:
                    home = parts[0]
                    away = ' '.join(parts[2:])
                else:
                    home = 'Unknown'
                    away = 'Unknown'
            
            # Try to extract scores
            wl = game.get('WL', '')
            # Check if wl is a string with W or L
            if isinstance(wl, str) and ('W' in wl or 'L' in wl):
                # Game completed
                status = 3
            else:
                # Game in progress or upcoming (WL is NaN for these)
                status = 2
            
            game_list.append({
                'game_id': game.get('GAME_ID'),
                'home_team': home,
                'away_team': away,
                'matchup': matchup,
                'status': status
            })
        
        return game_list
        
    except Exception as e:
        print(f"Error fetching games: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    games = get_todays_games()
    
    print("\n" + "=" * 70)
    print("TODAY'S GAMES (Feb 1, 2026)")
    print("=" * 70)
    
    for i, game in enumerate(games, 1):
        status_text = {
            1: "Upcoming",
            2: "In Progress",
            3: "Final"
        }.get(game['status'], "Unknown")
        
        print(f"\n{i}. {game['game_id']}: {game['away_team']} @ {game['home_team']}")
        print(f"   Status: {status_text}")
        print(f"   Matchup: {game['matchup']}")


if __name__ == "__main__":
    main()
