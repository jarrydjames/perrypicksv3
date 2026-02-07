#!/usr/bin/env python3
"""Fetch NBA Game Schedule with ID Mapping

Fetches game schedule from ESPN API and maps ESPN IDs to NBA.com IDs
so predictions can run immediately.

Usage:
    python fetch_game_schedule.py
    python fetch_game_schedule.py --date 2026-02-07
    python fetch_game_schedule.py --json
    python fetch_game_schedule.py --nba-ids
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import requests
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def fetch_espn_schedule(date_str: str) -> dict:
    """Fetch game schedule from ESPN API."""
    date_formatted = date_str.replace('-', '')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_formatted}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    
    return {}


def fetch_nba_schedule(date_str: str) -> dict:
    """Fetch game schedule from NBA.com API."""
    date_formatted = date_str.replace('-', '')
    url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00_{date_formatted}.json"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    
    return {}


def create_espn_to_nba_mapping(espn_data: dict, nba_data: dict) -> dict:
    """
    Map ESPN game IDs to NBA.com game IDs by matching games.
    
    Matches games by:
    1. Away team tricode
    2. Home team tricode
    3. Game time (date/time)
    """
    mapping = {}
    
    if 'events' not in espn_data:
        return mapping
    
    espn_games = espn_data['events']
    nba_games = []
    
    if 'scoreboard' in nba_data and 'games' in nba_data['scoreboard']:
        nba_games = nba_data['scoreboard']['games']
    
    # Create lookup for NBA games
    nba_lookup = {}
    for nba_game in nba_games:
        away_tri = nba_game.get('awayTeam', {}).get('teamTricode', '')
        home_tri = nba_game.get('homeTeam', {}).get('teamTricode', '')
        game_time = nba_game.get('gameTimeUTC', '')
        
        key = (away_tri, home_tri, game_time)
        nba_lookup[key] = nba_game
    
    # Map ESPN games to NBA games
    for espn_game in espn_games:
        espn_id = espn_game.get('id')
        
        # Get teams from ESPN
        competitors = espn_game.get('competitions', [{}])[0].get('competitors', [])
        
        if len(competitors) < 2:
            continue
        
        # Determine home/away
        if competitors[0].get('homeAway') == 'home':
            home_tri = competitors[0].get('team', {}).get('abbreviation', '')
            away_tri = competitors[1].get('team', {}).get('abbreviation', '')
        else:
            home_tri = competitors[1].get('team', {}).get('abbreviation', '')
            away_tri = competitors[0].get('team', {}).get('abbreviation', '')
        
        # Get game time from ESPN
        espn_time = espn_game.get('date', '')
        
        # Try to find matching NBA game
        nba_game = nba_lookup.get((away_tri, home_tri, espn_time))
        
        # If exact time match fails, try fuzzy match by teams only
        if not nba_game:
            for key, nba_g in nba_lookup.items():
                if key[0] == away_tri and key[1] == home_tri:
                    nba_game = nba_g
                    break
        
        if nba_game:
            nba_id = nba_game.get('gameId')
            mapping[espn_id] = nba_id
        else:
            mapping[espn_id] = None
    
    return mapping


def print_schedule(espn_data: dict, nba_data: dict, mapping: dict, date_str: str, json_output: bool = False):
    """Print game schedule with mapped NBA.com IDs."""
    print("=" * 100)
    print(f"NBA GAME SCHEDULE FOR {date_str}")
    print("=" * 100)
    print()
    
    if 'events' not in espn_data:
        print("No games found for this date.")
        print()
        print("=" * 100)
        return
    
    games = espn_data['events']
    
    if not games:
        print("No games found for this date.")
        print()
        print("=" * 100)
        return
    
    # Check if we have NBA mapping
    mapped_count = sum(1 for v in mapping.values() if v is not None)
    unmapped_count = len(games) - mapped_count
    has_mapping = mapped_count > 0
    mapping_source = "Mixed (ESPN + NBA.com mapping)" if has_mapping else "ESPN only"
    
    print(f"Found {len(games)} games")
    print(f"Mapped: {mapped_count}, Unmapped: {unmapped_count}")
    print(f"Source: {mapping_source}")
    print()
    
    if json_output:
        # Print JSON format
        output_games = []
        for game in games:
            espn_id = game.get('id')
            nba_id = mapping.get(espn_id)
            
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            
            if len(competitors) >= 2:
                if competitors[0].get('homeAway') == 'home':
                    home_team = competitors[0].get('team', {}).get('abbreviation', 'HOME')
                    away_team = competitors[1].get('team', {}).get('abbreviation', 'AWAY')
                else:
                    home_team = competitors[1].get('team', {}).get('abbreviation', 'HOME')
                    away_team = competitors[0].get('team', {}).get('abbreviation', 'AWAY')
            else:
                home_team = 'HOME'
                away_team = 'AWAY'
            
            status = game.get('status', {}).get('type', {}).get('name', 'Unknown')
            date_time = game.get('date', 'Unknown')
            
            output_games.append({
                'espn_game_id': espn_id,
                'nba_game_id': nba_id,
                'away_team': away_team,
                'home_team': home_team,
                'status': status,
                'date_time': date_time
            })
        
        print(json.dumps(output_games, indent=2))
    else:
        # Print table format
        print(f"{'ESPN ID':<12} | {'NBA ID':<12} | {'Away':<6} @ {'Home':<6} | {'Status':<20} | {'Time (UTC)'}")
        print("-" * 100)
        
        for game in games:
            espn_id = game.get('id')
            nba_id = mapping.get(espn_id)
            
            # Get teams
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            
            if len(competitors) >= 2:
                if competitors[0].get('homeAway') == 'home':
                    home_team = competitors[0].get('team', {}).get('abbreviation', 'HOME')
                    away_team = competitors[1].get('team', {}).get('abbreviation', 'AWAY')
                else:
                    home_team = competitors[1].get('team', {}).get('abbreviation', 'HOME')
                    away_team = competitors[0].get('team', {}).get('abbreviation', 'AWAY')
            else:
                home_team = 'HOME'
                away_team = 'AWAY'
            
            status = game.get('status', {}).get('type', {}).get('name', 'Unknown')
            date_time = game.get('date', 'Unknown')
            
            # Extract just time portion
            if 'T' in str(date_time):
                time_str = date_time.split('T')[1][:5]  # Get HH:MM
            else:
                time_str = str(date_time)
            
            nba_display = nba_id if nba_id else "NOT MAPPED"
            
            print(f"{espn_id:<12} | {nba_display:<12} | {away_team:<6} @ {home_team:<6} | {status:<20} | {time_str}")
    
    print()
    print("=" * 100)


def get_nba_ids_for_predictions(mapping: dict, espn_data: dict) -> list:
    """Get list of NBA.com game IDs that are ready for predictions."""
    nba_ids = []
    
    for espn_id, nba_id in mapping.items():
        if nba_id is not None:
            nba_ids.append(nba_id)
    
    return nba_ids


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA game schedule with ESPN to NBA.com ID mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Today's games
  python fetch_game_schedule.py
  
  # Specific date
  python fetch_game_schedule.py --date 2026-02-07
  
  # JSON output
  python fetch_game_schedule.py --json
  
  # Get NBA IDs for predictions (space-separated)
  python fetch_game_schedule.py --nba-ids
  
  # Save to file
  python fetch_game_schedule.py --date 2026-02-07 --output schedule.json

Note: ESPN IDs are used for fetching (no rate limiting).
      NBA.com IDs are used for predictions.
"""
    )
    
    parser.add_argument(
        '--date', '-d',
        default=None,
        help='Date in YYYY-MM-DD format (default: today)',
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output in JSON format',
    )
    
    parser.add_argument(
        '--nba-ids',
        action='store_true',
        help='Output only NBA.com game IDs (space-separated)',
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Save output to file',
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch schedules
    espn_data = fetch_espn_schedule(date_str)
    nba_data = fetch_nba_schedule(date_str)
    
    # Create mapping
    mapping = create_espn_to_nba_mapping(espn_data, nba_data)
    
    # Print schedule
    print_schedule(espn_data, nba_data, mapping, date_str, args.json)
    
    # Output NBA IDs if requested
    if args.nba_ids:
        nba_ids = get_nba_ids_for_predictions(mapping, espn_data)
        if nba_ids:
            print(' '.join(nba_ids))
        else:
            print("No NBA game IDs mapped")
    
    # Save to file if requested
    if args.output:
        output_data = {
            'date': date_str,
            'mapping': mapping,
            'games': espn_data.get('events', [])
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Schedule saved to {args.output}")

if __name__ == '__main__':
    main()
