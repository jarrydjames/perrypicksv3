#!/usr/bin/env python3
"""Fetch NBA Game Schedule

Fetches game schedule for a given date from ESPN API (fallback
when NBA.com API is rate-limited).

Usage:
    python fetch_game_schedule.py
    python fetch_game_schedule.py --date 2026-02-07
    python fetch_game_schedule.py --json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import requests
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def fetch_schedule(date_str: str) -> dict:
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


def print_schedule(data: dict, date_str: str, json_output: bool = False):
    """Print game schedule in formatted table."""
    print("=" * 100)
    print(f"NBA GAME SCHEDULE FOR {date_str}")
    print("=" * 100)
    print()
    
    if 'events' not in data:
        print("No games found for this date.")
        print()
        print("=" * 100)
        return
    
    games = data['events']
    
    if not games:
        print("No games found for this date.")
        print()
        print("=" * 100)
        return
    
    print(f"Found {len(games)} games")
    print(f"Source: ESPN API")
    print()
    
    if json_output:
        # Print JSON format
        output_games = []
        for game in games:
            game_id = game.get('id')
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            
            if len(competitors) >= 2:
                # Determine home/away
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
                'game_id': game_id,
                'away_team': away_team,
                'home_team': home_team,
                'status': status,
                'date_time': date_time
            })
        
        print(json.dumps(output_games, indent=2))
    else:
        # Print table format
        print(f"{'Game ID':<12} | {'Away':<6} @ {'Home':<6} | {'Status':<20} | {'Time (UTC)'}")
        print("-" * 100)
        
        for game in games:
            game_id = game.get('id')
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            
            # Determine home/away
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
            
            # Extract just the time portion
            if 'T' in str(date_time):
                time_str = date_time.split('T')[1][:5]  # Get HH:MM
            else:
                time_str = str(date_time)
            
            print(f"{game_id:<12} | {away_team:<6} @ {home_team:<6} | {status:<20} | {time_str}")
    
    print()
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA game schedule for a given date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Today's games
  python fetch_game_schedule.py
  
  # Specific date
  python fetch_game_schedule.py --date 2026-02-07
  
  # JSON output
  python fetch_game_schedule.py --json
  
  # Save to file
  python fetch_game_schedule.py --date 2026-02-07 --output schedule.json
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
    
    # Fetch schedule
    data = fetch_schedule(date_str)
    
    # Print schedule
    print_schedule(data, date_str, args.json)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Schedule saved to {args.output}")


if __name__ == '__main__':
    main()
