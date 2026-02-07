#!/usr/bin/env python3
"""Fetch NBA Game Schedule with ID Mapping

Fetches game schedule from ESPN API and maps ESPN IDs to NBA.com IDs
using NBA's public CDN schedule feed (no rate limiting).

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
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# Team abbreviation normalization: ESPN -> NBA (Complete mapping for all 30 teams)
# ESPN sometimes uses different abbreviations than NBA.com
# This mapping normalizes ESPN abbreviations to NBA.com format
TEAM_ABBR_NORMALIZATION = {
    # Atlanta Hawks
    'ATL': 'ATL',
    
    # Boston Celtics
    'BOS': 'BOS',
    'BOSTON': 'BOS',
    
    # Brooklyn Nets
    'BKN': 'BKN',
    'BROOKLYN': 'BKN',
    
    # Charlotte Hornets
    'CHA': 'CHA',
    'CHARLOTTE': 'CHA',
    
    # Chicago Bulls
    'CHI': 'CHI',
    'CHICAGO': 'CHI',
    
    # Cleveland Cavaliers
    'CLE': 'CLE',
    'CLEVELAND': 'CLE',
    'Cavs': 'CLE',
    
    # Dallas Mavericks
    'DAL': 'DAL',
    'DALLAS': 'DAL',
    
    # Denver Nuggets
    'DEN': 'DEN',
    'DENVER': 'DEN',
    
    # Detroit Pistons
    'DET': 'DET',
    'DETROIT': 'DET',
    
    # Golden State Warriors
    'GSW': 'GSW',
    'GS': 'GSW',
    'GOLDEN STATE': 'GSW',
    'WARRIORS': 'GSW',
    
    # Houston Rockets
    'HOU': 'HOU',
    'HOUSTON': 'HOU',
    
    # Indiana Pacers
    'IND': 'IND',
    'INDIANA': 'IND',
    
    # Los Angeles Clippers
    'LAC': 'LAC',
    'LA CLIPPERS': 'LAC',
    'CLIPPERS': 'LAC',
    
    # Los Angeles Lakers
    'LAL': 'LAL',
    'LA LAKERS': 'LAL',
    'LAKERS': 'LAL',
    
    # Memphis Grizzlies
    'MEM': 'MEM',
    'MEMPHIS': 'MEM',
    
    # Miami Heat
    'MIA': 'MIA',
    'MIAMI': 'MIA',
    
    # Milwaukee Bucks
    'MIL': 'MIL',
    'MILWAUKEE': 'MIL',
    
    # Minnesota Timberwolves
    'MIN': 'MIN',
    'MINNESOTA': 'MIN',
    
    # New Orleans Pelicans
    'NOP': 'NOP',
    'NO': 'NOP',
    'NEW ORLEANS': 'NOP',
    'PELICANS': 'NOP',
    
    # New York Knicks
    'NYK': 'NYK',
    'NY': 'NYK',
    'NEW YORK': 'NYK',
    'KNICKS': 'NYK',
    
    # Oklahoma City Thunder
    'OKC': 'OKC',
    'OKLAHOMA CITY': 'OKC',
    'THUNDER': 'OKC',
    
    # Orlando Magic
    'ORL': 'ORL',
    'ORLANDO': 'ORL',
    
    # Philadelphia 76ers
    'PHI': 'PHI',
    'PHILADELPHIA': 'PHI',
    '76ERS': 'PHI',
    
    # Phoenix Suns
    'PHX': 'PHX',
    'PHO': 'PHX',
    'PHOENIX': 'PHX',
    'SUNS': 'PHX',
    
    # Portland Trail Blazers
    'POR': 'POR',
    'PORTLAND': 'POR',
    'TRAIL BLAZERS': 'POR',
    'BLAZERS': 'POR',
    
    # Sacramento Kings
    'SAC': 'SAC',
    'SACRAMENTO': 'SAC',
    'KINGS': 'SAC',
    
    # San Antonio Spurs
    'SAS': 'SAS',
    'SA': 'SAS',
    'SAN ANTONIO': 'SAS',
    'SPURS': 'SAS',
    
    # Toronto Raptors
    'TOR': 'TOR',
    'TORONTO': 'TOR',
    'RAPTORS': 'TOR',
    
    # Utah Jazz
    'UTA': 'UTA',
    'UTAH': 'UTA',
    'JAZZ': 'UTA',
    
    # Washington Wizards
    'WAS': 'WAS',
    'WSH': 'WAS',
    'WASHINGTON': 'WAS',
    'WIZARDS': 'WAS',
}

# Also handle reverse normalization for consistency (already normalized values)
TEAM_ABBR_NORMALIZATION.update({v: v for v in [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET',
    'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
    'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
    'TOR', 'UTA', 'WAS'
]})


def normalize_team_abbr(team_abbr: str) -> str:
    """
    Normalize team abbreviation to NBA.com format.
    
    Args:
        team_abbr: Team abbreviation (ESPN or NBA format)
        
    Returns:
        Normalized team abbreviation (NBA.com format)
    """
    if not team_abbr:
        return ''
    return TEAM_ABBR_NORMALIZATION.get(team_abbr.upper(), team_abbr.upper())


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


def fetch_nba_cdn_schedule() -> dict:
    """
    Fetch full season schedule from NBA CDN (no rate limiting).
    
    Uses scheduleLeagueV2.json which is publicly accessible
    and includes NBA game IDs.
    
    Returns:
        dict: Full season schedule
    """
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    
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


def extract_nba_games_for_date(nba_data: dict, date_str: str) -> List[Dict]:
    """
    Extract NBA games for a specific date from CDN schedule.
    
    Args:
        nba_data: Full NBA CDN schedule data
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        List of games for the specified date
    """
    games = []
    
    if 'leagueSchedule' not in nba_data:
        return games
    
    game_dates = nba_data['leagueSchedule'].get('gameDates', [])
    
    # Format target date for matching
    # NBA CDN uses MM/DD/YYYY format
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        target_date_str_1 = target_date.strftime('%m/%d/%Y')
        target_date_str_2 = target_date.strftime('%Y-%m-%d')
    except ValueError:
        return games
    
    # Find matching date
    for date_entry in game_dates:
        entry_date = date_entry.get('gameDate', '')
        
        # Match date in either format
        if target_date_str_1 in str(entry_date) or target_date_str_2 in str(entry_date):
            # Extract games from this date
            for game in date_entry.get('games', []):
                game_id = game.get('gameId')
                away_team = normalize_team_abbr(game.get('awayTeam', {}).get('teamTricode', ''))
                home_team = normalize_team_abbr(game.get('homeTeam', {}).get('teamTricode', ''))
                game_time_utc = game.get('gameDateTimeUTC', game.get('gameDateUTC', ''))
                
                games.append({
                    'game_id': game_id,
                    'away_team': away_team,
                    'home_team': home_team,
                    'game_time_utc': game_time_utc
                })
            break
    
    return games


def create_espn_to_nba_mapping(espn_data: dict, nba_games: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Map ESPN game IDs to NBA.com game IDs by matching games.
    
    Matches games by:
    1. Away team tricode (normalized)
    2. Home team tricode (normalized)
    3. Game time (UTC) - for disambiguation
    
    Args:
        espn_data: ESPN API response data
        nba_games: List of NBA games for the same date
        
    Returns:
        Dict mapping ESPN game IDs to NBA game IDs
    """
    mapping = {}
    
    if 'events' not in espn_data:
        return mapping
    
    espn_games = espn_data['events']
    
    # Create lookup for NBA games (key: away_team|home_team)
    nba_lookup = {}
    for nba_game in nba_games:
        away_tri = nba_game.get('away_team', '')
        home_tri = nba_game.get('home_team', '')
        game_time = nba_game.get('game_time_utc', '')
        
        # Normalize team abbreviations for lookup key
        away_tri_norm = normalize_team_abbr(away_tri)
        home_tri_norm = normalize_team_abbr(home_tri)
        
        key = f"{away_tri_norm}|{home_tri_norm}"
        
        # Store with time for disambiguation
        if key not in nba_lookup:
            nba_lookup[key] = []
        nba_lookup[key].append(nba_game)
    
    # Map ESPN games to NBA games
    for espn_game in espn_games:
        espn_id = espn_game.get('id')
        
        # Get teams from ESPN
        competitors = espn_game.get('competitions', [{}])[0].get('competitors', [])
        
        if len(competitors) < 2:
            mapping[espn_id] = None
            continue
        
        # Determine home/away
        if competitors[0].get('homeAway') == 'home':
            home_tri = competitors[0].get('team', {}).get('abbreviation', '')
            away_tri = competitors[1].get('team', {}).get('abbreviation', '')
        else:
            home_tri = competitors[1].get('team', {}).get('abbreviation', '')
            away_tri = competitors[0].get('team', {}).get('abbreviation', '')
        
        # Normalize team abbreviations
        away_tri_norm = normalize_team_abbr(away_tri)
        home_tri_norm = normalize_team_abbr(home_tri)
        
        # Get game time from ESPN
        espn_time = espn_game.get('date', '')
        
        # Find matching NBA game
        key = f"{away_tri_norm}|{home_tri_norm}"
        
        if key in nba_lookup:
            # Get first match (could be multiple games with same teams)
            nba_match = nba_lookup[key][0]
            nba_id = nba_match.get('game_id')
            mapping[espn_id] = nba_id
            
            # Remove from lookup to prevent duplicate mappings
            if len(nba_lookup[key]) > 1:
                nba_lookup[key].pop(0)
            else:
                del nba_lookup[key]
        else:
            mapping[espn_id] = None
    
    return mapping


def print_schedule(espn_data: dict, nba_games: List[Dict], mapping: dict, date_str: str, json_output: bool = False):
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
    mapping_source = "ESPN + NBA CDN (mapped)" if has_mapping else "ESPN only (unmapped)"
    
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
                    home_team = normalize_team_abbr(competitors[0].get('team', {}).get('abbreviation', 'HOME'))
                    away_team = normalize_team_abbr(competitors[1].get('team', {}).get('abbreviation', 'AWAY'))
                else:
                    home_team = normalize_team_abbr(competitors[1].get('team', {}).get('abbreviation', 'HOME'))
                    away_team = normalize_team_abbr(competitors[0].get('team', {}).get('abbreviation', 'AWAY'))
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
                    home_team = normalize_team_abbr(competitors[0].get('team', {}).get('abbreviation', 'HOME'))
                    away_team = normalize_team_abbr(competitors[1].get('team', {}).get('abbreviation', 'AWAY'))
                else:
                    home_team = normalize_team_abbr(competitors[1].get('team', {}).get('abbreviation', 'HOME'))
                    away_team = normalize_team_abbr(competitors[0].get('team', {}).get('abbreviation', 'AWAY'))
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


def get_nba_ids_for_predictions(mapping: dict, espn_data: dict) -> List[str]:
    """Get list of NBA.com game IDs that are ready for predictions."""
    nba_ids = []
    
    for espn_id, nba_id in mapping.items():
        if nba_id is not None:
            nba_ids.append(nba_id)
    
    return nba_ids


def main_with_output(date_str: str) -> Dict:
    """Fetch game schedule and return as dict (for programmatic use).
    
    Args:
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        Dict containing:
            - 'date': Date string
            - 'mapping': ESPN ID to NBA ID mapping
            - 'games': List of game dicts with nba_id, espn_id, teams, etc.
    """
    # Fetch schedules
    espn_data = fetch_espn_schedule(date_str)
    nba_data = fetch_nba_cdn_schedule()
    
    # Extract NBA games for target date
    nba_games = extract_nba_games_for_date(nba_data, date_str)
    
    # Create mapping
    mapping = create_espn_to_nba_mapping(espn_data, nba_games)
    
    # Build games list
    games = []
    
    if 'events' in espn_data:
        for game in espn_data['events']:
            espn_id = game.get('id')
            nba_id = mapping.get(espn_id)
            
            # Get teams
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            
            if len(competitors) >= 2:
                if competitors[0].get('homeAway') == 'home':
                    home_team = normalize_team_abbr(competitors[0].get('team', {}).get('abbreviation', 'HOME'))
                    away_team = normalize_team_abbr(competitors[1].get('team', {}).get('abbreviation', 'AWAY'))
                else:
                    home_team = normalize_team_abbr(competitors[1].get('team', {}).get('abbreviation', 'HOME'))
                    away_team = normalize_team_abbr(competitors[0].get('team', {}).get('abbreviation', 'AWAY'))
            else:
                home_team = 'HOME'
                away_team = 'AWAY'
            
            status = game.get('status', {}).get('type', {}).get('name', 'Unknown')
            date_time = game.get('date', 'Unknown')
            
            # Extract time portion
            if 'T' in str(date_time):
                time_utc = date_time.split('T')[1][:5]  # Get HH:MM
            else:
                time_utc = str(date_time)
            
            games.append({
                'espn_id': espn_id,
                'nba_id': nba_id,
                'away_team': away_team,
                'home_team': home_team,
                'status': status,
                'time_utc': time_utc,
                'date_time': date_time
            })
    
    return {
        'date': date_str,
        'mapping': mapping,
        'games': games
    }


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
      NBA CDN schedule is used for ID mapping (also no rate limiting).
      Team abbreviations are automatically normalized (e.g., WSH -> WAS, GS -> GSW).
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
    print(f"Fetching ESPN schedule for {date_str}...")
    espn_data = fetch_espn_schedule(date_str)
    
    print(f"Fetching NBA CDN schedule...")
    nba_data = fetch_nba_cdn_schedule()
    
    # Extract NBA games for target date
    print(f"Extracting NBA games for {date_str}...")
    nba_games = extract_nba_games_for_date(nba_data, date_str)
    
    # Create mapping
    print(f"Creating ESPN to NBA ID mapping...")
    mapping = create_espn_to_nba_mapping(espn_data, nba_games)
    
    print()
    
    # Print schedule
    print_schedule(espn_data, nba_games, mapping, date_str, args.json)
    
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
