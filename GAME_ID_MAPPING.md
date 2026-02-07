# ESPN to NBA.com Game ID Mapping - ✅ SOLVED

## The Challenge

ESPN API and NBA.com use different game ID formats:
- **ESPN IDs**: 10-digit numeric (e.g., `401810602`)
- **NBA.com IDs**: 10-digit numeric (e.g., `0022500747`)

Previous solutions failed because:
- NBA.com stats API is rate-limited (403 errors)
- ESPN API doesn't include NBA.com IDs
- No reliable way to map IDs in real-time

---

## ✅ SOLUTION: Use NBA's Public CDN Schedule Feed

Instead of hitting the rate-limited stats.nba.com API, we now use NBA's publicly accessible CDN schedule:

**URL**: `https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`

**Benefits**:
- ✅ No rate limiting (publicly accessible)
- ✅ Complete season schedule in one request
- ✅ Includes official NBA.com game IDs
- ✅ No API key or authentication needed

---

## How It Works

### Step 1: Fetch ESPN Schedule
```bash
python fetch_game_schedule.py --date 2026-02-07
```
- Fetches from ESPN API (no rate limiting)
- Gets ESPN game IDs, teams, times

### Step 2: Fetch NBA CDN Schedule
- Pulls full season schedule from NBA CDN
- Extracts games for target date
- Gets NBA.com game IDs

### Step 3: Match Games by Teams
- Normalizes team abbreviations (e.g., WSH → WAS, GS → GSW)
- Matches ESPN games to NBA games by:
  - Away team tricode (normalized)
  - Home team tricode (normalized)
  - Date/time (for disambiguation)

### Step 4: Return Mapping
- ESPN ID → NBA ID mapping
- Ready for predictions

---

## Team Abbreviation Normalization

| ESPN | NBA | Normalized |
|-------|-------|------------|
| WSH   | WAS   | WAS        |
| SA     | SAS   | SAS        |
| UTAH   | UTA   | UTA        |
| GS     | GSW   | GSW        |
| NY     | NYK   | NYK        |
| NO     | NOP   | NOP        |
| PHX    | PHO   | PHO        |

---

## Usage Examples

### Fetch Schedule
```bash
# Table format
python fetch_game_schedule.py --date 2026-02-07

# JSON format
python fetch_game_schedule.py --date 2026-02-07 --json

# Get NBA IDs only (for automation)
python fetch_game_schedule.py --date 2026-02-07 --nba-ids

# Save to file
python fetch_game_schedule.py --date 2026-02-07 --output schedule.json
```

### Run Predictions on Mapped NBA IDs
```bash
# Use NBA IDs from fetch_game_schedule.py output
python run_pregame_predictions.py --games 0022500747 0022500748 0022500749
python run_halftime_predictions.py --games 0022500747 0022500748 0022500749
python run_q3_predictions.py --games 0022500747 0022500748 0022500749
```

---

## Example Output

```
====================================================================================================
NBA GAME SCHEDULE FOR 2026-02-07
====================================================================================================

Found 10 games
Mapped: 10, Unmapped: 0
Source: ESPN + NBA CDN (mapped)

ESPN ID      | NBA ID       | Away   @ Home   | Status               | Time (UTC)
----------------------------------------------------------------------------------------------------
401810602    | 0022500747   | WSH    @ BKN    | STATUS_SCHEDULED     | 20:00
401810603    | 0022500748   | HOU    @ OKC    | STATUS_SCHEDULED     | 20:30
401810604    | 0022500749   | DAL    @ SA     | STATUS_SCHEDULED     | 23:00
...
====================================================================================================
```

---

## Technical Details

### NBA CDN Schedule Format

```json
{
  "meta": {"version": 1.0, ...},
  "leagueSchedule": {
    "seasonYear": 2026,
    "gameDates": [
      {
        "gameDate": "02/07/2026 00:00:00",
        "games": [
          {
            "gameId": "0022500747",
            "awayTeam": {"teamTricode": "WAS", ...},
            "homeTeam": {"teamTricode": "BKN", ...},
            "gameDateTimeUTC": "2026-02-07T20:00:00Z"
          },
          ...
        ]
      },
      ...
    ]
  }
}
```

---

## Files

- **fetch_game_schedule.py** - Main script for fetching and mapping
- **GAME_ID_MAPPING.md** - This documentation
- **schedule_*.json** - Sample output files (generated)

---

**Status**: ✅ Production Ready
**Last Updated**: 2026-02-07
**Version**: 2.0 (CDN-based mapping)
