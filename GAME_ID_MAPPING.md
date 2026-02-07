# ESPN to NBA.com Game ID Mapping Guide

## Challenge

ESPN API and NBA.com API use different game ID formats:
- **ESPN IDs**: 10-digit numeric (e.g., `401810588`)
- **NBA.com IDs**: 10-digit numeric (e.g., `0022500733`)

## Current Limitations

1. **NBA.com API Rate Limiting**
   - Returns 403 errors when frequently accessed
   - Makes real-time mapping unreliable
   
2. **No Direct Mapping Field**
   - ESPN API does not include NBA.com IDs
   - NBA.com API is rate-limited
   - Cannot reliably map in real-time

## Proposed Solutions

### Option 1: Manual Mapping (Quick Fix)

Create a manual mapping file for known games:

```json
{
  "401810588": "0022500733",
  "401810589": "0022500734",
  "401810590": "0022500735",
  ...
}
```

### Option 2: Schedule-Based Mapping (Automation)

Modify prediction scripts to:
1. Fetch ESPN schedule (no rate limiting)
2. Store game IDs as-is (ESPN format)
3. Update prediction scripts to work with ESPN IDs
4. Internally handle ESPN to NBA.com ID conversion when needed

### Option 3: Persistent Mapping Database

Create a mapping database that grows over time:

```python
# First run: Store ESPN IDs
# Second run: When NBA.com API available, add mappings
# Future runs: Use existing mappings

mapping_db = {
  "401810588": {"nba_id": "0022500733", "date": "2026-02-05"},
  "401810589": {"nba_id": "0022500734", "date": "2026-02-05"},
  ...
}
```

## Current Recommendation

For now, use ESPN IDs for schedule fetching and store as-is.
When NBA.com API is available (not rate-limited), build the mapping incrementally.

```bash
# Fetch schedule (always works)
python fetch_game_schedule.py --date 2026-02-05

# Run predictions on known NBA.com IDs
python run_pregame_predictions.py --games 0022500733 0022500734 0022500735
```

## Next Steps

1. Update prediction scripts to accept ESPN game IDs
2. Create persistent mapping database
3. Add incremental mapping building when NBA.com API is available
4. Implement fallback for unmapped games

---

Last Updated: 2026-02-07
