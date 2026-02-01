# PerryPicks v3 - NBA API Data Fetching Guide

## 📚 Documentation

For detailed NBA API data fetching methodology, best practices, and troubleshooting, see:

**[docs/NBA_API_DATA_FETCHING.md](docs/NBA_API_DATA_FETCHING.md)**

## 🎯 Quick Start

### Fetching Games

```python
from datetime import date
from src.data.scoreboard import fetch_scoreboard, format_game_label

# Fetch games with scores
games = fetch_scoreboard(date.today(), include_live=True)

# Filter completed games
completed = [g for g in games if g.home_score is not None and g.home_score > 0]

for game in completed:
    print(f"{game.away} @ {game.home}")
    print(f"  Score: {game.away_score} - {game.home_score}")
    print(f"  Status: {game.status_text}")
```

### Key Points

1. **Always use `include_live=True`** to get actual scores
2. **Filter by `home_score > 0`** to confirm games are completed
3. **Handle exceptions gracefully** - APIs can be flaky
4. **Statistics are in `game.homeTeam.statistics`** object

## 📊 Model Performance

### Out-of-Sample Test Results (Jan 27-30, 2026)

- **Games Tested:** 33 completed games
- **Total MAE:** 3.37 points
- **Margin MAE:** 3.80 points
- **Winner Accuracy:** 90.9% (30/33 correct)
- **Within 3 pts:** 51.5%
- **Within 5 pts:** 69.7%
- **Within 10 pts:** 97.0%

## 🔧 Common Issues

### No Scores Found

**Problem:** All games show `home_score=None`

**Solution:**
```python
games = fetch_scoreboard(date, include_live=True)  # Must use include_live=True!
```

### Statistics Are All 0

**Problem:** All eFG, FTR, etc. are 0

**Solution:**
```python
# Wrong:
home_efg = game_data.get('homeTeam', {}).get('effectiveFieldGoalPercentage')

# Correct:
home_stats = game_data.get('homeTeam', {}).get('statistics', {})
home_efg = home_stats.get('fieldGoalsEffectiveAdjusted', 0.5)
```

### 403 Forbidden Errors

**Problem:** API returns 403

**Solution:** We use `scheduleLeagueV2.json` from public CDN (no 403 errors)

## 📖 Full Documentation

See [docs/NBA_API_DATA_FETCHING.md](docs/NBA_API_DATA_FETCHING.md) for:
- Complete API endpoint documentation
- Response structure examples
- Feature extraction formulas
- Common pitfalls and solutions
- Troubleshooting guide
- Out-of-sample testing methodology

## 🚀 Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run streamlit: `streamlit run app.py`
3. Pick a date and select a game
4. View predictions and betting recommendations

## 📝 Model Card

See `model_card_v1_ridge.txt` for complete model documentation.

## 🤝 Contributing

This is a research project for basketball analytics. All predictions are for educational purposes only.
