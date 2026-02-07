# PerryPredictions UI - Streamlit App

A temporary Streamlit app to manually trigger and view predictions while building full automated posting system.

---

## 🚀 Quick Start

```bash
# Navigate to project directory
cd /path/to/PerryPicks v3

# Run Streamlit app
streamlit run perry_predictions_ui.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📋 Features

### 1. Date Selection
- Pick any date (past or future)
- Default: Tomorrow's games
- Max date: 1 year in future
- Min date: 1 year in past

### 2. Prediction Type
Choose which model to run:
- **Pregame** - Predict final scores before game starts
- **Halftime** - Predict final scores at halftime (using H1 scores)
- **Q3** - Predict final scores after Q3 (using Q3 cumulative scores)

### 3. Advanced Options
- **Fetch Odds** - Include betting odds in predictions (default: ON)
- **Show Raw Output** - Display raw prediction JSON (default: OFF)

### 4. Game Schedule
- Displays all games for selected date
- Shows ESPN ID, NBA ID, teams, status, time (UTC)
- 100% ESPN→NBA ID mapping

### 5. Predictions
- Run predictions for all games
- Display results in table format
- Shows:
  - Game ID
  - Teams
  - Predicted total
  - Predicted margin
  - Predicted winner
  - Confidence score
  - Odds (if enabled)

### 6. Formatted Posts
- Auto-generates social media posts
- Copy and paste ready
- Includes:
  - Emojis 🏀🔥⚡🎯🏆💰
  - Team names
  - Scores (H1, Q3, etc.)
  - Predictions
  - Odds (if enabled)
  - Hashtags
- Separate post for each game
- Expandable sections for easy viewing

---

## 📊 Output Examples

### Pregame Post
```
🏀 Pregame Prediction: Washington Wizards @ Brooklyn Nets

📊 Predicted Total: 221.8
📈 Predicted Margin: -3.2 (WAS)
🎯 Predicted Winner: WAS

💰 Odds: Spread WAS -3.5, O/U 219.5

#NBA #PerryPredictions #NBAWAS #NBABKN
```

### Halftime Post
```
🔥 Halftime Update: Washington Wizards @ Brooklyn Nets

📊 Halftime: 56 - 52
📈 Projected 2H: 60.8 - 54.0
🎯 Projected Final: 116.8 - 106.0
🏆 Projected Winner: WAS by 10.8

💰 Odds: Spread WAS -4.5, O/U 222.5

#NBA #PerryPredictions #NBAWAS #NBABKN
```

### Q3 Post
```
⚡ Q3 Update: Washington Wizards @ Brooklyn Nets

📊 Q3 Cumulative: 95.0 - 84.0
📈 Estimated Q4: 30.8 - 26.4
🎯 Projected Final: 125.8 - 110.4
🏆 Projected Winner: WAS by 15.4

💰 Odds: Spread WAS -6.5, O/U 236.5

#NBA #PerryPredictions #NBAWAS #NBABKN
```

---

## 🎯 Use Cases

### 1. Pre-Game Research
```bash
# 1. Open app
streamlit run perry_predictions_ui.py

# 2. Select tomorrow's date

# 3. Choose "Pregame"

# 4. Run predictions

# 5. Copy formatted posts

# 6. Post to social media manually
```

### 2. In-Game Updates
```bash
# 1. Wait for halftime or Q3

# 2. Select current date

# 3. Choose "Halftime" or "Q3"

# 4. Run predictions

# 5. Copy formatted posts

# 6. Post updates to social media
```

### 3. Testing Models
```bash
# 1. Select past date (completed games)

# 2. Run all 3 models (pregame, halftime, q3)

# 3. Compare predictions to actual results

# 4. Analyze model performance
```

---

## 🔧 Technical Details

### Architecture
```
User Interface (Streamlit)
    ↓
Date Selection → Prediction Type Selection
    ↓
Fetch Schedule (fetch_game_schedule.py)
    ↓
Display Games Table
    ↓
User Clicks "Run Predictions"
    ↓
Run Predictions (src/predict_api.py)
    ↓
Format Results (Tables + Posts)
    ↓
Display to User
```

### Components Used
- **Streamlit** - UI framework
- **fetch_game_schedule.py** - Schedule fetching & ID mapping
- **src/predict_api.py** - Prediction API
- **Pandas** - Dataframe formatting

### Data Flow
1. User selects date and prediction type
2. App fetches schedule from ESPN + NBA CDN
3. App displays games table
4. User clicks "Run Predictions"
5. App runs predictions for each game
6. App formats results into tables
7. App generates formatted posts
8. User copies posts and posts manually

---

## 🐛 Troubleshooting

### Issue: No games found
**Solution:**
- Check if date is correct
- Verify date has NBA games (offseason = no games)
- Try a different date

### Issue: Predictions fail
**Solution:**
- Check internet connection
- Verify NBA API is accessible
- Check log messages in app
- Try again later (API rate limiting)

### Issue: Odds not showing
**Solution:**
- Make sure "Fetch Odds" is enabled
- Check odds API status
- Some games may not have odds available

### Issue: Streamlit won't start
**Solution:**
```bash
# Install Streamlit if not installed
pip install streamlit

# Or using uv
uv add streamlit
```

---

## 🚀 Future Enhancements

### Phase 1: Immediate Improvements
- [ ] Add specific game selection (not just all games)
- [ ] Add real-time game state indicator
- [ ] Add prediction history (save to database)
- [ ] Add comparison to actual results
- [ ] Add confidence thresholds

### Phase 2: Automated Posting (Future)
- [ ] Twitter API integration
- [ ] Bluesky API integration
- [ ] Auto-post predictions
- [ ] Duplicate detection
- [ ] Error handling & retry

### Phase 3: Advanced Features
- [ ] Multiple date selection
- [ ] Batch predictions for multiple dates
- [ ] Model comparison (side-by-side)
- [ ] Performance analytics
- [ ] Custom post templates

---

## 📚 Documentation

- **AUTOMATION_FLOW.md** - Complete automation flow
- **AUTOMATION_SUMMARY.md** - Quick reference
- **STREAMLIT_UI.md** - This document (you're reading it!)
- **GAME_ID_MAPPING.md** - ESPN to NBA ID mapping
- **README_MODELS.md** - Model documentation

---

## 💡 Tips

1. **Pre-Game** - Run predictions 1-2 hours before tipoff for best accuracy
2. **Halftime** - Run at halftime for in-game updates
3. **Q3** - Run after Q3 for final quarter projections
4. **Odds** - Enable odds for betting context
5. **Copy Posts** - Use the formatted posts for easy posting

---

**Last Updated:** 2026-02-07
**Status:** Production Ready
**Version:** 1.0
