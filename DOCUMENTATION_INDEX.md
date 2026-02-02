# PerryPicks v3 - Documentation Index

**Last Updated:** Feb 1, 2026  
**Project:** NBA Game Prediction Tool

---

## 📚 Essential Documentation

### 🚀 Quick Start
1. **[SEASON_DATA_GUIDE.md](SEASON_DATA_GUIDE.md)** ⭐ START HERE
   - NBA API usage guide
   - Season format rules (`2025-26`)
   - Column reference (Advanced vs Base mode)
   - Common mistake prevention

2. **[docs/PREDICTION_CHECKLIST.md](docs/PREDICTION_CHECKLIST.md)** ⭐ READ BEFORE PREDICTING
   - 30-checkbox pre-prediction checklist
   - Common mistakes with solutions
   - Step-by-step workflow
   - Quick reference guide

---

## 📋 Recent Updates

### Feb 1, 2026
3. **[docs/FEB1_2026_UPDATES.md](docs/FEB1_2026_UPDATES.md)**
   - Today's bug fixes and improvements
   - Technical changes (season config, matchup parsing, deduplication)
   - Today's predictions (7 games)

4. **[docs/DAILY_WORK_SUMMARY.md](docs/DAILY_WORK_SUMMARY.md)**
   - Complete summary of today's work
   - Files created/modified
   - Prediction quality metrics
   - Achievement unlocked badges

---

## 🏗️ System Architecture

### Core Documentation
5. **[FINAL_REPORT.md](FINAL_REPORT.md)**
   - Full system documentation
   - Model architecture (Ridge, RF, XGBoost)
   - Feature engineering (72 features)
   - Training pipeline

6. **[docs/MODEL_DOCUMENTATION.md](docs/MODEL_DOCUMENTATION.md)**
   - Model types and parameters
   - Feature definitions
   - Training procedures

7. **[docs/v1-architecture.md](docs/v1-architecture.md)**
   - Original architecture
   - Design decisions

8. **[docs/v2-plan.md](docs/v2-plan.md)**
   - V2 planning notes
   - Improvement ideas

9. **[docs/v3-technical-plan.md](docs/v3-technical-plan.md)**
   - v3 technical implementation
   - ML improvements

---

## 🛠️ Technical Guides

### Data Fetching
10. **[docs/NBA_API_DATA_FETCHING.md](docs/NBA_API_DATA_FETCHING.md)**
    - Detailed API usage
    - Endpoint parameters
    - Response formats

11. **[README_API_GUIDE.md](README_API_GUIDE.md)**
    - Quick API reference
    - Common endpoints

### Model Training
12. **[docs/full_training_plan_20250131.md](docs/full_training_plan_20250131.md)**
    - Complete training workflow
    - Data preparation
    - Model evaluation

13. **[docs/complete_training_run_20250131.md](docs/complete_training_run_20250131.md)**
    - Training execution
    - Results summary

### Backtesting
14. **[docs/BACKTEST_SUMMARY.md](docs/BACKTEST_SUMMARY.md)**
    - Backtest methodology
    - Results analysis

15. **[docs/offline_backtest.md](docs/offline_backtest.md)**
    - Offline testing guide

---

## 🐛 Bug Fixes & Issues

### Recent Fixes (Jan 31 - Feb 1, 2026)
- **AttributeError:** `docs/comprehensive_attributeerror_fix_20250131.md`
- **KeyError:** `docs/derived_keyerror_fix_20250131.md`
- **ImportError:** `docs/import_error_fix_20250131.md`
- **Feature count:** `docs/feature_count_43_to_35_fix_20250131.md`
- **Missing data:** `docs/missing_team_data_fix_20250131.md`
- **Odds API:** `docs/odds_api_optimization_implementation_20250131.md`

### Streamlit Cloud
- **403 Error:** `docs/STREAMLIT_CLOUD_403_INVESTIGATION.md`
- **Slow load:** `docs/streamlit_slow_load_fix_20250131.md`
- **Module fixes:** `docs/streamlit_cloud_fix_20250119_module_level_imports.md`

---

## 📊 Analysis Reports

### Comprehensive Reports
16. **[FINAL_COMPREHENSIVE_REPORT.md](FINAL_COMPREHENSIVE_REPORT.md)**
    - Complete system analysis
    - Performance metrics

17. **[COMPREHENSIVE_ANALYSIS_REPORT.md](COMPREHENSIVE_ANALYSIS_REPORT.md)**
    - Feature analysis
    - Model comparison

18. **[ML_IMPROVEMENTS_REPORT.md](ML_IMPROVEMENTS_REPORT.md)**
    - ML enhancements
    - Feature engineering

19. **[RESEARCH_IMPROVEMENTS.md](RESEARCH_IMPROVEMENTS.md)**
    - Research findings
    - Improvement ideas

20. **[ENHANCEMENT_REPORT.md](ENHANCEMENT_REPORT.md)**
    - System enhancements
    - Future improvements

---

## 🎯 Prediction Tools

### Scripts
- `predict_todays_games.py` - Main prediction script
- `make_pregame_prediction.py` - Single game prediction
- `get_todays_games.py` - Game fetching utility
- `debug_data_fetch.py` - Data verification
- `debug_todays_games.py` - Game list debugging

### Predictions
- `data/predictions/todays_predictions_2026-02-01.csv` - Today's predictions

---

## 🔧 Troubleshooting

### Common Issues

**Season Data Problems**
→ See: `SEASON_DATA_GUIDE.md`
→ Always use `SEASON = '2025-26'` format

**Missing Features**
→ See: `docs/PREDICTION_CHECKLIST.md`
→ Use Advanced mode for PACE, OFF_RATING, DEF_RATING

**Unknown Teams**
→ See: `docs/PREDICTION_CHECKLIST.md`
→ Verify all 30 teams in TEAM_IDS

**Duplicate Games**
→ See: `docs/PREDICTION_CHECKLIST.md`
→ Always deduplicate by GAME_ID

---

## 📖 Reading Order

### New to the Project?
1. Start: `README.md` - Project overview
2. Then: `FINAL_REPORT.md` - System documentation
3. Then: `SEASON_DATA_GUIDE.md` - Data fetching guide
4. Then: `docs/PREDICTION_CHECKLIST.md` - Pre-prediction checklist

### Ready to Make Predictions?
1. Read: `docs/PREDICTION_CHECKLIST.md` (Check all 30 boxes!)
2. Run: `predict_todays_games.py`
3. Review: `data/predictions/todays_predictions_YYYY-MM-DD.csv`

### Planning V2?
1. Review: `docs/FEB1_2026_UPDATES.md` - Recent fixes
2. Review: `docs/v2-plan.md` - V2 ideas
3. Review: `FINAL_COMPREHENSIVE_REPORT.md` - Current performance

---

## 🏆 Documentation Stats

- **Total Documentation Files:** 80+
- **Bug Fix Documents:** 50+
- **Analysis Reports:** 10+
- **Technical Guides:** 15+
- **Most Recent:** Feb 1, 2026

---

## 📞 Quick Reference

### Critical Constants
```python
SEASON = '2025-26'  # Always use this format
API_TIMEOUT = 30  # Seconds
N_RECENT_GAMES = 10  # For recent form
HOME_COURT_ADVANTAGE = 3.0  # Points
```

### Key Team IDs
```python
OKC = 1610612760
DEN = 1610612743
# See SEASON_DATA_GUIDE.md for all 30 teams
```

### Model Files
```python
models = {
    'total': 'models/total_ridge_model.joblib',
    'home': 'models/home_ridge_model.joblib',
    'away': 'models/away_ridge_model.joblib'
}
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run predictions
uv run python predict_todays_games.py

# Check predictions
cat data/predictions/todays_predictions_*.csv
```

---

**Remember:** Always check `docs/PREDICTION_CHECKLIST.md` before making predictions! 🐶

---

*Last updated: Feb 1, 2026*
