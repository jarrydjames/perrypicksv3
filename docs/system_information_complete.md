# PerryPicks v3 - System Information for Model Design & Evaluation

**Date:** January 29, 2026  
**System:** PerryPicks v3 (NBA Halftime Prediction)  
**Status:** Production-ready baseline model, temporal features analyzed

---

## 1) DATA & TIME STRUCTURE

### Seasons Covered
- **Seasons:** 2023-2025 (partial 2025 season)
- **Type:** Regular season + playoffs (mixed, not separated)
- **Games per season:**
  - 2023: 2,196 games
  - 2024: 5,604 games
  - 2025: 3,384 games (through June 23, 2025)
  - **Total:** 11,184 games
- **25-26 season:** 0 games (7 fetched but not merged - lack halftime stats)

### Timestamp Granularity
- **Format:** `datetime64[us, UTC]` (microsecond precision)
- **Timezone:** UTC
- **Date range:** 2023-10-05 16:00:00 UTC to 2025-06-23 00:00:00 UTC
- **Granularity:** Full datetime (not date-only)

### Double-Header / Same-Day Games
- **Same-day games:** 466 occurrences (games on same date)
- **Double-headers:** Possible (NBA occasionally has double-headers)
- **Impact:** Minimal for halftime predictions (game-day stats are same)

### Time Ordering
- **Time-ordered:** NO (not strictly monotonic)
- **Time ties:** 9,352 occurrences (games with same timestamp)
- **Sorting:** Uses pseudo-timestamp for stable ordering: `season_end_yy + game_id`

---

## 2) PREDICTION MOMENTS

### Current Architecture
- **Prediction moment:** HALFTIME (end of 2nd quarter)
- **Model task:** Predict 2H (second half) total and margin
- **Training approach:** Single model with halftime features

### Trained As
- **Architecture:** Separate models per cut
- **Current:** HALFTIME model (predicts H2 from H1 stats)
- **No time marker:** No explicit "minute" or "quarter" feature
- **Not trained:** Pre-game or Q3 prediction models

### Feature Availability
| Feature | When Available | Use |
|---------|----------------|------|
| Baseline 13 | HALFTIME | H1 stats (score, events, eFG, efficiency) |
| Temporal (rolling) | HALFTIME (pre-computed) | Last 5/10 games, streaks, rest |
| Market lines | PRE-GAME (if available) | Opening/closing lines |

---

## 3) TARGET DEFINITION

### Current Targets
- **Primary target:** `h2_total` (second half total points)
- **Secondary target:** `h2_margin` (second half margin: home_score - away_score)
- **Target type:** Aggregate (total/margin), NOT home_final/away_final

### Willingness to Switch
- **Current:** Aggregate targets (h2_total, h2_margin)
- **Open to:** home_final/away_final (but would require feature re-engineering)
- **Trade-off:** Aggregate targets currently work well (MAE: 7.07)

### Probability Outputs
- **Required:** NO (point predictions only)
- **Available:** Uncertainty intervals (Gaussian, 95% coverage)
- **ROI simulation:** Synthetic betting ROI available (edge threshold: 6%)

---

## 4) FEATURE INVENTORY (EXACT LIST)

### Baseline Features (17 total) - HALFTIME

| Feature Name | When Available | How Computed | Uses Historical |
|--------------|----------------|-----------------|------------------|
| `h1_home` | HALFTIME | First half home score | NO |
| `h1_away` | HALFTIME | First half away score | NO |
| `h1_total` | HALFTIME | First half total score (home + away) | NO |
| `h1_margin` | HALFTIME | First half margin (home - away) | NO |
| `h1_events` | HALFTIME | Total events in first half | NO |
| `h1_n_2pt` | HALFTIME | 2-point shots made in first half | NO |
| `h1_n_3pt` | HALFTIME | 3-point shots made in first half | NO |
| `h1_n_turnover` | HALFTIME | Turnovers in first half | NO |
| `h1_n_rebound` | HALFTIME | Rebounds in first half | NO |
| `h1_n_foul` | HALFTIME | Fouls in first half | NO |
| `h1_n_timeout` | HALFTIME | Timeouts in first half | NO |
| `h1_n_sub` | HALFTIME | Substitutions in first half | NO |
| `home_efg` | HALFTIME | Home team effective FG % | NO |
| `away_efg` | HALFTIME | Away team effective FG % | NO |
| `home_tpar` | HALFTIME | Home team true shooting % | NO |
| `home_tor` | HALFTIME | Home team turnover rate | NO |
| `home_orbp` | HALFTIME | Home team offensive rebound % | NO |

### Temporal Features (12 total) - HALFTIME (pre-computed)

| Feature Name | When Available | How Computed | Uses Historical |
|--------------|----------------|-----------------|------------------|
| `home_pts_scored_avg_5` | HALFTIME | Rolling avg points scored (last 5 games) | YES |
| `home_pts_allowed_avg_5` | HALFTIME | Rolling avg points allowed (last 5 games) | YES |
| `home_margin_avg_5` | HALFTIME | Rolling avg margin (last 5 games) | YES |
| `home_current_streak_5` | HALFTIME | Current win/loss streak (last 5 games) | YES |
| `home_days_since_last` | HALFTIME | Days since previous game | NO |
| `home_is_back_to_back` | HALFTIME | 1 if playing back-to-back | NO |
| `away_pts_scored_avg_5` | HALFTIME | Rolling avg points scored (last 5 games) | YES |
| `away_pts_allowed_avg_5` | HALFTIME | Rolling avg points allowed (last 5 games) | YES |
| `away_margin_avg_5` | HALFTIME | Rolling avg margin (last 5 games) | YES |
| `away_current_streak_5` | HALFTIME | Current win/loss streak (last 5 games) | YES |
| `away_days_since_last` | HALFTIME | Days since previous game | NO |
| `away_is_back_to_back` | HALFTIME | 1 if playing back-to-back | NO |

### Rolling Windows (Additional)
| Feature | Window | Description |
|---------|---------|-------------|
| `*_avg_10` | 10 games | Rolling averages over last 10 games |
| `*_wins_10` | 10 games | Total wins in last 10 games |

### Market Features (0 total)
- **Market priors:** DROPPED in baseline backtests
- **Features available (if used):**
  - `market_total_line` (total line)
  - `market_home_spread_line` (spread)
  - `market_home_team_total_line` (home team total)
  - `market_away_team_total_line` (away team total)
- **Usage:** Opening/closing lines not distinguished (single value)
- **Current status:** Dropped from backtests (--drop-market-priors flag)

---

## 5) TEAM & ID FEATURES

### Team IDs
- **Team IDs:** YES
- **Home team ID:** `home_team_id` (integer)
- **Away team ID:** `away_team_id` (integer)

### Team Persistence
- **Teams persist across seasons:** YES
- **Season tracking:** `season_end_yy` (season end year)
- **ID stability:** Consistent team IDs across 2023-2025

### Relocations / Renames
- **Normalization:** NOT applied (team IDs as-is)
- **Relocations:** Not handled (NBA team relocations possible)

### Home/Away Flags
- **Home flag:** YES (inferred: `home_team_id` in home team role)
- **Away flag:** YES (inferred: `away_team_id` in away team role)
- **Explicit flag:** NO (no `is_home` boolean feature)

---

## 6) PLAY-BY-PLAY AVAILABILITY

### PBP Data Status
- **Full PBP data:** YES (7 files fetched from 25-26 season)
- **Format:** JSON
- **Coverage:** Minimal (only 7 games from Jan 26-29, 2026)
- **Main data:** Boxscore aggregates (not full PBP)

### Aggregation Level
- **Per-minute bins:** NO
- **Per-possession bins:** NO
- **Current aggregation:** Boxscore at halftime ONLY
- **Granularity:** Half-level aggregates (H1, H2, full game)

### PBP Utilization
- **Status:** PBP data exists but not used for features
- **Reason:** PBP aggregation not implemented
- **Feature source:** Boxscore (halftime aggregates)

---

## 7) TEMPORAL FEATURE CONSTRUCTION

### Rolling Window Sizes
- **Window sizes:** [5, 10]
- **Rolling averages:** pts_scored_avg_5/10, pts_allowed_avg_5/10, margin_avg_5/10
- **Rolling wins:** wins_5/10
- **Rolling streaks:** current_streak_5 (only 5-game streak available)

### Season Boundary Reset
- **Reset at season boundaries:** NO
- **Cross-season rolling:** YES (rolling stats include games from previous season)
- **Implication:** Early season games use previous season data

### Opponent Strength
- **Opponent strength in rolling stats:** NO
- **Pure performance:** Rolling stats based on team's own performance only
- **No SOS:** No strength of schedule adjustment

### Forward-Only Pass
- **Forward-only pass:** YES
- **Method:** Pre-computed features (forward-only, no leakage)
- **Computation:** Rolling features computed once, stored, merged at prediction time

---

## 8) CURRENT BACKTEST SPLIT (CRITICAL)

### Outer Split Strategy
- **Strategy:** Walkforward (rolling blocks)
- **Method:** Train → Test → Step → Train → Test → ...
- **Contiguity:** YES (folds are contiguous in time)

### Fold Specifications
- **Min train size:** 500 games
- **Test size:** 200 games
- **Step size:** 200 games
- **Total folds:** ~55 folds (11,184 games / 200)

### Fold Structure
```
Fold 1: Train 0-500, Test 500-700
Fold 2: Train 0-700, Test 700-900
Fold 3: Train 0-900, Test 900-1100
...
Fold N: Train 0-(N*200), Test (N*200)-((N+1)*200)
```

### Validation Approach
- **Time-ordered:** YES (uses gameTimeUTC for sorting)
- **Leakage-free:** YES (no future data in training)
- **Stationarity:** Assumed (no explicit stationarity tests)

---

## 9) HYPERPARAMETER TUNING

### Retuning Strategy
- **Retune per fold:** NO (fixed parameters)
- **Tuning method:** None (uses defaults)
- **Manual tuning:** Tested specific depths (3, 6, 10) in analysis

### Model Defaults

#### Gradient Boosting Trees (GBT)
- `max_depth`: 6 (default), tested 3/6/10
- `learning_rate`: 0.05
- `max_iter`: 500
- `min_samples_leaf`: 30

#### Random Forest
- `n_estimators`: 400
- `max_depth`: None (unlimited)
- `min_samples_leaf`: 2

#### Ridge Regression
- `alpha`: 2.0 (L2 regularization)

### Validation Split
- **Inside training:** Not specified (uses full training data)
- **Cross-validation:** None
- **Hyperparameter search:** Grid search (if implemented)

### Max Trials
- **Per fold:** Not specified (uses default parameters)
- **Search space:** Manual exploration (depth 3/6/10)

---

## 10) UNCERTAINTY METHOD

### Current Interval Method
- **Method:** Gaussian (normal distribution)
- **Implementation:** `sigma_from_residuals()` - residual standard deviation
- **Prediction:** `mean ± 1.96 * sigma` for 95% CI

### Coverage Targets
- **Default coverage:** 95%
- **Confidence:** Not configurable (hardcoded)
- **Method:** Normal distribution assumption

### Conditional Coverage
- **Evaluate conditional coverage:** NO
- **Uniform coverage:** YES (same 95% for all predictions)
- **Stratification:** None (no conditional by game type, etc.)

---

## 11) METRICS & DECISION RULES

### Primary Metric
- **Primary metric:** MAE (Mean Absolute Error)
- **Calculation:** `mean(|y_pred - y_actual|)`
- **Focus:** 2H total prediction accuracy

### Secondary Metrics
- **RMSE:** Root Mean Squared Error (penalizes large errors)
- **Margin MAE:** MAE for margin predictions
- **Margin RMSE:** RMSE for margin predictions

### Minimal Improvement Threshold
- **Go/no-go threshold:** NOT specified
- **Decision rule:** Visual comparison (no formal threshold)
- **Baseline:** MAE = 7.0702 (GBT, depth=3, 13 features)

### Statistical Tests
- **Diebold-Mariano (DM): NO
- **Bootstrap:** NO
- **Confidence intervals:** NO
- **Test used:** None (basic metric comparison)

### Confidence Level
- **Required confidence:** NOT specified
- **Default:** 95% (for uncertainty intervals)
- **Metric significance:** Not tested (no p-values)

---

## 12) DEPLOYMENT CONSTRAINTS

### Retraining Frequency
- **Frequency:** NOT specified (likely daily or weekly)
- **Data freshness:** 25-26 season games fetched but not merged
- **Recommendation:** Daily retraining for in-season updates

### Training Time Budget
- **Max time:** NOT specified
- **Model training:** <60 seconds for current dataset (11,184 games)
- **Feature building:** <120 seconds (rolling stats)

### Inference Latency
- **Latency constraints:** NOT specified
- **Model type:** Tree-based (fast inference)
- **Per prediction:** <1ms typical

### Model Interpretability
- **Requirement:** NOT required (ensemble models acceptable)
- **Current model:** Gradient Boosting Trees (black-box)
- **Feature importance:** Available (but not currently extracted)

---

## Summary of Key Findings

### Optimal Configuration
- **Model:** Gradient Boosting Trees (sklearn GBT)
- **Depth:** 3 (shallow trees prevent overfitting)
- **Features:** 13 baseline features (halftime stats)
- **MAE:** 7.0702 (best performing)

### What Doesn't Work
- **Temporal features:** No benefit (-0.04% to -6.87%)
- **Deeper models:** Overfitting (5-10% worse with depth 6/10)
- **More complex models:** XGBoost/CatBoost not needed

### Data Limitations
- **Time granularity:** Same-day games (466), time ties (9,352)
- **PBP data:** Available but not aggregated (only boxscore used)
- **25-26 season:** 0 games in training data (not merged)

### Recommendations for Statistically Valid Models

1. **Keep walkforward backtest:** Leaks-free, time-ordered
2. **Use fixed hyperparameters:** Default params work well
3. **Baseline features sufficient:** Don't use temporal features
4. **Simple model (depth=3):** Prevents overfitting
5. **No market priors needed:** Halftime stats are predictive enough

---

## Files Referenced

- `data/processed/halftime_with_temporal_features_total.parquet` - Main dataset
- `data/processed/rolling_features.parquet` - Rolling statistics
- `src/modeling/backtest_utils.py` - Walkforward fold logic
- `src/modeling/walkforward_backtest.py` - Backtest execution
- `src/modeling/sklearn_models.py` - Model definitions
- `src/modeling/uncertainty.py` - Uncertainty intervals

---

**Date:** January 29, 2026  
**Status:** SYSTEM INFORMATION COMPLETE  
**Purpose:** Enable statistically valid model design, pipelines, and leakage-free backtests
