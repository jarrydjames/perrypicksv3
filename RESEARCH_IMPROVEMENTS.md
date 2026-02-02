# Research: Improvements for NBA Prediction ML Models

## Executive Summary

This document researches improvements to enhance accuracy for:
- **Total Points Prediction** (currently 15.54 MAE)
- **Margin/Spread Prediction** (currently 12.00 MAE)
- **Winner Prediction** (currently 61.0% accuracy)

Goal: Reduce MAE by 2-3 points and improve winner accuracy to 65%+.

---

## Part 1: Feature Engineering Improvements

### Current Features (34)
- Team ratings (off/def, pace, 4 factors)
- Win percentages
- Matchup differentials
- Home court advantage
- Expected metrics

### New Features to Add

#### 1. Rest Days (High Impact, Easy)
**What:** Days since each team's last game

**Why important:** Fresh teams perform better
- 1-2 days: Fatigue from recent game
- 3-4 days: Optimal rest
- 5+ days: Rust from too much time off

**Expected improvement:** 0.5-1.0 points MAE reduction

**Implementation:**
```python
# For each game, calculate days since last game for both teams
df['home_rest_days'] = days_since_last_game(home_id, game_date)
df['away_rest_days'] = days_since_last_game(away_id, game_date)
df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
```

#### 2. Back-to-Back Games (High Impact, Easy)
**What:** Is a team playing consecutive days?

**Why important:** Back-to-backs cause 2-3 point drop in scoring
- Second night of B2B: ~-2.5 points
- Travel + B2B: ~-3.5 points

**Expected improvement:** 0.5-1.5 points MAE reduction

**Implementation:**
```python
df['home_is_b2b'] = days_since_last_game(home_id, game_date) == 1
df['away_is_b2b'] = days_since_last_game(away_id, game_date) == 1
df['home_b2b_x_home'] = df['home_is_b2b']
df['away_b2b_x_away'] = df['away_is_b2b']
```

#### 3. Travel Distance (Medium Impact, Medium difficulty)
**What:** Miles traveled to current game location

**Why important:** Travel fatigue affects performance
- 0-500 miles: No impact
- 500-1500 miles: -1 to -2 points
- 1500+ miles: -2 to -3 points
- Time zone changes: Additional fatigue

**Expected improvement:** 0.5-1.0 points MAE reduction

**Implementation:**
```python
# Need team arena coordinates or geocoding
df['home_travel_miles'] = get_travel_distance(home_last_city, home_arena)
df['away_travel_miles'] = get_travel_distance(away_last_city, home_arena)
df['away_travel_category'] = pd.cut(df['away_travel_miles'], 
                                        bins=[0, 500, 1500, 10000],
                                        labels=['short', 'medium', 'long'])
```

#### 4. Recent Form (High Impact, Easy)
**What:** Performance in last N games (3, 5, 10)

**Why important:** Teams on hot/cold streaks continue
- Last 5 games: Current form indicator
- Momentum effect documented in sports
- Captures short-term rating changes

**Expected improvement:** 1.0-2.0 points MAE reduction

**Implementation:**
```python
# For each team, calculate last 5 games before current game
df['home_recent_points'] = avg_last_n_games(home_id, game_date, n=5)
df['away_recent_points'] = avg_last_n_games(away_id, game_date, n=5)
df['home_recent_margin'] = avg_margin_last_n_games(home_id, game_date, n=5)
df['away_recent_margin'] = avg_margin_last_n_games(away_id, game_date, n=5)
df['home_recent_wins'] = wins_last_n_games(home_id, game_date, n=5)
df['away_recent_wins'] = wins_last_n_games(away_id, game_date, n=5)
df['form_diff'] = df['home_recent_wins'] - df['away_recent_wins']
```

#### 5. Head-to-Head History (Medium Impact, Easy)
**What:** Historical matchup results

**Why important:** Style mismatches affect outcomes
- Some teams historically dominate others
- Offensive style vs defensive style
- Coaching matchups

**Expected improvement:** 0.3-0.8 points MAE reduction

**Implementation:**
```python
df['h2h_home_wins'] = head_to_head_wins(home_id, away_id, before_date)
df['h2h_away_wins'] = head_to_head_wins(away_id, home_id, before_date)
df['h2h_total_games'] = df['h2h_home_wins'] + df['h2h_away_wins']
df['h2h_home_win_pct'] = df['h2h_home_wins'] / df['h2h_total_games']
```

#### 6. Schedule Strength (Medium Impact, Medium)
**What:** Average rating of recent opponents

**Why important:** Wins vs bad teams ≠ true strength
- Beating easy teams inflates ratings
- Playing tough schedule = lower margin but better team

**Expected improvement:** 0.5-1.0 points MAE reduction

**Implementation:**
```python
# Average opponent rating from last 10 games
df['home_sos_rating'] = avg_opponent_rating(home_id, last_n=10)
df['away_sos_rating'] = avg_opponent_rating(away_id, last_n=10)
df['sos_diff'] = df['home_sos_rating'] - df['away_sos_rating']
```

#### 7. Season Phase (Low-Medium Impact, Easy)
**What:** Point in season (early, mid, late, playoffs)

**Why important:** Teams improve/regress over season
- Early: Roster learning
- Mid: Consistent performance
- Late: Tanking or playoff push
- Playoffs: Different dynamics

**Expected improvement:** 0.2-0.5 points MAE reduction

**Implementation:**
```python
# Map game date to season phase
df['games_played'] = total_games_for_team_before_date(team_id, game_date)
df['season_phase'] = pd.cut(df['games_played'],
                             bins=[0, 20, 50, 82],
                             labels=['early', 'mid', 'late'])
```

#### 8. Injury Impact (High Impact, Hard - need data source)
**What:** Key players injured/resting

**Why important:** Stars drive team performance
- Missing 1 star: -3 to -5 points
- Missing multiple: -5 to -8 points
- Role players: -1 to -2 points

**Expected improvement:** 1.0-2.5 points MAE reduction

**Implementation:**
```python
# Would need injury data API (ESPN, NBA, etc.)
df['home_injury_score'] = injury_impact_score(home_id, game_date)
df['away_injury_score'] = injury_impact_score(away_id, game_date)
df['injury_diff'] = df['home_injury_score'] - df['away_injury_score']
```

#### 9. Player-Level Features (High Impact, Hard)
**What:** Individual player stats and minutes

**Why important:** Stars matter more than depth
- Top 3 players drive ~60% of team offense
- Usage rate, efficiency, matchups
- Rest for key players

**Expected improvement:** 1.5-3.0 points MAE reduction

**Implementation:**
```python
# Aggregate player boxscores by team
df['home_top3_usage'] = avg_usage_top3_players(home_id, game_date)
df['away_top3_usage'] = avg_usage_top3_players(away_id, game_date)
df['home_star_efficiency'] = avg_efficiency_stars(home_id, game_date)
```

#### 10. Advanced Team Stats (Medium Impact, Easy)
**What:** Additional team efficiency metrics

**Why important:** More complete picture of team ability
- Offensive rating (already have)
- Defensive rating (already have)
- Net rating (off - def)
- Effective field goal % (already have)
- True shooting % (TS%)
- Assist ratio, turnover ratio

**Expected improvement:** 0.3-0.8 points MAE reduction

**Implementation:**
```python
df['home_net_rating'] = df['home_off_rating'] - df['home_def_rating']
df['away_net_rating'] = df['away_off_rating'] - df['away_def_rating']
df['net_rating_diff'] = df['home_net_rating'] - df['away_net_rating']
df['home_ts_pct'] = points / (2 * (fga + 0.44 * fta))
```

---

## Part 2: ML Model Improvements

### Current Models
- Ridge Regression (total)
- Random Forest (margin)

### Alternative Models to Test

#### 1. XGBoost (High Impact, Already imported)
**What:** Gradient boosting with regularization

**Why better than current:**
- Handles non-linear relationships
- Built-in feature importance
- Regularization prevents overfitting
- Fast training
- Often SOTA for tabular data

**Expected improvement:** 0.5-1.5 points MAE reduction

**Key hyperparameters:**
```python
XGBRegressor(
    n_estimators=200-500,      # Number of trees
    max_depth=3-6,             # Tree depth
    learning_rate=0.05-0.1,    # Step size
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,        # Column sampling
    reg_alpha=0.1-1.0,         # L1 regularization
    reg_lambda=1.0-5.0,         # L2 regularization
    min_child_weight=1-5,         # Regularization
    gamma=0-0.5,                 # Minimum loss reduction
)
```

**Implementation priority:** HIGH (easy to add, likely big impact)

#### 2. LightGBM (High Impact, Already imported)
**What:** Gradient boosting with histogram-based training

**Why better than current:**
- Faster than XGBoost
- Handles categorical features natively
- Lower memory usage
- Often SOTA for tabular data

**Expected improvement:** 0.5-1.5 points MAE reduction

**Key hyperparameters:**
```python
LGBMRegressor(
    n_estimators=200-500,
    max_depth=3-6,
    learning_rate=0.05-0.1,
    num_leaves=31-127,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_data_in_leaf=20-50,
)
```

**Implementation priority:** HIGH (easy to add, likely big impact)

#### 3. CatBoost (High Impact, Need to install)
**What:** Gradient boosting with ordered boosting

**Why better than current:**
- Best for categorical features
- Automatic handling of categorical data
- Less hyperparameter tuning needed
- Often outperforms XGBoost/LightGBM

**Expected improvement:** 0.5-2.0 points MAE reduction

**Key hyperparameters:**
```python
CatBoostRegressor(
    depth=6-10,
    learning_rate=0.05-0.1,
    iterations=500-1000,
    l2_leaf_reg=1-10,
    bagging_temperature=0-1,
    border_count=128-254,
)
```

**Implementation priority:** MEDIUM (need pip install catboost, but likely worth it)

#### 4. Neural Networks (Medium-High Impact, Hard)
**What:** Deep learning models

**Why better than current:**
- Captures complex interactions
- Can learn non-linear patterns
- Embeddings for categorical features

**Expected improvement:** 1.0-2.5 points MAE reduction

**Architecture:**
```python
# MLP for tabular data
nn = Sequential([
    Dense(64, activation='relu', input_dim=34),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Output
])
```

**Implementation priority:** LOW (harder, more complex, may not beat GBM for this data size)

#### 5. Ensembling (High Impact, Medium difficulty)
**What:** Combine multiple model predictions

**Why better than current:**
- Reduces variance
- Combines strengths of different models
- More robust predictions

**Methods:**
```python
# Simple average
prediction = (xgb_pred + lgb_pred + rf_pred) / 3

# Weighted average (by validation MAE)
weights = [1/mae_xgb, 1/mae_lgb, 1/mae_rf]
weights = weights / sum(weights)
prediction = xgb_pred * weights[0] + lgb_pred * weights[1] + ...

# Stacking (train meta-model)
base_preds = [xgb_pred, lgb_pred, rf_pred]
meta_features = base_preds
final_prediction = meta_model.predict(meta_features)
```

**Expected improvement:** 0.5-1.0 points MAE reduction

**Implementation priority:** HIGH (easy to do, proven to work)

---

## Part 3: Training Improvements

### Current Method
- Simple 70/15/15 time-based split
- Default hyperparameters
- No hyperparameter tuning

### Improvements

#### 1. Time Series Cross-Validation (Medium Impact, Medium)
**What:** Rolling validation that respects time

**Why better than current:**
- More realistic performance estimate
- Prevents look-ahead bias
- Better than simple split

**Implementation:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # Train and validate
```

**Expected improvement:** 0.2-0.8 points MAE reduction (better model selection)

#### 2. Hyperparameter Tuning (High Impact, Medium)
**What:** Optimize model parameters

**Why better than current:**
- Default params rarely optimal
- Custom tuning for your data
- Significant gains possible

**Methods:**
```python
# Grid Search (exhaustive but slow)
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [100, 200, 500],
              'max_depth': [3, 5, 7],
              'learning_rate': [0.05, 0.1, 0.2]}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X, y)

# Randomized Search (faster)
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5)

# Bayesian Optimization (most efficient)
from skopt import BayesSearchCV
bayes = BayesSearchCV(model, param_space, n_iter=30, cv=5)
```

**Expected improvement:** 0.5-1.5 points MAE reduction

#### 3. Feature Selection (Medium Impact, Easy)
**What:** Remove irrelevant/noisy features

**Why better than current:**
- Reduces overfitting
- Faster training
- Better generalization

**Methods:**
```python
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(model, n_features_to_select=20)
rfe.fit(X, y)

# SelectFromModel (tree-based)
from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(RandomForest(), threshold='median')
X_selected = selector.fit_transform(X, y)

# Permutation Importance
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10)
```

**Expected improvement:** 0.3-0.8 points MAE reduction

---

## Part 4: Advanced Techniques

#### 1. Calibration Improvements
**What:** Better probability calibration

**Why important:**
- Improves betting decisions
- Better confidence estimates
- Platt scaling for tree models

**Implementation:**
```python
from sklearn.calibration import CalibratedClassifierCV
# For classification (win probability)
calibrated = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
```

**Expected improvement:** 1-2% winner accuracy improvement

#### 2. Monte Carlo Simulation
**What:** Generate thousands of simulated games

**Why better:**
- Provides confidence intervals
- Accounts for uncertainty
- Better risk management

**Implementation:**
```python
# Run 10,000 simulations
simulations = []
for _ in range(10000):
    # Add noise to predictions
    sim_total = pred_total + np.random.normal(0, total_error_std)
    sim_margin = pred_margin + np.random.normal(0, margin_error_std)
    simulations.append((sim_total, sim_margin))

# Get percentiles
total_5th = np.percentile([s[0] for s in simulations], 5)
total_95th = np.percentile([s[0] for s in simulations], 95)
```

**Expected improvement:** Better risk management, not direct MAE improvement

---

## Part 5: Prioritized Implementation Plan

### Phase A: Quick Wins (Easy, High Impact) - DO FIRST

#### Phase 9: Add Rest & B2B Features
**Implementation time:** 2-3 hours
**Expected improvement:** 1.0-2.5 points MAE
**Features to add:**
- rest_days (home/away)
- rest_days_diff
- is_b2b (home/away)
- b2b_x_home_away
- recent_form_5games (points, margin, wins)

**Impact:** HIGH (rest and fatigue are huge factors)

#### Phase 10: Train XGBoost & LightGBM
**Implementation time:** 1-2 hours
**Expected improvement:** 0.5-1.5 points MAE
**Models to test:**
- XGBoost with tuned parameters
- LightGBM with tuned parameters
- Compare to Ridge and RF

**Impact:** HIGH (GBM models usually beat baseline)

#### Phase 11: Build Ensemble
**Implementation time:** 1-2 hours
**Expected improvement:** 0.5-1.0 points MAE
**Ensemble methods:**
- Simple average (best 3 models)
- Weighted average (by validation MAE)
- Compare individual vs ensemble

**Impact:** HIGH (ensembling consistently improves results)

### Phase B: Medium Effort (Medium Impact) - DO SECOND

#### Phase 12: Add Advanced Features
**Implementation time:** 4-6 hours
**Expected improvement:** 1.0-2.0 points MAE
**Features to add:**
- Head-to-head history
- Schedule strength
- Net rating
- Advanced team stats

**Impact:** MEDIUM (additional context, but smaller effect than rest/B2B)

#### Phase 13: Hyperparameter Tuning
**Implementation time:** 3-4 hours
**Expected improvement:** 0.5-1.5 points MAE
**Methods:**
- Bayesian optimization for top models
- Grid search on narrowed ranges

**Impact:** MEDIUM (optimizes what you already have)

### Phase C: Advanced (High Effort, High Impact) - DO THIRD

#### Phase 14: Travel Distance
**Implementation time:** 6-8 hours
**Expected improvement:** 0.5-1.0 points MAE
**Requirements:**
- Arena coordinates
- Distance calculation API

**Impact:** MEDIUM (impactful but hard to implement)

#### Phase 15: CatBoost + Neural Network
**Implementation time:** 4-6 hours
**Expected improvement:** 0.5-2.0 points MAE
**Models:**
- CatBoost (if available data size supports it)
- Simple neural network

**Impact:** MEDIUM (more complex, may not beat GBM)

---

## Expected Cumulative Impact

| Phase | Improvement | Cumulative MAE | Time to Implement |
|--------|-------------|------------------|-------------------|
| Current baseline | 15.54 (total) / 12.00 (margin) | - | - |
| Phase 9 (Rest/B2B/Recent) | -1.5 | 14.04 / 10.50 | 2-3h |
| Phase 10 (XGBoost/LightGBM) | -1.0 | 13.04 / 9.50 | 1-2h |
| Phase 11 (Ensemble) | -0.5 | 12.54 / 9.00 | 1-2h |
| **TOTAL (Quick Wins)** | **-3.0** | **12.54 / 9.00** | **4-7h** |
| Phase 12 (Advanced Features) | -0.5 | 12.04 / 8.50 | 4-6h |
| Phase 13 (Hyperparameter Tuning) | -0.5 | 11.54 / 8.00 | 3-4h |
| **TOTAL (with Medium)** | **-4.0** | **11.54 / 8.00** | **11-17h** |
| Phase 14 (Travel) | -0.3 | 11.24 / 7.70 | 6-8h |
| Phase 15 (CatBoost/NN) | -0.5 | 10.74 / 7.20 | 4-6h |
| **TOTAL (All)** | **-4.8** | **10.74 / 7.20** | **21-31h** |

**Winner accuracy goal:**
- Current: 61.0%
- After Quick Wins: 65-66%
- After All: 67-68%

---

## Part 6: Alternative Data Sources

### Potential API Data Sources

#### 1. Basketball-Reference
- **URL:** basketball-reference.com
- **Data:** Advanced stats, team ratings, player stats
- **Cost:** Free
- **Difficulty:** Web scraping needed

#### 2. NBA API (Official)
- **URL:** api.nba.com
- **Data:** Scores, schedules, boxscores
- **Cost:** Free (rate limited)
- **Difficulty:** API integration

#### 3. ESPN
- **URL:** site.api.espn.com/apis
- **Data:** Injuries, transactions
- **Cost:** Free
- **Difficulty:** API integration

#### 4. Odds APIs
- **URL:** the-odds-api.com, oddsportal.com
- **Data:** Betting lines, Vegas consensus
- **Cost:** Some free, some paid
- **Difficulty:** API integration

---

## Conclusion

### Best ROI Improvements (Quick Wins)
1. **Add Rest Days & Back-to-Back** (2-3h, -1.5 MAE)
2. **Train XGBoost** (1-2h, -1.0 MAE)
3. **Add Recent Form** (1-2h, -0.5 MAE)
4. **Build Ensemble** (1-2h, -0.5 MAE)

**Total: 4-7 hours for 3.0 points MAE reduction!**

### Long-term Improvements
1. Travel distance
2. Injury data integration
3. Hyperparameter tuning
4. Advanced features

---

**Research completed. Ready for implementation!**
