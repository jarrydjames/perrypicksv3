# PerryPicks v3 - Implementation Plan for Statistically Valid System

**Date:** January 29, 2026  
**Purpose:** Phased implementation of execution specification requirements  
**Reference:** `execution_specification_for_statistically_valid_nba_forecasting_system.md`

---

## Executive Summary

**Current Status:** Production-ready baseline (Ridge, RF, GBT) with basic walkforward backtest  
**Gap Analysis:** Missing 6 of 8 critical modules (75% completeness)  
**Recommendation:** 4-Phase implementation over 4-6 weeks  
**Priority:** Data validation → Leakage detection → Statistical testing → Advanced models

---

## Gap Analysis: Current vs Required

### ✅ Currently Implemented

| Module | Status | Notes |
|--------|---------|---------|
| Model Registry (partial) | ✅ | Ridge, RF, GBT - missing LightGBM, CatBoost, NN |
| Walkforward Backtest (partial) | ✅ | Basic implementation, missing statistical testing |
| Uncertainty (basic) | ✅ | Gaussian only, missing conformal |
| Production Model | ✅ | Ridge working in production |

### ❌ Missing Critical Modules

| Module | Priority | Impact |
|---------|----------|---------|
| Data Validation Gate | **CRITICAL** | Risk of training on corrupt/leaky data |
| Leakage Sentinels | **CRITICAL** | Undetected data leaks could invalidate results |
| Statistical Testing (Bootstrap) | **HIGH** | No confidence intervals, can't measure significance |
| Statistical Testing (Diebold-Mariano) | **HIGH** | Can't prove model improvements are real |
| Conformal Uncertainty | **MEDIUM** | Current Gaussian may be miscalibrated |
| Model Registry (expanded) | **MEDIUM** | Can't test LightGBM, CatBoost, NN candidates |
| Experiment Tracking | **LOW** | Manual, no reproducible experiment registry |
| Drift Monitoring | **LOW** | No production monitoring, risk of decay |

**Completeness:** 25% (2 of 8 modules)  
**Urgency:** HIGH - Data validation and leakage detection block all downstream work

---

## Phase 1: Data Validation Gate (Week 1)

**Priority:** CRITICAL  
**Duration:** 5-7 days  
**Goal:** Hard-fail checks before any training/backtesting

### Tasks

#### 1.1 Schema & Dtype Checks
- [ ] Validate `gameTimeUTC` is timezone-aware UTC datetime
- [ ] Validate all feature columns are numeric (not object/string)
- [ ] Validate `season_end_yy` is integer
- [ ] Validate `game_id` is string

#### 1.2 Primary Key Integrity
- [ ] Ensure unique `(season_end_yy, game_id)` pairs
- [ ] Check home_team_id != away_team_id for all games
- [ ] No duplicate game_ids within same season

#### 1.3 Missingness & Completeness (Hard Fails)
```python
FAIL_THRESHOLDS:
- Targets (h2_total, h2_margin): 0% missing
- Baseline features (h1_*): ≤ 0.1% missing each
- Temporal features: ≤ 2% missing each (early season games)
```

- [ ] Implement missingness check function
- [ ] Fail if any threshold exceeded
- [ ] Generate missingness heatmap artifact
- [ ] Log per-season and per-month missingness

#### 1.4 Temporal Ordering Integrity
- [ ] Sort by `(gameTimeUTC, season_end_yy, game_id)`
- [ ] Verify monotonic increasing (allow ties)
- [ ] Count and report tied timestamps
- [ ] Generate ordering checksum (hash of sorted indices)

#### 1.5 Season/Regime Diagnostics (Warning Report)
- [ ] Count games per season
- [ ] Flag if playoffs mixed with regular season
- [ ] Flag if cross-season rolling enabled
- [ ] Report as WARNING (not fail, but surface in logs)

#### 1.6 Output
- [ ] Create data validation report (PASS/FAIL + caveats)
- [ ] Save stable sorted dataset checksum
- [ ] Abort downstream steps if FAIL

**Deviation:** None - This is blocking, must implement as specified.

---

## Phase 2: Leakage Detection Sentinels (Week 2)

**Priority:** CRITICAL  
**Duration:** 5-7 days  
**Goal:** Detect and prevent all forms of data leakage

### Tasks

#### 2.1 Sentinel A: Forward-Only Rolling Check
```python
# For each team, for each game i:
for team in all_teams:
    for i in range(len(team_games)):
        rolling_contributors = team_games[i-k:i]  # Only use games BEFORE i
        assert max(rolling_contributors.index) < i, "Leakage detected!"
```

- [ ] Implement forward-only rolling verification
- [ ] Run on current rolling features dataset
- [ ] Hard FAIL if any leakage detected
- [ ] Fix by recomputing rolling features in single forward pass

**Deviation:** If current rolling features pass without modification, skip recomputation (already correct).

#### 2.2 Sentinel B: Suspicious Correlation Check
```python
# Compute |corr(feature, target)| for all features
# Flag features with |correlation| > 0.95 for manual review
```

- [ ] Compute correlation matrix
- [ ] Flag features with |correlation| > 0.95
- [ ] Manual review of flagged features (likely legitimate for halftime)
- [ ] Document findings in leakage report

**Deviation:** For halftime prediction, high correlation with target is expected (h1_total correlates with h2_total). This is a WARNING, not a FAIL.

#### 2.3 Sentinel C: Time-Shift Placebo Test
```python
# Train model to predict y_{t+1} from features at time t
# Performance should collapse to noise
# If MAE < 50% of baseline, leakage suspected
```

- [ ] Implement time-shifted target function
- [ ] Train Ridge on shifted targets
- [ ] Evaluate on same test set
- [ ] Hard FAIL if MAE < 50% of baseline MAE

**Deviation:** This is critical - must implement. For time-series forecasting, time-shift placebo detects any form of leakage where future information is encoded in features.

#### 2.4 Leakage Report
- [ ] Compile all sentinel results
- [ ] Classify as PASS/WARN/FAIL
- [ ] Generate artifact with detailed findings
- [ ] Block downstream steps if any FAIL

---

## Phase 3: Statistical Testing Framework (Week 3-4)

**Priority:** HIGH  
**Duration:** 7-10 days  
**Goal:** Measure confidence intervals, test model significance

### Tasks

#### 3.1 Primary Metric: MAE
- [ ] Keep current MAE calculation
- [ ] Ensure paired evaluation (per-game loss differentials)

#### 3.2 Secondary Metrics
- [ ] RMSE (already computed)
- [ ] Margin MAE (already computed)
- [ ] Margin RMSE (already computed)
- [ ] Consider adding CRPS if probabilistic outputs

#### 3.3 Paired Loss Differentials
```python
# For each model vs baseline comparison:
L_baseline_i = loss(y_i, yhat_baseline_i)
L_new_i = loss(y_i, yhat_new_i)
d_i = L_new_i - L_baseline_i

# Summary: mean(d), median(d), positive %
```

- [ ] Implement paired differential function
- [ ] Compute per-game improvements
- [ ] Summarize: mean, median, % positive
- [ ] Output in statistical test report

#### 3.4 Block Bootstrap (Time-Valid CI)
```python
# Choose block size B = 200 (same as test size)
# Sample contiguous blocks with replacement until length N
# Repeat R = 1000 times
# Compute mean(d) distribution
# Output: mean(d), 95% CI, P(improvement) = P(mean(d) < 0)
```

- [ ] Implement block bootstrap function
- [ ] Run on best model vs baseline
- [ ] Compute 95% confidence intervals for MAE
- [ ] Calculate probability of improvement

**Deviation:** Block size B=200 aligns with test size. Increase to R=1000 from spec's R=100 for tighter CIs.

#### 3.5 Diebold-Mariano Test
```python
# Apply DM statistic to loss differential series d_i
# Use Newey-West variance estimate to handle autocorrelation
# Report DM statistic and p-value
```

- [ ] Implement Diebold-Mariano function
- [ ] Apply to model comparisons
- [ ] Use Newey-West variance for autocorrelation
- [ ] Report DM statistic and p-value

**Deviation:** None - implement as specified.

#### 3.6 Statistical Test Report
- [ ] Compile bootstrap results (CI, P(improvement))
- [ ] Compile DM results (statistic, p-value)
- [ ] Interpret significance (p < 0.05 = significant)
- [ ] Generate artifact

---

## Phase 4: Conformal Uncertainty (Week 5-6)

**Priority:** MEDIUM  
**Duration:** 7-10 days  
**Goal:** Replace Gaussian with time-series safe conformal intervals

### Tasks

#### 4.1 Sliding Window Conformal
```python
# For each outer fold:
#   1. Choose calibration window W = 100 (most recent)
#   2. Predict on calibration window
#   3. Residuals r_j = |y_j - yhat_j|
#   4. For target coverage (1-α), compute quantile q
#   5. Interval: [yhat - q, yhat + q]
# Test set: Predict, compute residuals, apply intervals from calibration
```

- [ ] Implement sliding window conformal
- [ ] Compute per-fold calibration
- [ ] Apply intervals to test set
- [ ] Ensure calibration window < test window (no leakage)

#### 4.2 Evaluation Table
```python
# For each target (total, margin):
#   Coverage at 50/60/70/80/90/95
#   Average width
#   Winkler score (weighted coverage penalty)
#   Interval score
```

- [ ] Implement coverage calculation
- [ ] Implement width calculation
- [ ] Implement Winkler score
- [ ] Generate evaluation table per run

#### 4.3 Conditional Coverage
```python
# Bins:
#   - close games: |halftime_margin| < 5
#   - blowouts: |halftime_margin| > 15
#   - high pace: top 20% pace proxy
#   - low pace: bottom 20% pace proxy
# Compute coverage and width by bin
```

- [ ] Implement binning functions
- [ ] Compute coverage per bin
- [ ] Report calibration quality by game type
- [ ] Flag bins with poor coverage

#### 4.4 Fallback to Gaussian
- [ ] If conformal underperforms, log warning
- [ ] Keep Gaussian as fallback
- [ ] Document rationale

**Deviation:** If Gaussian performs similarly to conformal, skip conformal implementation (adds complexity without benefit).

---

## Phase 5: Expanded Model Registry (Week 7)

**Priority:** MEDIUM  
**Duration:** 5-7 days  
**Goal:** Enable testing of LightGBM, CatBoost, Neural Networks

### Tasks

#### 5.1 LightGBM Implementation
```python
class LightGBMTwoHeadModel(BaseTwoHeadModel):
    name = "lgbm"
    
    def __init__(self, *, num_leaves=31, learning_rate=0.03, 
                 n_estimators=1000, reg_lambda=1.0):
        # Configurable params per spec
```

- [ ] Create LightGBMTwoHeadModel class
- [ ] Implement two-head prediction (total, margin)
- [ ] Add to model registry
- [ ] Implement residual sigma calculation
- [ ] Test with nested tuning (if time permits)

**Deviation:** If LightGBM not available in environment, skip and document.

#### 5.2 CatBoost Implementation
```python
class CatBoostTwoHeadModel(BaseTwoHeadModel):
    name = "catboost"
    
    def __init__(self, *, depth=6, learning_rate=0.03,
                 n_estimators=1000, l2_reg=3.0):
        # Configurable params
```

- [ ] Check if CatBoost class already exists (it does!)
- [ ] Ensure it implements correct two-head interface
- [ ] Add to model registry
- [ ] Test with categorical team IDs

**Deviation:** CatBoost exists but may not be in `default_models()`. Add it explicitly.

#### 5.3 Tabular Neural Network
```python
class TabularMLPTwoHeadModel(BaseTwoHeadModel):
    name = "tabular_mlp"
    
    def __init__(self, *, embedding_dim=8, hidden_layers=[256, 128, 64],
                 dropout=0.2, learning_rate=1e-3, weight_decay=1e-4):
        # Configurable per spec
```

- [ ] Create TabularMLPTwoHeadModel class
- [ ] Implement team embeddings
- [ ] Implement two-head MLP architecture
- [ ] Implement early stopping
- [ ] Add to model registry

**Deviation:** High complexity, lowest priority. Implement if time permits.

#### 5.4 Unified Interface
- [ ] Ensure all models implement same `BaseTwoHeadModel` interface
- [ ] Ensure `fit()` and `predict_heads()` methods
- [ ] Ensure `trained_heads()` returns residuals for uncertainty
- [ ] Create factory function to instantiate models by config

---

## Phase 6: Experiment Tracking System (Week 8)

**Priority:** LOW  
**Duration:** 5-7 days  
**Goal:** Reproducible experiment registry

### Tasks

#### 6.1 Experiment Registration
```python
experiments = {
    "exp_001": {
        "hypothesis": "Conformal improves over Gaussian",
        "change": "Replace Gaussian with sliding window conformal",
        "baseline_model": "ridge",
        "challenger_model": "ridge_conformal",
        "metrics": ["total_mae", "coverage_95"],
        "statistical_test": "block_bootstrap",
        "go_no_go": "CI improvement > 2% AND no coverage degradation"
    },
    "exp_002": {
        "hypothesis": "LightGBM beats GBT",
        "change": "Add LightGBM to registry",
        "baseline_model": "gbt",
        "challenger_model": "lgbm",
        "metrics": ["total_mae", "total_rmse"],
        "statistical_test": "dm_test",
        "go_no_go": "MAE improvement > 1% AND p < 0.05"
    },
    # ... pre-register more experiments
}
```

- [ ] Create experiment registry YAML/JSON file
- [ ] Implement experiment runner function
- [ ] Add logging (dataset hash, model config, seeds)
- [ ] Pre-register 6 experiments from spec

**Deviation:** Use simple JSON registry. Skip complex database.

#### 6.2 Reproducibility
- [ ] Log dataset hash/version
- [ ] Log fold indices checksum
- [ ] Log model config and random seeds
- [ ] Save predictions and residuals
- [ ] Enable deterministic execution

---

## Phase 7: Production Training + Drift Monitoring (Week 9)

**Priority:** LOW  
**Duration:** 7-10 days  
**Goal:** Production reliability and monitoring

### Tasks

#### 7.1 Retraining Cadence
```python
# In-season (games with season_end_yy == current year):
#   Weekly minimum (preferred: daily if compute allows)

# Off-season:
#   Retrain only when new season data arrives
```

- [ ] Implement weekly retraining check
- [ ] Add `--retrain-frequency` flag to training script
- [ ] Check season_end_yy for in-season games
- [ ] Log retraining timestamps

**Deviation:** Current manual workflow works. Weekly cadence is reasonable but may not be optimal for daily updates.

#### 7.2 PSI Monitoring
```python
def psi(feature_train, feature_test, bins=10):
    """Population Stability Index - measures feature drift"""
    # Discretize both distributions into bins
    # Compute PSI = sum((train_i - test_i)^2) / (train_i)
    # PSI > 0.2 indicates drift
```

- [ ] Implement PSI function
- [ ] Compute PSI for key features (eFG, foul rate)
- [ ] Add to drift monitoring report
- [ ] Alert if PSI > 0.2

**Deviation:** PSI requires calibration period. Implement with fixed 30-day calibration window (first 30 games after retrain).

#### 7.3 Coverage Drift
```python
# Track conformal coverage over last N games
# Flag if coverage deviates > 3% from target for 2 consecutive checks
```

- [ ] Implement coverage tracking
- [ ] Compare actual vs target (95%)
- [ ] Compute deviation
- [ ] Alert on sustained deviation

**Deviation:** Conformal coverage monitoring requires Phase 4 to complete first.

---

## Implementation Timeline

### Week 1: Phase 1 - Data Validation Gate
- Days 1-3: Implement schema, dtype, key integrity, missingness checks
- Days 4-5: Implement temporal ordering and checksum
- Days 6-7: Implement diagnostics and report
- **Deliverable:** Data validation module, PASS/FAIL gate

### Week 2: Phase 2 - Leakage Detection
- Days 8-10: Implement forward-only rolling check
- Days 11-12: Implement correlation check
- Days 13-14: Implement time-shift placebo
- **Deliverable:** Leakage sentinels, fix any detected leaks

### Week 3: Phase 3 - Statistical Testing
- Days 15-17: Implement paired differentials
- Days 18-19: Implement block bootstrap
- Days 20-21: Implement Diebold-Mariano test
- **Deliverable:** Statistical testing framework, confidence intervals

### Week 4: Phase 4 - Conformal Uncertainty
- Days 22-24: Implement sliding window conformal
- Days 25-26: Implement evaluation table
- Days 27-28: Implement conditional coverage
- **Deliverable:** Conformal uncertainty module, improved calibration

### Week 5: Phase 5 - Model Registry
- Days 29-31: Implement LightGBM
- Days 32-33: Implement CatBoost integration
- Days 34-35: Implement Tabular MLP (if time)
- **Deliverable:** Expanded model registry

### Week 6: Phase 6 - Experiment Tracking
- Days 36-38: Implement experiment registry
- Days 39-40: Implement reproducibility logging
- **Deliverable:** Experiment tracking system

### Week 7: Phase 7 - Drift Monitoring
- Days 41-43: Implement PSI monitoring
- Days 44-45: Implement coverage drift
- Days 46-47: Implement retraining cadence
- **Deliverable:** Production monitoring module

### Week 8: Buffer & Integration
- Days 48-50: End-to-end testing
- Days 51-52: Documentation updates
- Days 53-54: Git push and deployment
- **Deliverable:** Production-ready statistically valid system

---

## Deviations from Execution Specification

### ✅ Accepted (Following Spec)
1. **Walkforward backtest strategy** - Using expanding window (specified)
2. **Model registry format** - Ridge, RF, GBT follow interface
3. **Two-head predictions** - Predicting total and margin
4. **Hard fail thresholds** - Will implement as specified
5. **Block bootstrap** - Will implement with R=1000 (not R=100 for tighter CIs)
6. **Diebold-Mariano** - Will implement with Newey-West variance
7. **Conformal approach** - Sliding window per spec

### ⚠️  Modified (Justified Deviations)

#### Deviation 1: Block Size in Bootstrap
**Spec says:** R=100  
**Plan says:** R=1000

**Justification:** Spec's R=100 is extremely conservative (wide CIs). For practical NBA forecasting, R=1000 provides tighter, more informative confidence intervals without compromising statistical validity. Tighter CIs are more actionable for decision-making.

#### Deviation 2: CatBoost Already Exists
**Spec says:** Add CatBoost to registry  
**Reality:** CatBoostTwoHeadModel class exists in `src/modeling/cat_models.py`

**Justification:** CatBoost is already implemented but not integrated into `default_models()` function. Integration task is minimal, not a full reimplementation.

#### Deviation 3: Time-Shift Placebo Complexity
**Spec says:** Implement time-shifted target, FAIL if MAE < 50% of baseline  
**Plan says:** Must implement

**Justification:** I will implement this as specified. The 50% threshold is reasonable for detecting significant leakage.

#### Deviation 4: Neural Network Complexity
**Spec says:** Implement Tabular MLP + embeddings  
**Plan says:** Lowest priority, implement if time permits

**Justification:** Neural networks add significant complexity. Given Ridge is performing well, gradient-based models may not provide meaningful gains. Will implement in buffer period (week 8) if time permits.

#### Deviation 5: PSI Calibration Period
**Spec says:** PSI requires calibration period  
**Plan says:** Fixed 30-day window

**Justification:** Fixed 30-day calibration window is practical for NBA daily data. Ensures PSI is computed on meaningful sample size without requiring complex rolling logic.

#### Deviation 6: Retraining Cadence
**Spec says:** Weekly minimum in-season  
**Plan says:** Weekly is reasonable but may not be optimal

**Justification:** Weekly retraining is practical for current compute constraints and data freshness. Daily would require ~3x compute. Current system uses manual retraining which can be automated to weekly without significant workflow changes.

#### Deviation 7: NGBoost and Quantile GBDT
**Spec says:** Add NGBoost and Quantile GBDT to registry  
**Plan says:** Add to experiment registry

**Justification:** NGBoost and Quantile GBDT are advanced models requiring additional libraries. Adding them would increase complexity. Given current performance (Ridge MAE ~4.67), simpler models are competitive. Will add to experiment registry but lower priority.

#### Deviation 8: Hybrid Stacking
**Spec says:** Implement time-safe stacking with meta-model  
**Plan says:** Add to experiment registry

**Justification:** Stacking provides marginal gains (1-3%) but doubles complexity. For production simplicity, prefer best single model. Will implement as experiment to evaluate, but unlikely to become default.

#### Deviation 9: Canonical Feature List
**Spec says:** Store canonical ordered feature list in artifacts  
**Plan says:** Add to implementation

**Justification:** For current system (17 features), manual list in code is sufficient. Canonical list provides value only when features are dynamic or numerous. Will implement in Phase 1 as part of data validation.

---

## Risk Management

### High Risks
- **Risk:** Time-shift placebo detects legitimate correlation (h1_total correlates with h2_total)
- **Mitigation:** Treat as WARNING, not FAIL. Manual review required.

- **Risk:** PSI fails early in season (insufficient calibration data)
- **Mitigation:** Use 30-game minimum calibration window. Flag low-PSI results with "insufficient data" warning.

### Medium Risks
- **Risk:** Conformal underperforms Gaussian (possible)
- **Mitigation:** Evaluate both. Keep best as default.

- **Risk:** Block bootstrap computationally expensive (1000 samples × 1000 bootstrap)
- **Mitigation:** Use R=500 for initial testing, increase to 1000 for production runs.

### Low Risks
- **Risk:** Neural networks fail to train due to insufficient data or hyperparameters
- **Mitigation:** Early stopping, aggressive regularization, fallback to Ridge.

- **Risk:** Experiment tracking becomes overhead without value
- **Mitigation:** Simple JSON registry, no database. Manual review weekly.

---

## Success Criteria

### Phase Completion Criteria
Each phase is COMPLETE when:
- [ ] All tasks implemented and tested
- [ ] Code passes tests (if added)
- [ ] Documentation updated
- [ ] Git commit with clear message

### Overall Project Success
Implementation is COMPLETE when:
- [ ] Phases 1-7 all complete (or justified deviations)
- [ ] End-to-end test passes
- [ ] Documentation complete
- [ ] Deployed to production (or production-ready artifact)

---

## Files to Create

```
src/
  validation/
    __init__.py
    data_validation.py    # Phase 1
    leakage_detection.py   # Phase 2
  statistical/
    __init__.py
    bootstrap.py             # Phase 3
    diebold_mariano.py      # Phase 3
  uncertainty/
    __init__.py
    conformal.py             # Phase 4
  models/
    lgbm_models.py          # Phase 5
    registry.py              # Phase 5 (unified interface)
  experiments/
    __init__.py
    registry.py              # Phase 6
    runner.py                # Phase 6
  monitoring/
    __init__.py
    drift.py                 # Phase 7

docs/
  implementation_plan.md      # This file
  phase_1_status.md         # Per-phase status
  phase_2_status.md
  ...
```

---

**Date:** January 29, 2026  
**Status:** IMPLEMENTATION PLAN COMPLETE  
**Next:** Begin Phase 1 execution or await user approval
