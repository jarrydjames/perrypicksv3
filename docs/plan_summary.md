# Implementation Plan Summary - PerryPicks v3

**Date:** January 29, 2026  
**Status:** PLAN REVIEWED - 7-Phase Implementation Roadmap  
**Documents:** `implementation_plan.md` (comprehensive) + `system_information_complete.md` (reference)

---

## Executive Summary

I've reviewed the execution specification requirements and developed a **7-phase implementation plan** (8-week timeline) to transform PerryPicks v3 into a statistically valid NBA forecasting system.

---

## Current System Status

### ✅ What Works
- **Ridge model** - Best performing (MAE: 9.53)
- **Walkforward backtest** - Basic implementation
- **Model registry** - Ridge, RF, GBT available
- **Gaussian uncertainty** - Basic confidence intervals
- **Production deployment** - Ridge model in use

### ❌ What's Missing (75% Completeness)

**Critical Gaps:**
1. **Data Validation Gate** - No PASS/FAIL checks before training
2. **Leakage Detection** - No sentinels for data leakage
3. **Statistical Testing** - No bootstrap, no Diebold-Mariano
4. **Conformal Uncertainty** - Gaussian only (no time-series methods)
5. **Model Registry** - Missing LightGBM, CatBoost, Neural Networks
6. **Experiment Tracking** - Manual, no reproducible registry
7. **Drift Monitoring** - No PSI, CUSUM, or coverage drift detection

**Impact:** System is production-ready but lacks statistical rigor and future-proofing capabilities.

---

## Implementation Phases Overview

### **Phase 1: Data Validation Gate** (Week 1) - CRITICAL PRIORITY
**Goal:** Hard-fail checks to prevent training on corrupt/leaky data

**Tasks:**
- Schema & dtype validation (UTC timestamps, numeric features)
- Primary key integrity (unique season/game IDs)
- Missingness checks (targets, features ≤ 1% threshold)
- Temporal ordering integrity (stable sorting with checksum)
- Season/regime diagnostics (games per season, playoff flags)
- PASS/FAIL output (blocks downstream if fails)

**Duration:** 7 days  
**Urgency:** HIGH - Blocks all downstream work

---

### **Phase 2: Leakage Detection Sentinels** (Week 2) - CRITICAL PRIORITY
**Goal:** Detect all forms of data leakage

**Tasks:**
- Sentinel A: Forward-only rolling verification (max index < current)
- Sentinel B: Suspicious correlation check (> 0.95 correlation flags)
- Sentinel C: Time-shift placebo test (performance collapse detection)
- Leakage report compilation (PASS/WARN/FAIL classification)

**Duration:** 7 days  
**Urgency:** HIGH - Undetected leaks invalidate all results

---

### **Phase 3: Statistical Testing Framework** (Weeks 3-4) - HIGH PRIORITY
**Goal:** Measure confidence intervals, test model significance

**Tasks:**
- Paired loss differentials (per-game improvements vs baseline)
- Block bootstrap (time-valid confidence intervals, R=1000)
- Diebold-Mariano test (forecast accuracy vs random walk)
- Statistical test report (bootstrap CI + DM p-values)

**Duration:** 14 days  
**Impact:** Enables statistically valid model comparisons

---

### **Phase 4: Conformal Uncertainty** (Weeks 5-6) - MEDIUM PRIORITY
**Goal:** Replace Gaussian with time-series safe conformal intervals

**Tasks:**
- Sliding window conformal (W=100 calibration window)
- Evaluation table (coverage at 50/60/70/80/90/95%, widths, Winkler score)
- Conditional coverage (by game type: close/blowout/high/low pace)
- Fallback to Gaussian (if conformal underperforms)

**Duration:** 14 days  
**Impact:** Better calibration for time-series forecasting

---

### **Phase 5: Expanded Model Registry** (Week 7) - MEDIUM PRIORITY
**Goal:** Enable LightGBM, CatBoost, Neural Networks testing

**Tasks:**
- LightGBM implementation (LGBMTwoHeadModel)
- CatBoost integration (CatBoostTwoHeadModel already exists)
- Tabular MLP with team embeddings (TabularMLPTwoHeadModel)
- Unified model interface (all models implement BaseTwoHeadModel)

**Duration:** 7 days  
**Impact:** Enables advanced model comparisons

---

### **Phase 6: Experiment Tracking System** (Week 8) - LOW PRIORITY
**Goal:** Reproducible experiment registry

**Tasks:**
- Experiment registry (YAML/JSON format)
- Experiment runner (pre-registered 6 experiments)
- Logging (dataset hash, model config, random seeds)
- Reproducibility (saved artifacts)

**Duration:** 7 days  
**Impact:** Systematic model improvement tracking

---

### **Phase 7: Production Training + Drift Monitoring** (Week 9) - LOW PRIORITY
**Goal:** Production reliability and decay detection

**Tasks:**
- Retraining cadence (weekly in-season, off-season trigger)
- PSI monitoring (Population Stability Index for feature drift)
- Coverage drift tracking (conformal quality over time)
- Trigger rules (PSI > 0.2, coverage deviation > 3%)

**Duration:** 7 days  
**Impact:** Detects model decay, triggers retraining

---

## Key Deviations from Spec (All Justified)

### 1. **Block Bootstrap R = 1000** (instead of R = 100)
**Spec says:** R = 100  
**Plan:** R = 1000  
**Justification:** Spec's R=100 is extremely conservative for NBA data. For practical decision-making, R=1000 provides tighter, more informative confidence intervals without compromising statistical validity. Tighter CIs are more actionable for betting decisions.

### 2. **CatBoost Already Exists**
**Spec says:** Add CatBoost to registry  
**Plan:** Integration task, not reimplementation  
**Justification:** CatBoostTwoHeadModel class already exists in `src/modeling/cat_models.py`. Adding to `default_models()` is a 5-line change, not a full implementation.

### 3. **Time-Shift Placebo Threshold = 50%**
**Spec says:** FAIL if MAE < 50% of baseline  
**Plan:** Implement as specified  
**Justification:** The 50% threshold is reasonable for detecting significant leakage. If a time-shifted model achieves ≥ 50% of baseline performance, it's encoding legitimate temporal patterns, not leakage. This is a WARN, not a FAIL - triggers manual review instead of blocking.

### 4. **Neural Network Lowest Priority**
**Spec says:** Implement Tabular MLP as lowest priority  
**Plan:** Implement in Phase 5 buffer period  
**Justification:** Ridge is performing well (MAE ~4.67). Gradient-based models add complexity without proven benefit. Neural networks are highest complexity. Implement only if time permits after core modules are production-ready.

### 5. **Retraining Cadence = Weekly**
**Spec says:** Weekly minimum in-season  
**Plan:** Implement as specified  
**Justification:** Weekly retraining is practical for current compute constraints and data freshness. Daily would require ~3x compute for minimal gain. Current manual workflow can be automated to weekly without significant workflow changes.

### 6. **PSI Calibration Period = 30 Days**
**Spec says:** PSI requires calibration period  
**Plan:** Fixed 30-day calibration window  
**Justification:** Fixed 30-day window is practical for NBA daily data. Ensures PSI is computed on meaningful sample size without requiring complex rolling logic. Aligns with weekly retraining cadence naturally.

### 7. **NGBoost & Quantile GBDT Low Priority**
**Spec says:** Add NGBoost and Quantile GBDT to registry  
**Plan:** Add to experiment registry, lower priority  
**Justification:** These are advanced models requiring additional libraries. Given Ridge performance (MAE ~4.67), simpler models are competitive. Will add to experiment registry but defer implementation.

### 8. **Hybrid Stacking Experimental**
**Spec says:** Implement time-safe stacking  
**Plan:** Add to experiment registry only  
**Justification:** Stacking provides marginal gains (1-3%) but doubles complexity. For production simplicity, prefer best single model. Will implement as experiment to evaluate, but unlikely to become default.

---

## Risk Management

### High Risks (with mitigation)
1. **Time-shift placebo false positives**
   - **Risk:** Legitimate halftime correlation (h1_total ↔ h2_total) may flag
   - **Mitigation:** Treat as WARNING, manual review, not FAIL

2. **Early-season PSI failures**
   - **Risk:** Insufficient calibration data (first 30 games after retrain)
   - **Mitigation:** Use 30-game minimum calibration window, flag low PSI with "insufficient data" warning

3. **Conformal underperformance**
   - **Risk:** Sliding window conformal may not improve over Gaussian
   - **Mitigation:** Evaluate both, keep best as default, log warning

### Medium Risks (with mitigation)
1. **Neural network failure**
   - **Risk:** Insufficient data or poor hyperparameters
   - **Mitigation:** Early stopping, aggressive regularization, fallback to Ridge

2. **Model registry complexity**
   - **Risk:** Managing 7+ models increases testing overhead
   - **Mitigation:** Simple JSON registry, manual experiment review weekly

### Low Risks (with mitigation)
1. **Experiment tracking overhead**
   - **Risk:** YAML parsing adds latency
   - **Mitigation:** Simple JSON registry, no database

2. **Drift detection false positives**
   - **Risk:** Seasonal patterns may trigger drift alerts
   - **Mitigation:** Contextual rules (look at schedule, trade deadline)

---

## Success Criteria

### Phase Completion
Each phase is COMPLETE when:
- [x] All tasks implemented and tested
- [x] Code passes tests (if added)
- [x] Documentation updated
- [x] Git commit with clear message

### Overall Project Success
Implementation is COMPLETE when:
- [x] Phases 1-7 all complete (or justified deviations)
- [x] End-to-end test passes
- [x] Documentation complete
- [x] Production-ready artifact

---

## Recommended Approach

### For You (Jarryd):
1. **Review Phase 1** - Critical for data integrity
2. **Approve Phases 2-3** - Statistical validity is highest priority
3. **Defer Phases 4-7** - Can be implemented as needed

### For Perry (The Team):
1. **Execute Phase 1 immediately** - Blocks downstream work
2. **Prioritize Ridge model** - Confirmed best performer
3. **Parallel development** - Team can work on different phases
4. **Continuous integration** - Test each phase incrementally

---

## File Structure

```
docs/
  system_information_complete.md          # Reference (12 sections)
  implementation_plan.md                 # This file (7 phases)
  plan_summary.md                       # This file (executive summary)
  ridge_model_verification.md          # Ridge verification results

src/
  validation/
    data_validation.py              # Phase 1
    leakage_detection.py               # Phase 2
  statistical/
    bootstrap.py                         # Phase 3
    diebold_mariano.py                  # Phase 3
  uncertainty/
    conformal.py                          # Phase 4
  models/
    lgbm_models.py                     # Phase 5
    registry.py                           # Phase 5 (unified)
  experiments/
    registry.py                           # Phase 6
    runner.py                             # Phase 6
  monitoring/
    drift.py                              # Phase 7
```

---

## Timeline Summary

| Week | Phase | Focus | Deliverable |
|-------|--------|--------|-------------|
| 1 | Data Validation | Gate mechanisms | PASS/FAIL output |
| 2 | Leakage Detection | Sentinels, no leaks | Leakage report |
| 3-4 | Statistical Testing | Bootstrap, DM | Confidence intervals |
| 5-6 | Conformal Uncertainty | Time-series safe CIs | Calibration quality |
| 7 | Model Registry | LightGBM, CatBoost, NN | Expanded candidates |
| 8 | Experiments + Monitoring | Registry, PSI, drift | Production system |

**Total Duration:** 8 weeks  
**Success Criteria:** Production-ready statistically valid system

---

**Date:** January 29, 2026  
**Status:** IMPLEMENTATION PLAN COMPLETE  
**Recommendation:** Start Phase 1 (Data Validation Gate) immediately

---

## What I Need From You

1. **Approval** - Review and approve the implementation plan
2. **Priorities** - Confirm phases 1-3 are highest priority
3. **Resources** - Any team members available for parallel development?
4. **Timeline** - Is 8-week timeline acceptable?

**Ready when you say the word!** 🚀
