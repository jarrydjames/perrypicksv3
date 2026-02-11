# Champion Rigor Enhancements (Defensibility + Future Refresh)

This document answers:

1. What to add to make results more rigorous/defensible.
2. How to set up long-term model refinement.
3. How to refresh data and recalibrate/retrain on a recurring schedule.

Companion components:
- `src/pipelines/champion_e2e.py`
- `config/champion_testing_v1.json`
- `src/pipelines/champion_refresh_cycle.py`
- `config/champion_refresh_policy_v1.json`

---

## A) Additions that materially improve defensibility

## 1) Stronger statistical protocol

Add all of these if not already present in your final leaderboard outputs:

- Paired significance tests on fold-level losses (DM + paired bootstrap CIs).
- Effect-size reporting (not just p-values).
- Multiple-comparison correction when many models are tested.
- Stability metrics:
  - std of MAE,
  - worst-fold MAE,
  - downside tail (P90/P95 absolute error).

Why: prevents championing models that win by noise.

## 2) Calibration as a hard gate

Require calibration metrics per state/market:

- ECE ≤ configured threshold,
- Brier score tracked per fold and OOS holdout,
- reliability plots archived with run artifacts,
- coverage targets at 50/60/70/80/90/95.

Why: high point accuracy without calibrated probabilities can still produce bad decision quality.

## 3) Market-realistic backtesting

If ROI is used at all, require:

- timestamp-valid lines,
- vig and slippage,
- delay between signal and execution,
- exposure caps,
- no training-time threshold tuning on evaluation windows.

Why: avoids inflated ROI from unrealistic assumptions.

## 4) Frozen data/manifests per run

Persist immutable run metadata:

- data hash,
- feature list hash,
- code SHA,
- dependency snapshot,
- fold boundaries.

Why: reproducibility and auditability.

---

## B) Future-proof model lifecycle (yes, this sets you up)

With the new orchestration + refresh policy pattern, you can run a recurring MLOps loop:

1. **Ingest / refresh data** (pregame, halftime, Q3 datasets).
2. **Run readiness checks** (`champion_refresh_cycle.py`).
3. **Decide action**:
   - calibration-only,
   - full retrain,
   - block until data issues fixed.
4. **Run canonical champion test** (`champion_e2e.py`).
5. **Promote only if gates pass**.
6. **Deploy via shadow/canary/rollback policy**.

This is sufficient to continuously improve models with future data while maintaining governance.

---

## C) Recommended recurring cadence

From `config/champion_refresh_policy_v1.json`:

- Drift checks every 6 hours.
- Calibration refresh weekly.
- Full retrain weekly (or when enough new games accrue).

Recommended production variant:

- In-season: full retrain weekly, calibration 2x/week.
- Off-season: full retrain monthly, calibration monthly.
- Always trigger immediate retrain if drift thresholds breach.

---

## D) Additional high-value (time-consuming) upgrades

1. **Hierarchical modeling by matchup context**
   - separate heads for pace tiers / travel fatigue regimes.
2. **Player-availability uncertainty model**
   - injury status probability integrated before final prediction.
3. **Line-movement aware target engineering**
   - model robustness to close/open line drift.
4. **Ensemble governance**
   - stacked champion only if it beats best single model on both MAE and calibration.
5. **Post-deployment drift attribution**
   - identify feature-level contributors to degradation (e.g., PSI + SHAP drift).
6. **Champion challenge framework**
   - current champion vs challenger run continuously in shadow for objective swap decisions.

---

## E) What is still required to be fully "definitive"

To be truly definitive, enforce these as non-optional CI checks:

- Leaderboard files for all three states exist and contain all required models.
- Fold-level metrics + calibration metrics must be present.
- Promotion is blocked unless all thresholds pass.
- Any manual override requires written justification artifact checked into repo.

---

## F) Operational commands

Refresh readiness:

```bash
python src/pipelines/champion_refresh_cycle.py --policy config/champion_refresh_policy_v1.json
```

Canonical testing:

```bash
python src/pipelines/champion_e2e.py --config config/champion_testing_v1.json
```

Promotion (gated):

```bash
python src/pipelines/champion_e2e.py --config config/champion_testing_v1.json --promote
```

---

## Bottom line

Yes — with these additions, you have a path to:

- stronger statistical defensibility,
- repeatable future retraining/calibration,
- controlled champion promotion,
- and measurable long-term model improvement.
