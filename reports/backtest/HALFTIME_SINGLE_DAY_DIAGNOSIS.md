# Halftime Single-Day Backtest vs 51-Fold Tuning: Diagnosis

## What is happening

The 51-fold tuning run and the single-day backtest are **not evaluating under equivalent conditions**.

- The 51-fold run reports strong CatBoost aggregate metrics (e.g., `mae_total ~= 7.96`, `mae_margin ~= 3.85`, `rmse_margin ~= 4.88`) across 51 walk-forward folds.  
- The single-day ESPN/CDN backtest for `2026-02-11` reports `mae_total = 9.88` but much worse `mae_margin = 16.53`, `win_accuracy = 42.9%`, and `brier_score = 0.781` over only 14 games.

This mismatch is expected given the current pipeline differences below.

## Root causes identified

### 1) Out-of-distribution date vs training horizon

The single-day test date is **2026-02-11**, while the script trains on historical rows before the target date and indicates training data ends around **2025-06-23**.

Implication: the one-day test is an OOD regime (new season context, roster/coaching/injury shifts, and potential stat distribution drift) that the 51-fold run did not validate on.

### 2) Team recency features are collapsing to defaults in the single-day run

The single-day detailed output shows `home_team_id = 0.0` and `away_team_id = 0.0`, and many recency/advanced features are constants across all games (e.g., identical `home_efg`, `away_efg`, `home_pts_scored_avg_5`, `away_pts_scored_avg_5`, etc.).

Implication: model inputs at inference are substantially less informative than the training feature distribution used in tuning.

### 3) Probability calibration mismatch vs tuning pipeline

The nested walk-forward tuning pipeline performs **post-hoc sigma scaling** on a tail calibration split (`sigma_k_total`, `sigma_k_margin`) before evaluating coverage/Brier. The single-day backtest script does not apply equivalent calibration when deriving `pred_win_prob`.

Implication: even when point predictions are reasonable, win probabilities can be miscalibrated (consistent with the very poor Brier result).

### 4) Metric target mismatch in expectations (winner accuracy vs tuning objective)

The tuning objective optimizes RMSE + coverage penalty + Brier components at fold level; it does not directly maximize one-day winner hit rate on tiny samples. Expecting single-day winner accuracy to match fold-aggregated model quality is statistically fragile.

### 5) Tiny evaluation sample (n=14)

Single-day variance is high. One bad slate can swing winner accuracy materially, especially with close games.

## What to address (priority order)

## P0: Fix feature parity between training and inference

1. **Fix team-id extraction path for NBA CDN payloads** so `home_team_id`/`away_team_id` are populated (non-zero) in single-day inference.
2. Add a **hard fail gate**: if team IDs are zero or if critical recency features are constant across the slate, abort scoring and emit a diagnostic.
3. Build a **feature parity report** (train vs inference): null-rate, zero-rate, constant-rate, min/max, and drift score for top features.

Success criterion: inference distributions for critical recency features are within expected train-time ranges.

## P1: Align calibration with champion evaluation

1. Reuse the same sigma calibration logic used in nested backtest (`k_t`, `k_m`) in the single-day path.
2. Persist calibration metadata from champion selection and apply it at inference.
3. Report calibrated vs uncalibrated Brier side-by-side.

Success criterion: Brier on rolling windows materially improves and becomes stable.

## P1: Match evaluation protocol before judging model quality

1. Run rolling multi-day holdouts (e.g., 14/30/60-day windows), not one day only.
2. Compare model on the **same targets** as tuning (h2_total/h2_margin first), then translate to full-game metrics.
3. Add confidence intervals for winner accuracy to avoid overreacting to 14-game slates.

## P2: Champion parameter selection hardening

1. Do not default to “fold 51 params” as a universal production parameter set.
2. Select robust params via median-of-top-k or retrain tuned model on full pre-cutoff data using chosen objective.
3. Store a model card with training horizon, feature schema hash, and calibration settings.

## Practical implementation checklist

- [ ] Add `--diagnose-features` mode to single-day backtest script.
- [ ] Enforce a `FeatureHealthGate` (ID validity, constant-feature threshold, missing critical columns).
- [ ] Export train/inference feature summary JSON to `reports/backtest/`.
- [ ] Add calibrated probability path shared with nested backtest.
- [ ] Add rolling-window evaluation script and publish dashboard table (14/30/60-day).

## Bottom line

The current gap is driven less by CatBoost model quality and more by **evaluation/protocol mismatch + degraded inference features + calibration inconsistency**. If feature parity and calibration are aligned to the champion pipeline, single-day/short-window performance should move materially closer to the 51-fold baseline.
