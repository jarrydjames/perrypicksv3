# Pregame Data Import Adjustments Recommendation
**Date:** 2026-02-07  
**Scope:** Follow-up on recently committed pregame prediction and data reliability findings

## Context Reviewed
Recent commits and analysis documents show the pregame model can still produce low-information outputs when imports are incomplete or stale. The latest fixes improved season selection and safer stat-row selection, but the remaining weak point is the **data import/control plane**, not the model itself.

## What Is Still Failing
1. **Upstream stats intermittently return empty payloads** for valid team/season requests.
2. **Historical dataset freshness lags game day**, creating gaps where feature extraction falls back to defaults.
3. **Placeholder schedule entries (e.g., `UNK`) leak into downstream workflows** and waste prediction cycles.
4. **Import/run scripts are inconsistent in environment loading and path assumptions**, which can break automation depending on execution context.

## Recommended Adjustments (Priority Order)

### P0 — Add a Pre-Prediction "Import Gate"
Before running pregame predictions, require a lightweight health gate that verifies:
- historical data watermark is within an acceptable freshness threshold (e.g., <= 36h old),
- at least one valid stat source is available for both teams,
- game metadata is valid (known team tricodes, valid game IDs, scheduled time).

If gate fails: skip prediction with explicit reason (`STALE_DATA`, `MISSING_TEAM_STATS`, `PLACEHOLDER_GAME`) instead of producing default-heavy output.

**Why:** This prevents silent default fallback from becoming user-facing predictions.

---

### P0 — Harden Data Import as Multi-Source with Watermarking
Implement import sequence per run:
1. Pull schedule from primary endpoint.
2. Pull/refresh recent historical games window (e.g., trailing 14 days).
3. Attempt team advanced stats import for inferred season; if empty, retry previous season.
4. Persist run metadata: source used, rows imported, latest game date, and any fallback used.

Store a **watermark table/file** (e.g., `latest_game_date_utc`, `imported_at_utc`, `row_count`) and read this during prediction gating.

**Why:** Moves from best-effort imports to observable, testable imports.

---

### P1 — Quarantine Placeholder/Invalid Schedule Rows Early
In the import stage, reject games where:
- home/away tricode is `UNK`, empty, or not in team mapping,
- game_id format is invalid,
- start time is missing when required for trigger scheduling.

Log these rows to a quarantine artifact (`data/diagnostics/quarantined_games_YYYYMMDD.json`) for later audit.

**Why:** Keeps invalid schedule payloads from polluting trigger creation and prediction runs.

---

### P1 — Standardize Environment Loading for Import Scripts
Create and use one shared env loader for all import/runner scripts. It should:
- resolve `.env` from repo root reliably,
- support cron/systemd/heredoc execution,
- return explicit status (loaded/not loaded, path used).

**Why:** Eliminates run-context drift causing imports to fail or operate in degraded mode.

---

### P2 — Add Import Reliability Metrics and Alerts
Track and alert on:
- import success rate,
- days since latest historical game,
- percentage of predictions using default-heavy features,
- count of quarantined schedule rows.

Trigger operational warning if thresholds are exceeded (example: >10% default-heavy predictions over 1 day).

**Why:** Detects degradation before predictions become uniformly low quality.

## Suggested Implementation Plan

### Phase 1 (Same Day)
- Implement pre-prediction import gate.
- Add placeholder-row quarantine in schedule ingestion.
- Emit structured logs for `data_source` and `fallback_reason`.

### Phase 2 (1–2 Days)
- Add watermark persistence and validation.
- Standardize env loading in scripts.
- Add an `--import-check-only` command for automation preflight.

### Phase 3 (2–4 Days)
- Add reliability metrics + dashboard tile/alerts.
- Backfill diagnostics for prior days to baseline normal behavior.

## Acceptance Criteria
- No pregame prediction is posted when both team feature sets are default-heavy due to stale/missing imports.
- Import job reports explicit source and freshness metadata on every run.
- `UNK`/invalid schedule rows are quarantined and never reach trigger scheduling.
- Automation health check reflects import freshness and source status, not just process liveness.

## Bottom-Line Recommendation
Treat this as a **data quality and import observability problem**. The model-level fixes are good, but durable improvement now depends on introducing a hard import gate, freshness watermarking, and strict schedule-row validation so predictions are blocked when imports are not trustworthy.
