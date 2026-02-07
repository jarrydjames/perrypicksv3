# PerryPicks V3 – Platform QoL & Feature Proposals

**Date:** 2026-02-06  
**Context reviewed:** automation runner flow, prediction pipelines, and existing review docs (`QUICK_START_GUIDE.md`, `SYSTEM_REVIEW_DOCUMENTATION.md`, `docs/platform_code_review_20260206.md`).

## 1) Product Intention (What the platform is trying to do)

PerryPicks V3 is best understood as a **real-time NBA prediction operations platform** with three core goals:

1. **Produce timing-sensitive predictions** (pregame / halftime / Q3) using model pipelines.
2. **Operationalize those predictions reliably** through scheduling, trigger firing, persistence, and Discord delivery.
3. **Enable confidence in outcomes** through backtesting, calibration, and historical reporting.

In other words: this is not just a model repo; it is a production workflow that needs both **ML quality** and **ops quality** to win.

---

## 2) Desired Outcome (What “better” should mean)

A stronger outcome for this platform is:

- **Higher decision quality:** predictions are not only accurate, but actionable (edge vs line, confidence, market context).
- **Higher trust:** every pick is explainable, reproducible, and tied to model/version/data lineage.
- **Higher uptime / lower operator burden:** the system self-heals, notifies proactively, and needs minimal babysitting.
- **Faster learning loop:** post-game grading feeds directly into model + policy improvements.

---

## 3) Recommended QoL Adjustments (Near-term)

## A. Operator Experience

### A1. “Single Pane” Ops Dashboard
Add a lightweight dashboard (Streamlit or internal UI route) showing:
- Current date status (CST), active games, fired/pending triggers.
- Last successful prediction time by market stage.
- Discord delivery status and retry queue depth.
- Error budget (failures in last 1h / 24h).

**Why it helps:** removes log-diving and shortens time-to-detection for failures.

### A2. Health Check Command
Create a single CLI, e.g. `python -m scripts.healthcheck`, that validates:
- API reachability,
- model artifact presence,
- DB readability/writability,
- queue backlog thresholds,
- environment config completeness.

**Why it helps:** pre-flight checks before game windows and faster incident triage.

### A3. Trigger Explainability Logs
For every fired or skipped trigger, log structured reasons:
- condition met/failed,
- clock/state snapshot,
- data availability gates.

**Why it helps:** significantly easier debugging for “why no pick posted?” incidents.

## B. Prediction UX

### B1. Confidence-First Output Format
Augment Discord/consumer output with:
- confidence tier (Low/Med/High),
- prediction interval width,
- implied uncertainty warning when interval is too wide.

**Why it helps:** users consume risk, not just point estimates.

### B2. Market Edge Overlay
Compute and display:
- model fair total/spread,
- current market number,
- edge in points and implied value tier.

**Why it helps:** converts model output into decision-ready betting context.

### B3. Canonical Pick IDs
Generate deterministic `pick_id` using game_id + market + trigger + model_version.

**Why it helps:** de-duplication, easier auditing, and cleaner joins across grading/reporting.

## C. Reliability & Automation

### C1. Retry Policies with Backoff + Dead Letter Queue
For Discord/API failures:
- bounded retries with exponential backoff,
- failed messages moved to DLQ for manual replay.

**Why it helps:** prevents silent loss and avoids blocking critical loops.

### C2. Idempotent Trigger Processor
Ensure each trigger execution is idempotent via durable lock/version checks.

**Why it helps:** protects against duplicate posts/picks during restarts and race conditions.

### C3. Game-Day Guardrails
Automatic “degraded mode” when upstream APIs fail:
- fallback data source or cached priors,
- reduced posting mode with explicit warning label.

**Why it helps:** graceful degradation instead of hard downtime.

---

## 4) Additional Features / Functions (High-Value)

## F1. Closing-Line Value (CLV) Tracking
Track opening line, posting-time line, and closing line for every pick.

- Add CLV metrics to daily/weekly report cards.
- Segment CLV by trigger window (T-3h, T-1h, T-10m, halftime).

**Impact:** measures market-beating quality even before final game outcome variance resolves.

## F2. Adaptive Bet Policy Engine
Extend `bet_policy` to support bankroll-aware staking:
- flat stake,
- confidence-scaled stake,
- capped Kelly variant.

Include hard caps by day, market, and correlation cluster.

**Impact:** bridges model predictions to practical risk-managed execution.

## F3. Post-Game Learning Pipeline
Automate post-game joins and diagnostics:
- prediction vs actual,
- calibration drift,
- error by feature cohort (rest disadvantage, travel, injury).

Publish nightly “what failed today” snapshots.

Add a **Plain-Language Miss Explainer** that compares pregame/halftime forecast path vs final game path and publishes a consumer-facing summary in **3 bullets max**:
- **What we expected:** predicted pace/scoring script and win path at tipoff or halftime.
- **What changed live:** the key deviation event(s) (e.g., unexpected foul trouble, bench run, injury, shooting outlier, OT).
- **Why this was path deviation (not model collapse):** concise evidence that baseline assumptions were reasonable at prediction time (line movement alignment, in-range confidence band until event, historical rarity of deviation).

This should be generated as plain language (no technical jargon) so users can quickly understand whether a miss came from an unusual game-state shock versus poor forecasting quality.

**Impact:** turns ops data into model improvement without manual analysis cycles.

## F4. Feature Availability Telemetry
For each prediction, persist feature completeness profile (missingness, stale age, fallback used).

**Impact:** quantifies data quality impact and supports model trust scoring.

## F5. Experiment Registry for Model/Policy A-B Tests
Introduce experiment IDs tying together:
- model version,
- calibration strategy,
- bet policy version,
- output template version.

**Impact:** makes it easy to prove which changes improve ROI/CLV/hit-rate.

## F6. User-Tailored Alerting Modes
Support Discord channel profiles:
- conservative mode (only high-confidence edges),
- balanced mode,
- exploratory mode.

**Impact:** better user fit and reduced alert fatigue.

---

## 5) Prioritized Roadmap (Suggested)

### Phase 1 (1–2 weeks): QoL + Trust
1. Healthcheck CLI.
2. Trigger explainability logs.
3. Canonical pick IDs.
4. Confidence-first output.

### Phase 2 (2–4 weeks): Reliability + Decision Value
1. Retry + DLQ.
2. Idempotent trigger processing.
3. Market edge overlay.
4. CLV tracking.

### Phase 3 (4+ weeks): Compounding Advantage
1. Adaptive bet policy engine.
2. Post-game learning pipeline.
3. Experiment registry.
4. Feature availability telemetry.

---

## 6) Success Metrics to Track

- **Operational:** trigger miss rate, duplicate pick rate, message delivery success %, MTTR.
- **Prediction Quality:** MAE/RMSE by stage, interval coverage, calibration error.
- **Decision Quality:** CLV rate, average edge captured, ROI by confidence tier.
- **User Experience:** alert engagement rate, mute/unsubscribe rate, time-to-detect failures.

---

## 7) Practical “First 3” Recommendation

If only three improvements are chosen immediately, implement:
1. **Trigger explainability + healthcheck** (faster reliability wins).
2. **Confidence + edge overlay in outputs** (better decision usability).
3. **CLV tracking** (strongest medium-term signal of real platform edge).

These three create immediate operator clarity, user value, and measurable learning momentum.
