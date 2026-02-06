# PerryPicks v3 Implementation Plan

**Purpose:** Provide a complete, comprehensive plan to stabilize and evolve PerryPicks v3 without making code changes until approval.

---

## 1) Goals & Success Criteria

### Primary Goals
1. **Reliability & reproducibility**: repeatable training runs with consistent artifacts.
2. **Prediction quality**: incremental MAE reduction (2–3 pts) and higher winner accuracy.
3. **Operational stability**: fewer runtime regressions, clearer observability, faster debugging.
4. **Maintainability**: reduce script sprawl and align on a canonical pipeline.

### Success Metrics
- **Model quality**: total MAE ≤ 13.5, margin MAE ≤ 10.5, winner accuracy ≥ 65% on recent OOS window.
- **Stability**: <1% failed prediction calls per week in the UI.
- **Reproducibility**: deterministic rebuilds (same data + code → same artifacts).
- **Operational**: single command to retrain and validate (no manual steps).

---

## 2) Current State Summary (Baseline)

- **Pregame pipeline** exists (team ratings → features → training → backtest) with documented outputs.
- **Multiple parallel scripts** exist for later-phase enhancements (head-to-head, travel, injuries, etc.).
- **Game-state detection and auto-model selection** are implemented for pregame/halftime/Q3.
- **Streamlit UI** is the primary interface.

This plan consolidates the patchwork into a coherent architecture, adds missing features in a structured order, and improves reliability/testing before new functionality is layered in.

---

## 3) Architecture Consolidation

### 3.1 Canonical Pipeline Layout
Create a single, versioned pipeline structure to avoid “phase script” confusion:

```
/ src/
  /pipelines/
    /pregame/
      ingest.py
      features.py
      train.py
      evaluate.py
      package.py
    /halftime/
    /q3/
  /registry/
    datasets.py
    models.py
    metrics.py
  /config/
    settings.py
    defaults.yaml
```

### 3.2 Artifact Versioning
- **Dataset versions** (e.g., `data/processed/v1/`) and **model versions** (`models/v1/`).
- Store metadata with each artifact: feature list, build time, data span, code hash.

### 3.3 Unified Entrypoints
- `make train` or `python -m src.pipelines.pregame.train`
- `make evaluate` for reproducible metrics
- `make predict` for locally verifying runtime predictions

**Deliverable**: A single, coherent pipeline with aligned data/model artifacts.

---

## 4) Data & Feature Engineering Roadmap

### 4.1 Phase A: “Low Friction” Features (High ROI)
**Goal**: Improve accuracy with minimal external dependencies.
- **Rest days** per team
- **Back-to-back indicator**
- **Recent form (last 5/10 games)**
- **Head-to-head history**
- **Schedule strength (recent opponents’ strength)**

**Deliverable**: Expanded feature set integrated into pregame dataset and retrained models.

### 4.2 Phase B: Medium Complexity Features
- **Travel distance** and time-zone changes
- **Season phase** (early/mid/late)

**Deliverable**: Enhanced features with documented impact analysis.

### 4.3 Phase C: High Complexity Features
- **Injury impact score**
- **Player-level features**

**Deliverable**: Injury + player pipeline with clear data source strategy and caching.

---

## 5) Modeling & Training Improvements

### 5.1 Baseline Models (Short Term)
- Re-train Ridge/RF/GB models with the improved feature set.
- Use **time-based splits** only.

### 5.2 Model Selection & Calibration
- Standardize evaluation with:
  - MAE, RMSE, bias
  - Win accuracy
  - Calibration (Brier score for probability outputs)

### 5.3 Ensemble Strategy
- Weighted ensemble of best models with per-target optimization (total/margin).

---

## 6) Runtime & UI Stability

### 6.1 UI Improvements
- Stable game selection (persisted state)
- Correct timezone-based default date
- Reduced odds API calls with persistent caching

### 6.2 Runtime Prediction Guardrails
- Feature availability checks before prediction
- Model/data compatibility validation (feature count and ordering)

### 6.3 Observability
- Structured logs by pipeline step and runtime mode
- Snapshot capture for predictions (date, model, feature set hash)

---

## 7) Automated Operations & Alerts

### 7.1 Scheduled Predictions
- Headless scanner to detect games at halftime/Q3.
- Auto-run predictions and write results to `data/predictions/`.

### 7.2 Discord / Slack Notifications
- Post top 3 bets with metadata (odds, edge, confidence).

### 7.3 Bet Grading
- Automated grading after final game result, stored in local DB.

---

## 8) Testing Strategy

### 8.1 Unit Tests
- Feature calculations (rest days, B2B, H2H)
- Model selection logic (pregame/halftime/Q3)

### 8.2 Integration Tests
- End-to-end pipeline run on sample data
- Prediction output schema validation

### 8.3 Regression Tests
- Re-run historical backtest to detect quality degradation

---

## 9) Data Governance & Reliability

### 9.1 Data Contracts
- Define expected schemas for each dataset stage.
- Validate on load (e.g., Great Expectations or custom checks).

### 9.2 Source Reliability
- NBA schedule endpoint health checks
- Cached fallback data for high-availability prediction runs

---

## 10) Delivery Plan & Milestones

### Milestone 1: Stabilization (2–3 weeks)
- Consolidate pipeline
- Add artifact versioning
- Add core tests and reproducibility checks

### Milestone 2: Feature Expansion (3–4 weeks)
- Phase A features integrated
- Retrain & compare baseline metrics

### Milestone 3: Advanced Features (4–6 weeks)
- Phase B + Phase C features
- Production-grade model selection + calibration

### Milestone 4: Automation & Operations (2–4 weeks)
- Automated scanning + notifications
- Grading & reporting pipeline

---

## 11) Open Decisions & Risks

### Decisions Needed
- **Primary data source** for injury + player-level stats
- **Preferred model family** for long-term maintenance (Ridge vs XGBoost vs ensemble)
- **Deployment mode** (local, Streamlit Cloud, containerized)

### Risks
- **Data sourcing** (injury/player data licensing constraints)
- **Data drift** over season and model degradation
- **Runtime API reliability** (NBA endpoints)

---

## 12) Next Steps (Pending Approval)

1. Confirm **feature roadmap order** (Phase A first recommended).
2. Approve **architecture consolidation** and new pipeline structure.
3. Decide **data source strategy** for injuries/player stats.
4. Identify **target metrics** and acceptable tradeoffs.

---

## Appendix: Reference Documents

- `SUMMARY.md` — current pipeline summary and results
- `FINAL_REPORT.md` — system overview and performance
- `RESEARCH_IMPROVEMENTS.md` — feature roadmap and expected impacts
- `docs/v3-technical-plan.md` — UI + automation roadmap
