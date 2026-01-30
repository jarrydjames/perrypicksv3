# PerryPicks V3 Technical Plan

## Overview
This plan addresses bug fixes, new features (Q3 evaluation, live tracking overhaul), and local automation with Discord integration. All changes maintain backward compatibility with existing halftime model.

---

## Phase 1: BUG FIXES (Highest Priority)

### Bug 1: Dropdown Selection Reverts
**Problem:** Selectbox defaults to index 0 on every rerun (mobile navigation, Streamlit refresh).

**Root Cause:** 
- Selectbox at line 294 lacks `key` parameter
- No `st.session_state` persistence for selected game index
- Each rerun recreates selectbox with fresh state

**Solution:**
```python
idx = st.selectbox(
    "Games",
    list(range(len(games))),
    format_func=lambda i: labels[i],
    key="pp_game_idx",
    index=st.session_state.get("pp_game_idx", 0),
)
st.session_state["pp_game_idx"] = idx
```
**File:** `app.py`

---

### Bug 2: Date Rollover Too Early
**Problem:** `pp_pick_date` defaults to `_dt.date.today()` (system UTC), causing premature rollover to tomorrow.

**Root Cause:**
- Uses `datetime.date.today()` which returns local system date (likely UTC)
- No timezone awareness
- Should use `America/Chicago` timezone per requirements

**Solution:**
```python
import pytz
TZ = pytz.timezone("America/Chicago")
st.session_state.setdefault("pp_pick_date", datetime.now(TZ).date())
TZ = pytz.timezone(os.getenv("TZ", "America/Chicago"))
```
**Dependencies:** Add `pytz` to requirements.txt
**File:** `app.py`

---

### Bug 3: Mobile Refresh Wastes Odds API Calls
**Problem:** Mobile navigation causes full app rerun, forcing re-download of odds.

**Root Cause:**
- Odds caching is Streamlit-only (`st.cache_data`), lost on rerun
- No persistent storage of odds responses
- No manual refresh button with cooldown

**Solution:**
1. Add explicit Refresh button with cooldown
2. Add persistent odds caching to SQLite
3. Modify odds_api.py to log all calls

**Files:** `app.py`, `src/storage/sqlite_store.py`, `src/odds/persistent_cache.py` (new), `src/odds/odds_api.py`

---

## Phase 2: Q3 EVALUATION ADDITIONS

### Feature A: Extend Evaluation Window to Q3
**Goal:** Compute picks at halftime AND end-of-Q3 without changing halftime model.

**Approach:**
1. Create `src/modeling/q3_model.py` - new Q3 evaluator (separate from halftime)
2. Add `eval_at_q3` flag to `predict_from_game_id()`
3. When `eval_at_q3=True`, call Q3 model instead of halftime model

**Files:** `src/modeling/q3_model.py` (new), `src/predict_from_gameid_v3_runtime.py` (update)

---

### Feature B: Train Separate Q3 Model
**Goal:** Train Q3 model using end-of-Q3 state, similar to halftime pipeline.

**Approach:**
1. Reuse existing training infrastructure
2. Add `build_dataset_q3.py` to create Q3-specific training data
3. Add `train_q3_model()` function

**Files:** `src/build_dataset_q3.py` (new), `src/modeling/q3_model.py` (update)

---

### Feature C: Live Tracking Overhaul
**Goal:** Replace inaccurate tracking with accurate time-series charting.

**Solution:**
1. Enhanced snapshot schema in SQLite (add period, clock, scores)
2. Update `src/ui/tracking.py` to capture and chart correctly

**Files:** `src/storage/sqlite_store.py` (enhanced), `src/ui/tracking.py` (enhanced)

---

## Phase 3: LOCAL AUTOMATION + DISCORD

### Architecture: UI vs Core vs Automation

```
PerryPicks V3 Architecture:

├── UI Layer (Streamlit)
│   ├── app.py (main app)
│   └── src/ui/* (components)
│
├── Core Prediction Engine
│   ├── src/predict_api.py (entry point)
│   ├── src/predict_from_gameid_v3_runtime.py (orchestrator)
│   ├── src/modeling/* (halftime + Q3 models)
│   └── src/domain/* (betting logic)
│
└── Automation Layer (Headless)
    ├── scripts/automation/
    │   ├── game_scanner.py  # Detects halftime/Q3
    │   ├── discord_poster.py  # Posts top 3 bets
    │   └── bet_grader.py  # Grades results
    └── src/automation/
        ├── scheduler.py  # Cron wrapper
        └── discord_client.py  # Discord bot integration
```

---

### 1. Game Scanner (Halftime/Q3 Detection)
**File:** `scripts/automation/game_scanner.py`

Scan today's games, detect halftime (period 2, clock 12:00) and end of Q3 (period 4, clock 12:00).

---

### 2. Discord Poster
**File:** `scripts/automation/discord_poster.py`, `src/automation/discord_client.py`

Use Discord Webhook URL. Format post with header, 3 ranked bets, footer.

---

### 3. Bet Grader
**File:** `scripts/automation/bet_grader.py`

Check if bets hit after game final.

---

### 4. Scheduler
**File:** `scripts/automation/scheduler.py`

Cron wrapper that runs every minute, scans games, triggers prediction and Discord posting.

---

## Phase 4: LOCAL STORAGE SCHEMA (SQLite)

### Complete Schema for V3

```sql
CREATE TABLE games (
  game_id TEXT PRIMARY KEY,
  created_ts_utc TEXT NOT NULL
);

CREATE TABLE bets (
  bet_id TEXT PRIMARY KEY,
  game_id TEXT NOT NULL,
  created_ts_utc TEXT NOT NULL,
  bet_type TEXT NOT NULL,
  side TEXT NOT NULL,
  line REAL,
  odds INTEGER,
  payload_json TEXT NOT NULL,
  FOREIGN KEY(game_id) REFERENCES games(game_id)
);

CREATE TABLE snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  ts_utc TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  UNIQUE(game_id, ts_utc),
  FOREIGN KEY(game_id) REFERENCES games(game_id)
);

-- NEW: Enhanced tracking snapshots
CREATE TABLE tracking_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  ts_utc TEXT NOT NULL,
  period INTEGER,
  clock TEXT,
  home_score INTEGER,
  away_score INTEGER,
  prediction_json TEXT NOT NULL,
  UNIQUE(game_id, ts_utc),
  FOREIGN KEY(game_id) REFERENCES games(game_id)
);

-- NEW: Odds cache (persistent)
CREATE TABLE odds_cache (
  cache_key TEXT PRIMARY KEY,
  home TEXT NOT NULL,
  away TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_ts_utc TEXT NOT NULL,
  expires_at_utc TEXT NOT NULL
);

-- NEW: Picks posted (for automation)
CREATE TABLE picks_posted (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  event TEXT NOT NULL,
  posted_ts_utc TEXT NOT NULL,
  picks_json TEXT NOT NULL,
  UNIQUE(game_id, event),
  FOREIGN KEY(game_id) REFERENCES games(game_id)
);

-- NEW: Discord messages (for replies)
CREATE TABLE discord_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  event TEXT NOT NULL,
  message_id TEXT NOT NULL,
  posted_ts_utc TEXT NOT NULL,
  UNIQUE(game_id, event),
  FOREIGN KEY(game_id) REFERENCES games(game_id)
);

-- NEW: Grading results
CREATE TABLE grading_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  bet_id TEXT NOT NULL,
  event TEXT NOT NULL,
  hit BOOLEAN NOT NULL,
  graded_ts_utc TEXT NOT NULL,
  FOREIGN KEY(game_id) REFERENCES games(game_id),
  FOREIGN KEY(bet_id) REFERENCES bets(bet_id)
);
```

**File:** `src/storage/sqlite_store.py` (enhanced)

---

## Phase 5: CONFIGURATION (.env)

### Environment Variables

```bash
ODDS_API_KEY=your_api_key_here
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_CHANNEL_ID=123456789
TZ=America/Chicago
ODDS_CACHE_TTL=600
ENABLE_AUTOMATION=true
SCAN_INTERVAL_SECONDS=60
```

**File:** `.env.example` (new)

---

## DELIVERABLES CHECKLIST

### Phase 1: Bug Fixes
- [ ] Fix dropdown selection reverts (app.py)
- [ ] Fix date rollover timezone issue (app.py)
- [ ] Add persistent odds caching (src/odds/persistent_cache.py)
- [ ] Add refresh button with cooldown (app.py)
- [ ] Add odds API call logging (src/odds/odds_api.py)

### Phase 2: Q3 Additions
- [ ] Create Q3 model stub (src/modeling/q3_model.py)
- [ ] Add Q3 training pipeline hooks (src/build_dataset_q3.py)
- [ ] Update runtime predictor for Q3 flag (src/predict_from_gameid_v3_runtime.py)

### Phase 3: Tracking Overhaul
- [ ] Enhance tracking snapshot schema (src/storage/sqlite_store.py)
- [ ] Update tracking UI with accurate charting (src/ui/tracking.py)

### Phase 4: Automation + Discord
- [ ] Create game scanner (scripts/automation/game_scanner.py)
- [ ] Create Discord poster (scripts/automation/discord_poster.py, src/automation/discord_client.py)
- [ ] Create bet grader (scripts/automation/bet_grader.py)
- [ ] Create scheduler (scripts/automation/scheduler.py)

### Phase 5: Storage & Config
- [ ] Implement complete SQLite schema (src/storage/sqlite_store.py)
- [ ] Add .env.example with all config vars
- [ ] Update requirements.txt with new dependencies (pytz, requests)

---

## PRIORITY ORDER

1. Phase 1 (Bug Fixes) - Blockers for reliable usage
2. Phase 2 (Q3 Additions) - Core new feature
3. Phase 3 (Tracking Overhaul) - UX improvement
4. Phase 5 (Storage & Config) - Foundation for automation
5. Phase 4 (Automation + Discord) - Deploy after core features stable

---

## BACKWARD COMPATIBILITY

- Halftime model code path COMPLETELY unchanged
- All existing v2 features work identically
- Q3 model is opt-in (via `eval_at_q3` flag)
- Automation is separate entry point (not integrated into Streamlit)
