# Platform Code Review (2026-02-06)

## Scope and method

This review focused on code paths that impact runtime reliability, model correctness, and maintainability:

- Streamlit apps (`app.py`, `app_v3.py`)
- Data ingestion and API fetchers (`core/data_sources.py`, `src/fetch_game_ids.py`, `src/fetch_games_by_id_range.py`)
- Model training pipeline scripts (`src/train_halftime_full_pipeline.py`, `train_final.py`)

Checks included:
- Repository-wide Python bytecode compilation for syntax validation
- Manual inspection of critical modules and supporting scripts

---

## Executive summary

The platform contains several high-severity defects that can prevent execution or silently degrade model quality:

1. **Hard runtime failures (syntax errors)** in training scripts.
2. **Data science correctness defects** in walk-forward CV logic (empty train window, degenerate targets).
3. **Reliability defects** in datetime parsing and request handling that can silently fallback to bad defaults.
4. **Maintainability/performance drag** from root-level script sprawl and environment-specific path assumptions.

---

## Findings (prioritized)

### 1) High — Syntax errors block script execution

- `src/train_halftime_full_pipeline.py` contains f-strings with escaped quotes inside expression segments, which is invalid Python syntax.
- `train_final.py` contains an invalid nested f-string for the divider print.

**Impact:** critical training/experimentation paths cannot run.

**Recommendation:** add a CI gate using `python -m compileall` (or `ruff check`) and fix f-string formatting consistently.

---

### 2) High — Walk-forward CV uses an empty training slice

In `walk_forward_cv`, `train_start` is initialized to `min_train_size`, then `train_df` is sliced as `df_sorted.iloc[train_start:train_end]` where `train_end = train_start`; this always produces an empty frame on first fold.

**Impact:** model training can fail or produce invalid metrics due to empty arrays.

**Recommendation:** use `train_df = df_sorted.iloc[:train_end]` for expanding-window CV, or maintain explicit `train_start=0` with a rolling window.

---

### 3) High — Degenerate synthetic target construction in halftime pipeline

`y_train_margin` is built as `h1_total - h1_total`, yielding all zeros; total target is `h1_total + h1_total`, which is a simplistic duplication.

**Impact:** margin model is guaranteed to learn a trivial constant; evaluation and model selection become misleading.

**Recommendation:** derive targets from true 2H outcomes, not transformations of halftime totals.

---

### 4) High — `app_v3.py` session state default can crash first render

`st.session_state.page` is initialized to `'home'` but the sidebar options use labels like `'🏠 Home'`, and index selection uses `pages.index(st.session_state.page)`.

**Impact:** first page load can throw `ValueError: 'home' is not in list`.

**Recommendation:** initialize to an actual option value (e.g., `'🏠 Home'`) or guard with fallback index.

---

### 5) Medium — Datetime parser references undefined `datetime` symbol

In `core/data_sources.py`, `_parse_nba_datetime` checks `isinstance(..., datetime)` but only `timedelta` is imported from `datetime`.

**Impact:** `NameError` is swallowed by broad exception handling, causing avoidable parse failures and potential downstream fallback behavior.

**Recommendation:** import `datetime` explicitly (`from datetime import datetime, timedelta`) and narrow exception scopes.

---

### 6) Medium — Request handling omits status checks and output directory safety

`src/fetch_game_ids.py` calls `requests.get(...).json()` directly and writes to `data/processed/...` without ensuring directory existence.

**Impact:** transient HTTP failures can be misinterpreted as JSON errors; file writes can fail on clean environments.

**Recommendation:** call `raise_for_status()`, handle JSON decode failures, and ensure `Path(out_path).parent.mkdir(parents=True, exist_ok=True)`.

---

### 7) Medium — Environment-specific absolute path and bare `except`

`src/fetch_games_by_id_range.py` modifies `sys.path` with a user-machine absolute path and uses a bare `except` around PBP fetch.

**Impact:** non-portable behavior and suppressed root-cause diagnostics.

**Recommendation:** remove absolute path mutation; package imports properly. Replace bare `except` with targeted exception classes and logging.

---

### 8) Medium — Monolithic root script layout increases operational risk

The repository root contains many similarly named phase/optional/fix scripts and backup variants.

**Impact:** raises risk of running stale scripts, duplicated logic, and inconsistent results; makes onboarding/debugging slower.

**Recommendation:** consolidate into `src/` packages and a small set of canonical entrypoints; archive legacy/experimental scripts outside runtime paths.

---

## Performance and architecture notes

- In-memory caches are present for API data, but error handling often falls back broadly, making cache usefulness harder to reason about under failures.
- Training scripts appear to repeat feature engineering/model definitions across many standalone files; centralizing training config and CV utilities would reduce divergence and improve reproducibility.
- Adding lightweight CI checks (compile, lint, minimal smoke test) would catch multiple issues identified above before merge.

---

## Suggested immediate remediation plan

1. **Stabilize execution**: fix syntax errors and add compile/lint pre-merge checks.
2. **Correct modeling logic**: repair walk-forward split and target definitions in halftime pipeline.
3. **Harden IO**: normalize request/response handling and filesystem preparation in fetch scripts.
4. **Reduce structural entropy**: define canonical pipelines and deprecate duplicate/backup scripts from active paths.

