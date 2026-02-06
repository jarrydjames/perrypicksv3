# Automation Reliability + Operations Backend Plan

## Goals
- Start/stop the Discord automation **predictably, safely, and repeatedly** with zero duplicate posting.
- Introduce an operations backend that gives full visibility and control: run status, upcoming triggers, queue health, manual trigger/retry, and audit trail.
- Improve resiliency (crash recovery, idempotency, alerting) so this can run unattended.

---

## 1) Reliable Start/Stop Strategy

## 1.1 Process model (recommended)
Use a **supervisor-managed worker** (systemd preferred on Linux) instead of ad-hoc terminal sessions.

- `perrypick-worker.service`: long-running scheduler/runner process.
- Optional `perrypick-api.service`: operations backend API.
- Both configured with:
  - `Restart=always`
  - `RestartSec=5`
  - environment file (`.env.production`)
  - structured log output to journald + rotated app logs.

### Why this is reliable
- Automatic restart on crash.
- Controlled shutdown (`SIGTERM`) with graceful cleanup.
- Startup ordering (DB/network dependencies) and boot-time auto-start.

## 1.2 Graceful lifecycle contract
Implement explicit worker lifecycle states:
- `STARTING` → `RUNNING` → `DRAINING` → `STOPPED`
- `ERROR` if startup checks fail.

On stop request:
1. Mark worker `DRAINING`.
2. Stop claiming new trigger jobs.
3. Finish in-flight jobs with timeout (e.g., 30s).
4. Persist heartbeat final state.
5. Exit cleanly.

This avoids half-sent Discord posts and partial DB writes.

## 1.3 Leader lock to prevent duplicate posters
If there is any chance of multiple worker instances, add a **distributed lock**:
- DB advisory lock or lock row (`worker_leader_lock`).
- Only lock-holder executes triggers.
- Others remain standby and expose `NOT_LEADER` status.

This prevents duplicate Discord posts after restarts/deploys.

## 1.4 Health checks before run
At startup, run a readiness bundle:
- DB writable check.
- Discord webhook/API test in non-post mode.
- Odds API connectivity check.
- Clock sanity check (NTP skew threshold).
- Model availability check.

If any critical check fails, stay in `ERROR` and alert.

## 1.5 Safe shutdown + panic controls
Provide two stop actions:
- **Graceful stop** (default): drain and exit.
- **Immediate kill** (break-glass): terminate now, mark interrupted jobs recoverable.

Add **global posting kill-switch**:
- Config or DB flag `posting_enabled=false`.
- Worker can continue evaluating triggers but skips outbound Discord calls.

## 1.6 Crash recovery rules
On startup:
- Reconcile `IN_PROGRESS` jobs older than timeout as `RETRYABLE`.
- Rebuild schedule for active game day windows.
- Re-run missed triggers within allowed lateness (e.g., <= 5 min) with idempotency keys.

---

## 2) Operations Backend (Monitoring + Management)

## 2.1 Architecture
Build a lightweight control plane:
- **Backend API** (FastAPI recommended).
- **Ops UI** (simple React/Next or server-rendered page).
- Uses existing SQLite initially; migrate to Postgres for production concurrency.

### Components
1. `worker` (executor)
2. `api` (query/control)
3. `db` (state, schedule, audit)
4. `notifier` (Slack/Discord ops alerts)

## 2.2 Core backend capabilities

### A) Real-time status dashboard
Show:
- Worker status (`RUNNING`, `DRAINING`, `ERROR`, uptime, version/commit hash).
- Last heartbeat and lag.
- Queue depth by trigger type.
- In-flight jobs and duration.
- Recent posts and failure counts.

### B) Trigger calendar + due timeline
Show:
- Upcoming triggers for next 24h/48h.
- Game-local and UTC times.
- Dependency status (odds fetched? model ready?).
- Risk flags (late data, stale odds, retry attempts).

### C) Manual operations
Allow operators to:
- Manually trigger specific game + trigger type.
- Retry failed trigger/post.
- Skip/cancel queued trigger.
- Pause/resume all posting or by trigger type.
- Recompute schedule for a date.

Every action writes to audit log with actor + reason.

### D) Event + audit logs
Central table for control actions and system events:
- who did what, when, from where
- before/after values
- correlation IDs linking API request → worker job → Discord post

### E) Alerting
Alert on:
- no heartbeat > N seconds
- trigger overdue > threshold
- repeated Discord API failures
- unusually high post error rate
- odds/model data stale

## 2.3 API surface (suggested)
- `GET /health`
- `GET /worker/status`
- `POST /worker/start`
- `POST /worker/stop?mode=graceful|immediate`
- `POST /worker/pause-posting`
- `POST /worker/resume-posting`
- `GET /triggers?from=&to=&status=`
- `POST /triggers/{id}/manual-fire`
- `POST /triggers/{id}/retry`
- `POST /triggers/{id}/cancel`
- `GET /posts/recent`
- `GET /incidents`

Protect all mutating endpoints with auth + RBAC.

## 2.4 Data model additions
Add/extend tables:
- `worker_heartbeats(worker_id, status, started_at, last_seen_at, meta_json)`
- `job_queue(job_id, game_id, trigger_type, due_at, status, attempts, locked_by, idempotency_key)`
- `job_runs(job_run_id, job_id, started_at, ended_at, result, error_json)`
- `control_actions(action_id, actor, action_type, payload_json, created_at, reason)`
- `incidents(incident_id, severity, signal, status, opened_at, resolved_at)`

Keep `discord_posts` as source-of-truth for delivery outcomes.

---

## 3) Reliability Engineering Patterns to Add

1. **Idempotency everywhere**
   - Use deterministic idempotency key: `game_id + trigger_type + schedule_slot`.
   - Ensure retries never duplicate Discord posts.

2. **Retry policy with backoff + jitter**
   - Transient failures retried with capped exponential backoff.
   - Separate permanent failures (e.g., validation) from transient network errors.

3. **Timeout budgets per step**
   - Odds fetch timeout, model inference timeout, Discord send timeout.
   - Fail fast to avoid queue blockage.

4. **Circuit breakers**
   - If Discord API fails repeatedly, open breaker and alert.
   - Continue internal processing with `POST_DEFERRED` state.

5. **Dead letter queue**
   - Jobs exceeding max retries move to DLQ for manual triage.

6. **Structured logging + metrics**
   - JSON logs with correlation IDs.
   - Metrics: due-to-fire latency, success ratio, retries, overdue count, API error rate.

7. **Versioned run metadata**
   - Store model version + code commit hash on every trigger run.
   - Makes outcome debugging and rollback much easier.

---

## 4) Security + Access Control
- Require authentication for ops dashboard.
- RBAC roles:
  - `viewer`: read-only status.
  - `operator`: manual trigger/retry/pause.
  - `admin`: config changes, force stop.
- Immutable audit logs for all mutating actions.
- Secret management for Discord/Odds keys (env or vault, never UI plaintext).

---

## 5) Rollout Plan (Phased)

### Phase 1 (1–2 days): Operational safety baseline
- Add graceful shutdown/drain behavior.
- Add worker heartbeat table + heartbeat writes.
- Add posting kill-switch.
- Add idempotency key checks on trigger execution.
- Create systemd service units + runbook.

### Phase 2 (2–4 days): Control API + basic dashboard
- FastAPI service with status + trigger listing + manual fire/retry endpoints.
- Minimal web dashboard (status, upcoming triggers, recent posts).
- API auth + audit logging.

### Phase 3 (2–5 days): Reliability hardening
- Queue abstraction (`job_queue`) replacing implicit polling-only execution.
- Retry policies, DLQ, circuit breaker.
- Alert pipeline (Discord/Slack ops channel).

### Phase 4 (ongoing): Advanced operations
- Incident workflow, annotations, postmortem links.
- Capacity analytics and forecast.
- Multi-worker HA with leader election and standby failover.

---

## 6) Suggested Extra Features (High Value)
1. **Simulation mode in dashboard**: run trigger pipeline without posting to validate today's slate.
2. **Dry-run diff view**: compare current planned post vs last successful post template.
3. **Manual approval gate** (optional): require operator approval for high-risk bets.
4. **SLA panel**: “% triggers fired within 30s of due time”.
5. **Replay tool**: replay a historical game day to diagnose issues.
6. **Template versioning for Discord messages** with preview before send.
7. **On-call digest**: daily summary of failures, retries, missed triggers.
8. **Auto-pause on anomaly**: if edge/probability outputs look out of band.
9. **Config registry**: centrally manage trigger windows/thresholds with change history.
10. **Data freshness monitor**: warn if odds or game feeds are stale before due triggers.

---

## 7) Operational Runbook (What your team does day-to-day)

### Start of day
- Check dashboard health, queue depth, and upcoming trigger timeline.
- Verify odds/model/data freshness all green.

### During games
- Watch overdue and retry panels.
- Use manual fire only with reason codes.
- If third-party outage: activate posting pause, continue internal processing.

### End of day
- Review incident panel.
- Resolve DLQ items.
- Export daily ops summary (success %, missed triggers, median latency).

---

## 8) Success Criteria (Definition of Done)
- 0 duplicate Discord posts across restart/deploy scenarios.
- >99% triggers executed within SLA window.
- Mean time to detect worker outage < 1 minute.
- Mean time to recover from transient outage < 5 minutes.
- Full auditability for every manual trigger and retry.

