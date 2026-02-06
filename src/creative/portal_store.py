from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_SCHEDULED = "scheduled"
STATUS_PUBLISHED = "published"


class CreativePortalStore:
    """SQLite-backed persistence for creative management workflows."""

    def __init__(self, db_path: str = "data/perrypicks.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS creative_requests (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  game_id TEXT,
                  channel TEXT NOT NULL,
                  confidence_tier TEXT,
                  prediction_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_by TEXT,
                  created_ts_utc TEXT NOT NULL,
                  updated_ts_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS creative_variants (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  request_id INTEGER NOT NULL,
                  variant_label TEXT NOT NULL,
                  prompt_text TEXT NOT NULL,
                  overlay_copy TEXT,
                  negative_prompt TEXT,
                  seed INTEGER,
                  model_name TEXT,
                  image_uri TEXT,
                  quality_score REAL,
                  status TEXT NOT NULL,
                  rejection_reason TEXT,
                  created_ts_utc TEXT NOT NULL,
                  updated_ts_utc TEXT NOT NULL,
                  FOREIGN KEY(request_id) REFERENCES creative_requests(id)
                );
                CREATE INDEX IF NOT EXISTS idx_cv_request_status ON creative_variants(request_id, status);

                CREATE TABLE IF NOT EXISTS creative_publish_records (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  variant_id INTEGER NOT NULL,
                  channel TEXT NOT NULL,
                  scheduled_ts_utc TEXT,
                  published_ts_utc TEXT,
                  publish_status TEXT NOT NULL,
                  external_post_id TEXT,
                  notes TEXT,
                  created_ts_utc TEXT NOT NULL,
                  FOREIGN KEY(variant_id) REFERENCES creative_variants(id)
                );
                CREATE INDEX IF NOT EXISTS idx_publish_status ON creative_publish_records(publish_status);

                CREATE TABLE IF NOT EXISTS creative_policies (
                  key TEXT PRIMARY KEY,
                  value_json TEXT NOT NULL,
                  updated_ts_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS creative_prompt_profiles (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  profile_name TEXT UNIQUE NOT NULL,
                  channel TEXT NOT NULL,
                  prompt_template TEXT NOT NULL,
                  negative_prompt TEXT,
                  is_active INTEGER NOT NULL DEFAULT 1,
                  version INTEGER NOT NULL DEFAULT 1,
                  created_ts_utc TEXT NOT NULL,
                  updated_ts_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS creative_audit_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  actor TEXT NOT NULL,
                  action TEXT NOT NULL,
                  target_type TEXT NOT NULL,
                  target_id TEXT NOT NULL,
                  payload_json TEXT,
                  created_ts_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS creative_system_state (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_ts_utc TEXT NOT NULL
                );
                """
            )

    def log_action(self, *, actor: str, action: str, target_type: str, target_id: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO creative_audit_log(actor, action, target_type, target_id, payload_json, created_ts_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (actor, action, target_type, target_id, json.dumps(payload or {}), self._now()),
            )

    def create_request(
        self,
        *,
        game_id: Optional[str],
        channel: str,
        confidence_tier: str,
        prediction_payload: Dict[str, Any],
        created_by: str,
    ) -> int:
        now = self._now()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO creative_requests(
                  game_id, channel, confidence_tier, prediction_json, status, created_by, created_ts_utc, updated_ts_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    channel,
                    confidence_tier,
                    json.dumps(prediction_payload),
                    STATUS_PENDING,
                    created_by,
                    now,
                    now,
                ),
            )
            req_id = int(cur.lastrowid)
        self.log_action(actor=created_by, action="request_created", target_type="creative_request", target_id=str(req_id))
        return req_id

    def add_variant(
        self,
        *,
        request_id: int,
        variant_label: str,
        prompt_text: str,
        overlay_copy: str,
        model_name: str,
        seed: Optional[int],
        image_uri: str,
        quality_score: float,
        actor: str,
        negative_prompt: str = "",
    ) -> int:
        now = self._now()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO creative_variants(
                  request_id, variant_label, prompt_text, overlay_copy, negative_prompt, seed, model_name,
                  image_uri, quality_score, status, created_ts_utc, updated_ts_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    variant_label,
                    prompt_text,
                    overlay_copy,
                    negative_prompt,
                    seed,
                    model_name,
                    image_uri,
                    float(quality_score),
                    STATUS_PENDING,
                    now,
                    now,
                ),
            )
            variant_id = int(cur.lastrowid)
        self.log_action(actor=actor, action="variant_created", target_type="creative_variant", target_id=str(variant_id))
        return variant_id

    def list_queue(self) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                  v.id AS variant_id,
                  r.id AS request_id,
                  r.game_id,
                  r.channel,
                  r.confidence_tier,
                  v.variant_label,
                  v.quality_score,
                  v.status,
                  v.rejection_reason,
                  v.overlay_copy,
                  v.prompt_text,
                  v.image_uri,
                  v.updated_ts_utc
                FROM creative_variants v
                JOIN creative_requests r ON r.id = v.request_id
                WHERE v.status IN (?, ?, ?)
                ORDER BY v.updated_ts_utc DESC
                """,
                (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_variant_status(self, *, variant_id: int, status: str, actor: str, reason: str = "") -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE creative_variants SET status=?, rejection_reason=?, updated_ts_utc=? WHERE id=?",
                (status, reason, self._now(), variant_id),
            )
        self.log_action(
            actor=actor,
            action=f"variant_{status}",
            target_type="creative_variant",
            target_id=str(variant_id),
            payload={"reason": reason},
        )

    def schedule_publish(self, *, variant_id: int, channel: str, scheduled_ts_utc: str, actor: str, notes: str = "") -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO creative_publish_records(
                  variant_id, channel, scheduled_ts_utc, publish_status, notes, created_ts_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (variant_id, channel, scheduled_ts_utc, STATUS_SCHEDULED, notes, self._now()),
            )
            con.execute(
                "UPDATE creative_variants SET status=?, updated_ts_utc=? WHERE id=?",
                (STATUS_SCHEDULED, self._now(), variant_id),
            )
        self.log_action(actor=actor, action="publish_scheduled", target_type="creative_variant", target_id=str(variant_id))

    def list_assets(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                  v.id AS variant_id,
                  v.variant_label,
                  v.model_name,
                  v.image_uri,
                  v.status,
                  v.quality_score,
                  r.game_id,
                  r.channel,
                  r.confidence_tier,
                  v.created_ts_utc
                FROM creative_variants v
                JOIN creative_requests r ON r.id = v.request_id
                ORDER BY v.created_ts_utc DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_policy(self, key: str, value: Dict[str, Any]) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO creative_policies(key, value_json, updated_ts_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value_json=excluded.value_json,
                  updated_ts_utc=excluded.updated_ts_utc
                """,
                (key, json.dumps(value), self._now()),
            )

    def get_policy(self, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
        with self._connect() as con:
            row = con.execute("SELECT value_json FROM creative_policies WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

    def upsert_prompt_profile(
        self,
        *,
        profile_name: str,
        channel: str,
        prompt_template: str,
        negative_prompt: str,
        is_active: bool,
    ) -> None:
        now = self._now()
        with self._connect() as con:
            row = con.execute(
                "SELECT id, version FROM creative_prompt_profiles WHERE profile_name=?",
                (profile_name,),
            ).fetchone()
            if row:
                con.execute(
                    """
                    UPDATE creative_prompt_profiles
                    SET channel=?, prompt_template=?, negative_prompt=?, is_active=?,
                        version=?, updated_ts_utc=?
                    WHERE profile_name=?
                    """,
                    (
                        channel,
                        prompt_template,
                        negative_prompt,
                        int(bool(is_active)),
                        int(row["version"]) + 1,
                        now,
                        profile_name,
                    ),
                )
            else:
                con.execute(
                    """
                    INSERT INTO creative_prompt_profiles(
                      profile_name, channel, prompt_template, negative_prompt, is_active, version, created_ts_utc, updated_ts_utc
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (profile_name, channel, prompt_template, negative_prompt, int(bool(is_active)), now, now),
                )

    def list_prompt_profiles(self) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT profile_name, channel, prompt_template, negative_prompt, is_active, version, updated_ts_utc
                FROM creative_prompt_profiles
                ORDER BY updated_ts_utc DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def set_system_toggle(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO creative_system_state(key, value, updated_ts_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts_utc=excluded.updated_ts_utc
                """,
                (key, value, self._now()),
            )

    def get_system_toggle(self, key: str, default: str = "false") -> str:
        with self._connect() as con:
            row = con.execute("SELECT value FROM creative_system_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def summarize_metrics(self) -> Dict[str, Any]:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) AS c FROM creative_variants").fetchone()["c"]
            approved = con.execute("SELECT COUNT(*) AS c FROM creative_variants WHERE status=?", (STATUS_APPROVED,)).fetchone()["c"]
            rejected = con.execute("SELECT COUNT(*) AS c FROM creative_variants WHERE status=?", (STATUS_REJECTED,)).fetchone()["c"]
            scheduled = con.execute("SELECT COUNT(*) AS c FROM creative_publish_records WHERE publish_status=?", (STATUS_SCHEDULED,)).fetchone()["c"]
            reasons = con.execute(
                """
                SELECT rejection_reason, COUNT(*) AS count
                FROM creative_variants
                WHERE status=? AND COALESCE(rejection_reason, '') <> ''
                GROUP BY rejection_reason
                ORDER BY count DESC
                LIMIT 5
                """,
                (STATUS_REJECTED,),
            ).fetchall()

        approval_rate = (float(approved) / float(total)) if total else 0.0
        return {
            "total_variants": int(total),
            "approved": int(approved),
            "rejected": int(rejected),
            "scheduled": int(scheduled),
            "approval_rate": approval_rate,
            "top_rejection_reasons": [dict(row) for row in reasons],
        }
