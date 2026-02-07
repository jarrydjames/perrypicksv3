from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


def _fetch_df(conn: sqlite3.Connection, query: str, params: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def render_ops_dashboard(db_path: str = "data/automation.db") -> None:
    st.subheader("Operations Dashboard")
    st.caption("Single-pane operational status for automation, triggers, delivery, and failures.")

    path = Path(db_path)
    if not path.exists():
        st.warning(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(path)
    try:
        games = _fetch_df(conn, "SELECT status, COUNT(*) as n FROM games GROUP BY status")
        triggers = _fetch_df(conn, "SELECT status, COUNT(*) as n FROM triggers GROUP BY status")
        pending = _fetch_df(
            conn,
            """
            SELECT trigger_type, COUNT(*) as n
            FROM triggers
            WHERE status IN ('scheduled','processing')
            GROUP BY trigger_type
            ORDER BY n DESC
            """,
        )
        fired_24h = _fetch_df(
            conn,
            """
            SELECT trigger_type, COUNT(*) as n
            FROM triggers
            WHERE status='fired' AND fired_at_utc >= datetime('now','-24 hours')
            GROUP BY trigger_type
            ORDER BY n DESC
            """,
        )
        dlq = _fetch_df(conn, "SELECT COUNT(*) AS dlq_count FROM discord_post_dlq")
        misses = _fetch_df(conn, "SELECT COUNT(*) AS miss_explanations FROM miss_explanations")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Games", int(games["n"].sum()) if not games.empty else 0)
        with c2:
            st.metric("Triggers pending", int(pending["n"].sum()) if not pending.empty else 0)
        with c3:
            st.metric("DLQ backlog", int(dlq.iloc[0]["dlq_count"]) if not dlq.empty else 0)
        with c4:
            st.metric("Miss explainers", int(misses.iloc[0]["miss_explanations"]) if not misses.empty else 0)

        st.markdown("### Trigger Status")
        st.dataframe(triggers if not triggers.empty else pd.DataFrame([{"status": "none", "n": 0}]), width="stretch")

        st.markdown("### Pending by Trigger Type")
        st.dataframe(pending if not pending.empty else pd.DataFrame([{"trigger_type": "none", "n": 0}]), width="stretch")

        st.markdown("### Fired in Last 24h")
        st.dataframe(fired_24h if not fired_24h.empty else pd.DataFrame([{"trigger_type": "none", "n": 0}]), width="stretch")

        st.markdown("### Active Incidents")
        dlq_rows = _fetch_df(
            conn,
            """
            SELECT created_at_utc, game_id, trigger_type, retry_count, error_text
            FROM discord_post_dlq
            ORDER BY created_at_utc DESC
            LIMIT 50
            """,
        )
        st.dataframe(dlq_rows if not dlq_rows.empty else pd.DataFrame([{"created_at_utc": "n/a", "game_id": "none", "trigger_type": "none", "retry_count": 0, "error_text": "none"}]), width="stretch")
    finally:
        conn.close()
