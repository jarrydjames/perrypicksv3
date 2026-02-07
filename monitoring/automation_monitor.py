"""
PerryPicks v3 - Automation Monitoring Portal

Monitor automation status, view scheduled games, and manually trigger predictions.
"""

import os
import streamlit as st
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3
import pytz
from typing import List, Dict, Any, Optional
import subprocess

# Page config
st.set_page_config(
    page_title="PerryPicks Automation Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "automation.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_automation_status() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python -m worker.runner"],
            capture_output=True,
            text=True
        )
        is_running = len(result.stdout.strip()) > 0
        
        log_path = Path(__file__).parent.parent / "logs" / "automation.log"
        last_log_time = None
        if log_path.exists():
            last_log_time = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
        
        return {
            "running": is_running,
            "last_log_time": last_log_time,
            "db_exists": DB_PATH.exists()
        }
    except Exception as e:
        st.error(f"Error checking status: {e}")
        return {"running": False, "db_exists": DB_PATH.exists()}


def get_scheduled_games() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cst_tz = pytz.timezone('America/Chicago')
    today_cst = datetime.now(timezone.utc).astimezone(cst_tz)
    today_date_str = today_cst.strftime('%Y-%m-%d')
    
    cursor.execute("""
        SELECT game_id, home_team, away_team, start_time_utc, status, 
               current_period, game_clock, score_home, score_away
        FROM games
        WHERE game_date = ?
        ORDER BY start_time_utc
    """, (today_date_str,))
    
    games = []
    for row in cursor.fetchall():
        game = dict(row)
        game['start_time_cst'] = datetime.fromisoformat(game['start_time_utc']).astimezone(cst_tz)
        games.append(game)
    
    conn.close()
    return games


def get_game_triggers(game_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT trigger_type, scheduled_time_utc, fired_at_utc, status
        FROM triggers
        WHERE game_id = ?
        ORDER BY scheduled_time_utc
    """, (game_id,))
    
    triggers = []
    for row in cursor.fetchall():
        trigger = dict(row)
        trigger['scheduled_time_cst'] = datetime.fromisoformat(trigger['scheduled_time_utc']).astimezone(pytz.timezone('America/Chicago'))
        if trigger['fired_at_utc']:
            trigger['fired_time_cst'] = datetime.fromisoformat(trigger['fired_at_utc']).astimezone(pytz.timezone('America/Chicago'))
        triggers.append(trigger)
    
    conn.close()
    return triggers


def get_next_trigger(game_id: str) -> Optional[Dict[str, Any]]:
    triggers = get_game_triggers(game_id)
    now_utc = datetime.now(timezone.utc)
    
    for trigger in triggers:
        if trigger['status'] == 'scheduled':
            trigger_time = datetime.fromisoformat(trigger['scheduled_time_utc'])
            if trigger_time > now_utc:
                return trigger
    return None


def format_countdown(trigger_time: datetime) -> str:
    now = datetime.now(timezone.utc)
    delta = trigger_time - now
    
    if delta.total_seconds() < 0:
        return "NOW"
    
    total_seconds = delta.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def trigger_prediction(game_id: str, trigger_type: str) -> bool:
    try:
        from worker.triggers import TriggerFirer
        
        db_path = Path(__file__).parent.parent / "data" / "automation.db"
        firer = TriggerFirer(db_path, dry_run=False)
        
        success = firer.fire_trigger(game_id, trigger_type)
        
        if success:
            st.success(f"Successfully triggered {trigger_type} prediction for {game_id}")
            st.rerun()
        else:
            st.error(f"Failed to trigger {trigger_type} prediction for {game_id}")
        
        return success
    except Exception as e:
        st.error(f"Error triggering prediction: {e}")
        return False


st.title("PerryPicks Automation Monitor")
st.markdown("---")

st.subheader("Automation Status")

col1, col2, col3 = st.columns(3)
status = get_automation_status()

with col1:
    if status['running']:
        st.success("RUNNING")
    else:
        st.error("STOPPED")
    st.caption("Process Status")

with col2:
    st.info(f"{status['db_exists']}")
    st.caption("Database")

with col3:
    if status['last_log_time']:
        time_ago = datetime.now(timezone.utc) - status['last_log_time']
        st.caption(f"Last log: {time_ago.seconds // 60}m ago")
    else:
        st.caption("No logs found")

st.markdown("---")
st.caption("Auto-refreshes every 30 seconds")

st.subheader(f"Games Scheduled for {datetime.now(pytz.timezone('America/Chicago')).strftime('%B %d, %Y')}")

games = get_scheduled_games()

if not games:
    st.info("No games scheduled for today")
    st.caption("Games will appear here when automation detects them from NBA API")
else:
    total_games = len(games)
    completed_games = len([g for g in games if g['status'] != 'Scheduled'])
    in_progress = len([g for g in games if g['status'] == 'In Progress'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games", total_games)
    with col2:
        st.metric("Scheduled", total_games - completed_games)
    with col3:
        st.metric("In Progress", in_progress)
    with col4:
        st.metric("Completed", completed_games)
    
    st.markdown("---")
    
    for i, game in enumerate(games, 1):
        with st.expander(f"{i}. {game['away_team']} @ {game['home_team']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Game ID:** `{game['game_id']}`")
                st.write(f"**Status:** {game['status']}")
                
                if game['status'] == 'In Progress':
                    st.write(f"**Score:** {game['away_team']} {game['score_away']} - {game['score_home']} {game['home_team']}")
                    if game['current_period'] > 0:
                        st.write(f"**Period:** Q{game['current_period']} - {game['game_clock']}")
                elif game['status'] == 'Final':
                    st.write(f"**Final Score:** {game['away_team']} {game['score_away']} - {game['score_home']} {game['home_team']}")
            
            with col2:
                st.write(f"**Start Time:** {game['start_time_cst'].strftime('%I:%M %p')}")
                st.write(f"**Date:** {game['start_time_cst'].strftime('%A, %B %d')}")
            
            st.markdown("---")
            
            triggers = get_game_triggers(game['game_id'])
            next_trigger = get_next_trigger(game['game_id'])
            
            if triggers:
                st.write("**Triggers:**")
                
                for trigger in triggers:
                    time_str = trigger['scheduled_time_cst'].strftime('%I:%M %p')
                    
                    if trigger['status'] == 'fired':
                        st.caption(f"~~{trigger['trigger_type']}~~ at {time_str} Fired at {trigger['fired_time_cst'].strftime('%I:%M %p')}")
                    elif trigger['status'] == 'scheduled':
                        st.caption(f"⏳ {trigger['trigger_type']} at {time_str}")
                    else:
                        is_next = (next_trigger and next_trigger['trigger_type'] == trigger['trigger_type'])
                        if is_next:
                            countdown = format_countdown(datetime.fromisoformat(trigger['scheduled_time_utc']))
                            st.markdown(f"🔴 **{trigger['trigger_type']}** - **{countdown}**")
                        else:
                            st.caption(f"📅 {trigger['trigger_type']} at {time_str}")
            
            st.markdown("---")
            
            st.write("**Manual Trigger:**")
            
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button(f"Pre-Game Prediction", key=f"pregame_{game['game_id']}"):
                    trigger_prediction(game['game_id'], 'PRE_GAME')
            
            with btn_col2:
                if st.button(f"Halftime Prediction", key=f"halftime_{game['game_id']}"):
                    trigger_prediction(game['game_id'], 'HALFTIME')

st.markdown("---")
st.caption("PerryPicks v3 Automation Monitor • Auto-refreshes every 30s")
