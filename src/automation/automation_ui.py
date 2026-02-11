"""UI Components and Helpers for PerryPicks v3 Automation Manager.

Provides reusable components for the Streamlit automation interface.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, date
from pathlib import Path
import streamlit as st
import pandas as pd
import datetime as dt
import pandas as pd
import threading
import time
from core.storage import GameStorage
from core.env import load_environment
from src.data.game_data import fetch_game_by_id
from core.env import load_environment
from src.automation import (
    PostQueue,
    SocialMediaManager,
    AutomationOrchestrator,
)
from src.automation.post_queue import PostStatus

logger = logging.getLogger(__name__)

# Project root path for running scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Global variables to track running automation
_automation_thread = None
_automation_stop_event = None
_automation_monitor = None  # Keep reference to monitor for stopping
_queue_processor = None  # Background queue processor
_queue_processor_thread = None  # Queue processor thread

# Session state keys
SESSION_STATE_ORCHESTRATOR = "automation_orchestrator"
SESSION_STATE_PLATFORMS = "automation_platforms"
SESSION_STATE_SCHEDULE = "automation_schedule"
SESSION_STATE_AUTOMATION_RUNNING = "automation_running"
SESSION_STATE_AUTOMATION_STATUS = "automation_status"
SESSION_STATE_QUEUE_PROCESSOR_RUNNING = "queue_processor_running"
SESSION_STATE_QUEUE_PROCESSOR_STATUS = "queue_processor_status"


def init_session_state():
    """Initialize Streamlit session state for automation manager."""
    if SESSION_STATE_ORCHESTRATOR not in st.session_state:
        st.session_state[SESSION_STATE_ORCHESTRATOR] = None
        st.session_state["orchestrator_error"] = None
    if SESSION_STATE_PLATFORMS not in st.session_state:
        st.session_state[SESSION_STATE_PLATFORMS] = None
    if SESSION_STATE_SCHEDULE not in st.session_state:
        st.session_state[SESSION_STATE_SCHEDULE] = {
            "enabled": False,
            "poll_interval": 15,
        }
    if SESSION_STATE_AUTOMATION_RUNNING not in st.session_state:
        st.session_state[SESSION_STATE_AUTOMATION_RUNNING] = False
    if SESSION_STATE_AUTOMATION_STATUS not in st.session_state:
        st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
            "status": "idle",
            "message": "",
            "last_update": None,
        }
    # FIX: Store automation thread/monitor in session state to survive reruns
    if "automation_thread" not in st.session_state:
        st.session_state["automation_thread"] = None
    if "automation_monitor" not in st.session_state:
        st.session_state["automation_monitor"] = None
    if "automation_stop_event" not in st.session_state:
        st.session_state["automation_stop_event"] = None


def get_orchestrator(dry_run: bool = False) -> Optional[AutomationOrchestrator]:
    """Get or create orchestrator instance."""
    # Check for previous error
    if st.session_state.get("orchestrator_error"):
        st.error(f"Previous error: {st.session_state['orchestrator_error']}")
    
    if st.session_state.get(SESSION_STATE_ORCHESTRATOR) is None:
        try:
            st.session_state[SESSION_STATE_ORCHESTRATOR] = AutomationOrchestrator(
                dry_run=dry_run,
            )
            logger.info("Automation orchestrator initialized")
        except Exception as e:
            logger.error(f"Error creating orchestrator: {e}")
            st.session_state["orchestrator_error"] = str(e)
            st.error(f"Error initializing automation: {e}")
            return None
    
    return st.session_state.get(SESSION_STATE_ORCHESTRATOR)


def reset_orchestrator():
    """Reset orchestrator instance."""
    if SESSION_STATE_ORCHESTRATOR in st.session_state:
        del st.session_state[SESSION_STATE_ORCHESTRATOR]


def get_queue() -> PostQueue:
    """Get post queue instance."""
    orchestrator = get_orchestrator()
    if orchestrator:
        try:
            return orchestrator.social_manager.queue
        except Exception as e:
            logger.error(f"Error getting queue: {e}")
            # Return empty queue for display
            return PostQueue()
    else:
        return PostQueue()


def delete_post_from_queue(post_id: str) -> Dict[str, Any]:
    """Delete a post from the queue.
    
    Args:
        post_id: Post ID to delete
        
    Returns:
        Result dict with success flag
    """
    queue = get_queue()
    success = queue.delete_post(post_id)
    
    if success:
        return {
            "success": True,
            "message": f"Post {post_id} deleted from queue",
        }
    else:
        return {
            "success": False,
            "message": f"Post {post_id} not found in queue",
        }


def render_status_card(
    title: str,
    value: str,
    color: str = "blue",
    icon: str = "📊",
):
    """Render a status card."""
    color_map = {
        "blue": "primary",
        "green": "success",
        "red": "danger",
        "yellow": "warning",
    }
    
    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: white;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 0.875rem; color: #666; margin-top: 0.5rem;">{title}</div>
            <div style="
                font-size: 1.5rem;
                font-weight: bold;
                color: #1976d2;
                margin-top: 0.25rem;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_platform_status(
    platforms: List[str],
    enabled_platforms: set,
):
    """Render platform status indicators."""
    cols = st.columns(len(platforms))
    
    platform_info = {
        "twitter": {"name": "Twitter/X", "icon": "🐦"},
        "bluesky": {"name": "Bluesky", "icon": "🦋"},
        "discord": {"name": "Discord", "icon": "💬"},
    }
    
    for col, platform in zip(cols, platforms):
        info = platform_info.get(platform, {"name": platform, "icon": "⚙️"})
        enabled = platform in enabled_platforms
        
        with col:
            status_color = "green" if enabled else "gray"
            status_text = "✅ Enabled" if enabled else "❌ Disabled"
            
            st.markdown(
                f"""
                <div style="
                    padding: 0.75rem;
                    border-radius: 0.5rem;
                    background-color: {"#e8f5e9" if enabled else "#f5f5f5"};
                    border: 1px solid {"#c8e6c9" if enabled else "#e0e0e0"};
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem;">{info['icon']}</div>
                    <div style="font-weight: bold; margin-top: 0.25rem;">{info['name']}</div>
                    <div style="font-size: 0.875rem; color: {"#2e7d32" if enabled else "#757575"}; margin-top: 0.25rem;">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_queue_table(posts: List[Any], max_rows: int = 20):
    """Render queue as table with delete buttons."""
    if not posts:
        st.info("No posts in queue")
        return
    
    # Limit rows
    posts = posts[:max_rows]
    
    # Render each post as a row with delete button
    for i, post in enumerate(posts):
        # Parse created_at_utc (ISO 8601 string) and format it
        try:
            created_dt = datetime.fromisoformat(post.created_at_utc.replace("Z", "+00:00"))
            created_str = created_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing created_at_utc for {post.post_id}: {e}")
            created_str = post.created_at_utc[:16] if post.created_at_utc else "Unknown"
        
        # Status color
        status_colors = {
            "pending": "🟡",
            "posting": "🔄",
            "posted": "✅",
            "failed": "❌",
            "retrying": "⚠️",
        }
        status_emoji = status_colors.get(post.status.value, "❓")
        
        # Create row with columns
        col1, col2, col3, col4, col5, col6 = st.columns([0.15, 0.1, 0.1, 0.1, 0.3, 0.25])
        
        with col1:
            st.text(post.game_id)
        
        with col2:
            st.text(post.platform)
        
        with col3:
            st.text(status_emoji)
        
        with col4:
            st.text(created_str)
        
        with col5:
            content_preview = post.content[:40] + "..." if len(post.content) > 40 else post.content
            st.text(content_preview)
        
        with col6:
            # Delete button
            button_key = f"delete_{post.post_id}_{i}"
            if st.button("🗑️", key=button_key, help=f"Delete post {post.post_id}"):
                result = delete_post_from_queue(post.post_id)
                if result.get("success"):
                    st.toast(f"Deleted post: {post.post_id[:20]}...", icon="🗑️")
                    st.rerun()
                else:
                    st.error(f"Failed to delete post: {result.get('message')}")
        
        st.divider()


def render_post_content(content: str, max_chars: int = 500):
    """Render post content in a nice format."""
    truncated = content[:max_chars]
    if len(content) > max_chars:
        truncated += "..."
    
    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f5f5f5;
            border-left: 4px solid #1976d2;
            font-family: monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
        ">
{truncated}
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_posts_by_status(
    posts: List[Any],
    status_filter: List[PostStatus],
) -> List[Any]:
    """Filter posts by status."""
    if not status_filter:
        return posts
    
    return [p for p in posts if p.status in status_filter]


def filter_posts_by_platform(
    posts: List[Any],
    platform_filter: Optional[str],
) -> List[Any]:
    """Filter posts by platform."""
    if not platform_filter:
        return posts
    
    return [p for p in posts if p.platform == platform_filter]


def filter_posts_by_game(
    posts: List[Any],
    game_id_filter: Optional[str],
) -> List[Any]:
    """Filter posts by game ID."""
    if not game_id_filter:
        return posts
    
    return [p for p in posts if game_id_filter.lower() in p.game_id.lower()]


def get_game_options(date: dt.date = None) -> list:
    """Get list of available games for a specific date.
    
    Args:
        date: Date to fetch games for (default: today)
    
    Returns:
        List of ScoreboardGame objects
    """
    try:
        import datetime as dt
        from src.data.scoreboard import fetch_scoreboard, format_game_label
        
        # Use provided date or default to today
        target_date = date if date else dt.date.today()
        games = fetch_scoreboard(target_date, include_live=False)
        
        if not games:
            logger.warning(f"No games available for {target_date}")
            return []
        
        logger.info(f"Found {len(games)} games for {target_date}")
        return games
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        st.warning(f"Could not fetch games: {e}")
        return []

def get_game_ids(date: dt.date = None) -> List[str]:
    """Get list of game IDs for a specific date.
    
    Args:
        date: Date to fetch games for (default: today)
    
    Returns:
        List of game IDs
    """
    games = get_game_options(date)
    return [game.game_id for game in games]


def run_prediction(
    game_id: str,
    trigger_type: str = "pregame",
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    fetch_odds: bool = True,
    allow_duplicates: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run prediction for a single game.
    
    Args:
        game_id: Game ID to predict
        trigger_type: Trigger type (pregame, halftime, q3, halftime_retroactive, q3_retroactive)
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        fetch_odds: If True, fetch odds from API (default True). Set False for testing.
        allow_duplicates: If True, bypass duplicate detection
        progress_callback: Optional callback(progress, message) for UI updates
    
    Returns:
        Prediction results dictionary
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # CRITICAL LOG: Show what parameters we received
    logger.info(f"="*60)
    logger.info(f"RUN_PREDICTION CALLED")
    logger.info(f"  game_id: {game_id}")
    logger.info(f"  trigger_type: {trigger_type}")
    logger.info(f"  platforms: {platforms}")
    logger.info(f"  dry_run: {dry_run}")
    logger.info(f"  fetch_odds: {fetch_odds}")
    logger.info(f"  allow_duplicates: {allow_duplicates}")
    logger.info(f"="*60)
    
    orchestrator = get_orchestrator(dry_run=dry_run)
    if not orchestrator:
        logger.error("Orchestrator not initialized")
        return {"success": False, "error": "Orchestrator not initialized"}
    
    # Map retroactive trigger types to actual prediction modes
    mode_mapping = {
        "pregame": "pregame",
        "halftime": "halftime",
        "halftime_retroactive": "halftime",
        "q3": "q3",
        "q3_retroactive": "q3",
        "total_day": "pregame",
    }
    
    mode = mode_mapping.get(trigger_type, trigger_type)
    
    results = orchestrator.run_predictions(
        game_ids=[game_id],
        trigger_type=trigger_type,
        mode=mode,  # Use mapped prediction mode
        fetch_odds=fetch_odds,
        allow_duplicates=allow_duplicates,
        progress_callback=progress_callback,
    )
    
    # Convert orchestrator results to simple success/error format
    # CRITICAL: Flatten the results so UI can access predictions and posted directly
    if results.get("errors") and len(results["errors"]) > 0:
        # Check if all errors were just duplicate posts
        all_duplicates = all("duplicate" in str(e.get("error", "")) for e in results["errors"])
        if all_duplicates:
            return {
                "success": True,
                "message": "Post already exists (duplicate)",
                "results": results,
                "predictions": results.get("predictions", []),  # FLATTEN
                "posted": results.get("posted", []),  # FLATTEN
                "errors": results.get("errors", []),  # FLATTEN
            }
        else:
            # Return the first error
            first_error = results["errors"][0].get("error", "Unknown error")
            return {
                "success": False,
                "error": first_error,
                "results": results,
                "predictions": results.get("predictions", []),  # FLATTEN
                "posted": results.get("posted", []),  # FLATTEN
                "errors": results.get("errors", []),  # FLATTEN
            }
    elif len(results.get("predictions", [])) > 0:
        # Successfully generated predictions - FLATTEN the results structure
        return {
            "success": True,
            "results": results,
            "predictions": results.get("predictions", []),  # FLATTEN - UI can access directly
            "posted": results.get("posted", []),  # FLATTEN - UI can access directly
            "errors": results.get("errors", []),  # FLATTEN - UI can access directly
        }
    else:
        return {
            "success": False,
            "error": "No predictions generated",
            "results": results,
            "predictions": results.get("predictions", []),  # FLATTEN
            "posted": results.get("posted", []),  # FLATTEN
            "errors": results.get("errors", []),  # FLATTEN
        }


def process_queue(max_posts: int = 10) -> Dict[str, Any]:
    """Process pending posts from queue."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"success": False, "error": "Orchestrator not initialized"}
    
    return orchestrator.process_post_queue(batch_size=max_posts)


def clear_processed_cache() -> Dict[str, Any]:
    """Clear processed predictions cache.
    
    This allows re-running predictions that were previously marked as processed.
    Useful for testing or re-posting failed predictions.
    
    Returns:
        Status dictionary with count of cleared entries
    """
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"success": False, "error": "Orchestrator not initialized"}
    
    try:
        count = orchestrator.clear_processed_predictions()
        return {
            "success": True,
            "message": f"Cleared {count} processed prediction entries from cache",
            "count": count,
        }
    except Exception as e:
        logger.error(f"Error clearing processed cache: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def get_statistics() -> Dict[str, Any]:
    """Get automation statistics."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {
            "error": "Orchestrator not initialized",
            "processed_predictions": 0,
            "queue_stats": {
                "total": 0,
                "pending": 0,
                "posted": 0,
                "failed": 0,
            },
            "enabled_platforms": [],
        }
    
    try:
        return orchestrator.get_stats()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {
            "error": str(e),
            "processed_predictions": 0,
            "queue_stats": {
                "total": 0,
                "pending": 0,
                "posted": 0,
                "failed": 0,
            },
            "enabled_platforms": [],
        }


def render_automation_status():
    """Render automation status and controls.
    
    Shows current automation status and provides buttons to start/stop monitoring.
    """
    st.subheader("🤖 Automation Status")
    
    # Get status
    status = get_automation_status()
    
    # Status card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status["running"]:
            st.success("✅ Running")
        else:
            st.warning("⏸️  Stopped")
    
    with col2:
        if status["thread_alive"]:
            st.success(f"🧵 {status['thread_name']}")
        else:
            st.info("🧵 No thread")
    
    with col3:
        status_text = status["status"].get("status", "unknown")
        if status_text == "running":
            st.info(f"📊 Monitoring")
        elif status_text == "stopped":
            st.info(f"⏹️  Idle")
        elif status_text == "error":
            st.error(f"❌ Error")
        else:
            st.info(f"ℹ️  {status_text}")
    
    # Details
    status_data = status["status"]
    if status_data:
        if status_data.get("message"):
            st.info(f"📝 {status_data['message']}")
        
        if status_data.get("last_update"):
            try:
                last_update = datetime.fromisoformat(status_data["last_update"])
                time_ago = datetime.now() - last_update
                time_str = format_timedelta(time_ago)
                st.caption(f"🕐 Last update: {time_str} ago")
            except:
                pass
    
    # Controls
    st.subheader("Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏹️  Stop Automation", disabled=not status["running"], key="auto_stop"):
            result = stop_automation()
            if result["success"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])
    
    with col2:
        if st.button("🔄 Refresh Status", key="auto_refresh"):
            st.rerun()
    
    with col3:
        if st.button("🚀 Force Evaluate", key="auto_force_eval"):
            with st.spinner("Evaluating triggers..."):
                result = force_evaluate_triggers(
                    platforms=["twitter"],
                    dry_run=True,
                )
                
                if result["success"]:
                    st.success(f"✓ Evaluated {result['games_evaluated']} games")
                    if result["triggers_fired"]:
                        st.info(f"⚡ Fired {len(result['triggers_fired'])} triggers")
                    if result["errors"]:
                        st.warning(f"⚠️  {len(result['errors'])} errors")
                else:
                    st.error("Failed to evaluate triggers")


def render_queue_processor_status():
    """Render queue processor status and controls.
    
    Shows queue processor status, statistics, and controls.
    """
    st.subheader("📨 Queue Processor")
    
    # Get status
    status = get_queue_processor_status()
    
    # Status card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status.get("running"):
            st.success("✅ Running")
        else:
            st.warning("⏸️  Stopped")
    
    with col2:
        if status.get("thread_alive"):
            st.success(f"🧵 {status.get('thread_name')}")
        else:
            st.info("🧵 No thread")
    
    with col3:
        poll_interval = status.get("poll_interval", 0)
        batch_size = status.get("batch_size", 0)
        st.info(f"⏱️  {poll_interval}s / {batch_size} posts")
    
    # Queue statistics
    queue_stats = status.get("queue", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", queue_stats.get("total", 0))
    
    with col2:
        st.metric("Pending", queue_stats.get("pending", 0), delta_color="normal")
    
    with col3:
        st.metric("Posted", queue_stats.get("posted", 0), delta_color="normal")
    
    with col4:
        st.metric("Failed", queue_stats.get("failed", 0), delta_color="inverse")
    
    # Processor statistics
    processor_stats = status.get("stats", {})
    
    if processor_stats:
        st.subheader("📊 Processor Stats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processed", processor_stats.get("processed", 0))
        
        with col2:
            st.metric("Failed Batches", processor_stats.get("failed", 0), delta_color="inverse")
        
        with col3:
            last_processed = processor_stats.get("last_processed_at", "Never")
            if last_processed != "Never":
                try:
                    dt = datetime.fromisoformat(last_processed)
                    time_ago = datetime.now() - dt
                    last_processed = format_timedelta(time_ago) + " ago"
                except:
                    pass
            st.caption(f"Last processed: {last_processed}")
    
    # Controls
    st.subheader("Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status.get("running"):
            if st.button("⏹️  Stop Queue Processor", key="qp_stop"):
                result = stop_queue_processor()
                if result["success"]:
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])
        else:
            if st.button("▶️  Start Queue Processor", key="qp_start"):
                result = start_queue_processor(
                    poll_interval=15,
                    batch_size=10,
                )
                if result["success"]:
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])
    
    with col2:
        if st.button("🔄 Refresh Status", key="qp_refresh"):
            st.rerun()
    
    with col3:
        if st.button("⚡ Process Now"):
            with st.spinner("Processing queue..."):
                result = process_queue_now(max_posts=50)
                
                if result["success"]:
                    st.success(f"✓ Processed {result['processed']} posts")
                else:
                    st.error(f"Failed: {result.get('error', 'Unknown error')}")


def format_timedelta(td: timedelta) -> str:
    """Format timedelta as human-readable string.
    
    Args:
        td: timedelta to format
        
    Returns:
        Formatted string (e.g., "5m 30s")
    """
    seconds = int(td.total_seconds())
    
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def get_platforms() -> set:
    """Get enabled platforms."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return set()
    
    return orchestrator.social_manager.enabled_platforms


def get_platform_status() -> Dict[str, bool]:
    """Get status of all platforms."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {}
    
    sm = orchestrator.social_manager
    return {
        "twitter": sm.is_platform_enabled("twitter"),
        "bluesky": sm.is_platform_enabled("bluesky"),
        "discord": sm.is_platform_enabled("discord"),
    }


def run_predictions_for_all_games(
    date: dt.date = None,
    trigger_type: str = "pregame",
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    fetch_odds: bool = True,
    allow_duplicates: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run predictions for all games on a specific date.
    
    Args:
        date: Date to predict for (default: today)
        trigger_type: Trigger type (pregame, halftime, q3)
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        fetch_odds: If True, fetch odds from API (default True). Set False for testing.
        allow_duplicates: If True, bypass duplicate detection
        progress_callback: Optional callback(progress, message) for UI updates
    
    Returns:
        Prediction results dictionary
    """
    orchestrator = get_orchestrator(dry_run=dry_run)
    if not orchestrator:
        return {"success": False, "error": "Orchestrator not initialized"}
    
    # Get game IDs for the date
    game_ids = get_game_ids(date)
    
    if not game_ids:
        return {
            "success": False,
            "error": "No games found for the selected date",
        }
    
    return orchestrator.run_predictions(
        game_ids=game_ids,
        trigger_type=trigger_type,
        mode=trigger_type,  # Use trigger type as prediction mode (pregame, halftime, q3)
        fetch_odds=fetch_odds,
        allow_duplicates=allow_duplicates,
        progress_callback=progress_callback,
    )

def queue_gamestate_conscious_posts(
    game_id: str,
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    allow_duplicates: bool = False,
) -> Dict[str, Any]:
    """Queue posts that will trigger at different game states.
    
    This creates 3 posts for each game:
    - Pregame: Triggers immediately
    - Halftime: Triggers when halftime is reached
    - Q3: Triggers when Q3 is reached
    
    Args:
        game_id: Game ID to queue posts for
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        allow_duplicates: If True, bypass duplicate detection
    
    Returns:
        Dictionary with results for each trigger type
    """
    results = {
        "pregame": None,
        "halftime": None,
        "q3": None,
        "errors": [],
    }
    
    trigger_types = ["pregame", "halftime", "q3"]
    
    for trigger_type in trigger_types:
        # OPTIMIZATION: Don't fetch odds when queueing posts
        # Odds should only be fetched when triggers actually fire (halftime/Q3)
        # This saves API credits - we don't need odds for posts that haven't triggered yet
        # Pregame posts: fetch odds=False (will fetch when posted)
        # Halftime posts: fetch odds=False (will fetch when halftime trigger fires)
        # Q3 posts: fetch odds=False (will fetch when Q3 trigger fires)
        result = run_prediction(
            game_id=game_id,
            trigger_type=trigger_type,
            platforms=platforms,
            dry_run=dry_run,
            fetch_odds=False,  # Don't fetch odds when queueing - save API credits
            allow_duplicates=allow_duplicates,
        )
        
        if result.get("success"):
            results[trigger_type] = result
        else:
            results["errors"].append({
                "trigger_type": trigger_type,
                "error": result.get("error", "Unknown error"),
            })
    
    return results

def run_total_day_view(
    date: dt.date = None,
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    fetch_odds: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """Generate total day view post with all games in a single table.
    
    This runs predictions for all games and creates a single aggregated post
    using the Option 3 table format (Discord full slate).
    
    Args:
        date: Date to predict for (default: today)
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        fetch_odds: If True, fetch odds from API (default True). Set False for testing.
        progress_callback: Optional callback(progress, message) for UI updates
    
    Returns:
        Prediction results dictionary
    """
    from src.predict_api import predict_game
    from src.automation.post_generator import PostGenerator
    
    # Initialize post generator
    post_generator = PostGenerator(
        include_odds=True,
        include_confidence=True,
        use_emojis=True,
        hashtags=["#NBAPredictions", "#PerryPicks"],
    )
    
    results = {
        "date": date,
        "total_games": 0,
        "predictions": [],
        "errors": [],
        "total_day_post": None,
    }
    
    # Get game IDs for the date
    game_ids = get_game_ids(date)
    
    if not game_ids:
        return {
            "success": False,
            "error": "No games found for the selected date",
            **results,
        }
    
    results["total_games"] = len(game_ids)
    successful_predictions = []
    
    # Run predictions for all games (without posting)
    for i, game_id in enumerate(game_ids, 1):
        try:
            # Update progress
            progress = i / len(game_ids) * 0.8  # Use 80% of progress for predictions
            message = f"Processing {game_id} ({i}/{len(game_ids)})..."
            logger.info(message)
            if progress_callback:
                progress_callback(progress, message)
            
            # Run prediction
            if progress_callback:
                progress_callback(progress, f"Predicting {game_id}...")
            
            prediction = predict_game(game_id, mode="pregame", fetch_odds=fetch_odds)
            
            if prediction and prediction.get("status") in ("success", "warning"):
                successful_predictions.append(prediction)
                results["predictions"].append(prediction)
                logger.info(f"Prediction successful for {game_id}")
            else:
                error_msg = prediction.get("error", "Unknown error") if isinstance(prediction, dict) else "Unknown error"
                results["errors"].append({
                    "game_id": game_id,
                    "error": error_msg,
                })
                logger.error(f"Prediction failed for {game_id}: {error_msg}")
        
        except Exception as e:
            results["errors"].append({
                "game_id": game_id,
                "error": str(e),
            })
            logger.error(f"Error processing {game_id}: {e}")
        
        if progress_callback:
            status_msg = f"✓ Completed {i}/{len(game_ids)} games"
            if len(results["errors"]) > 0:
                status_msg += f" ({len(results['errors'])} errors)"
            progress_callback(progress, status_msg)
    
    # Generate total day view post if we have predictions
    if successful_predictions:
        try:
            if progress_callback:
                progress_callback(0.9, "Generating total day view post...")
            
            # Create aggregated prediction dict with all games
            aggregated_prediction = {
                "status": "success",
                "games": successful_predictions,
                "model_used": "PREGAME_V3_FINAL",
                "game_id": f"total_day_{date.strftime('%Y%m%d')}",
                "trigger_type": "pregame",
            }
            
            # Generate the post
            post_content = post_generator.generate_pregame_post(
                aggregated_prediction,
                platform='discord'
            )
            
            # Queue the post for all platforms
            if progress_callback:
                progress_callback(0.95, "Queueing total day view post...")
            
            orchestrator = get_orchestrator(dry_run=dry_run)
            if orchestrator:
                # Determine which platforms to post to
                target_platforms = platforms or list(orchestrator.social_manager.enabled_platforms)
                target_platforms = [p for p in target_platforms if p in orchestrator.social_manager.enabled_platforms]
                
                # Queue post for each platform
                platform_results = {}
                # Add timestamp to make post_id unique (prevents duplicate detection on re-runs)
                from datetime import datetime
                timestamp = int(datetime.now().timestamp())
                
                for platform in target_platforms:
                    try:
                        post_id = orchestrator.social_manager.queue.enqueue(
                            game_id=f"total_day_{date.strftime('%Y%m%d')}_{timestamp}",
                            platform=platform,
                            content=post_content,
                            trigger_type="pregame_total_day",
                            max_retries=3,
                        )
                        
                        if post_id:
                            platform_results[platform] = {
                                "post_id": post_id,
                                "status": "queued",
                            }
                        else:
                            # Duplicate post
                            platform_results[platform] = {
                                "status": "duplicate",
                                "reason": "Duplicate post detected",
                            }
                    except Exception as e:
                        platform_results[platform] = {
                            "status": "error",
                            "error": str(e),
                        }
                
                results["total_day_post"] = {
                    "content": post_content,
                    "platforms": platform_results,
                    "success": True,
                }
                
                if progress_callback:
                    progress_callback(1.0, f"✓ Total day view post queued!")
            else:
                results["total_day_post"] = {
                    "content": post_content,
                    "error": "Orchestrator not available to queue post",
                    "success": False,
                }
        
        except Exception as e:
            results["errors"].append({
                "game_id": "total_day",
                "error": f"Failed to generate total day post: {str(e)}",
            })
            logger.error(f"Error generating total day post: {e}")
    else:
        results["total_day_post"] = {
            "error": "No successful predictions to aggregate",
            "success": False,
        }
    
    results["success"] = len(results["errors"]) == 0 and results["total_day_post"] and results["total_day_post"].get("success")
    
    return results

def run_full_day_automation(
    date: dt.date = None,
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    fetch_odds: bool = True,
    allow_retroactive: bool = False,
    allow_duplicates: bool = False,
    enable_background_monitoring: bool = False,
    rate_limit_delay: float = 1.0,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run complete full day automation - one click for everything.
    
    This creates:
    1. Individual pregame predictions for all games
    2. Total day summary post (Option 3 table format)
    3. Halftime triggers for each game (game-time aware, auto-posts at halftime)
    4. Q3 triggers for each game (game-time aware, auto-posts at Q3)
    
    All posts are queued automatically and will post at the appropriate times.
    
    Args:
        date: Date to predict for (default: today)
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        fetch_odds: If True, fetch odds from API (default True). Set False for testing.
        allow_retroactive: If True, generate halftime/Q3 predictions for completed games
        allow_duplicates: If True, bypass duplicate detection (can be combined with allow_retroactive)
        enable_background_monitoring: If True, start background monitoring for real-time triggers
        rate_limit_delay: Seconds to wait between API calls (default 1.0s)
        progress_callback: Optional callback(progress, message) for UI updates
        
    Returns:
        Comprehensive results dictionary with all automation results
    """
    global _automation_monitor  # Declare global for assignment
    
    # Auto-freshen data: Run game scanner to import today's games and update watermark
    # This ensures fresh data before any predictions are generated
    try:
        if progress_callback:
            progress_callback(0.01, "Freshening game data...")
        
        logger.info("Running game scanner to freshen data...")
        result = agent_run_shell_command(
            command=f"uv run python scripts/automation/game_scanner.py --date {date.isoformat()}",
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        
        if result.get("success"):
            logger.info("Game scanner completed successfully - data freshened")
            if progress_callback:
                progress_callback(0.03, "✓ Data freshened")
        else:
            logger.warning(f"Game scanner failed: {result.get('error')}")
            # Don't fail the whole automation if scanner fails, just log a warning
            if progress_callback:
                progress_callback(0.03, "⚠ Game scanner failed (continuing...)")
    
    except Exception as e:
        logger.warning(f"Error running game scanner: {e}")
        # Don't fail the whole automation if scanner fails, just log a warning
        if progress_callback:
            progress_callback(0.03, "⚠ Game scanner error (continuing...)")
    
    # Get game IDs for the date
    game_ids = get_game_ids(date)
    
    if not game_ids:
        return {
            "success": False,
            "error": "No games found for the selected date",
        }
    
    results = {
        "date": date,
        "total_games": len(game_ids),
        "pregame_individual": None,
        "pregame_day_summary": None,
        "halftime_triggers": None,
        "q3_triggers": None,
        "background_monitoring": None,
        "errors": [],
    }
    
    # Progress stages:
    # 0-25%: Individual pregame predictions
    # 25-50%: Total day summary
    # 50-75%: Halftime triggers
    # 75-100%: Q3 triggers
    
    # Stage 1: Individual Pregame Predictions
    try:
        if progress_callback:
            progress_callback(0.05, "Starting full day automation...")
        
        pregame_individual = run_predictions_for_all_games(
            date=date,
            trigger_type="pregame",
            platforms=platforms,
            dry_run=dry_run,
            fetch_odds=False,  # Don't fetch odds for pregame (save API calls)
            allow_duplicates=allow_duplicates,
            progress_callback=lambda p, m: progress_callback(0.05 + (p * 0.20), m) if progress_callback else None,
        )
        
        results["pregame_individual"] = pregame_individual
        
        # Collect errors
        if pregame_individual.get("errors"):
            results["errors"].extend(pregame_individual.get("errors", []))
        
        if progress_callback:
            success_count = len(pregame_individual.get("predictions", []))
            error_count = len(pregame_individual.get("errors", []))
            progress_callback(0.25, f"✓ Pregame predictions: {success_count} games, {error_count} errors")
    
    except Exception as e:
        results["errors"].append({
            "stage": "pregame_individual",
            "error": str(e),
        })
        logger.error(f"Error in pregame individual predictions: {e}")
    
    # Stage 2: Total Day Summary
    try:
        if progress_callback:
            progress_callback(0.26, "Generating total day summary...")
        
        pregame_day_summary = run_total_day_view(
            date=date,
            platforms=platforms,
            dry_run=dry_run,
            fetch_odds=False,  # Don't fetch odds for pregame total day view (save API calls)
            progress_callback=lambda p, m: progress_callback(0.26 + (p * 0.24), m) if progress_callback else None,
        )
        
        results["pregame_day_summary"] = pregame_day_summary
        
        # Collect errors
        if pregame_day_summary.get("errors"):
            results["errors"].extend(pregame_day_summary.get("errors", []))
        
        if progress_callback:
            success = pregame_day_summary.get("success", False)
            progress_callback(0.50, f"✓ Total day summary: {'Success' if success else 'Failed'}")
    
    except Exception as e:
        results["errors"].append({
            "stage": "pregame_day_summary",
            "error": str(e),
        })
        logger.error(f"Error in total day summary: {e}")
    
    # Stage 3: Halftime Triggers (game-time aware)
    halftime_trigger_results = {
        "total_games": len(game_ids),
        "successful": [],
        "errors": [],
        "skipped_not_started": [],
        "skipped_not_halftime": [],
        "skipped_completed": [],  # Games already finished
    }
    
    try:
        if progress_callback:
            progress_callback(0.51, "Evaluating halftime triggers...")
        
        for i, game_id in enumerate(game_ids, 1):
            try:
                # Check game state before attempting prediction
                game_data = fetch_game_by_id(game_id)
                
                if not game_data:
                    # Game not found - might be in future
                    halftime_trigger_results["skipped_not_started"].append(game_id)
                    logger.info(f"Halftime trigger skipped (game not found): {game_id}")
                else:
                    game_status = game_data.get("gameStatus")
                    period = game_data.get("period", 0)
                    
                    # gameStatus: 0=not started, 1=Q1, 2=Q2, 3=Q3, 4=Q4, 5=OT, 6=Final
                    # Skip if game already completed (unless retroactive is enabled or duplicates allowed)
                    if game_status >= 6:  # Final
                        if allow_retroactive or allow_duplicates:
                            # Retroactive mode or duplicate override - generate prediction anyway
                            result = run_prediction(
                                game_id=game_id,
                                trigger_type="halftime_retroactive",
                                platforms=platforms,
                                dry_run=dry_run,
                                fetch_odds=True,  # Fetch odds for retroactive halftime predictions
                                allow_duplicates=True,  # Allow duplicates for retroactive posts
                            )
                            
                            if result.get("success"):
                                halftime_trigger_results["successful"].append(game_id)
                                logger.info(f"Retroactive halftime prediction generated: {game_id}")
                            else:
                                error = result.get("error", "Unknown error")
                                halftime_trigger_results["errors"].append({
                                    "game_id": game_id,
                                    "error": error,
                                })
                                logger.error(f"Failed to generate retroactive halftime prediction for {game_id}: {error}")
                        else:
                            # Normal mode - skip completed games
                            halftime_trigger_results["skipped_completed"].append(game_id)
                            logger.info(f"Halftime trigger skipped (game already completed): {game_id}")
                    elif game_status < 2 or (game_status == 2 and period < 3):
                        # Game not at halftime yet - skip for now
                        halftime_trigger_results["skipped_not_started"].append(game_id)
                        logger.info(f"Halftime trigger skipped (not at halftime yet): {game_id} (status={game_status}, period={period})")
                    else:
                        # Game is at halftime or past - generate prediction
                        result = run_prediction(
                            game_id=game_id,
                            trigger_type="halftime",
                            platforms=platforms,
                            dry_run=dry_run,
                            fetch_odds=True,  # Fetch odds for halftime predictions
                        )
                        
                        if result.get("success"):
                            halftime_trigger_results["successful"].append(game_id)
                            logger.info(f"Halftime prediction generated: {game_id}")
                        else:
                            error = result.get("error", "Unknown error")
                            halftime_trigger_results["errors"].append({
                                "game_id": game_id,
                                "error": error,
                            })
                            logger.error(f"Failed to generate halftime prediction for {game_id}: {error}")
                        
                        # Rate limiting - add delay between requests
                        if rate_limit_delay > 0:
                            import time
                            time.sleep(rate_limit_delay)
                
                # Update progress
                progress = 0.51 + (i / len(game_ids) * 0.24)
                if progress_callback:
                    progress_callback(progress, f"Halftime: {i}/{len(game_ids)} games")
            
            except Exception as e:
                halftime_trigger_results["errors"].append({
                    "game_id": game_id,
                    "error": str(e),
                })
                logger.error(f"Error in halftime trigger for {game_id}: {e}")
        
        results["halftime_triggers"] = halftime_trigger_results
        results["errors"].extend(halftime_trigger_results["errors"])
        
        if progress_callback:
            success_count = len(halftime_trigger_results["successful"])
            skipped_count = (len(halftime_trigger_results["skipped_not_started"]) + 
                           len(halftime_trigger_results["skipped_not_halftime"]) +
                           len(halftime_trigger_results["skipped_completed"]))
            error_count = len(halftime_trigger_results["errors"])
            status_msg = f"✓ Halftime: {success_count} generated, {skipped_count} skipped, {error_count} errors"
            progress_callback(0.75, status_msg)
    
    except Exception as e:
        results["errors"].append({
            "stage": "halftime_triggers",
            "error": str(e),
        })
        logger.error(f"Error in halftime triggers: {e}")
    
    # Stage 4: Q3 Triggers (game-time aware)
    q3_trigger_results = {
        "total_games": len(game_ids),
        "successful": [],
        "errors": [],
        "skipped_not_started": [],
        "skipped_not_q3": [],
        "skipped_completed": [],  # Games already finished
    }
    
    try:
        if progress_callback:
            progress_callback(0.76, "Evaluating Q3 triggers...")
        
        for i, game_id in enumerate(game_ids, 1):
            try:
                # Check game state before attempting prediction
                game_data = fetch_game_by_id(game_id)
                
                if not game_data:
                    # Game not found - might be in future
                    q3_trigger_results["skipped_not_started"].append(game_id)
                    logger.info(f"Q3 trigger skipped (game not found): {game_id}")
                else:
                    game_status = game_data.get("gameStatus")
                    period = game_data.get("period", 0)
                    
                    # gameStatus: 0=not started, 1=Q1, 2=Q2, 3=Q3, 4=Q4, 5=OT, 6=Final
                    # Skip if game already completed (unless retroactive is enabled or duplicates allowed)
                    if game_status >= 6:  # Final
                        if allow_retroactive or allow_duplicates:
                            # Retroactive mode or duplicate override - generate prediction anyway
                            result = run_prediction(
                                game_id=game_id,
                                trigger_type="q3_retroactive",
                                platforms=platforms,
                                dry_run=dry_run,
                                fetch_odds=True,  # Fetch odds for retroactive Q3 predictions
                                allow_duplicates=True,  # Allow duplicates for retroactive posts
                            )
                            
                            if result.get("success"):
                                q3_trigger_results["successful"].append(game_id)
                                logger.info(f"Retroactive Q3 prediction generated: {game_id}")
                            else:
                                error = result.get("error", "Unknown error")
                                q3_trigger_results["errors"].append({
                                    "game_id": game_id,
                                    "error": error,
                                })
                                logger.error(f"Failed to generate retroactive Q3 prediction for {game_id}: {error}")
                        else:
                            # Normal mode - skip completed games
                            q3_trigger_results["skipped_completed"].append(game_id)
                            logger.info(f"Q3 trigger skipped (game already completed): {game_id}")
                    elif game_status < 3 or (game_status == 3 and period < 4):
                        # Game not at Q3 yet - skip for now
                        q3_trigger_results["skipped_not_started"].append(game_id)
                        logger.info(f"Q3 trigger skipped (not at Q3 yet): {game_id} (status={game_status}, period={period})")
                    else:
                        # Game is at Q3 or past - generate prediction
                        result = run_prediction(
                            game_id=game_id,
                            trigger_type="q3",
                            platforms=platforms,
                            dry_run=dry_run,
                            fetch_odds=True,  # Fetch odds for Q3 predictions
                        )
                        
                        if result.get("success"):
                            q3_trigger_results["successful"].append(game_id)
                            logger.info(f"Q3 prediction generated: {game_id}")
                        else:
                            error = result.get("error", "Unknown error")
                            q3_trigger_results["errors"].append({
                                "game_id": game_id,
                                "error": error,
                            })
                            logger.error(f"Failed to generate Q3 prediction for {game_id}: {error}")
                        
                        # Rate limiting - add delay between requests
                        if rate_limit_delay > 0:
                            import time
                            time.sleep(rate_limit_delay)
                
                # Update progress
                progress = 0.76 + (i / len(game_ids) * 0.24)
                if progress_callback:
                    progress_callback(progress, f"Q3: {i}/{len(game_ids)} games")
            
            except Exception as e:
                q3_trigger_results["errors"].append({
                    "game_id": game_id,
                    "error": str(e),
                })
                logger.error(f"Error in Q3 trigger for {game_id}: {e}")
        
        results["q3_triggers"] = q3_trigger_results
        results["errors"].extend(q3_trigger_results["errors"])
        
        if progress_callback:
            success_count = len(q3_trigger_results["successful"])
            skipped_count = (len(q3_trigger_results["skipped_not_started"]) + 
                           len(q3_trigger_results["skipped_not_q3"]) +
                           len(q3_trigger_results["skipped_completed"]))
            error_count = len(q3_trigger_results["errors"])
            status_msg = f"✓ Q3: {success_count} generated, {skipped_count} skipped, {error_count} errors"
            progress_callback(1.0, status_msg)
    
    except Exception as e:
        results["errors"].append({
            "stage": "q3_triggers",
            "error": str(e),
        })
        logger.error(f"Error in Q3 triggers: {e}")
    
    # Stage 5: Start Full Background Monitoring with Trigger Engine (if enabled)
    global _automation_monitor
    
    if enable_background_monitoring:
        try:
            if progress_callback:
                progress_callback(1.0, "Starting full background monitoring...")
            
            # Import required classes
            from src.automation.game_state_monitor import GameStateMonitor
            from src.automation.trigger_engine import TriggerEngine
            from src.automation.auto_queue_processor import AutoQueueProcessor
            
            # Initialize monitor
            monitor = GameStateMonitor(poll_interval_seconds=30)
            # Note: Don't assign to _automation_monitor here to avoid scope issues
            # The monitor is captured by closure in monitor_loop()
            
            # Initialize queue processor
            queue_processor = AutoQueueProcessor(
                social_manager=None,  # Will use orchestrator's social manager
            )
            
            # Initialize trigger engine
            trigger_engine = TriggerEngine(
                game_state_monitor=monitor,
                queue_processor=queue_processor,
                storage=GameStorage(),
            )
            
            # Store monitor globally so we can stop it later
            # (Must do this before defining monitor_loop to avoid scope issues)
            _automation_monitor = monitor
            
            # Initialize game states for all games
            for game_id in game_ids:
                monitor.update_game_state(game_id)
            
            # Start monitoring in background thread
            def monitor_loop():
                """Background monitoring loop."""
                try:
                    logger.info("Starting monitoring loop...")
                    st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                        "status": "running",
                        "message": "Monitoring games for halftime/Q3 triggers...",
                        "last_update": datetime.now().isoformat(),
                    }
                    
                    # Run monitoring loop
                    monitor.start()
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                        "status": "error",
                        "message": str(e),
                        "last_update": datetime.now().isoformat(),
                    }
                    import traceback
                    traceback.print_exc()
                finally:
                    st.session_state[SESSION_STATE_AUTOMATION_RUNNING] = False
                    logger.info("Monitoring loop stopped")
            
            # Start monitoring in background thread
            global _automation_thread, _automation_stop_event  # Note: monitor is captured by closure, not global
            
            # Stop any existing thread
            if _automation_thread and _automation_thread.is_alive():
                logger.info("Stopping existing automation thread...")
                if _automation_monitor is not None:
                    logger.info(f"Stopping monitor (running={_automation_monitor.running})...")
                    _automation_monitor.stop()
                _automation_thread.join(timeout=5)
            
            # Create stop event
            _automation_stop_event = threading.Event()
            
            # Start new thread
            _automation_thread = threading.Thread(
                target=monitor_loop,
                daemon=True,
                name="AutomationMonitor"
            )
            _automation_thread.start()
            
            # Update session state
            st.session_state[SESSION_STATE_AUTOMATION_RUNNING] = True
            
            # Return monitoring status
            results["background_monitoring"] = {
                "status": "running",
                "games_monitored": len(monitor.game_states),
                "poll_interval": 30,
                "thread_name": _automation_thread.name,
                "thread_alive": _automation_thread.is_alive(),
                "message": "Full background monitoring is running! Games are being monitored for halftime/Q3 triggers. Predictions will be auto-generated and queued when triggers fire.",
            }
            
            logger.info(f"Full background monitoring started for {len(game_ids)} games")
            logger.info(f"Monitoring thread started: {_automation_thread.name}")
            
            # Stage 6: Start Background Queue Processor
            if progress_callback:
                progress_callback(1.0, "Starting background queue processor...")
            
            logger.info("Starting background queue processor...")
            
            # Start queue processor
            queue_processor_result = start_queue_processor(
                poll_interval=15,
                batch_size=10,
            )
            
            if queue_processor_result.get("success"):
                results["queue_processor"] = {
                    "status": "running",
                    "message": queue_processor_result.get("message"),
                    "poll_interval": 15,
                    "batch_size": 10,
                }
                logger.info("Background queue processor started")
            else:
                results["errors"].append({
                    "stage": "queue_processor",
                    "error": queue_processor_result.get("message", "Unknown error"),
                })
                logger.error("Failed to start background queue processor")
        
        except Exception as e:
            results["errors"].append({
                "stage": "background_monitoring",
                "error": str(e),
            })
            logger.error(f"Error starting background monitoring: {e}")
            import traceback
            traceback.print_exc()
    
    # Finalize results
    total_errors = len(results["errors"])
    results["success"] = total_errors == 0
    
    return results


def stop_automation() -> Dict[str, Any]:
    """Stop running background automation.
    
    Returns:
        Status dictionary
    """
    global _automation_thread, _automation_stop_event, _automation_monitor
    
    result = {
        "success": False,
        "message": "",
        "thread_stopped": False,
    }
    
    try:
        if _automation_thread and _automation_thread.is_alive():
            logger.info("Stopping automation thread...")
            
            # Try to stop monitor via session state
            if SESSION_STATE_AUTOMATION_STATUS in st.session_state:
                st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                    "status": "stopping",
                    "message": "Stopping monitoring...",
                    "last_update": datetime.now().isoformat(),
                }
            
            # Stop queue processor first
            logger.info("Stopping queue processor...")
            queue_result = stop_queue_processor()
            if queue_result.get("success"):
                logger.info("Queue processor stopped")
            else:
                logger.warning(f"Queue processor did not stop: {queue_result.get('message')}")
            
            # Stop monitor
            if _automation_monitor is not None:
                logger.info(f"Stopping game state monitor (running={_automation_monitor.running})...")
                _automation_monitor.stop()
            
            # Set stop event
            if _automation_stop_event:
                _automation_stop_event.set()
            
            # Wait for thread to stop (timeout 10s - thread may be sleeping)
            _automation_thread.join(timeout=10)
            
            if _automation_thread.is_alive():
                result["thread_stopped"] = False
                result["message"] = "Thread still running (may be waiting on sleep)"
                logger.warning("Automation thread still running - monitor may be sleeping")
            else:
                result["thread_stopped"] = True
                result["message"] = "Automation stopped successfully"
                logger.info("Automation thread stopped")
            
            # Update session state
            st.session_state[SESSION_STATE_AUTOMATION_RUNNING] = False
            st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                "status": "stopped",
                "message": "Monitoring stopped",
                "last_update": datetime.now().isoformat(),
            }
        else:
            result["message"] = "No automation thread running"
            logger.info("No automation thread running")
        
        result["success"] = True
    
    except Exception as e:
        result["message"] = f"Error stopping automation: {e}"
        logger.error(f"Error stopping automation: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def get_automation_status() -> Dict[str, Any]:
    """Get current automation status.
    
    Returns:
        Status dictionary with running state and details
    """
    # FIX: Read from session state to survive Streamlit reruns
    automation_thread = st.session_state.get("automation_thread")
    
    return {
        "running": st.session_state.get(SESSION_STATE_AUTOMATION_RUNNING, False),
        "thread_alive": automation_thread.is_alive() if automation_thread else False,
        "thread_name": automation_thread.name if automation_thread else None,
        "status": st.session_state.get(SESSION_STATE_AUTOMATION_STATUS, {}),
    }


def get_monitored_games() -> Dict[str, Any]:
    """Get currently monitored game states.
    
    Returns:
        Dictionary with game_id -> game_state mapping
    """
    # FIX: Read from session state to survive Streamlit reruns
    automation_monitor = st.session_state.get("automation_monitor")
    
    try:
        if automation_monitor is None:
            return {}
        
        # Get game states from monitor
        # Works with both GameStateMonitor and GameStateService
        if hasattr(automation_monitor, 'game_monitor'):
            # GameStateService - get from game_monitor attribute
            states = automation_monitor.game_monitor.get_all_states()
        elif hasattr(automation_monitor, 'get_all_states'):
            # GameStateMonitor - direct call
            states = automation_monitor.get_all_states()
        else:
            logger.warning(f"automation_monitor type: {type(automation_monitor)} - no get_all_states method")
            states = {}
        
        # Convert to serializable dict
        result = {}
        for game_id, state in states.items():
            result[game_id] = state.to_dict()
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting monitored games: {e}")
        return {}


def stop_monitoring_game(game_id: str) -> Dict[str, Any]:
    """Stop monitoring a specific game.
    
    Args:
        game_id: Game ID to stop monitoring
        
    Returns:
        Status dictionary with success/failure result
    """
    # FIX: Read from session state to survive Streamlit reruns
    automation_monitor = st.session_state.get("automation_monitor")
    
    result = {
        "success": False,
        "message": "",
        "game_id": game_id,
    }
    
    try:
        if automation_monitor is None:
            result["message"] = "Game monitor not running"
            logger.warning("Cannot stop monitoring: automation_monitor is None")
            return result
        
        # Works with both GameStateMonitor and GameStateService
        if hasattr(automation_monitor, 'game_monitor'):
            # GameStateService - get from game_monitor attribute
            stopped = automation_monitor.game_monitor.stop_monitoring_game(game_id)
        elif hasattr(automation_monitor, 'stop_monitoring_game'):
            # GameStateMonitor - direct call
            stopped = automation_monitor.stop_monitoring_game(game_id)
        else:
            result["message"] = "Monitor does not support stop_monitoring_game"
            logger.warning(f"automation_monitor type: {type(automation_monitor)} - no stop_monitoring_game method")
            return result
        
        if stopped:
            result["success"] = True
            result["message"] = f"Stopped monitoring for {game_id}"
            logger.info(f"Successfully stopped monitoring for {game_id}")
        else:
            result["message"] = f"Game {game_id} was not being monitored"
            logger.warning(f"Failed to stop monitoring for {game_id}: game not in monitoring list")
        
        return result
    
    except Exception as e:
        result["message"] = f"Error stopping monitoring: {e}"
        logger.error(f"Error stopping monitoring for {game_id}: {e}")
        return result


def test_game_state_service_import() -> Dict[str, Any]:
    """Test GameStateService import and initialization.
    
    This is a diagnostic function to help troubleshoot startup issues.
    
    Returns:
        Dictionary with test results
    """
    result = {
        "success": False,
        "message": "",
        "steps": [],
    }
    
    # Step 1: Import test
    try:
        from src.automation.game_state_service import GameStateService
        result["steps"].append({"step": "import", "status": "success"})
    except Exception as e:
        result["steps"].append({"step": "import", "status": "error", "error": str(e)})
        result["message"] = f"Import failed: {e}"
        result["success"] = False
        return result
    
    # Step 2: Instantiation test
    try:
        service = GameStateService(
            poll_interval_seconds=30,
            platforms=None,
            dry_run=True,  # Use dry run for testing
        )
        result["steps"].append({"step": "instantiate", "status": "success"})
    except Exception as e:
        result["steps"].append({"step": "instantiate", "status": "error", "error": str(e)})
        result["message"] = f"Instantiation failed: {e}"
        result["success"] = False
        return result
    
    # Step 3: Attributes check
    try:
        has_monitor = hasattr(service, 'game_monitor')
        has_trigger = hasattr(service, 'trigger_engine')
        has_queue = hasattr(service, 'queue_processor')
        result["steps"].append({
            "step": "attributes",
            "status": "success",
            "has_game_monitor": has_monitor,
            "has_trigger_engine": has_trigger,
            "has_queue_processor": has_queue,
        })
    except Exception as e:
        result["steps"].append({"step": "attributes", "status": "error", "error": str(e)})
        result["message"] = f"Attributes check failed: {e}"
        result["success"] = False
        return result
    
    # Cleanup - don't actually start the service
    try:
        if hasattr(service, 'stop'):
            service.stop()
        result["steps"].append({"step": "cleanup", "status": "success"})
    except Exception as e:
        result["steps"].append({"step": "cleanup", "status": "warning", "error": str(e)})
    
    result["success"] = True
    result["message"] = "GameStateService is ready to use"
    return result


def start_game_state_monitor(
    poll_interval_seconds: int = 30,
) -> Dict[str, Any]:
    """Start game state service in background thread.
    
    This starts the full GameStateService which includes:
    - Game State Monitor (tracks live games)
    - Trigger Engine (evaluates halftime/Q3 triggers)
    - Auto Queue Processor (posts predictions automatically)
    
    This is the main service for dashboard toggle control.
    
    Args:
        poll_interval_seconds: How often to poll for game updates
    
    Returns:
        Status dictionary with success/failure result
    """
    # FIX: Use session state instead of globals to survive Streamlit reruns
    _automation_stop_event = st.session_state.get("automation_stop_event")
    _automation_thread = st.session_state.get("automation_thread")
    _automation_monitor = st.session_state.get("automation_monitor")
    
    result = {
        "success": False,
        "message": "",
        "running": False,
    }
    
    try:
        # Check if already running
        if _automation_thread and _automation_thread.is_alive():
            result["message"] = "Game state monitor is already running"
            result["running"] = True
            logger.warning("Game state monitor is already running")
            return result
        
        logger.info("Starting game state service...")
        
        # Import required classes
        try:
            from src.automation.game_state_service import GameStateService
            logger.info("Successfully imported GameStateService")
        except Exception as e:
            result["message"] = f"Failed to import GameStateService: {e}"
            logger.error(f"Failed to import GameStateService: {e}")
            import traceback
            traceback.print_exc()
            return result
        
        # Initialize GameStateService (includes monitor, trigger engine, and queue processor)
        try:
            service = GameStateService(
                poll_interval_seconds=poll_interval_seconds,
                platforms=None,  # Post to all enabled platforms
                dry_run=False,  # Actually post (not test mode)
            )
            logger.info("Successfully created GameStateService instance")
        except Exception as e:
            result["message"] = f"Failed to create GameStateService: {e}"
            logger.error(f"Failed to create GameStateService: {e}")
            import traceback
            traceback.print_exc()
            return result
        
        # FIX: Store service in session state so it survives Streamlit reruns
        st.session_state["automation_monitor"] = service
        
        # Start monitoring in background thread
        def monitor_loop():
            """Background monitoring loop."""
            try:
                logger.info("Starting game state service loop...")
                logger.info(f"Service poll interval: {service.poll_interval}s")
                logger.info(f"Service running flag: {service.running}")
                st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                    "status": "running",
                    "message": "Monitoring games for halftime/Q3 triggers and processing queue automatically...",
                    "last_update": datetime.now().isoformat(),
                }
                
                # Run service loop (includes monitor, triggers, and queue processing)
                logger.info("Calling service.start()...")
                service.start()
                logger.info("service.start() returned (service stopped)")
            
            except Exception as e:
                logger.error(f"Error in game state service loop: {e}")
                st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                    "status": "error",
                    "message": str(e),
                    "last_update": datetime.now().isoformat(),
                }
                import traceback
                traceback.print_exc()
            finally:
                st.session_state[SESSION_STATE_AUTOMATION_RUNNING] = False
                logger.info("Game state service loop stopped")
        
        # Start monitoring in background thread
        
        # Stop any existing thread
        if _automation_thread and _automation_thread.is_alive():
            logger.info("Stopping existing automation thread...")
            if _automation_monitor is not None:
                logger.info(f"Stopping service (running={_automation_monitor.running})...")
                _automation_monitor.stop()
            _automation_thread.join(timeout=5)
        
        # Create stop event
        _automation_stop_event = threading.Event()
        
        # Start new thread
        try:
            _automation_thread = threading.Thread(
                target=monitor_loop,
                daemon=True,
                name="GameStateService"
            )
            # FIX: Store thread in session state to survive Streamlit reruns
            st.session_state["automation_thread"] = _automation_thread
            st.session_state["automation_stop_event"] = threading.Event()
            _automation_thread.start()
            logger.info(f"Successfully started background thread: {_automation_thread.name}")
            
            # Small delay to ensure thread has started before we continue
            import time
            time.sleep(0.1)
            
            # Verify thread is alive
            if not _automation_thread.is_alive():
                raise RuntimeError("Thread failed to start or died immediately")
            
        except Exception as e:
            result["message"] = f"Failed to start background thread: {e}"
            logger.error(f"Failed to start background thread: {e}")
            import traceback
            traceback.print_exc()
            return result
        
        # Update session state
        try:
            st.session_state[SESSION_STATE_AUTOMATION_RUNNING] = True
            logger.info("Set SESSION_STATE_AUTOMATION_RUNNING = True")
        except Exception as e:
            logger.error(f"Failed to set session state: {e}")
        
        # Update status
        try:
            st.session_state[SESSION_STATE_AUTOMATION_STATUS] = {
                "status": "running",
                "message": f"Game state service started. Polling every {poll_interval_seconds}s. Triggers: halftime, Q3-5min.",
                "last_update": datetime.now().isoformat(),
            }
            logger.info("Updated SESSION_STATE_AUTOMATION_STATUS")
        except Exception as e:
            logger.error(f"Failed to update automation status: {e}")
        
        result["success"] = True
        result["message"] = f"Game state service started (monitoring, triggers, queue processing) - polling every {poll_interval_seconds}s"
        result["running"] = True
        result["thread_alive"] = _automation_thread.is_alive()
        result["thread_name"] = _automation_thread.name
        
        logger.info(f"Game state service started successfully")
        logger.info(f"Monitoring thread started: {_automation_thread.name}")
        logger.info(f"Thread alive: {_automation_thread.is_alive()}")
    
    except Exception as e:
        result["message"] = f"Error starting game state service: {e}"
        logger.error(f"Error starting game state service: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def force_evaluate_triggers(
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Force evaluation of triggers for all monitored games.
    
    Useful for manual trigger testing or catching up.
    
    Args:
        platforms: Platforms to post to
        dry_run: If True, don't actually post
        
    Returns:
        Evaluation results
    """
    from src.automation.trigger_engine import TriggerEngine
    from src.automation.game_state_monitor import GameStateMonitor
    from src.automation.auto_queue_processor import AutoQueueProcessor
    
    result = {
        "success": False,
        "games_evaluated": 0,
        "triggers_fired": [],
        "errors": [],
    }
    
    try:
        # Create temporary monitor
        monitor = GameStateMonitor(poll_interval_seconds=30)
        
        # Get today's games
        from core.storage import GameStorage
        storage = GameStorage()
        games = storage.load_games()
        
        # Evaluate all games
        game_ids = list(games.keys()) if games else []
        
        for game_id in game_ids:
            try:
                # Update game state
                monitor.update_game_state(game_id)
            except Exception as e:
                result["errors"].append({
                    "game_id": game_id,
                    "error": f"Failed to update game state: {e}",
                })
        
        result["games_evaluated"] = len(game_ids)
        result["success"] = True
        
        logger.info(f"Force evaluation complete: {len(game_ids)} games")
    
    except Exception as e:
        result["errors"].append({
            "error": f"Force evaluation failed: {e}",
        })
        logger.error(f"Error in force evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def start_queue_processor(
    poll_interval: int = 15,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """Start background queue processor.
    
    Args:
        poll_interval: Seconds between queue polls
        batch_size: Max posts to process per poll
        
    Returns:
        Status dictionary
    """
    global _queue_processor, _queue_processor_thread
    
    result = {
        "success": False,
        "message": "",
        "running": False,
    }
    
    try:
        # Check if already running
        if _queue_processor and _queue_processor.running:
            result["message"] = "Queue processor already running"
            result["running"] = True
            return result
        
        # Import here to avoid circular imports
        from src.automation.background_queue_processor import BackgroundQueueProcessor
        from src.automation.social_media_manager import SocialMediaManager
        
        # Get social manager
        social_manager = None
        orchestrator = get_orchestrator()
        if orchestrator and orchestrator.social_manager:
            social_manager = orchestrator.social_manager
        
        # Initialize queue processor
        _queue_processor = BackgroundQueueProcessor(
            poll_interval=poll_interval,
            batch_size=batch_size,
            social_manager=social_manager,
        )
        
        # Start queue processor
        success = _queue_processor.start()
        
        if success:
            # Update session state
            st.session_state[SESSION_STATE_QUEUE_PROCESSOR_RUNNING] = True
            st.session_state[SESSION_STATE_QUEUE_PROCESSOR_STATUS] = {
                "status": "running",
                "message": "Queue processor is running",
                "last_update": datetime.now().isoformat(),
            }
            
            result["success"] = True
            result["message"] = "Queue processor started successfully"
            result["running"] = True
            
            logger.info("Background Queue Processor started")
        else:
            result["message"] = "Failed to start queue processor"
            result["running"] = False
            logger.error("Failed to start Background Queue Processor")
    
    except Exception as e:
        result["message"] = f"Error starting queue processor: {e}"
        result["running"] = False
        logger.error(f"Error in start_queue_processor: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def stop_queue_processor() -> Dict[str, Any]:
    """Stop background queue processor.
    
    Returns:
        Status dictionary
    """
    global _queue_processor
    
    result = {
        "success": False,
        "message": "",
        "stopped": False,
    }
    
    try:
        if not _queue_processor or not _queue_processor.running:
            result["message"] = "Queue processor not running"
            result["stopped"] = True
            return result
        
        # Stop queue processor
        stopped = _queue_processor.stop(timeout=10)
        
        if stopped:
            # Update session state
            st.session_state[SESSION_STATE_QUEUE_PROCESSOR_RUNNING] = False
            st.session_state[SESSION_STATE_QUEUE_PROCESSOR_STATUS] = {
                "status": "stopped",
                "message": "Queue processor stopped",
                "last_update": datetime.now().isoformat(),
            }
            
            result["success"] = True
            result["message"] = "Queue processor stopped successfully"
            result["stopped"] = True
            
            logger.info("Background Queue Processor stopped")
        else:
            result["message"] = "Queue processor did not stop gracefully"
            result["stopped"] = False
            logger.warning("Queue processor did not stop gracefully")
    
    except Exception as e:
        result["message"] = f"Error stopping queue processor: {e}"
        result["stopped"] = False
        logger.error(f"Error in stop_queue_processor: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def get_queue_processor_status() -> Dict[str, Any]:
    """Get queue processor status.
    
    Returns:
        Status dictionary
    """
    global _queue_processor
    
    if not _queue_processor:
        return {
            "running": False,
            "initialized": False,
            "status": st.session_state.get(SESSION_STATE_QUEUE_PROCESSOR_STATUS, {}),
        }
    
    # Get detailed status
    status = _queue_processor.get_status()
    
    return {
        "running": status.get("running", False),
        "thread_alive": status.get("thread_alive", False),
        "thread_name": status.get("thread_name"),
        "poll_interval": status.get("poll_interval"),
        "batch_size": status.get("batch_size"),
        "stats": status.get("stats", {}),
        "queue": status.get("queue", {}),
        "status": st.session_state.get(SESSION_STATE_QUEUE_PROCESSOR_STATUS, {}),
    }


def process_queue_now(max_posts: int = None) -> Dict[str, Any]:
    """Process queue immediately (one-off).
    
    Args:
        max_posts: Max posts to process
        
    Returns:
        Processing results
    """
    global _queue_processor
    
    result = {
        "success": False,
        "processed": 0,
        "error": None,
    }
    
    try:
        if not _queue_processor:
            result["error"] = "Queue processor not initialized"
            return result
        
        # Process queue
        status = _queue_processor.process_now(max_posts=max_posts)
        
        if status.get("success"):
            processed = status.get("processed_predictions", 0)
            result["success"] = True
            result["processed"] = processed
            
            logger.info(f"Queue processed: {processed} posts")
        else:
            result["error"] = status.get("error", "Unknown error")
            logger.error(f"Failed to process queue: {result['error']}")
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing queue: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def refresh_data(): 
    """Refresh automation data (force reload)."""
    reset_orchestrator()
    
    # Set selected date to today if it exists in session state
    if "selected_manual_date" in st.session_state:
        st.session_state["selected_manual_date"] = dt.date.today()
    if "selected_dashboard_date" in st.session_state:
        st.session_state["selected_dashboard_date"] = dt.date.today()
    
    st.toast("Data refreshed!", icon="🔄")