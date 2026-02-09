"""PerryPicks v3 - Automation Manager.

Streamlit GUI for managing social media automation.

This is a SEPARATE app from the main PerryPicks v3 tool.
It does not impact manual research/reviews.

Usage:
    streamlit run pages/04_Automation_Manager.py
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

# Add project root to Python path (must be BEFORE any other imports)
# We're in pages/ directory, so we need to go up one level to get project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Page config
st.set_page_config(
    page_title="Automation Manager | PerryPicks v3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
from src.automation.automation_ui import (
    init_session_state,
    get_orchestrator,
    get_queue,
    get_statistics,
    get_platform_status,
    get_platforms,
    get_game_options,
    get_game_ids,
    run_prediction,
    run_predictions_for_all_games,
    run_total_day_view,
    run_full_day_automation,
    queue_gamestate_conscious_posts,
    process_queue,
    refresh_data,
    render_status_card,
    render_platform_status,
    render_queue_table,
    render_post_content,
    filter_posts_by_status,
    filter_posts_by_platform,
    filter_posts_by_game,
    SESSION_STATE_PLATFORMS,
    # Automation status functions
    get_automation_status,
    get_queue_processor_status,
    start_queue_processor,
    stop_queue_processor,
    render_automation_status,
    render_queue_processor_status,
)

from src.automation.game_state_monitor import GameStateMonitor, GameState
from src.automation.game_state_service import GameStateService

from src.automation.post_queue import PostStatus
from src.data.scoreboard import format_game_label

# Initialize session state
init_session_state()

# Initialize logging
logger = logging.getLogger(__name__)


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("# 🤖 Automation Manager")
        st.markdown("---")
        
        # Simple refresh button
        if st.button("🔄 Sidebar Refresh"):
            st.rerun()


def render_dashboard():
    """Render dashboard with statistics and game schedule."""
    st.markdown("## 📊 Dashboard")
    
    # Date filter for game schedule
    st.markdown("### Select Date")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "selected_dashboard_date" not in st.session_state:
            st.session_state["selected_dashboard_date"] = datetime.now().date()
        
        selected_date = st.date_input(
            "Dashboard Date",
            value=st.session_state["selected_dashboard_date"],
        )
        st.session_state["selected_dashboard_date"] = selected_date
    
    with col2:
        if st.button("🔄 Go to Dashboard Today"):
            st.session_state["selected_dashboard_date"] = datetime.now().date()
            st.rerun()
    
    st.markdown("---")
    
    # Get statistics
    stats = get_statistics()
    
    if "error" in stats:
        st.error(f"Error loading statistics: {stats['error']}")
        return
    
    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        processed = stats.get("processed_predictions", 0)
        render_status_card("Processed", str(processed), icon="🎯")
    
    with col2:
        queue_stats = stats.get("queue_stats", {})
        pending = queue_stats.get("pending", 0)
        render_status_card("Pending", str(pending), color="yellow", icon="⏳")
    
    with col3:
        posted = queue_stats.get("posted", 0)
        render_status_card("Posted", str(posted), color="green", icon="✅")
    
    with col4:
        failed = queue_stats.get("failed", 0)
        render_status_card("Failed", str(failed), color="red", icon="❌")
    
    st.markdown("---")
    
    # Platform status
    st.markdown("### Platform Status")
    enabled_platforms = stats.get("enabled_platforms", [])
    render_platform_status(
        platforms=["twitter", "bluesky", "discord"],
        enabled_platforms=set(enabled_platforms),
    )
    
    st.markdown("---")
    
    # Service Status (NEW)
    st.markdown("### 🚦 Service Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎮 Game Monitoring")
        
        # Get automation status
        automation_status = get_automation_status()
        
        # Status indicator
        if automation_status.get("running"):
            st.success("🟢 **LIVE**")
        else:
            st.warning("🔴 **STOPPED**")
        
        # Thread status
        if automation_status.get("thread_alive"):
            st.caption("Thread: Running")
        else:
            st.caption("Thread: Inactive")
        
        # Add quick link to Game State tab
        if st.button("Go to Game State Settings", key="dashboard_game_state_btn"):
            st.info("👆 Go to the 'Game State' tab to control game monitoring")
    
    with col2:
        st.markdown("#### 📨 Queue Processing")
        
        # Get queue processor status
        queue_status = get_queue_processor_status()
        
        # Status indicator
        if queue_status.get("running"):
            st.success("🟢 **LIVE**")
        else:
            st.warning("🔴 **STOPPED**")
        
        # Thread status
        if queue_status.get("thread_alive"):
            st.caption("Thread: Running")
        else:
            st.caption("Thread: Inactive")
        
        # Stats
        stats = queue_status.get("stats", {})
        processed = stats.get("processed", 0)
        st.caption(f"Posts processed: {processed}")
        
        # Add quick toggle
        if st.button("🔘 Toggle Queue Processing", key="dashboard_toggle_queue"):
            if queue_status.get("running"):
                # Stop it
                stop_queue_processor()
                st.success("Queue processing stopped")
                st.rerun()
            else:
                # Start it
                start_queue_processor(poll_interval=15, batch_size=10)
                st.success("Queue processing started")
                st.rerun()
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Dashboard Queue", use_container_width=True, key="dashboard_process_queue"):
            with st.spinner("Processing queue..."):
                result = process_queue(max_posts=10)
                processed = result.get('processed', 0)
                successful = result.get('successful', 0)
                failed = result.get('failed', 0)
                
                if processed > 0:
                    if successful > 0:
                        st.success(f"✅ Processed {processed} posts! ({successful} successful, {failed} failed)")
                        st.toast(f"Sent {successful} posts successfully!", icon="✅")
                    else:
                        st.error(f"❌ Processed {processed} posts but all failed ({failed} failures)")
                        st.toast("All posts failed to send", icon="❌")
                    
                    # Show error details if any posts failed
                    if failed > 0 and result.get('posts'):
                        with st.expander("🔍 Error Details", expanded=False):
                            for post in result['posts']:
                                if post.get('status') == 'failed':
                                    st.markdown(f"**Post:** `{post.get('post_id')}`")
                                    st.markdown(f"**Platform:** `{post.get('platform')}`")
                                    st.markdown(f"**Error:** `{post.get('error', 'Unknown error')}`")
                else:
                    st.info("ℹ️ No posts to process")
    
    with col2:
        st.info("Use the 'Queue' tab above to view the queue")
    
    with col3:
        st.info("Use the 'Settings' tab above to configure settings")
    
    st.markdown("---")
    
    # Game schedule for selected date
    st.markdown(f"### Game Schedule for {selected_date}")
    games = get_game_options(selected_date)
    
    if not games:
        st.info(f"No games scheduled for {selected_date}")
    else:
        # Display games in a table
        game_data = []
        for game in games:
            game_data.append({
                "Game ID": game.game_id,
                "Matchup": format_game_label(game),
                "Status": game.status_text or "Scheduled",
                "Period": game.period or "-",
                "Clock": game.clock or "-",
                "Score": f"{game.away_score}-{game.home_score}" if game.away_score is not None and game.home_score is not None else "-",
            })
        
        import pandas as pd
        df = pd.DataFrame(game_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Recent activity
    st.markdown("### Recent Activity")
    queue = get_queue()
    all_posts = queue.get_all_posts()
    
    # Get recent posts (last 10)
    # Parse created_at_utc for sorting (ISO 8601 string)
    def parse_created_at(post):
        try:
            from datetime import datetime
            return datetime.fromisoformat(post.created_at_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Return old date for posts that fail to parse (they'll sort to end)
            from datetime import datetime
            return datetime.min
    
    recent_posts = sorted(
        all_posts,
        key=parse_created_at,
        reverse=True,
    )[:10]
    
    if recent_posts:
        render_queue_table(recent_posts, max_rows=10)
    else:
        st.info("No recent activity")


def render_manual_predictions():
    """Render manual predictions interface."""
    st.markdown("## 🎮 Manual Predictions")
    
    st.markdown("Trigger predictions manually for specific games.")
    
    # Date filter
    st.markdown("### Select Date")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "selected_manual_date" not in st.session_state:
            st.session_state["selected_manual_date"] = datetime.now().date()
        
        selected_date = st.date_input(
            "Manual Date",
            value=st.session_state["selected_manual_date"],
        )
        st.session_state["selected_manual_date"] = selected_date
    
    with col2:
        if st.button("🔄 Go to Manual Today"):
            st.session_state["selected_manual_date"] = datetime.now().date()
            st.rerun()
    
    # Get available games for selected date
    games = get_game_options(selected_date)
    
    if not games:
        st.warning(f"No games available for {selected_date}. Try selecting a different date.")
        return
    
    st.markdown(f"**Found {len(games)} games for {selected_date}**")
    st.markdown("---")
    
    # Game selection with team names
    st.markdown("### Select Game")
    
    # Create options with team names
    game_options = {}
    for game in games:
        label = format_game_label(game)
        game_options[game.game_id] = label
    
    selected_game_id = st.selectbox(
        "Game",
        options=list(game_options.keys()),
        format_func=lambda x: game_options.get(x, x),
    )
    
    # Mode selection
    st.markdown("### Prediction Mode")
    mode = st.radio(
        "Mode",
        ["Single Game Prediction", "Generate All Pregame Predictions", "Full Day Automation", "Queue Gamestate-Conscious Posts"],
        help="Choose how to generate predictions",
        horizontal=True,
    )
    
    # Trigger type (for single game mode)
    st.markdown("### Trigger Type")
    trigger_type = st.selectbox(
        "Trigger Type",
        ["pregame", "halftime", "q3"],
        help="When to trigger the prediction (only for single game mode)",
        disabled=(mode != "Single Game Prediction"),
    )
    
    # Platform selection
    st.markdown("### Platforms")
    platforms = st.multiselect(
        "Select Platforms",
        ["twitter", "bluesky", "discord"],
        default=["discord"],
        help="Leave empty to post to all enabled platforms",
    )
    
    # Dry run toggle
    dry_run = st.checkbox("🧪 Test Mode (don't actually post)", value=False)
    
    # Allow duplicates toggle
    allow_duplicates = st.checkbox(
        "♻️ Allow Duplicate Posts",
        value=False,
        help="If checked, bypass duplicate detection and allow posting the same prediction multiple times within the 24-hour window.\n\n⚠️ Use with caution! This may result in posting the same content multiple times.",
    )
    
    # Submit
    st.markdown("---")
    
    if mode == "Single Game Prediction":
        col1, col2 = st.columns(2)
        
        with col1:
            # Toggle for fetching odds
            fetch_odds = st.toggle(
                "📊 Single Game: Fetch Odds from API",
                value=True,
                help="If OFF, predictions will be generated without odds data. Useful for testing.",
            )
            
            if st.button("🚀 Run Prediction", use_container_width=True):
                with st.spinner(f"Running {trigger_type} prediction for {selected_game_id}..."):
                    result = run_prediction(
                        game_id=selected_game_id,
                        trigger_type=trigger_type,
                        platforms=platforms if platforms else None,
                        dry_run=dry_run,
                        fetch_odds=fetch_odds,
                        allow_duplicates=allow_duplicates,
                    )
                    
                    st.markdown("### Result")
                    
                    # Check for error result
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    
                    # Predictions
                    predictions = result.get("predictions", [])
                    if predictions:
                        st.success(f"✅ Successfully generated {len(predictions)} prediction(s)")
                        for pred in predictions:
                            st.markdown(f"- **Game ID:** {pred.get('game_id')}")
                            st.markdown(f"  **Status:** {pred.get('status')}")
                            st.markdown(f"  **Trigger:** {trigger_type}")
                    
                    # Posted
                    posted = result.get("posted", [])
                    if posted:
                        st.markdown("---")
                        st.success(f"✅ Queued {len(posted)} post(s)")
                        
                        # Show detailed post information
                        for post_result in posted:
                            game_id = post_result.get("game_id", "unknown")
                            platforms = post_result.get("platforms", {})
                            
                            with st.expander(f"📋 Post: {game_id}"):
                                if platforms:
                                    for platform, platform_result in platforms.items():
                                        status = platform_result.get("status", "unknown")
                                        st.markdown(f"**{platform}**: `{status}`")
                                        
                                        if status == "queued":
                                            post_id = platform_result.get("post_id")
                                            st.markdown(f"- Post ID: `{post_id}`")
                                            
                                            content = platform_result.get("content", "")
                                            if content:
                                                st.markdown("**Content:**")
                                                st.code(content, language="text")
                                        
                                        elif status == "duplicate":
                                            reason = platform_result.get("reason", "Duplicate post")
                                            st.warning(f"- Reason: {reason}")
                                        
                                        elif status == "error":
                                            error = platform_result.get("error", "Unknown error")
                                            st.error(f"- Error: {error}")
                    
                    # Errors
                    errors = result.get("errors", [])
                    if errors:
                        st.markdown("---")
                        st.error(f"❌ Errors: {len(errors)}")
                        for error in errors:
                            st.markdown(f"- {error.get('game_id')}: {error.get('error')}")
                    
                    # Show message if nothing happened
                    if not predictions and not posted and not errors and not result.get("error"):
                        st.warning("⚠️ No predictions generated. Game may have already been processed.")
                    
                    # Check queue to confirm posts are queued
                    if posted:
                        st.markdown("---")
                        st.markdown("### 📋 Queue Verification")
                        queue = get_queue()
                        all_posts = queue.get_all_posts()
                        pending_posts = [p for p in all_posts if p.status.value in ["pending", "posting"]]
                        
                        st.markdown(f"**Current Queue Status:**")
                        st.markdown(f"- Total posts in queue: {len(all_posts)}")
                        st.markdown(f"- Pending/posting: {len(pending_posts)}")
                        
                        if pending_posts:
                            st.markdown("**Recent Posts in Queue:**")
                            for post in pending_posts[:3]:  # Show last 3
                                st.markdown(f"- `{post.game_id}` → `{post.platform}` ({post.status.value})")
                            
                            st.markdown("---")
                            st.markdown("### 🚀 Process Queue Now")
                            st.info("💡 Posts are queued but not yet sent to platforms. Click below to send them now!")
                            
                            if st.button("📤 Send Posts to Platforms", use_container_width=True):
                                with st.spinner("Processing queue..."):
                                    orchestrator = get_orchestrator()
                                    process_result = orchestrator.process_post_queue(batch_size=50)
                                    
                                    processed = process_result.get('processed', 0)
                                    successful = process_result.get('successful', 0)
                                    failed = process_result.get('failed', 0)
                                    
                                    st.markdown("### Process Result")
                                    if successful > 0:
                                        st.success(f"✅ Processed {processed} posts! ({successful} successful, {failed} failed)")
                                    else:
                                        st.error(f"❌ Processed {processed} posts but all failed ({failed} failures)")
                                    st.markdown(f"- **Successful:** {successful}")
                                    st.markdown(f"- **Failed:** {failed}")
                                    st.toast(f"Sent {successful} posts successfully!", icon="✅")
                                    
                                    if process_result.get('posts'):
                                        st.markdown("**Posts Processed:**")
                                        for post in process_result['posts']:
                                            post_id = post.get('post_id', 'unknown')
                                            platform = post.get('platform', 'unknown')
                                            status = post.get('status', 'unknown')
                                            if status == 'posted':
                                                st.markdown(f"✓ `{post_id}` → `{platform}`: **{status}**")
                                            else:
                                                error = post.get('error', 'Unknown error')
                                                st.markdown(f"✗ `{post_id}` → `{platform}`: **{status}**")
                                                st.markdown(f"   Error: `{error}`")
                                        
                                        # Show summary of errors
                                        if failed > 0:
                                            failed_posts = [p for p in process_result['posts'] if p.get('status') == 'failed']
                                            with st.expander("🔍 Error Details", expanded=False):
                                                for post in failed_posts:
                                                    st.markdown(f"**Post:** `{post.get('post_id')}`")
                                                    st.markdown(f"**Platform:** `{post.get('platform')}`")
                                                    st.markdown(f"**Error:** `{post.get('error', 'Unknown error')}`")
        
        with col2:
            st.info(f"Selected: {game_options.get(selected_game_id, selected_game_id)}")
    
    elif mode == "Generate All Pregame Predictions":
        col1, col2 = st.columns(2)
        
        with col1:
            # Toggle for fetching odds
            fetch_odds = st.toggle(
                "📊 Pregame: Fetch Odds from API",
                value=True,
                help="If OFF, predictions will be generated without odds data. Useful for testing.",
            )
            
            st.markdown("---")
            
            # Individual game predictions
            st.markdown("#### Individual Game Predictions")
            if st.button(f"🚀 Generate Pregame Predictions for All {len(games)} Games", use_container_width=True):
                # Create progress bar and status placeholder
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                def progress_callback(progress, message):
                    """Update progress in UI."""
                    progress_bar.progress(progress)
                    status_placeholder.markdown(f"🔄 {message}")
                    logger.info(f"Progress: {progress:.0%} - {message}")
                
                try:
                    result = run_predictions_for_all_games(
                        date=selected_date,
                        trigger_type="pregame",
                        platforms=platforms if platforms else None,
                        dry_run=dry_run,
                        fetch_odds=fetch_odds,
                        allow_duplicates=allow_duplicates,
                        progress_callback=progress_callback,
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_placeholder.empty()
                    
                    st.markdown("### Result")
                    
                    # Check for error result
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    
                    # Show summary
                    predictions = result.get("predictions", [])
                    posted = result.get("posted", [])
                    errors = result.get("errors", [])
                    skipped = result.get("skipped", 0)
                    total_games = result.get("total_games", len(games))
                    
                    st.markdown("**Summary:**")
                    st.markdown(f"- Total games: {total_games}")
                    st.markdown(f"- Predictions generated: {len(predictions)}")
                    st.markdown(f"- Posts queued: {len(posted)}")
                    st.markdown(f"- Errors: {len(errors)}")
                    if skipped > 0:
                        st.markdown(f"- Skipped (already processed): {skipped}")
                    
                    # Success message
                    if len(predictions) > 0 and len(errors) == 0:
                        st.success(f"🎉 All {len(predictions)} predictions generated successfully!")
                    
                    # Show predictions
                    if predictions:
                        st.markdown("---")
                        st.success(f"✅ Successfully generated {len(predictions)} prediction(s)")
                        with st.expander("View predictions"):
                            for pred in predictions:
                                st.markdown(f"- {pred.get('game_id')}: {pred.get('status')}")
                    
                    # Show posted
                    if posted:
                        st.markdown("---")
                        st.success(f"✅ Queued {len(posted)} post(s)")
                        
                        # Show detailed post information
                        for i, post_result in enumerate(posted, 1):
                            game_id = post_result.get("game_id", "unknown")
                            trigger_type = post_result.get("trigger_type", "unknown")
                            platforms = post_result.get("platforms", {})
                            
                            with st.expander(f"📋 Post #{i}: {game_id} ({trigger_type})"):
                                st.markdown(f"**Game ID:** `{game_id}`")
                                st.markdown(f"**Trigger Type:** `{trigger_type}`")
                                
                                if platforms:
                                    st.markdown(f"**Platforms:**")
                                    for platform, platform_result in platforms.items():
                                        status = platform_result.get("status", "unknown")
                                        st.markdown(f"- **{platform}**: `{status}`")
                                        
                                        if status == "queued":
                                            post_id = platform_result.get("post_id")
                                            st.markdown(f"  - Post ID: `{post_id}`")
                                            
                                            content = platform_result.get("content", "")
                                            if content:
                                                st.markdown("  - **Content:**")
                                                st.code(content, language="text")
                                        
                                        elif status == "duplicate":
                                            reason = platform_result.get("reason", "Duplicate post")
                                            st.markdown(f"  - Reason: {reason}")
                                        
                                        elif status == "error":
                                            error = platform_result.get("error", "Unknown error")
                                            st.error(f"  - Error: {error}")
                    
                    # Show errors
                    if errors:
                        st.markdown("---")
                        st.error(f"❌ Errors: {len(errors)}")
                        for error in errors:
                            st.markdown(f"- {error.get('game_id')}: {error.get('error')}")
                    
                    # Show message if nothing happened
                    if not predictions and not posted and not errors and not result.get("error"):
                        st.warning("⚠️ No predictions were generated. All games may have been already processed.")
                    
                    # Check queue to confirm posts are queued
                    if posted:
                        st.markdown("---")
                        st.markdown("### 📋 Queue Verification")
                        queue = get_queue()
                        all_posts = queue.get_all_posts()
                        pending_posts = [p for p in all_posts if p.status.value in ["pending", "posting"]]
                        
                        st.markdown(f"**Current Queue Status:**")
                        st.markdown(f"- Total posts in queue: {len(all_posts)}")
                        st.markdown(f"- Pending/posting: {len(pending_posts)}")
                        
                        if pending_posts:
                            st.markdown("**Recent Posts in Queue:**")
                            for post in pending_posts[:5]:  # Show last 5
                                st.markdown(f"- `{post.game_id}` → `{post.platform}` ({post.status.value})")
                
                except Exception as e:
                    st.error(f"Error running predictions: {e}")
                    logger.error(f"Error running predictions: {e}", exc_info=True)
            
            st.markdown("---")
            
            # Total day view
            st.markdown("#### Total Day View")
            st.info(
                "Generate a single post with all games in a table format "
                "(Option 3 table). This creates one consolidated post instead of "
                "individual posts for each game."
            )
            
            if st.button(f"📊 Generate Total Day View for {len(games)} Games", use_container_width=True):
                # Create progress bar and status placeholder
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                def progress_callback(progress, message):
                    """Update progress in UI."""
                    progress_bar.progress(progress)
                    status_placeholder.markdown(f"🔄 {message}")
                    logger.info(f"Progress: {progress:.0%} - {message}")
                
                try:
                    from src.automation.automation_ui import run_total_day_view
                    
                    result = run_total_day_view(
                        date=selected_date,
                        platforms=platforms if platforms else None,
                        dry_run=dry_run,
                        fetch_odds=fetch_odds,
                        progress_callback=progress_callback,
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_placeholder.empty()
                    
                    st.markdown("### Result")
                    
                    # Check for error result
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    
                    # Show summary
                    predictions = result.get("predictions", [])
                    total_day_post = result.get("total_day_post", {})
                    errors = result.get("errors", [])
                    total_games = result.get("total_games", len(games))
                    
                    st.markdown("**Summary:**")
                    st.markdown(f"- Total games: {total_games}")
                    st.markdown(f"- Predictions generated: {len(predictions)}")
                    st.markdown(f"- Total day post: {'✅ Generated' if total_day_post.get('success') else '❌ Failed'}")
                    st.markdown(f"- Errors: {len(errors)}")
                    
                    # Success message
                    if result.get("success") and total_day_post.get("success"):
                        st.success(f"🎉 Total day view post generated successfully!")
                    
                    # Show predictions summary
                    if predictions:
                        st.markdown("---")
                        st.success(f"✅ Successfully generated {len(predictions)} prediction(s)")
                        with st.expander("View predictions summary"):
                            for pred in predictions:
                                game_id = pred.get('game_id', 'unknown')
                                home = pred.get('home_name', 'Home')
                                away = pred.get('away_name', 'Away')
                                total = pred.get('total', 0)
                                margin = pred.get('margin', 0)
                                winner = home if margin > 0 else away
                                st.markdown(f"- {game_id}: {away} @ {home} → Winner: {winner} ({total:.1f})")
                    
                    # Show total day post
                    if total_day_post and total_day_post.get("content"):
                        st.markdown("---")
                        st.success(f"✅ Total Day View Post Generated")
                        st.markdown(f"**Game ID:** `total_day_{selected_date.strftime('%Y%m%d')}`")
                        st.markdown("**Post Content:**")
                        st.code(total_day_post["content"], language="text")
                        
                        # Show platform results
                        platforms = total_day_post.get("platforms", {})
                        if platforms:
                            st.markdown("**Platforms:**")
                            for platform, platform_result in platforms.items():
                                status = platform_result.get("status", "unknown")
                                st.markdown(f"- **{platform}**: `{status}`")
                                
                                if status == "queued":
                                    post_id = platform_result.get("post_id")
                                    st.markdown(f"  - Post ID: `{post_id}`")
                                
                                elif status == "error":
                                    error = platform_result.get("error", "Unknown error")
                                    st.error(f"  - Error: {error}")
                    
                    # Show errors
                    if errors:
                        st.markdown("---")
                        st.error(f"❌ Errors: {len(errors)}")
                        for error in errors:
                            st.markdown(f"- {error.get('game_id')}: {error.get('error')}")
                    
                    # Show message if nothing happened
                    if not predictions and not total_day_post and not errors and not result.get("error"):
                        st.warning("⚠️ No predictions were generated.")
                    
                    # Check queue to confirm posts are queued
                    if total_day_post and total_day_post.get("content"):
                        st.markdown("---")
                        st.markdown("### 📋 Queue Verification")
                        queue = get_queue()
                        all_posts = queue.get_all_posts()
                        pending_posts = [p for p in all_posts if p.status.value in ["pending", "posting"]]
                        
                        st.markdown(f"**Current Queue Status:**")
                        st.markdown(f"- Total posts in queue: {len(all_posts)}")
                        st.markdown(f"- Pending/posting: {len(pending_posts)}")
                        
                        if pending_posts:
                            st.markdown("**Recent Posts in Queue:**")
                            for post in pending_posts[:5]:  # Show last 5
                                st.markdown(f"- `{post.game_id}` → `{post.platform}` ({post.status.value})")
                
                except Exception as e:
                    st.error(f"Error running total day view: {e}")
                    logger.error(f"Error running total day view: {e}", exc_info=True)
        
        with col2:
            st.info(f"Will generate pregame predictions for all {len(games)} games on {selected_date}")
    
    elif mode == "Full Day Automation":
        st.markdown("### 🚀 Full Day Automation")
        st.success(
            "**ONE CLICK FOR EVERYTHING!**\n\n"
            "This will create:\n"
            "✅ Individual pregame predictions for all games\n"
            "✅ Total day summary post (Option 3 table)\n"
            "✅ Halftime triggers for each game (auto-posts at halftime)\n"
            "✅ Q3 triggers for each game (auto-posts at Q3)\n\n"
            "All posts are queued automatically and will post at the appropriate times!"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Toggle for fetching odds
            fetch_odds = st.toggle(
                "📊 Full Day: Fetch Odds from API",
                value=True,
                help="If OFF, predictions will be generated without odds data. Useful for testing.",
            )
            
            if st.button(f"🎮 Run Full Day Automation for {len(games)} Games", use_container_width=True, type="primary"):
                # Create progress bar and status placeholder
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                def progress_callback(progress, message):
                    """Update progress in UI."""
                    progress_bar.progress(progress)
                    status_placeholder.markdown(f"🔄 {message}")
                    logger.info(f"Progress: {progress:.0%} - {message}")
                
                try:
                    from src.automation.automation_ui import run_full_day_automation
                    
                    result = run_full_day_automation(
                        date=selected_date,
                        platforms=platforms if platforms else None,
                        dry_run=dry_run,
                        fetch_odds=fetch_odds,
                        progress_callback=progress_callback,
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_placeholder.empty()
                    
                    st.markdown("### Result")
                    
                    # Overall success
                    if result.get("success"):
                        st.success(f"🎉 Full day automation completed successfully!")
                    else:
                        st.warning("⚠️ Full day automation completed with errors")
                    
                    # Show summary
                    st.markdown("---")
                    st.markdown("### 📊 Summary")
                    
                    total_games = result.get("total_games", 0)
                    total_errors = len(result.get("errors", []))
                    
                    st.markdown(f"**Total Games:** {total_games}")
                    st.markdown(f"**Total Errors:** {total_errors}")
                    
                    # Pregame Individual
                    pregame_individual = result.get("pregame_individual", {})
                    st.markdown("---")
                    st.markdown("### 1️⃣ Individual Pregame Predictions")
                    if pregame_individual:
                        predictions_count = len(pregame_individual.get("predictions", []))
                        posted_count = len(pregame_individual.get("posted", []))
                        errors_count = len(pregame_individual.get("errors", []))
                        
                        st.markdown(f"- Predictions: {predictions_count}")
                        st.markdown(f"- Posts queued: {posted_count}")
                        st.markdown(f"- Errors: {errors_count}")
                        
                        if posted_count > 0:
                            st.success(f"✅ {posted_count} individual pregame posts queued")
                    else:
                        st.error("❌ Failed to generate individual pregame predictions")
                    
                    # Pregame Day Summary
                    pregame_day_summary = result.get("pregame_day_summary", {})
                    st.markdown("---")
                    st.markdown("### 2️⃣ Total Day Summary")
                    if pregame_day_summary:
                        success = pregame_day_summary.get("success", False)
                        total_day_post = pregame_day_summary.get("total_day_post", {})
                        
                        if success and total_day_post.get("success"):
                            st.success("✅ Total day summary post queued")
                            
                            with st.expander("View total day summary content"):
                                st.code(total_day_post.get("content", ""), language="text")
                        else:
                            st.error("❌ Failed to generate total day summary")
                    else:
                        st.error("❌ Failed to generate total day summary")
                    
                    # Halftime Triggers
                    halftime_triggers = result.get("halftime_triggers", {})
                    st.markdown("---")
                    st.markdown("### 3️⃣ Halftime Triggers")
                    if halftime_triggers:
                        successful = len(halftime_triggers.get("successful", []))
                        errors = len(halftime_triggers.get("errors", []))
                        
                        st.markdown(f"- Successful: {successful}/{halftime_triggers.get('total_games', 0)}")
                        st.markdown(f"- Errors: {errors}")
                        
                        if successful > 0:
                            st.success(f"✅ {successful} halftime triggers queued (will auto-post at halftime)")
                        
                        if errors > 0:
                            st.warning(f"⚠️ {errors} games failed to queue halftime triggers")
                            with st.expander("View halftime trigger errors"):
                                for error in halftime_triggers.get("errors", []):
                                    st.markdown(f"- `{error.get('game_id')}`: {error.get('error')}")
                    else:
                        st.error("❌ Failed to set up halftime triggers")
                    
                    # Q3 Triggers
                    q3_triggers = result.get("q3_triggers", {})
                    st.markdown("---")
                    st.markdown("### 4️⃣ Q3 Triggers")
                    if q3_triggers:
                        successful = len(q3_triggers.get("successful", []))
                        errors = len(q3_triggers.get("errors", []))
                        
                        st.markdown(f"- Successful: {successful}/{q3_triggers.get('total_games', 0)}")
                        st.markdown(f"- Errors: {errors}")
                        
                        if successful > 0:
                            st.success(f"✅ {successful} Q3 triggers queued (will auto-post at Q3)")
                        
                        if errors > 0:
                            st.warning(f"⚠️ {errors} games failed to queue Q3 triggers")
                            with st.expander("View Q3 trigger errors"):
                                for error in q3_triggers.get("errors", []):
                                    st.markdown(f"- `{error.get('game_id')}`: {error.get('error')}")
                    else:
                        st.error("❌ Failed to set up Q3 triggers")
                    
                    # Overall errors
                    if total_errors > 0:
                        st.markdown("---")
                        st.error(f"### ⚠️ Overall Errors ({total_errors})")
                        with st.expander("View all errors"):
                            for i, error in enumerate(result.get("errors", []), 1):
                                stage = error.get("stage", "unknown")
                                game_id = error.get("game_id", "unknown")
                                error_msg = error.get("error", "unknown")
                                st.markdown(f"{i}. **{stage}** - `{game_id}`: {error_msg}")
                    
                    # Check queue
                    st.markdown("---")
                    st.markdown("### 📋 Queue Verification")
                    queue = get_queue()
                    all_posts = queue.get_all_posts()
                    pending_posts = [p for p in all_posts if p.status.value in ["pending", "posting"]]
                    
                    st.markdown(f"**Current Queue Status:**")
                    st.markdown(f"- Total posts in queue: {len(all_posts)}")
                    st.markdown(f"- Pending/posting: {len(pending_posts)}")
                    
                    if pending_posts:
                        st.markdown("**Posts by Trigger Type:**")
                        pregame_count = len([p for p in pending_posts if 'pregame' in p.trigger_type])
                        halftime_count = len([p for p in pending_posts if 'halftime' in p.trigger_type])
                        q3_count = len([p for p in pending_posts if 'q3' in p.trigger_type])
                        
                        st.markdown(f"- Pregame: {pregame_count}")
                        st.markdown(f"- Halftime: {halftime_count}")
                        st.markdown(f"- Q3: {q3_count}")
                        
                        st.markdown("**Recent Posts in Queue:**")
                        for post in pending_posts[:10]:  # Show last 10
                            st.markdown(f"- `{post.game_id}` → `{post.platform}` ({post.trigger_type})")
                
                except Exception as e:
                    # Clear progress indicators
                    progress_bar.empty()
                    status_placeholder.empty()
                    
                    st.markdown("### Result")
                    st.error(f"❌ Unexpected error occurred: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    logger.exception("Error in full day automation:")
                    st.toast("Failed to run full day automation", icon="❌")
        
        with col2:
            st.success(
                f"**Ready to automate {len(games)} games on {selected_date}!**\n\n"
                f"Click the button to set up all posts automatically.\n\n"
                f"What happens:\n"
                f"1. {len(games)} individual pregame posts\n"
                f"2. 1 total day summary post\n"
                f"3. {len(games)} halftime triggers\n"
                f"4. {len(games)} Q3 triggers\n\n"
                f"Total: {1 + len(games) * 3} posts queued"
            )
    
    elif mode == "Queue Gamestate-Conscious Posts":
        st.markdown("### Gamestate-Conscious Posting")
        st.info(
            "This mode will queue posts that trigger at different game states: "
            "**Pregame** - Triggers immediately\n"
            "**Halftime** - Triggers when halftime is reached\n"
            "**Q3** - Triggers when Q3 is reached"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🎯 Queue Gamestate-Conscious Posts for {selected_game_id}", use_container_width=True):
                with st.spinner(f"Queueing gamestate-conscious posts for {selected_game_id}..."):
                    results = queue_gamestate_conscious_posts(
                        game_id=selected_game_id,
                        platforms=platforms if platforms else None,
                        dry_run=dry_run,
                    )
                    
                    st.markdown("### Results")
                    
                    # Summary
                    success_count = sum(1 for t in ['pregame', 'halftime', 'q3'] if results.get(t))
                    st.markdown(f"**Summary:** {success_count}/3 posts queued successfully")
                    
                    # Pregame result
                    if results.get("pregame"):
                        st.success("✅ Pregame post queued successfully")
                    else:
                        st.error("❌ Pregame post failed")
                    
                    # Halftime result
                    if results.get("halftime"):
                        st.success("✅ Halftime post queued successfully")
                    else:
                        st.error("❌ Halftime post failed")
                    
                    # Q3 result
                    if results.get("q3"):
                        st.success("✅ Q3 post queued successfully")
                    else:
                        st.error("❌ Q3 post failed")
                    
                    # Show any errors
                    if results.get("errors"):
                        st.markdown("---")
                        st.error(f"❌ Errors: {len(results['errors'])}")
                        for error in results["errors"]:
                            st.markdown(f"- {error.get('trigger_type')}: {error.get('error')}")
                    
                    # Show overall success message
                    if success_count == 3:
                        st.success("🎉 All 3 gamestate-conscious posts queued successfully!")
                        st.toast("All 3 posts queued successfully!", icon="🎉")
                    elif success_count > 0:
                        st.toast(f"{success_count}/3 posts queued successfully", icon="✅")
        
        with col2:
            st.info(f"Selected: {game_options.get(selected_game_id, selected_game_id)}")


def render_queue_manager():
    """Render queue management interface."""
    st.markdown("## 📋 Queue Manager")
    
    queue = get_queue()
    all_posts = queue.get_all_posts()
    
    if not all_posts:
        st.info("No posts in queue")
        return
    
    # Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Status",
            ["pending", "posting", "posted", "failed", "retrying"],
            default=["pending", "posting"],
        )
    
    with col2:
        platforms = list({p.platform for p in all_posts})
        platform_filter = st.selectbox(
            "Platform",
            [None] + platforms,
            format_func=lambda x: "All" if x is None else x,
        )
    
    with col3:
        game_id_filter = st.text_input("Game ID", placeholder="Search by game ID...")
    
    # Apply filters
    filtered_posts = all_posts
    
    if status_filter:
        filtered_posts = [
            p for p in filtered_posts
            if p.status.value in status_filter
        ]
    
    if platform_filter:
        filtered_posts = filter_posts_by_platform(filtered_posts, platform_filter)
    
    if game_id_filter:
        filtered_posts = filter_posts_by_game(filtered_posts, game_id_filter)
    
    st.markdown(f"**Showing {len(filtered_posts)} posts**")
    st.markdown("---")
    
    # Queue table
    if filtered_posts:
        render_queue_table(filtered_posts, max_rows=50)
        
        # Actions
        st.markdown("---")
        st.markdown("### Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Process Queue", use_container_width=True):
                with st.spinner("Processing queue..."):
                    result = process_queue(max_posts=10)
                    processed = result.get('processed', 0)
                    successful = result.get('successful', 0)
                    failed = result.get('failed', 0)
                    
                    if processed > 0:
                        if successful > 0:
                            st.success(f"✅ Processed {processed} posts! ({successful} successful, {failed} failed)")
                            st.toast(f"Sent {successful} posts successfully!", icon="✅")
                        else:
                            st.error(f"❌ Processed {processed} posts but all failed ({failed} failures)")
                            st.toast("All posts failed to send", icon="❌")
                        
                        # Show error details if any posts failed
                        if failed > 0 and result.get('posts'):
                            with st.expander("🔍 Error Details", expanded=False):
                                for post in result['posts']:
                                    if post.get('status') == 'failed':
                                        st.markdown(f"**Post:** `{post.get('post_id')}`")
                                        st.markdown(f"**Platform:** `{post.get('platform')}`")
                                        st.markdown(f"**Error:** `{post.get('error', 'Unknown error')}`")
                    else:
                        st.info("ℹ️ No pending posts to process")
        
        with col2:
            # Clear queue with confirmation
            if st.button("🗑️ Clear Queue", use_container_width=True):
                st.session_state["show_clear_queue_confirm"] = True
                st.rerun()
            
            if st.session_state.get("show_clear_queue_confirm"):
                st.warning("⚠️ Are you sure you want to clear the queue? This cannot be undone.")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Yes, Clear Queue", type="primary"):
                        queue.clear_queue()
                        st.session_state["show_clear_queue_confirm"] = False
                        st.success("Queue cleared!")
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel"):
                        st.session_state["show_clear_queue_confirm"] = False
                        st.rerun()
    else:
        st.info("No posts match the filters")


def render_history():
    """Render post history interface."""
    st.markdown("## 📜 Post History")
    
    queue = get_queue()
    all_posts = queue.get_all_posts()
    
    if not all_posts:
        st.info("No post history")
        return
    
    # Get posted posts only
    posted_posts = [p for p in all_posts if p.status == PostStatus.POSTED]
    
    if not posted_posts:
        st.info("No posted posts in history")
        return
    
    # Sort by created date (newest first)
    # Parse created_at_utc for sorting (ISO 8601 string)
    def parse_created_at(post):
        try:
            from datetime import datetime
            return datetime.fromisoformat(post.created_at_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Return old date for posts that fail to parse (they'll sort to end)
            from datetime import datetime
            return datetime.min
    
    posted_posts = sorted(
        posted_posts,
        key=parse_created_at,
        reverse=True,
    )
    
    st.markdown(f"**Total Posted: {len(posted_posts)} posts**")
    st.markdown("---")
    
    # Show posts
    for i, post in enumerate(posted_posts[:50]):
        # Format created_at_utc for display
        try:
            from datetime import datetime
            created_dt = datetime.fromisoformat(post.created_at_utc.replace("Z", "+00:00"))
            created_str = created_dt.strftime('%Y-%m-%d %H:%M')
        except (ValueError, AttributeError):
            created_str = post.created_at_utc[:16] if post.created_at_utc else "Unknown"
        
        with st.expander(
            f"{post.game_id} | {post.platform} | {created_str}"
        ):
            st.markdown(f"**Post ID:** `{post.post_id}`")
            st.markdown(f"**Game ID:** `{post.game_id}`")
            st.markdown(f"**Platform:** `{post.platform}`")
            st.markdown(f"**Created:** `{post.created_at_utc}`")
            
            if post.posted_at_utc:
                st.markdown(f"**Posted:** `{post.posted_at_utc}`")
            
            if post.message_id:
                st.markdown(f"**Message ID:** `{post.message_id}`")
            
            st.markdown("**Content:**")
            render_post_content(post.content)


def render_settings():
    """Render settings interface."""
    st.markdown("## ⚙️ Settings")
    
    st.warning(
        "Settings are managed via environment variables. "
        "Edit `.env` file to change settings."
    )
    
    st.markdown("### Platform Configuration")
    st.markdown("Configure platform credentials in `.env`:")
    
    st.markdown("""
    ```env
    # Discord (Required)
    DISCORD_WEBHOOK_URL=...

    # Twitter/X (Optional)
    TWITTER_CONSUMER_KEY=...
    TWITTER_CONSUMER_SECRET=...
    TWITTER_ACCESS_TOKEN=...
    TWITTER_ACCESS_TOKEN_SECRET=...

    # Bluesky (Optional)
    BLUESKY_HANDLE=...
    BLUESKY_APP_PASSWORD=...
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("### Automation Settings")
    st.markdown("Configure automation behavior in `.env`:")
    
    st.markdown("""
    ```env
    # Platform selection
    SOCIAL_MEDIA_PLATFORMS=twitter bluesky discord

    # Deduplication window (default: 24h)
    POST_DEDUPE_WINDOW_HOURS=24

    # Retry settings (via discord_poster.py)
    DISCORD_MAX_RETRIES=3
    DISCORD_RETRY_BACKOFF_SECONDS=1.5
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("### Current Configuration")
    platform_status = get_platform_status()
    
    st.markdown("**Platforms:**")
    for platform, enabled in platform_status.items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        st.markdown(f"- {platform}: {status}")
    
    st.markdown("---")
    
    st.markdown("### Actions")
    if st.button("🔄 Refresh Configuration", use_container_width=True):
        refresh_data()
        st.success("Configuration refreshed!")
        st.rerun()


def render_logs():
    """Render logs interface."""
    st.markdown("## 📝 Logs")
    
    st.info(
        "Logs are not yet integrated with Streamlit. "
        "Check the console output or log files for now."
    )
    
    st.markdown("### View Logs")
    st.markdown("To view logs:")
    st.markdown("1. Run automation from CLI: `python scripts/automation/social_poster.py --schedule`")
    st.markdown("2. Check console output for real-time logs")
    st.markdown("3. Logs are also written to standard error")
    
    st.markdown("---")
    
    st.markdown("### Log Levels")
    st.markdown("- **INFO**: General information")
    st.markdown("- **WARNING**: Warnings (e.g., platform not configured)")
    st.markdown("- **ERROR**: Errors (e.g., API failures)")
    st.markdown("- **DEBUG**: Detailed debugging information")
    
    st.markdown("---")
    
    st.markdown("### Enable Verbose Logging")
    st.code(
        "python scripts/automation/social_poster.py --schedule --verbose",
        language="bash",
    )


def main():
    """Main app."""
    # Initialize session state
    if "show_clear_queue_confirm" not in st.session_state:
        st.session_state["show_clear_queue_confirm"] = False
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.markdown("# 🤖 Automation Manager")
    st.markdown("Manage PerryPicks v3 social media automation.")
    
    # Tabs
    tab_dashboard, tab_manual, tab_queue, tab_history, tab_settings, tab_logs, tab_game_state = st.tabs(
        ["Dashboard", "Manual", "Queue", "History", "Settings", "Logs", "Game State"]
    )
    
    # Render each tab's content
    with tab_dashboard:
        render_dashboard()
    
    with tab_manual:
        render_manual_predictions()
    
    with tab_queue:
        render_queue_manager()
    
    with tab_history:
        render_history()
    
    with tab_settings:
        render_settings()
    
    with tab_logs:
        render_logs()
    
    with tab_game_state:
        render_game_state_monitor()

def render_game_state_monitor():
    """Render game state monitoring tab.
    
    This tab allows monitoring and control of the live game state
    monitoring service that automatically generates predictions at halftime and Q3-5min.
    """
    st.markdown("### 🎮 Game State Monitor")
    
    st.info(
        """**Live Game State Monitoring**\n\n"
        "This service monitors NBA games in real-time and automatically:\n"
        "• Generates predictions when games reach **halftime**\n"
        "• Generates predictions when games have **5 minutes left in Q3**\n"
        "• Automatically processes queue to post to platforms\n"
        "• Runs hands-off - no manual intervention needed"""
    )
    
    st.markdown("---")
    
    # Status Flags for both services
    st.markdown("### 🚦 Service Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Game Monitoring Status
        st.markdown("#### 🎮 Game Monitoring")
        
        # Get automation status
        automation_status = get_automation_status()
        
        # Status indicator
        if automation_status.get("running"):
            st.success("🟢 **LIVE** - Game State Monitor is active")
        else:
            st.warning("🔴 **STOPPED** - Game State Monitor is inactive")
        
        # Thread status
        if automation_status.get("thread_alive"):
            st.caption(f"Thread: {automation_status.get('thread_name', 'N/A')}")
        else:
            st.caption("Thread: Not running")
        
        # Last update
        if "status" in automation_status:
            status_data = automation_status["status"]
            if "last_update" in status_data and status_data["last_update"]:
                from datetime import datetime
                try:
                    last_update = datetime.fromisoformat(status_data["last_update"])
                    time_ago = (datetime.now() - last_update).total_seconds()
                    if time_ago < 60:
                        time_str = f"{int(time_ago)}s ago"
                    elif time_ago < 3600:
                        time_str = f"{int(time_ago // 60)}m ago"
                    else:
                        time_str = f"{int(time_ago // 3600)}h ago"
                    st.caption(f"Last activity: {time_str}")
                except:
                    pass
    
    with col2:
        # Queue Processing Status
        st.markdown("#### 📨 Queue Processing")
        
        # Get queue processor status
        queue_status = get_queue_processor_status()
        
        # Status indicator
        if queue_status.get("running"):
            st.success("🟢 **LIVE** - Queue Processor is active")
        else:
            st.warning("🔴 **STOPPED** - Queue Processor is inactive")
        
        # Thread status
        if queue_status.get("thread_alive"):
            st.caption(f"Thread: {queue_status.get('thread_name', 'N/A')}")
        else:
            st.caption("Thread: Not running")
        
        # Stats
        stats = queue_status.get("stats", {})
        processed = stats.get("processed", 0)
        st.caption(f"Posts processed: {processed}")
        
        if "last_processed_at" in stats and stats["last_processed_at"]:
            from datetime import datetime
            try:
                last_processed = datetime.fromisoformat(stats["last_processed_at"])
                time_ago = (datetime.now() - last_processed).total_seconds()
                if time_ago < 60:
                    time_str = f"{int(time_ago)}s ago"
                elif time_ago < 3600:
                    time_str = f"{int(time_ago // 60)}m ago"
                else:
                    time_str = f"{int(time_ago // 3600)}h ago"
                st.caption(f"Last processed: {time_str}")
            except:
                pass
    
    st.markdown("---")
    
    # Automated Queue Processing Toggle
    st.markdown("### 🎛️ Automated Queue Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Toggle switch
        st.markdown("#### ⚡ Queue Processor Control")
        
        # Get current status
        queue_status = get_queue_processor_status()
        is_running = queue_status.get("running", False)
        
        # Toggle switch
        auto_queue = st.toggle(
            "🤖 Enable Automated Queue Processing",
            value=is_running,
            key="auto_queue_toggle",
            help="When enabled, queue processor runs continuously in the background",
        )
        
        # Check if state changed and take action
        if "auto_queue_enabled_prev" not in st.session_state:
            st.session_state["auto_queue_enabled_prev"] = is_running
        
        if auto_queue != st.session_state["auto_queue_enabled_prev"]:
            # State changed, update
            st.session_state["auto_queue_enabled_prev"] = auto_queue
            
            # Take action
            if auto_queue:
                # Enable
                with st.spinner("Starting queue processor..."):
                    result = start_queue_processor(
                        poll_interval=st.session_state.get("queue_poll_interval", 15),
                        batch_size=st.session_state.get("queue_batch_size", 10),
                    )
                    
                    if result.get("success"):
                        st.success(result.get("message"))
                        st.rerun()
                    else:
                        st.error(result.get("message"))
                        st.session_state["auto_queue_enabled_prev"] = False
            else:
                # Disable
                with st.spinner("Stopping queue processor..."):
                    result = stop_queue_processor()
                    
                    if result.get("success"):
                        st.success(result.get("message"))
                        st.rerun()
                    else:
                        st.error(result.get("message"))
                        st.session_state["auto_queue_enabled_prev"] = True
        
        # Show config if enabled
        if auto_queue:
            st.success("✅ Automated queue processing is **ENABLED**")
            st.caption("Queue will be processed every 15 seconds automatically")
        else:
            st.warning("⏸️  Automated queue processing is **DISABLED**")
            st.caption("Queue processing is manual - use 'Process Queue' button")
    
    with col2:
        # Configuration
        st.markdown("#### ⚙️ Configuration")
        
        poll_interval = st.number_input(
            "Poll Interval (seconds)",
            value=15,
            min_value=5,
            max_value=300,
            step=5,
            key="queue_poll_interval",
            help="How often to check queue for pending posts (default: 15s)",
        )
        
        batch_size = st.number_input(
            "Batch Size",
            value=10,
            min_value=1,
            max_value=100,
            step=1,
            key="queue_batch_size",
            help="Maximum posts to process per poll (default: 10)",
        )
        
        # Apply configuration button
        if st.button("⚙️ Apply Configuration", use_container_width=True, key="manual_apply_config"):
            # Apply configuration (in production, this would save to config)
            st.success("Configuration applied!")
            st.rerun()
    
    st.markdown("---")
    
    # Manual controls
    st.markdown("### 🎛️ Manual Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Start Queue Processor
        if st.button("▶️ Start Queue Processor", use_container_width=True, type="primary", key="manual_start_queue"):
            with st.spinner("Starting queue processor..."):
                result = start_queue_processor(
                    poll_interval=st.session_state.get("queue_poll_interval", 15),
                    batch_size=st.session_state.get("queue_batch_size", 10),
                )
                
                if result.get("success"):
                    st.success(result.get("message"))
                    st.rerun()
                else:
                    st.error(result.get("message"))
    
    with col2:
        # Stop Queue Processor
        if st.button("⏹️ Stop Queue Processor", use_container_width=True, key="manual_stop_queue"):
            with st.spinner("Stopping queue processor..."):
                result = stop_queue_processor()
                
                if result.get("success"):
                    st.success(result.get("message"))
                    st.rerun()
                else:
                    st.error(result.get("message"))
    
    with col3:
        # Process Queue Now (one-off)
        if st.button("⚡ Process Queue Now", use_container_width=True, key="manual_process_now"):
            with st.spinner("Processing queue..."):
                result = process_queue(max_posts=50)
                
                processed = result.get("processed_predictions", 0)
                successful = result.get("successful", 0)
                
                if processed > 0 or successful > 0:
                    st.success(f"✓ Processed {processed} post(s)")
                else:
                    st.info("No pending posts to process")
    
    st.markdown("---")
    
    # Detailed status
    st.markdown("### 📊 Detailed Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Game Monitor Details")
        render_automation_status()
    
    with col2:
        st.markdown("#### Queue Processor Details")
        render_queue_processor_status()
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 How It Works", expanded=False):
        st.markdown(
            """**Game State Monitoring Flow:**\n\n"
            "1. **Service starts** - Polls NBA API every 30 seconds\n"
            "2. **Game tracking** - Monitors period and time for all active games\n"
            "3. **Halftime trigger** - When game reaches end of Q2, generates halftime prediction\n"
            "4. **Q3 trigger** - When game reaches 5 minutes left in Q3, generates Q3 prediction\n"
            "5. **Auto-process** - Automatically processes queue to post to Discord\n"
            "6. **Repeat** - Continues monitoring until games finish or service stops\n\n\n"
            "**Trigger Logic:**\n"
            "• **Halftime**: period=2 AND time_remaining=0:00\n"
            "• **Q3-5min**: period=3 AND time_remaining≈5:00\n\n\n"
            "**Duplicate Prevention:**\n"
            "Each trigger is marked as fired after first execution, "
            "preventing duplicate posts for the same game."""
        )

if __name__ == "__main__":
    main()