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
)

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
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            refresh_data()
            st.rerun()
        
        st.markdown("---")
        
        # Platform status
        st.markdown("### Platform Status")
        platform_status = get_platform_status()
        render_platform_status(
            platforms=["twitter", "bluesky", "discord"],
            enabled_platforms=set(
                p for p, enabled in platform_status.items() if enabled
            ),
        )
        
        st.markdown("---")

        # Navigation
        st.markdown("### Navigation")
        st.markdown("Use the tabs above to navigate:")
        st.markdown("- **Dashboard**: Overview & stats")
        st.markdown("- **Manual**: Trigger predictions")
        st.markdown("- **Queue**: Manage queued posts")
        st.markdown("- **History**: Post history")
        st.markdown("- **Settings**: Configuration")
        st.markdown("- **Logs**: View logs")
        
        st.markdown("---")
        
        # Quick Start Guide
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("**Step 1:** Select 'Manual' tab")
        st.markdown("**Step 2:** Choose game(s) and prediction mode")
        st.markdown("**Step 3:** Click 'Generate Predictions' button")
        st.markdown("**Step 4:** Click 'Send Posts to Platforms' when it appears")
        st.markdown("**Step 5:** Posts appear on your social platforms!")
        st.markdown("")
        st.markdown("ℹ️ **Test Mode** is OFF by default")
        st.markdown("   Toggle 'Test Mode' to preview without posting")
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ Info")
        st.markdown("This automation manager is **separate** from the main PerryPicks v3 app.")
        st.markdown("It does not impact manual research or reviews.")


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
            "Date",
            value=st.session_state["selected_dashboard_date"],
            key="dashboard_date_input",
        )
        st.session_state["selected_dashboard_date"] = selected_date
    
    with col2:
        if st.button("🔄 Go to Today", key="goto_today"):
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
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Queue", use_container_width=True, key="dashboard_process_queue"):
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
            "Date",
            value=st.session_state["selected_manual_date"],
            key="manual_date_input",
        )
        st.session_state["selected_manual_date"] = selected_date
    
    with col2:
        if st.button("🔄 Go to Today", key="manual_goto_today"):
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
        ["Single Game Prediction", "Generate All Pregame Predictions", "Queue Gamestate-Conscious Posts"],
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
    
    # Submit
    st.markdown("---")
    
    if mode == "Single Game Prediction":
        col1, col2 = st.columns(2)
        
        with col1:
            # Toggle for fetching odds
            fetch_odds = st.toggle(
                "📊 Fetch Odds from API",
                value=True,
                help="If OFF, predictions will be generated without odds data. Useful for testing.",
                key="single_game_fetch_odds"
            )
            
            if st.button("🚀 Run Prediction", use_container_width=True):
                with st.spinner(f"Running {trigger_type} prediction for {selected_game_id}..."):
                    result = run_prediction(
                        game_id=selected_game_id,
                        trigger_type=trigger_type,
                        platforms=platforms if platforms else None,
                        dry_run=dry_run,
                        fetch_odds=fetch_odds,
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
                            
                            if st.button("📤 Send Posts to Platforms", use_container_width=True, key="send_posts_single"):
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
                "📊 Fetch Odds from API",
                value=True,
                help="If OFF, predictions will be generated without odds data. Useful for testing.",
                key="pregame_fetch_odds"
            )
            
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
                            
                            st.markdown("---")
                            st.markdown("### 🚀 Process Queue Now")
                            st.info("💡 Posts are queued but not yet sent to platforms. Click below to send them now!")
                            
                            if st.button("📤 Send Posts to Platforms", use_container_width=True, key="send_posts_all_predictions"):
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
                
                except Exception as e:
                    # Clear progress indicators
                    progress_bar.empty()
                    status_placeholder.empty()
                    
                    st.markdown("### Result")
                    st.error(f"❌ Unexpected error occurred: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    logger.exception("Error in generate all predictions:")
                    st.toast("Failed to generate predictions", icon="❌")
        
        with col2:
            st.info(f"Will generate pregame predictions for all {len(games)} games on {selected_date}")
    
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
            if st.button("🔄 Process Queue", use_container_width=True, key="queue_tab_process_queue"):
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
            if st.button("🗑️ Clear Queue", use_container_width=True):
                if st.confirm("Are you sure you want to clear the queue?"):
                    queue.clear_queue()
                    st.success("Queue cleared!")
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
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.markdown("# 🤖 Automation Manager")
    st.markdown("Manage PerryPicks v3 social media automation.")
    
    # Tabs
    tab_dashboard, tab_manual, tab_queue, tab_history, tab_settings, tab_logs = st.tabs(
        ["Dashboard", "Manual", "Queue", "History", "Settings", "Logs"]
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


if __name__ == "__main__":
    main()