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

from core.storage import GameStorage
from core.env import load_environment
from src.automation import (
    PostQueue,
    SocialMediaManager,
    AutomationOrchestrator,
)
from src.automation.post_queue import PostStatus

logger = logging.getLogger(__name__)

# Session state keys
SESSION_STATE_ORCHESTRATOR = "automation_orchestrator"
SESSION_STATE_PLATFORMS = "automation_platforms"
SESSION_STATE_SCHEDULE = "automation_schedule"


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
    """Render queue as table."""
    if not posts:
        st.info("No posts in queue")
        return
    
    # Limit rows
    posts = posts[:max_rows]
    
    # Convert to DataFrame
    data = []
    for post in posts:
        # Parse created_at_utc (ISO 8601 string) and format it
        try:
            created_dt = datetime.fromisoformat(post.created_at_utc.replace("Z", "+00:00"))
            created_str = created_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing created_at_utc for {post.post_id}: {e}")
            created_str = post.created_at_utc[:16] if post.created_at_utc else "Unknown"
        
        data.append({
            "Post ID": post.post_id[:20] + "..." if len(post.post_id) > 20 else post.post_id,
            "Game ID": post.game_id,
            "Platform": post.platform,
            "Status": post.status.value,
            "Created": created_str,
            "Content": post.content[:50] + "..." if len(post.content) > 50 else post.content,
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


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
    progress_callback=None,
) -> Dict[str, Any]:
    """Run prediction for a single game.
    
    Args:
        game_id: Game ID to predict
        trigger_type: Trigger type (pregame, halftime, q3)
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        fetch_odds: If True, fetch odds from API (default True). Set False for testing.
        progress_callback: Optional callback(progress, message) for UI updates
    
    Returns:
        Prediction results dictionary
    """
    orchestrator = get_orchestrator(dry_run=dry_run)
    if not orchestrator:
        return {"success": False, "error": "Orchestrator not initialized"}
    
    return orchestrator.run_predictions(
        game_ids=[game_id],
        trigger_type=trigger_type,
        mode=trigger_type,  # Use user's selected trigger type as prediction mode
        fetch_odds=fetch_odds,
        progress_callback=progress_callback,
    )


def process_queue(max_posts: int = 10) -> Dict[str, Any]:
    """Process pending posts from queue."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"success": False, "error": "Orchestrator not initialized"}
    
    return orchestrator.process_post_queue(batch_size=max_posts)


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
    progress_callback=None,
) -> Dict[str, Any]:
    """Run predictions for all games on a specific date.
    
    Args:
        date: Date to predict for (default: today)
        trigger_type: Trigger type (pregame, halftime, q3)
        platforms: Platforms to post to (None = all enabled)
        dry_run: If True, don't actually post
        fetch_odds: If True, fetch odds from API (default True). Set False for testing.
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
        progress_callback=progress_callback,
    )

def queue_gamestate_conscious_posts(
    game_id: str,
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
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
        result = run_prediction(
            game_id=game_id,
            trigger_type=trigger_type,
            platforms=platforms,
            dry_run=dry_run,
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
                for platform in target_platforms:
                    try:
                        post_id = orchestrator.social_manager.queue.enqueue(
                            game_id=f"total_day_{date.strftime('%Y%m%d')}",
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

def refresh_data():
    """Refresh automation data (force reload)."""
    reset_orchestrator()
    
    # Set selected date to today if it exists in session state
    if "selected_manual_date" in st.session_state:
        st.session_state["selected_manual_date"] = dt.date.today()
    if "selected_dashboard_date" in st.session_state:
        st.session_state["selected_dashboard_date"] = dt.date.today()
    
    st.toast("Data refreshed!", icon="🔄")