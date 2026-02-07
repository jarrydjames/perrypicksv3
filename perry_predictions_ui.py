#!/usr/bin/env python3
"""
PerryPredictions UI - Streamlit App

A temporary Streamlit app to manually trigger and view predictions
while we build full automated posting system.

Usage:
    streamlit run perry_predictions_ui.py
"""

import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import prediction functions
from src.predict_api import predict_game
import fetch_game_schedule


# Page config
st.set_page_config(
    page_title="PerryPredictions UI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div class="main-header">🏀 PerryPredictions UI</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    Manually trigger and view NBA predictions (Pregame, Halftime, Q3)
</div>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.header("⚙️ Configuration")

# Date selector
today = datetime.now()
default_date = today + timedelta(days=1)  # Tomorrow by default
date_input = st.sidebar.date_input(
    "Select Date",
    value=default_date,
    max_value=today + timedelta(days=365),
    min_value=today - timedelta(days=365)
)
date_str = date_input.strftime("%Y-%m-%d")

# Prediction type selector
st.sidebar.subheader("Prediction Type")
prediction_type = st.sidebar.selectbox(
    "Select Model",
    options=["Pregame", "Halftime", "Q3"],
    help="Choose which prediction model to run"
)

# Advanced options
st.sidebar.subheader("Advanced Options")
fetch_odds = st.sidebar.checkbox("Fetch Odds", value=False, help="Include betting odds in predictions")
show_raw_output = st.sidebar.checkbox("Show Raw Output", value=False, help="Show raw prediction output")


# Main content
st.markdown(f"---")
st.markdown(f"### 📅 Selected Date: **{date_str}**")
st.markdown(f"### 🎯 Prediction Model: **{prediction_type}**")


# Fetch schedule
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_schedule(date_str):
    """Fetch game schedule for a date."""
    try:
        schedule_data = fetch_game_schedule.main_with_output(date_str)
        return schedule_data
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return None


# Get schedule
with st.spinner("Fetching schedule..."):
    schedule_data = get_schedule(date_str)


if schedule_data is None:
    st.error("Failed to fetch schedule. Please check the date and try again.")
    st.stop()


# Display games
if 'games' in schedule_data and schedule_data['games']:
    st.markdown(f"### 📊 Games Found: {len(schedule_data['games'])}")
    
    # Create games dataframe
    games_df = pd.DataFrame(schedule_data['games'])
    
    # Reorder columns
    if 'nba_id' in games_df.columns and 'espn_id' in games_df.columns:
        display_cols = ['nba_id', 'away_team', 'home_team', 'status', 'time_utc']
        games_df = games_df[display_cols]
        games_df = games_df.rename(columns={
            'nba_id': 'NBA ID',
            'away_team': 'Away',
            'home_team': 'Home',
            'status': 'Status',
            'time_utc': 'Time (UTC)'
        })
    
    st.dataframe(games_df, use_container_width=True, hide_index=True)
else:
    st.warning("No games found for the selected date.")
    st.stop()


# Run predictions
st.markdown("---")
st.markdown("### 🚀 Run Predictions")

run_predictions = st.button(
    f"Run {prediction_type} Predictions",
    type="primary",
    use_container_width=True
)


if run_predictions:
    mode_map = {
        "Pregame": "pregame",
        "Halftime": "halftime",
        "Q3": "q3"
    }
    mode = mode_map[prediction_type]
    
    # Get games with team info
    games = schedule_data['games']
    
    # Run predictions
    with st.spinner(f"Running {prediction_type.lower()} predictions..."):
        results = []
        errors = []
        
        for i, game in enumerate(games):
            game_id = game['nba_id']
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            try:
                # Add delay between predictions to avoid NBA API rate limiting
                if i > 0:
                    time.sleep(1.0)  # 1 second delay between games
                
                result = predict_game(
                    game_input=game_id,
                    use_binned_intervals=False,
                    fetch_odds=fetch_odds,
                    mode=mode,
                    home_team=home_team,
                    away_team=away_team,
                    bypass_import_gate=True  # Bypass import gate for manual Streamlit UI predictions
                )
                results.append(result)
            except Exception as e:
                errors.append({"game_id": game_id, "error": str(e)})
                st.error(f"Error predicting game {game_id}: {e}")
    
    # Display results
    st.markdown('<div class="sub-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    
    if results:
        # Summary
        success_count = sum(1 for r in results if r.get('status') in ('success', 'warning'))
        total_count = len(games)
        error_count = total_count - success_count
        
        st.markdown(f"<div class='success-box'>✅ Successfully generated {success_count}/{total_count} predictions</div>", unsafe_allow_html=True)
        
        if error_count > 0:
            st.markdown(f"<div class='warning-box'>⚠️ {error_count} predictions failed due to team tricode errors</div>", unsafe_allow_html=True)
        
        # Create results dataframe
        results_data = []
        for r in results:
            if r.get('status') == 'success' or r.get('status') == 'warning':
                result = r
                
                # Get prediction values
                margin = result.get('margin', 0)
                total = result.get('total', 0)
                home_win_prob = result.get('home_win_prob', 0.5)
                
                # Get team names for winner calculation
                home_team = result.get('home_name', 'Home')
                away_team = result.get('away_name', 'Away')
                
                # Calculate projected scores
                # margin = home_score - away_score
                # total = home_score + away_score
                # home_score = (total + margin) / 2
                # away_score = (total - margin) / 2
                if isinstance(margin, (int, float)) and isinstance(total, (int, float)):
                    home_score = (total + margin) / 2
                    away_score = (total - margin) / 2
                    winner = home_team if margin > 0 else away_team
                    win_pct = home_win_prob if margin > 0 else (1 - home_win_prob)
                else:
                    home_score = 'N/A'
                    away_score = 'N/A'
                    winner = 'N/A'
                    win_pct = 'N/A'
                
                prediction_data = {
                    'Away Team': result.get('away_name', 'N/A'),
                    'Away Score': f"{away_score:.1f}" if isinstance(away_score, (int, float)) else 'N/A',
                    'Home Team': result.get('home_name', 'N/A'),
                    'Home Score': f"{home_score:.1f}" if isinstance(home_score, (int, float)) else 'N/A',
                    'Game Total': f"{total:.1f}" if isinstance(total, (int, float)) else 'N/A',
                    'Margin': f"{margin:+.1f}" if isinstance(margin, (int, float)) else 'N/A',
                    'Winner': winner,
                    'Win %': f"{win_pct*100:.1f}%" if isinstance(win_pct, float) else 'N/A',
                }
                
                # Add odds if available
                if fetch_odds and result.get('odds'):
                    odds = result.get('odds', {})
                    prediction_data['Spread'] = odds.get('spread', 'N/A')
                    prediction_data['O/U'] = odds.get('over_under', 'N/A')
                
                results_data.append(prediction_data)
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Generate formatted posts
        st.markdown('<div class="sub-header">📝 Formatted Posts (Copy & Paste)</div>', unsafe_allow_html=True)
        
        for i, r in enumerate(results):
            if r.get('status') in ('success', 'warning'):
                # Get prediction values
                game_id = r.get('game_id', f'game_{i}')
                margin = r.get('margin', 0)
                total = r.get('total', 0)
                home_win_prob = r.get('home_win_prob', 0.5)
                away_team = r.get('away_name', 'Away')
                home_team = r.get('home_name', 'Home')
                
                # Calculate projected scores
                if isinstance(margin, (int, float)) and isinstance(total, (int, float)):
                    home_score = (total + margin) / 2
                    away_score = (total - margin) / 2
                    winner = home_team if margin > 0 else away_team
                    win_prob = home_win_prob if margin > 0 else (1 - home_win_prob)
                    win_pct_str = f"{win_prob*100:.1f}%"
                else:
                    home_score = 'N/A'
                    away_score = 'N/A'
                    winner = 'N/A'
                    win_pct_str = 'N/A'
                
                # Generate post based on prediction type
                if mode == 'pregame':
                    post = f"""🏀 Pregame Prediction: {away_team} @ {home_team}

📊 Predicted Score: {away_score:.1f} - {home_score:.1f}
🎯 Predicted Total: {total:.1f}
📈 Predicted Margin: {margin:+.1f}
🏆 Predicted Winner: {winner} ({win_pct_str})"""
                
                elif mode == 'halftime':
                    h1_away = r.get('h1_away', 'N/A')
                    h1_home = r.get('h1_home', 'N/A')
                    pred_2h_away = r.get('pred_2h_away', 'N/A')
                    pred_2h_home = r.get('pred_2h_home', 'N/A')
                    pred_final_away = r.get('pred_final_away', 'N/A')
                    pred_final_home = r.get('pred_final_home', 'N/A')
                    
                    post = f"""🔥 Halftime Update: {away_team} @ {home_team}

📊 Halftime: {h1_away} - {h1_home}
📈 Projected 2H: {pred_2h_away:.1f} - {pred_2h_home:.1f}
🎯 Projected Final: {pred_final_away:.1f} - {pred_final_home:.1f}
🏆 Projected Winner: {r.get('predicted_winner', 'N/A')} by {r.get('predicted_margin', 'N/A'):.1f}"""
                
                elif mode == 'q3':
                    q3_cum_away = r.get('q3_cum_away', 'N/A')
                    q3_cum_home = r.get('q3_cum_home', 'N/A')
                    est_q4_away = r.get('est_q4_away', 'N/A')
                    est_q4_home = r.get('est_q4_home', 'N/A')
                    pred_final_away = r.get('pred_final_away', 'N/A')
                    pred_final_home = r.get('pred_final_home', 'N/A')
                    
                    post = f"""⚡ Q3 Update: {away_team} @ {home_team}

📊 Q3 Cumulative: {q3_cum_away:.1f} - {q3_cum_home:.1f}
📈 Estimated Q4: {est_q4_away:.1f} - {est_q4_home:.1f}
🎯 Projected Final: {pred_final_away:.1f} - {pred_final_home:.1f}
🏆 Projected Winner: {r.get('predicted_winner', 'N/A')} by {r.get('predicted_margin', 'N/A'):.1f}"""
                
                # Add odds if available
                if fetch_odds and r.get('odds'):
                    odds = r['odds']
                    spread = odds.get('spread', 'N/A')
                    ou = odds.get('over_under', 'N/A')
                    post += f"\n\n💰 Odds: Spread {spread}, O/U {ou}"
                
                # Add hashtags
                post += f"\n\n#NBA #PerryPredictions #NBA{away_team.replace(' ', '')} #NBA{home_team.replace(' ', '')}"
                
                # Display post with copy button
                with st.expander(f"📄 Post for {away_team} @ {home_team}", expanded=i == 0):
                    st.text_area("Post Content", post, height=200, key=f"post_{i}")
                    
                    # Add copy button (simulated)
                    if st.button(f"📋 Copy to Clipboard", key=f"copy_{i}"):
                        st.code(post, language=None)
                        st.success("Post copied! Use Ctrl+C to copy from above.")
    
    if errors:
        st.markdown(f"<div class='warning-box'>⚠️ {len(errors)} predictions failed. See errors above.</div>", unsafe_allow_html=True)
    
    # Show raw output if requested
    if show_raw_output:
        st.markdown('<div class="sub-header">🔍 Raw Output</div>', unsafe_allow_html=True)
        st.json(results)

else:
    st.info("Click 'Run Predictions' to generate predictions.")


# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #999; margin-top: 2rem;">
    <small>
        PerryPredictions UI v1.0 | 
        🏀 Built by Perry (code-puppy) | 
        For questions: Check AUTOMATION_FLOW.md
    </small>
</div>
""", unsafe_allow_html=True)
