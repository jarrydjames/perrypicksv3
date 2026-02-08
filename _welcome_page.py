"""PerryPicks v3 - Main Entry Point

This is the main entry point for the PerryPicks v3 Streamlit application.
It configures the app and automatically loads all pages from the 'pages/' directory.

Pages found in pages/:
- 04_Automation_Manager.py
"""

import streamlit as st
import os

# Configure Streamlit
st.set_page_config(
    page_title="PerryPicks v3",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page content
st.title("🐶 PerryPicks v3")
st.markdown("""Welcome to PerryPicks v3!

Use the sidebar to navigate to different pages:
- **Automation Manager**: Manage predictions and automation
""")

# Display info about available pages
st.info(
    """ℹ️ **Navigate using the sidebar**

    Pages are automatically loaded from the `pages/` directory.
    The main page is the Automation Manager which you've been working on."""
)

st.write("")
st.write("---")
st.write("## Quick Start")
st.markdown("""
1. Click on **Automation Manager** in the sidebar
2. Use the full day automation to generate predictions
3. Start the game state monitor for hands-off posting
4. Monitor predictions and automation status
""")

# Show file info
st.write("---")
st.write("## Available Pages")

pages_dir = "pages"
if os.path.exists(pages_dir):
    pages = [f for f in os.listdir(pages_dir) if f.endswith('.py') and not f.startswith('_')]
    
    for page in sorted(pages):
        st.write(f"- **{page}**")
else:
    st.warning("No pages directory found!")