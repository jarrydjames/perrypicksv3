# Streamlit Cloud Deployment Guide

Deploy PerryPredictions UI to Streamlit Cloud from the `streamlit-ui` branch.

---

## 📋 Prerequisites

1. ✅ Streamlit account (free at [share.streamlit.io](https://share.streamlit.io))
2. ✅ GitHub account with `perrypicksv3` repo
3. ✅ `streamlit-ui` branch pushed to GitHub

---

## 🚀 Deploy to Streamlit Cloud (5 Steps)

### Step 1: Go to Streamlit Cloud
Visit: https://share.streamlit.io

### Step 2: Click "New App"
- Sign in or create account
- Click "+ New app" button

### Step 3: Connect GitHub
- Click "Connect GitHub"
- Authorize Streamlit to access your repositories

### Step 4: Configure App

**Repository:** `jarrydjames/perrypicksv3`

**Branch:** `streamlit-ui` ← IMPORTANT! Not `main`

**Main File Path:** `perry_predictions_ui.py`

**Python Version:** `3.11` (or latest available)

**App URL (Optional):** `perrypicks-v3-ui` (or custom name)

### Step 5: Deploy
- Click "Deploy" button
- Wait 2-3 minutes for build
- App will be live!

**Your App URL:** `https://your-name-perrypicks-v3-ui.streamlit.app`

---

## 📝 Requirements.txt

Streamlit Cloud automatically detects dependencies, but you can create `requirements.txt`:

```txt
streamlit>=1.53.0
pandas>=2.0.0
requests>=2.31.0
scikit-learn>=1.3.0
xgboost>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

**To create requirements.txt:**
```bash
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks v3
pip freeze > requirements.txt
# OR
uv pip freeze > requirements.txt
```

---

## 🔧 Advanced Configuration

### Add Secrets (Environment Variables)

If you need API keys or secrets:

1. Go to your app on Streamlit Cloud
2. Click "⋮" (three dots) → "Settings"
3. Go to "Secrets" section
4. Add key-value pairs:
   ```
   ODDS_API_KEY=your_key_here
   NBA_API_KEY=your_key_here
   ```

**Access in app:**
```python
import streamlit as st
api_key = st.secrets["ODDS_API_KEY"]
```

### Custom Port (Not Needed)
Streamlit Cloud uses default port 8501 - no configuration needed.

### Custom Domain (Optional)
1. Go to app Settings
2. Click "Custom Domain"
3. Add your domain (e.g., `picks.yourdomain.com`)
4. Update DNS records

---

## 🐛 Troubleshooting

### Issue: Build Failed

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
- Create `requirements.txt` with all dependencies
- Add to `streamlit-ui` branch
- Push and redeploy

**Error:** `ImportError: cannot import name 'predict'`

**Solution:**
- Ensure `src/predict_api.py` exists and is accessible
- Check file permissions
- Verify import paths

### Issue: App Won't Start

**Error:** Blank page or 500 error

**Solution:**
- Check Streamlit Cloud logs (app → Settings → Logs)
- Verify `perry_predictions_ui.py` is correct path
- Check for Python syntax errors
- Ensure all dependencies are installed

### Issue: Predictions Fail

**Error:** `Error fetching schedule` or `API rate limit`

**Solution:**
- Check internet connectivity
- Verify NBA CDN API is accessible
- Add caching/timing to avoid rate limits
- Check logs for detailed errors

### Issue: Streamlit Cloud Not Showing Latest Branch

**Problem:** Changes on `streamlit-ui` not reflected

**Solution:**
```bash
# Push latest changes to streamlit-ui
git checkout streamlit-ui
git pull origin streamlit-ui
git push origin streamlit-ui

# Redeploy on Streamlit Cloud
# Go to app → Settings → Re-deploy
```

---

## 🔄 Update Deployment

### Method 1: Auto-Deploy (Recommended)
1. Make changes to `streamlit-ui` branch
2. Push to GitHub
3. Streamlit Cloud auto-detects and redeploys!

**Time:** 2-3 minutes

### Method 2: Manual Redeploy
1. Go to app on Streamlit Cloud
2. Click "⋮" → Settings
3. Scroll to "Deploy" section
4. Click "Re-deploy" button

**Time:** 2-3 minutes

---

## 📊 Monitor App Performance

### Check Logs
1. Go to app on Streamlit Cloud
2. Click "⋮" → Settings
3. Scroll to "Logs" section
4. View real-time logs

### Monitor Resources
Streamlit Cloud shows:
- CPU usage
- Memory usage
- Network traffic
- Request count

**Free Tier Limits:**
- CPU: Shared
- Memory: 1 GB
- Requests: Unlimited
- Bandwidth: Unlimited

---

## 🔒 Branch Management

### Current Branch Structure
```
main               ← Production automation (future)
  │
  └── streamlit-ui  ← Streamlit app (current deployment)
```

### Merge Back to Main (When Ready)

**When automation is complete:**

```bash
# 1. Switch to main
git checkout main
git pull origin main

# 2. Merge streamlit-ui
git merge streamlit-ui --no-ff

# 3. Review merge
# (Check for conflicts, test locally)

# 4. Push
git push origin main

# 5. Delete streamlit-ui branch (optional)
git branch -d streamlit-ui
git push origin --delete streamlit-ui
```

**After merging:**
- Streamlit UI becomes part of main
- Can deploy from main branch instead
- Streamlit Cloud app can be updated to use main

---

## 💡 Pro Tips

### 1. Use Streamlit Cloud's Features
- **Theming**: Custom colors and fonts
- **Caching**: `@st.cache_data` for faster loads
- **Session State**: Maintain user data across reruns

### 2. Optimize Performance
```python
# Cache schedule (don't refetch every interaction)
@st.cache_data(ttl=300)
def get_schedule(date_str):
    # ... fetch schedule
```

### 3. Error Handling
```python
try:
    result = predict(...)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()
```

### 4. Loading States
```python
with st.spinner("Running predictions..."):
    results = run_predictions()
```

---

## 📚 Useful Links

- **Streamlit Cloud**: https://share.streamlit.io
- **Streamlit Docs**: https://docs.streamlit.io
- **Deploy Guide**: https://docs.streamlit.io/streamlit-cloud/get-started/deploy-your-app
- **GitHub Repo**: https://github.com/jarrydjames/perrypicksv3/tree/streamlit-ui

---

## 🎯 Deployment Checklist

Before deploying, verify:

- [ ] `streamlit-ui` branch pushed to GitHub
- [ ] `perry_predictions_ui.py` exists and works locally
- [ ] `requirements.txt` includes all dependencies
- [ ] No import errors when running locally
- [ ] `fetch_game_schedule.py` has `main_with_output()` function
- [ ] `src/predict_api.py` exists and is importable
- [ ] All team mappings (83 variations) in place

---

## 🎉 Success!

Once deployed, you can:
- Access app from any browser
- Share URL with others
- Run predictions manually
- Copy formatted posts
- No local setup needed!

**Example URL:** `https://jarrydhames-perrypicks-v3-ui.streamlit.app`

---

**Last Updated:** 2026-02-07  
**Branch:** `streamlit-ui`  
**Status:** Ready to Deploy
