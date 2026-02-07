# Branch Structure - PerryPicks V3

## 📊 Current Branches

### `main` (Production)
**Purpose:** Production automation system (future)  
**Status:** Clean, ready for automation development  
**Latest Commit:** `2a0ad1c` - Move Streamlit UI to separate branch  

**Contains:**
- ✅ Prediction models (pregame, halftime, Q3)
- ✅ Schedule fetching with ESPN→NBA ID mapping
- ✅ Automation scripts (schedule_predictions.py)
- ✅ Documentation (AUTOMATION_FLOW.md, etc.)
- ✅ All core functionality
- ❌ No Streamlit UI (moved to separate branch)

**Next Steps:**
- Build post generator
- Add social media APIs
- Implement full automation
- Merge streamlit-ui when ready

---

### `streamlit-ui` (Temporary Solution)
**Purpose:** Streamlit web UI for manual predictions  
**Status:** Production ready, deployable to Streamlit Cloud  
**Latest Commit:** `c402d1f` - Add Streamlit Cloud deployment guide  

**Contains:**
- ✅ `perry_predictions_ui.py` - Streamlit web UI
- ✅ `STREAMLIT_UI.md` - Full UI documentation
- ✅ `START_UI.md` - Quick start guide
- ✅ `STREAMLIT_CLOUD_DEPLOY.md` - Deployment guide
- ✅ All main branch code (from merge)
- ✅ `main_with_output()` in fetch_game_schedule.py

**Usage:**
```bash
git checkout streamlit-ui
streamlit run perry_predictions_ui.py
```

**Deployment:**
- Go to: https://share.streamlit.io
- Repository: `jarrydjames/perrypicksv3`
- Branch: `streamlit-ui` ← IMPORTANT!
- Main file: `perry_predictions_ui.py`

---

## 🔄 Workflow

### Current Workflow
```
streamlit-ui branch
    ↓
Deploy to Streamlit Cloud
    ↓
Access app from any browser
    ↓
Run predictions manually
    ↓
Copy formatted posts
    ↓
Post to social media manually
```

### Future Workflow (After Automation Complete)
```
main branch
    ↓
Automated predictions (cron)
    ↓
Automated post generation
    ↓
Automated social media posting
    ↓
Hands-off operation!
```

---

## 🌿 Branch Relationships

```
main (production automation)
  │
  ├── streamlit-ui (temporary UI - current deployment)
  │
  └── (other branches)
```

---

## 🔧 Branch Management

### Working on Streamlit UI
```bash
# Switch to streamlit-ui
git checkout streamlit-ui

# Make changes
# (edit files, commit, push)

# Push to streamlit-ui
git push origin streamlit-ui
```

### Working on Automation
```bash
# Stay on main
git checkout main

# Make changes
# (edit files, commit, push)

# Push to main
git push origin main
```

### Merge streamlit-ui into main (When Ready)
```bash
# Switch to main
git checkout main
git pull origin main

# Merge streamlit-ui
git merge streamlit-ui --no-ff

# Test locally
# (run tests, verify changes)

# Push to main
git push origin main

# Delete streamlit-ui (optional)
git branch -d streamlit-ui
git push origin --delete streamlit-ui
```

---

## 📦 What's in Each Branch

### `main` Branch Files
- `src/predict_api.py` - Prediction API
- `fetch_game_schedule.py` - Schedule fetching
- `run_pregame_predictions.py` - Pregame runner
- `run_halftime_predictions.py` - Halftime runner
- `run_q3_predictions.py` - Q3 runner
- `schedule_predictions.py` - Unified scheduler
- `run_automated_predictions.py` - Continuous monitor
- `models/` - Trained models directory
- `data/` - Data directory
- `AUTOMATION_FLOW.md` - Automation documentation
- `AUTOMATION_SUMMARY.md` - Quick reference
- `GAME_ID_MAPPING.md` - ESPN to NBA ID mapping
- `README_MODELS.md` - Model documentation
- And all other production files

### `streamlit-ui` Branch Files
- All main branch files (from merge)
- `perry_predictions_ui.py` - Streamlit UI app
- `STREAMLIT_UI.md` - UI documentation
- `START_UI.md` - Quick start guide
- `STREAMLIT_CLOUD_DEPLOY.md` - Deployment guide

---

## 🚀 Deployment Strategy

### Current Strategy (Temporary)
1. Use `streamlit-ui` branch
2. Deploy to Streamlit Cloud
3. Access from web browser
4. Manual predictions & posting
5. Work on automation in `main` branch

### Future Strategy (Production)
1. Complete automation in `main` branch
2. Merge `streamlit-ui` into `main`
3. Deploy `main` to Streamlit Cloud (optional)
4. Use cron for automated predictions
5. Auto-post to social media

---

## 💡 Why Separate Branches?

### Benefits
1. ✅ **Clean Main** - `main` stays focused on automation
2. ✅ **Easy Deployment** - Deploy `streamlit-ui` directly
3. ✅ **Parallel Development** - Work on UI and automation simultaneously
4. ✅ **Clean Merge** - Easier to merge when ready
5. ✅ **Flexibility** - Keep UI temporary, delete later if needed

### Alternatives Considered
- ❌ Single branch with feature flags - More complex
- ❌ Separate repo - Harder to maintain
- ❌ Git submodules - Overkill for this use case
- ✅ Separate branch - Perfect fit!

---

## 📋 Deployment Checklist

### Streamlit UI (Current)
- [x] `streamlit-ui` branch created
- [x] Streamlit UI app created
- [x] Documentation added
- [x] Deployment guide created
- [x] Pushed to GitHub
- [ ] Deploy to Streamlit Cloud ← YOU DO THIS
- [ ] Test on Streamlit Cloud
- [ ] Share URL

### Automation (Future)
- [ ] Build post generator
- [ ] Add Twitter API
- [ ] Add Bluesky API
- [ ] Add posting scheduler
- [ ] Add duplicate detection
- [ ] Add error handling
- [ ] Test end-to-end
- [ ] Deploy to production
- [ ] Merge `streamlit-ui` into `main`

---

## 🎯 Next Steps

### Immediate (Today)
1. **Deploy Streamlit UI**
   - Go to https://share.streamlit.io
   - Connect GitHub
   - Choose `jarrydjames/perrypicksv3` repo
   - Select `streamlit-ui` branch ← IMPORTANT!
   - Set main file: `perry_predictions_ui.py`
   - Deploy!

2. **Test App**
   - Run predictions for different dates
   - Test all 3 models (pregame, halftime, Q3)
   - Copy formatted posts
   - Verify everything works

3. **Share URL**
   - Get Streamlit Cloud URL
   - Share with team/followers
   - Collect feedback

### Future (This Week)
1. **Build Automation** (on `main` branch)
   - Post generator
   - Social media APIs
   - Posting scheduler

2. **Merge Branches**
   - Merge `streamlit-ui` into `main`
   - Test thoroughly
   - Delete `streamlit-ui` branch

3. **Full Deployment**
   - Deploy automation system
   - Set up cron jobs
   - Enable auto-posting

---

## 📚 Documentation

- **STREAMLIT_CLOUD_DEPLOY.md** - Streamlit Cloud deployment guide
- **STREAMLIT_UI.md** - Streamlit UI documentation (on streamlit-ui branch)
- **START_UI.md** - Quick start guide (on streamlit-ui branch)
- **AUTOMATION_FLOW.md** - Complete automation flow (on main)
- **AUTOMATION_SUMMARY.md** - Quick reference (on main)
- **GAME_ID_MAPPING.md** - ESPN to NBA ID mapping (on main)
- **BRANCH_STRUCTURE.md** - This document

---

## 🎉 Success!

You now have:
- ✅ Clean `main` branch for automation
- ✅ Separate `streamlit-ui` branch for UI
- ✅ Ready to deploy to Streamlit Cloud
- ✅ Parallel development capability
- ✅ Clear separation of concerns

**Deploy your Streamlit UI now!** 🚀

---

**Last Updated:** 2026-02-07  
**Status:** Ready to Deploy  
**Branches:** `main` and `streamlit-ui`
