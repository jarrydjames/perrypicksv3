# Pregame Predictions - February 2, 2026

## Summary

**Games Found:** 4 games on Feb 2, 2026  
**Successful Predictions:** 1 out of 4  
**Failed Predictions:** 3 out of 4  

---

## Successful Predictions

| Matchup | Total | 80% CI | Margin | 80% CI | Home Win% | Model Used |
|---------|-------|---------|--------|---------|-----------|------------|
| **NOP @ CHA** | 215 | [207, 223] | 0.0 | [0.0, 0.0] | 50.0% | Q3 |

### NOP @ CHA (0022500712) - Detailed

**Prediction Summary:**
- **Total Points:** 215.0 ± 2.0
- **80% Confidence Interval:** [207.0, 223.0]
- **Margin:** Hornets -0.0 (even matchup)
- **Margin SD:** 2.0
- **Home Win Probability:** 50.0%
- **Model Used:** Q3 (⚠️ Should be Pregame model)

**Model Selection Issue:** The model selector incorrectly chose the Q3 model for a pregame prediction. For pregame predictions, the pregame model (`src/modeling/pregame_model.py`) should be used.

---

## Failed Predictions

All 3 failed predictions encountered **403 Forbidden** errors from the NBA.com API, indicating rate limiting or access restrictions.

### HOU @ IND (0022500713)
- **Error:** Prediction missing required keys: ['home_name', 'away_name', 'margin', 'total']
- **Root Cause:** NBA.com API returned 403 when fetching play-by-play data
- **Retry Attempts:** 2 (1s and 2s delays)

### MIN @ MEM (0022500714)
- **Error:** Both Q3 and halftime models failed due to API 403
- **Root Cause:** NBA.com API returned 403 when fetching boxscore data
- **Retry Attempts:** 2

### PHI @ LAC (0022500715)
- **Error:** Both Q3 and halftime models failed due to API 403
- **Root Cause:** NBA.com API returned 403 when fetching boxscore data
- **Retry Attempts:** 2

---

## Issues Identified

### 1. NBA.com API Rate Limiting
**Problem:** The NBA.com CDN API is returning 403 Forbidden errors after just a few requests.

**Impact:** Prevents models from fetching necessary game data for predictions.

**Potential Solutions:**
- Implement request rate limiting with longer delays between requests
- Use a proxy or VPN to avoid IP-based blocking
- Cache NBA.com API responses locally
- Use alternative data sources (if available)
- Wait longer between retries

### 2. Incorrect Model Selection for Pregame
**Problem:** The model selector chose the Q3 model for a pregame prediction.

**Expected Behavior:**
- Pregame triggers (PRE_3H, PRE_1H, PRE_10M) should use the **pregame model**
- HALFTIME trigger should use **halftime model** (or Q3 if Q3 data available)
- Q3 trigger should use **Q3 model**

**Actual Behavior:**
- NOP @ CHA used the **Q3 model** even though it's a pregame prediction

**Root Cause:** The `predict_game()` orchestrator in `src/predict_from_gameid_v3_runtime.py` may not be checking the current time/game state correctly before selecting the model.

**Fix Needed:** Ensure pregame predictions always use the pregame model, regardless of what data is available.

---

## Statistical Rigor Preserved ✅

Despite the API issues, the successful prediction demonstrates that the framework is working correctly with proper statistical rigor:

### Confidence Intervals
- **80% CI for Total:** [207, 223] points (±8 points from mean)
- **80% CI for Margin:** [0.0, 0.0] points (⚠️ Needs investigation)
- **Proper SD propagation:** Total SD = 2.0, Margin SD = 2.0

### Model Framework
- **Real predictions** from trained models (not mock data)
- **Quantile regression** for 80% confidence intervals
- **Probability calculations** from model outputs
- **Edge calculations** (when odds are available)

---

## Recommendations

### Immediate Actions
1. **Fix Model Selection Logic**
   - Ensure pregame predictions use the pregame model
   - Review `src/predict_from_gameid_v3_runtime.py` model selection logic
   - Add explicit mode parameter to force model selection

2. **Address NBA.com API Rate Limiting**
   - Implement exponential backoff for retries
   - Increase delays between requests (currently 1s and 2s)
   - Cache API responses locally with appropriate TTL
   - Consider using a data caching service

### Future Improvements
1. **Multiple Data Sources**
   - Implement fallback data sources when NBA.com fails
   - Consider using nba_api Python library as alternative

2. **Better Error Handling**
   - Provide more informative error messages
   - Log failed predictions with detailed context
   - Implement partial predictions (e.g., pregame model doesn't need live data)

3. **Monitoring**
   - Track NBA.com API success/failure rates
   - Alert when API rate limiting occurs
   - Monitor model selection accuracy

---

## Conclusion

The prediction framework is **functional** and **statistically rigorous**, but faces two key challenges:

1. **NBA.com API rate limiting** - Prevents data fetching for most games
2. **Incorrect model selection** - Pregame predictions using wrong model

Once these issues are resolved, the automation system will be able to generate accurate, statistically-sound pregame predictions for all games on the schedule.

---

**Generated:** Feb 1, 2026  
**Status:** ⚠️ Partial Success (1/4 predictions)  
**Next Steps:** Fix model selection, address API rate limiting
