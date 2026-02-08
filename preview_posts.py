#!/usr/bin/env python3
"""Preview what posts will look like with odds and bet recommendations"""

print("=" * 80)
print("POST PREVIEWS - PerryPicks V3")
print("=" * 80)
print()

# ====================================
# PREGAME - SINGLE GAME (for reference)
# ====================================
print("🏀 " * 15)
print("PREGAME - SINGLE GAME (for reference)")
print("🏀 " * 15)
print()
print("📊 **WAS @ DET**")
print("")
print("📈 **Predicted Scores:**")
print("WAS 112.4 - 110.6 DET")
print("")
print("🎯 **Projected Winner:** WAS")
print("🏆 **Win Probability:** 57.2%")
print("📊 **Game Total:** 223.0")
print("📏 **Margin:** WAS +1.8")
print("")
print("📊 **Team Totals:**")
print(f"  WAS: 112.4 (Over 110.5 @ -110)")
print(f"  DET: 110.6 (Under 111.5 @ -110)")
print()
print("Model: PREGAME_V3_FINAL | Confidence: Medium")
print()
print("#NBAPredictions #PerryPicks")
print()
print("-" * 80)
print()

# ====================================
# PREGAME - FULL SLATE (DISCORD/BLUESKY)
# ====================================
print("📋 " * 15)
print("PREGAME - FULL SLATE (Discord/Bluesky - No character limits)")
print("📋 " * 15)
print()
print("📊 **NBA PREGAME PREDICTIONS - 2026-02-08**")
print("🏀 " * 20)
print()

# Game 1
print("🎯 **Game 1/10: WAS @ DET**")
print("   Scores: WAS 112.4 - 110.6 DET")
print("   Winner: WAS (57.2%) | Total: 223.0 | Margin: +1.8")
print("   Team Totals: WAS 112.4 (O:110.5 @ -110), DET 110.6 (U:111.5 @ -110)")
print()

# Game 2
print("🎯 **Game 2/10: BKN @ ORL**")
print("   Scores: BKN 108.3 - 120.7 ORL")
print("   Winner: ORL (66.1%) | Total: 229.0 | Margin: +12.4")
print("   Team Totals: BKN 108.3 (U:109.5 @ -105), ORL 120.7 (O:119.5 @ -105)")
print()

# Game 3
print("🎯 **Game 3/10: GSW @ LAL**")
print("   Scores: GSW 115.2 - 116.8 LAL")
print("   Winner: LAL (52.8%) | Total: 232.0 | Margin: +1.6")
print("   Team Totals: GSW 115.2 (O:114.5 @ -110), LAL 116.8 (U:117.5 @ -110)")
print()

# Game 4
print("🎯 **Game 4/10: CHA @ ATL**")
print("   Scores: CHA 109.7 - 112.3 ATL")
print("   Winner: ATL (54.9%) | Total: 222.0 | Margin: +2.6")
print("   Team Totals: CHA 109.7 (U:110.5 @ -108), ATL 112.3 (O:111.5 @ -108)")
print()

# Game 5
print("🎯 **Game 5/10: SAC @ PHX**")
print("   Scores: SAC 117.4 - 114.6 PHX")
print("   Winner: SAC (58.3%) | Total: 232.0 | Margin: +2.8")
print("   Team Totals: SAC 117.4 (O:116.5 @ -110), PHX 114.6 (U:115.5 @ -110)")
print()

# Games 6-10 (compact format)
print("🎯 **Games 6-10:**")
print("   6️⃣ **TOR @ BOS** | Winner: TOR (55.1%) | Total: 219.0 | Margin: TOR -1.2")
print("      Team Totals: TOR 110.5 (U:111.5), BOS 108.5 (O:107.5)")
print()
print("   7️⃣ **IND @ MIA** | Winner: MIA (61.4%) | Total: 225.0 | Margin: MIA +4.5")
print("      Team Totals: IND 110.2 (U:111.5), MIA 114.8 (O:113.5)")
print()
print("   8️⃣ **CLE @ MIL** | Winner: CLE (53.7%) | Total: 230.5 | Margin: CLE +2.1")
print("      Team Totals: CLE 116.3 (O:115.5), MIL 114.2 (U:115.5)")
print()
print("   9️⃣ **DEN @ UTA** | Winner: DEN (67.2%) | Total: 228.0 | Margin: DEN +9.6")
print("      Team Totals: DEN 118.8 (O:117.5), UTA 109.2 (U:110.5)")
print()
print("   🔟 **MIN @ OKC** | Winner: OKC (59.8%) | Total: 226.0 | Margin: OKC +4.2")
print("      Team Totals: MIN 110.9 (U:111.5), OKC 115.1 (O:114.5)")
print()

print("Model: PREGAME_V3_FINAL | Games: 10 | Confidence: High")
print()
print("#NBAPredictions #PerryPicks #NBA")
print()
print("-" * 80)
print()

# ====================================
# PREGAME - TWITTER THREAD
# ====================================
print("🐦 " * 15)
print("PREGAME - TWITTER THREAD (character limit: 280)")
print("🐦 " * 15)
print()

print("" * 40 + "🧵 Post 1 (Summary):" + " " * 40)
print("=" * 40)
print()
twitter_1 = """📊 **NBA PREGAME PREDICTIONS** 🏀

🗓️ 2026-02-08 | 10 Games Today

🔥 **Highest Win Probs:**
• DEN (67.2%) vs UTA
• ORL (66.1%) vs BKN
• MIA (61.4%) vs IND

📈 **Highest Totals:**
• SAC/PHX: 232.0
• GSW/LAL: 232.0
• CLE/MIL: 230.5

🧵 Full breakdowns below 👇"""
print(twitter_1)
print(f"Character count: {len(twitter_1)}/280")
print()

print("" * 40 + "🧵 Post 2 (Games 1-3):" + " " * 40)
print("=" * 40)
print()
twitter_2 = """🏀 Games 1-3:

1️⃣ WAS @ DET
WAS 112.4 - 110.6
WAS (57.2%) | T: 223.0

2️⃣ BKN @ ORL
BKN 108.3 - 120.7
ORL (66.1%) | T: 229.0

3️⃣ GSW @ LAL
GSW 115.2 - 116.8
LAL (52.8%) | T: 232.0

🧵 Games 4-7 👇"""
print(twitter_2)
print(f"Character count: {len(twitter_2)}/280")
print()

print("" * 40 + "🧵 Post 3 (Games 4-7):" + " " * 40)
print("=" * 40)
print()
twitter_3 = """🏀 Games 4-7:

4️⃣ CHA @ ATL
CHA 109.7 - 112.3
ATL (54.9%) | T: 222.0

5️⃣ SAC @ PHX
SAC 117.4 - 114.6
SAC (58.3%) | T: 232.0

6️⃣ TOR @ BOS
TOR 111.3 - 107.7
TOR (55.1%) | T: 219.0

7️⃣ IND @ MIA
IND 108.9 - 116.1
MIA (61.4%) | T: 225.0

🧵 Games 8-10 + recap 👇"""
print(twitter_3)
print(f"Character count: {len(twitter_3)}/280")
print()

print("" * 40 + "🧵 Post 4 (Games 8-10):" + " " * 40)
print("=" * 40)
print()
twitter_4 = """🏀 Games 8-10:

8️⃣ CLE @ MIL
CLE 114.8 - 115.7
CLE (53.7%) | T: 230.5

9️⃣ DEN @ UTA
DEN 119.6 - 108.4
DEN (67.2%) | T: 228.0

🔟 MIN @ OKC
MIN 112.7 - 113.3
OKC (59.8%) | T: 226.0

📊 **Summary:**
Best Win Prob: DEN (67.2%)
Highest Total: SAC/PHX (232.0)

Model: PREGAME_V3_FINAL

#NBAPredictions #PerryPicks #NBA"""
print(twitter_4)
print(f"Character count: {len(twitter_4)}/280")
print()
print("-" * 80)
print()

# ====================================
# HALFTIME (already approved - with bets)
# ====================================
print("⏸️  " * 20)
print("HALFTIME POST (with bets - already approved)")
print("⏸️  " * 20)
print()
print("⚡ **HALFTIME UPDATE: CHA @ ATL**")
print("")
print("📊 **Halftime Score:**")
print("CHA 60 - 60 ATL")
print("")
print("📈 **Predicted Final:**")
print("CHA 116.6 - 114.4 ATL")
print("")
print("🎯 **Projected Winner:** CHA")
print("📏 **Margin:** CHA +2.2")
print("📊 **Game Total:** 231.0")
print()
print("🎯 **Best Bets (Top 3 by Edge):**")
print("")
print("🔥 1. **Over 230.5 @ -110** (edge +6.8%)")
print("   P(Over): 55.8% | Kelly: 2.4%")
print("")
print("✅ 2. **CHA +1.5 @ -105** (edge +5.2%)")
print("   P(Cover): 55.2% | Kelly: 1.8%")
print("")
print("💰 3. **CHA Over 115.5 @ -110** (edge +4.1%)")
print("   P(Over): 54.9% | Kelly: 1.2%")
print()
print("Model: HALFTIME_V2_CI | Confidence: High")
print()
print("#NBAPredictions #PerryPicks #Halftime")
print()
print("-" * 80)
print()

# ====================================
# Q3 (already approved - with bets)
# ====================================
print("🏀 " * 20)
print("Q3 POST (with bets - already approved)")
print("🏀 " * 20)
print()
print("⚡ **Q3 UPDATE: GSW @ LAL**")
print("")
print("📊 **Q3 Score:**")
print("GSW 71.0 - 79.0 LAL")
print("")
print("📈 **Projected Final:**")
print("GSW 93.4 - 104.6 LAL")
print("")
print("🎯 **Projected Winner:** LAL")
print("📏 **Margin:** LAL +11.2")
print("📊 **Game Total:** 198.0")
print()
print("🎯 **Best Bets (Top 3 by Edge):**")
print("")
print("🔥 1. **Under 200.5 @ -110** (edge +8.3%)")
print("   P(Under): 56.4% | Kelly: 3.1%")
print("")
print("✅ 2. **LAL -10.5 @ -108** (edge +7.1%)")
print("   P(Cover): 56.1% | Kelly: 2.5%")
print("")
print("💰 3. **GSW Under 96.5 @ -110** (edge +5.4%)")
print("   P(Under): 55.5% | Kelly: 1.6%")
print()
print("Model: Q3 Neural Network | Confidence: High")
print()
print("#NBAPredictions #PerryPicks #Q3")
print()
print("=" * 80)

# ====================================
# SUMMARY
# ====================================
print("=" * 80)
print("SUMMARY - POST FORMATS")
print("=" * 80)
print()

print("""✅ PREGAME:
• Team totals, game total, winner, win %, margin
• NO BETS (informational only)
• Twitter: 4-post thread (3 games per post)
• Discord: Single post with all 10 games

✅ HALFTIME:
• Team totals, game total, winner, margin
• Top 3 bets with edge and probability
• Twitter/Discord: Single post

✅ Q3:
• Team totals, game total, winner, margin
• Top 3 bets with edge and probability
• Twitter/Discord: Single post
""")

print()
print("=" * 80)
print("APPROVAL NEEDED")
print("=" * 80)
print()
print("👀 Review above formats and let me know:")
print("   1. ✅ Approve all formats
   2. 🔧 Request changes to specific format
   3. 📝 Ask questions about any format")
print()
print("=" * 80)