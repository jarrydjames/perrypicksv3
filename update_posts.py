#!/usr/bin/env python3
"""Update post generator with bets section."""

# Read backup file
with open('src/automation/post_generator.py.backup', 'r') as f:
    content = f.read()

# Update generate_halftime_post
old_halftime_return = '''        return self._add_hashtags(post, platform)'''

new_halftime_return = '''        # Calculate final stats and generate bets
        final_total = pred_final_home + pred_final_away
        final_margin = pred_final_home - pred_final_away
        winner = home_team if final_margin > 0 else away_team
        bets = _generate_best_bets(prediction, "halftime", max_bets=3)
        
        if platform == "twitter":
            emoji = "🔥" if self.use_emojis else "[2H]"
            post = (
                f"{emoji} HALFTIME UPDATE\\n\\n"
                f"{away_team} @ {home_team}\\n\\n"
                f"Halftime: {away_team} {h1_away} - {h1_home} {home_team}\\n\\n"
                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\\n\\n"
                f"Winner: {winner} | Margin: {final_margin:+.1f} | Total: {final_total:.1f}\\n\\n"
            )
            post += self._format_bets_section(bets, platform)
        else:
            post = (
                f"🔥 HALFTIME UPDATE: {away_team} @ {home_team}\\n\\n"
                f"Halftime: {away_team} {h1_away} - {h1_home} {home_team}\\n\\n"
                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\\n\\n"
                f"Winner: {winner} | Margin: {final_margin:+.1f} | Total: {final_total:.1f}\\n\\n"
            )
            post += self._format_bets_section(bets, platform)
        
        return self._add_hashtags(post, platform)'''

# Replace in generate_halftime_post only
content = content.replace(
    '        post = (\n                f"🔥 HALFTIME UPDATE: {away_team} @ {home_team}\\n\\n"\n                f"Halftime: {away_team} {h1_away} - {h1_home} {home_team}\\n\\n"\n                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\\n"\n            )\n        \n        return self._add_hashtags(post, platform)',
    '        post = (\n                f"🔥 HALFTIME UPDATE: {away_team} @ {home_team}\\n\\n"\n                f"Halftime: {away_team} {h1_away} - {h1_home} {home_team}\\n\\n"\n                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\\n"\n            )'
)

# Write updated file
with open('src/automation/post_generator.py', 'w') as f:
    f.write(content)

print('Updated post_generator.py')
