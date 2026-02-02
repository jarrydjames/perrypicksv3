"""Debug team name parsing"""
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

SEASON = '2025-26'

gamefinder = leaguegamefinder.LeagueGameFinder(
    league_id_nullable='00',
    season_nullable=SEASON,
    season_type_nullable='Regular Season'
)
df = gamefinder.get_data_frames()[0]

df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
todays_games = df[df['GAME_DATE'].dt.date == pd.Timestamp('2026-02-01').date()]

print(f"Total games found: {len(todays_games)}")
print("\nRaw matchups from API:")
print(todays_games[['GAME_ID', 'MATCHUP', 'WL']].head(20))

print("\n\nDeduplicated:")
unique_games = todays_games.drop_duplicates(subset=['GAME_ID'], keep='first')
print(f"Unique games: {len(unique_games)}")
for _, game in unique_games.iterrows():
    matchup = game['MATCHUP']
    print(f"  {game['GAME_ID']}: {matchup}")
