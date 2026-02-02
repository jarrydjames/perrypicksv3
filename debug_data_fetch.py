"""Debug: Check what data we get from NBA API"""
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamgamelog

OKC_ID = 1610612760
DEN_ID = 1610612743

print("=" * 70)
print("DEBUG: Fetching OKC Stats (2025-26)")
print("=" * 70)
okc_stats = leaguedashteamstats.LeagueDashTeamStats(
    team_id_nullable=OKC_ID,
    season='2025-26',
    measure_type_detailed_defense='Base',
    per_mode_detailed='PerGame'
)
okc_df = okc_stats.get_data_frames()[0]
print(f"\nOKC Stats columns: {list(okc_df.columns)}")
print(f"\nFirst row (OKC):")
print(okc_df.iloc[0].to_string())

print("\n" + "=" * 70)
print("DEBUG: Fetching DEN Stats (2025-26)")
print("=" * 70)
den_stats = leaguedashteamstats.LeagueDashTeamStats(
    team_id_nullable=DEN_ID,
    season='2025-26',
    measure_type_detailed_defense='Base',
    per_mode_detailed='PerGame'
)
den_df = den_stats.get_data_frames()[0]
print(f"\nDEN Stats columns: {list(den_df.columns)}")
print(f"\nFirst row (DEN):")
print(den_df.iloc[0].to_string())

print("\n" + "=" * 70)
print("DEBUG: Fetching OKC Game Log (2025-26)")
print("=" * 70)
okc_gamelog = teamgamelog.TeamGameLog(
    team_id=OKC_ID,
    season='2025-26'
)
okc_log = okc_gamelog.get_data_frames()[0]
print(f"\nOKC Game Log columns: {list(okc_log.columns)}")
print(f"\nLast 5 games:")
print(okc_log.head(5).to_string())
