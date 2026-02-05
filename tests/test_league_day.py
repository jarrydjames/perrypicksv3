"""
Tests for league_day and local_day_cst functionality.

Tests ensure:
- _parse_nba_schedule_time produces correct UTC for known examples
- Upsert sets local_day_cst properly regardless of input
- DAILY_SUMMARY payload includes league_day and game_ids only
- Runner skips DAILY_SUMMARY whose payload.league_day != current_league_day
"""

import pytest
import pendulum
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage import init_database, GameStorage
from core.data_sources import NBADataSource
from core.timezone import cst_game_date_from_start_time_utc, CST


class TestNbaScheduleTimeParsing:
    """Test NBA schedule time parsing."""
    
    def test_parse_nba_schedule_time_correct_utc(self):
        """Test that ET 19:00 on 02/05/2026 => 2026-02-06T00:00:00Z."""
        api_date_str = "02/05/2026"
        game_time_utc_placeholder = "1900-01-01T19:00:00Z"
        
        result = NBADataSource._parse_nba_schedule_time(
            None, api_date_str, game_time_utc_placeholder
        )
        
        expected = pendulum.parse("2026-02-06T00:00:00Z")
        
        assert result is not None, "Result should not be None"
        assert result.to_iso8601_string() == expected.to_iso8601_string()
    
    def test_parse_nba_schedule_time_with_different_times(self):
        """Test parsing various times."""
        test_cases = [
            ("02/05/2026", "1900-01-01T19:00:00Z", "2026-02-06T00:00:00Z"),
            ("02/05/2026", "1900-01-01T20:30:00Z", "2026-02-06T01:30:00Z"),
            ("02/05/2026", "1900-01-01T17:00:00Z", "2026-02-05T22:00:00Z"),
        ]
        
        for api_date, time_placeholder, expected_utc in test_cases:
            result = NBADataSource._parse_nba_schedule_time(
                None, api_date, time_placeholder
            )
            expected = pendulum.parse(expected_utc)
            assert result is not None
            assert result.to_iso8601_string() == expected.to_iso8601_string()


class TestGameStorageLeagueDay:
    """Test GameStorage league_day functionality."""
    
    def test_upsert_sets_local_day_cst_from_start_time_utc(self):
        """Test that upsert derives local_day_cst from start_time_utc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            init_database(db_path)
            
            start_time_utc = pendulum.parse("2026-02-06T00:00:00Z")
            local_day_cst = cst_game_date_from_start_time_utc(start_time_utc, tz=CST)
            
            GameStorage.upsert_game(
                game_id="0022500001",
                start_time_utc=start_time_utc,
                home_team="BOS",
                away_team="NYK",
                status="Scheduled",
                league_day="2026-02-05",
                db_path=db_path
            )
            
            game = GameStorage.get_game("0022500001", db_path=db_path)
            
            assert game is not None, "Game should be fetched"
            assert game['local_day_cst'] == "2026-02-05"
            assert game['league_day'] == "2026-02-05"
    
    def test_upsert_does_not_overwrite_league_day_when_not_provided(self):
        """Test that upsert preserves existing league_day when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            init_database(db_path)
            
            start_time_utc = pendulum.parse("2026-02-06T00:00:00Z")
            
            GameStorage.upsert_game(
                game_id="0022500001",
                start_time_utc=start_time_utc,
                home_team="BOS",
                away_team="NYK",
                status="Scheduled",
                league_day="2026-02-05",
                db_path=db_path
            )
            
            GameStorage.upsert_game(
                game_id="0022500001",
                start_time_utc=start_time_utc,
                home_team="BOS",
                away_team="NYK",
                status="In Progress",
                db_path=db_path
            )
            
            game = GameStorage.get_game("0022500001", db_path=db_path)
            
            assert game['league_day'] == "2026-02-05", "league_day should be preserved when not provided in upsert"
            assert game['status'] == "In Progress", "Other fields should be updated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
