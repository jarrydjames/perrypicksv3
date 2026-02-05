"""
Regression tests for timezone game date calculation.

Tests that game_date is correctly derived from start_time_utc in CST,
preventing bugs where late-evening games get assigned to the wrong day.

Run with:
    pytest tests/test_timezone_game_date.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pendulum
from core.timezone import cst_game_date_from_start_time_utc, CST


def test_game_date_boundary_utc_to_cst():
    """
    Test that games starting in early morning UTC are assigned to previous CST day.
    
    Example: 2026-02-04T03:00:00Z == 2026-02-03 21:00 CST
    - UTC: Feb 4, 3:00 AM
    - CST: Feb 3, 9:00 PM (previous day!)
    - Therefore game_date should be '2026-02-03'
    """
    dt = pendulum.parse("2026-02-04T03:00:00Z")
    result = cst_game_date_from_start_time_utc(dt, tz=CST)
    assert result == "2026-02-03", f"Expected '2026-02-03', got '{result}'"


def test_game_date_feb4_evening_is_feb4_cst():
    """
    Test that games starting in early morning UTC (midnight to 6 AM) get the previous CST date.
    
    Example: 2026-02-05T00:00:00Z == 2026-02-04 18:00 CST
    - UTC: Feb 5, 12:00 AM (midnight)
    - CST: Feb 4, 6:00 PM
    - Therefore game_date should be '2026-02-04'
    """
    dt = pendulum.parse("2026-02-05T00:00:00Z")
    result = cst_game_date_from_start_time_utc(dt, tz=CST)
    assert result == "2026-02-04", f"Expected '2026-02-04', got '{result}'"


def test_game_date_midnight_utc_is_previous_cst():
    """
    Test that midnight UTC is 6 PM CST (previous day).
    
    Example: 2026-02-04T00:00:00Z == 2026-02-03 18:00 CST
    - UTC: Feb 4, 12:00 AM (midnight)
    - CST: Feb 3, 6:00 PM (previous day!)
    - Therefore game_date should be '2026-02-03'
    """
    dt = pendulum.parse("2026-02-04T00:00:00Z")
    result = cst_game_date_from_start_time_utc(dt, tz=CST)
    assert result == "2026-02-03", f"Expected '2026-02-03', got '{result}'"


def test_game_date_six_am_utc_is_midnight_cst():
    """
    Test that 6 AM UTC is midnight CST.
    
    Example: 2026-02-04T06:00:00Z == 2026-02-04 00:00 CST
    - UTC: Feb 4, 6:00 AM
    - CST: Feb 4, 12:00 AM (midnight)
    - Therefore game_date should be '2026-02-04'
    """
    dt = pendulum.parse("2026-02-04T06:00:00Z")
    result = cst_game_date_from_start_time_utc(dt, tz=CST)
    assert result == "2026-02-04", f"Expected '2026-02-04', got '{result}'"


def test_game_date_afternoon_utc_same_day_cst():
    """
    Test that afternoon UTC games have the same CST date.
    
    Example: 2026-02-04T18:00:00Z == 2026-02-04 12:00 CST
    - UTC: Feb 4, 6:00 PM
    - CST: Feb 4, 12:00 PM (noon)
    - Therefore game_date should be '2026-02-04'
    """
    dt = pendulum.parse("2026-02-04T18:00:00Z")
    result = cst_game_date_from_start_time_utc(dt, tz=CST)
    assert result == "2026-02-04", f"Expected '2026-02-04', got '{result}'"


def test_game_date_with_iso_string_input():
    """
    Test that function works with ISO string input (not just pendulum.DateTime).
    """
    iso_str = "2026-02-04T03:00:00Z"
    result = cst_game_date_from_start_time_utc(iso_str, tz=CST)
    assert result == "2026-02-03", f"Expected '2026-02-03', got '{result}'"


def test_game_date_dst_handling():
    """
    Test that function correctly handles DST transitions (using CDT).
    
    During CDT (daylight saving time), CST offset is UTC-5 instead of UTC-6.
    This test verifies the function uses the correct timezone.
    """
    # During summer (CDT): 2025-07-04T06:00:00Z == 2025-07-04 01:00 CDT
    dt = pendulum.parse("2025-07-04T06:00:00Z")
    result = cst_game_date_from_start_time_utc(dt, tz='America/Chicago')
    assert result == "2025-07-04", f"Expected '2025-07-04', got '{result}'"


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run manually
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed, running tests manually...")
        test_game_date_boundary_utc_to_cst()
        print("✓ test_game_date_boundary_utc_to_cst passed")
        
        test_game_date_feb4_evening_is_feb4_cst()
        print("✓ test_game_date_feb4_evening_is_feb4_cst passed")
        
        test_game_date_midnight_utc_is_previous_cst()
        print("✓ test_game_date_midnight_utc_is_previous_cst passed")
        
        test_game_date_six_am_utc_is_midnight_cst()
        print("✓ test_game_date_six_am_utc_is_midnight_cst passed")
        
        test_game_date_afternoon_utc_same_day_cst()
        print("✓ test_game_date_afternoon_utc_same_day_cst passed")
        
        test_game_date_with_iso_string_input()
        print("✓ test_game_date_with_iso_string_input passed")
        
        test_game_date_dst_handling()
        print("✓ test_game_date_dst_handling passed")
        
        print("\nAll tests passed! ✓")
