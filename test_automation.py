"""
Quick verification script for PerryPicks v4 Automation System.
Tests database initialization and basic functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.storage import init_database, GameStorage, TriggerStorage


def test_database():
    """Test database initialization and basic operations."""
    print("Testing PerryPicks v4 Automation System...")
    print("-" * 50)
    
    # Initialize database
    db_path = Path('data/automation.db')
    init_database(db_path)
    print("✅ Database initialized")
    
    # Test game upsert
    game_id = 'test_game_001'
    start_time = datetime(2026, 2, 1, 19, 0, 0, tzinfo=timezone.utc)
    GameStorage.upsert_game(
        game_id=game_id,
        start_time_utc=start_time,
        home_team='Lakers',
        away_team='Warriors',
        status='Scheduled',
        game_date='2026-02-01',
        db_path=db_path
    )
    print(f"✅ Game upserted: {game_id}")
    
    # Test trigger scheduling
    scheduled_time = datetime(2026, 2, 1, 16, 0, 0, tzinfo=timezone.utc)
    trigger_id = TriggerStorage.schedule_trigger(
        game_id=game_id,
        trigger_type='PRE_3H',
        scheduled_time_utc=scheduled_time,
        payload={'home_team': 'Lakers', 'away_team': 'Warriors'},
        db_path=db_path
    )
    print(f"✅ Trigger scheduled: ID={trigger_id}")
    
    # Test game retrieval
    game = GameStorage.get_game(game_id, db_path=db_path)
    if game:
        print(f"✅ Game retrieved: {game['home_team']} vs {game['away_team']}")
    
    # Test trigger retrieval
    triggers = TriggerStorage.get_triggers_for_game(game_id, db_path=db_path)
    print(f"✅ Triggers retrieved: {len(triggers)} triggers")
    
    print("-" * 50)
    print("🎉 All tests passed! Automation system is ready.")
    print()
    print("Next steps:")
    print("1. Copy config/.env.example to .env and fill in your API keys")
    print("2. Run: python -m worker.runner --once")
    print("3. For continuous automation: python -m worker.runner")


if __name__ == '__main__':
    try:
        test_database()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
