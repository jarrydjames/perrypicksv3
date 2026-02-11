#!/usr/bin/env python3
"""Test game state monitoring and trigger evaluation.

This script tests the full flow:
1. Update game states
2. Check for halftime games
3. Evaluate triggers
4. See what happens

Run this to debug why triggers aren't firing.
"""

import logging
import sys
from datetime import date

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

def main():
    print("="*80)
    print("TESTING GAME STATE MONITORING")
    print("="*80)
    
    try:
        # Import components
        print("\n[1] Importing components...")
        from src.automation.game_state_service import GameStateService
        from src.data.scoreboard import fetch_scoreboard
        print("✓ Components imported")
        
        # Initialize game state service (includes all components)
        print("\n[2] Initializing game state service...")
        service = GameStateService(
            poll_interval_seconds=30,
            platforms=None,  # All enabled platforms
            dry_run=False,  # Actually post
        )
        print("✓ Game state service initialized")
        
        # Get components from service
        monitor = service.game_monitor
        trigger_engine = service.trigger_engine
        processor = service.queue_processor
        
        # Check available platforms
        print("\n[3] Checking available platforms...")
        if processor.social_manager:
            platforms = list(processor.social_manager.enabled_platforms)
            print(f"✓ Found {len(platforms)} enabled platform(s): {platforms}")
        else:
            print("⚠️  No social manager available")
        
        # Fetch today's games
        print(f"\n[4] Fetching today's games ({date.today()})...")
        games = fetch_scoreboard(date.today())
        print(f"✓ Fetched {len(games) if games else 0} game(s)")
        
        if not games:
            print("\n⚠️  No games found for today!")
            return
        
        # Update all game states
        print("\n[5] Updating game states...")
        updated_states = monitor.update_all_games()
        print(f"✓ Updated {len(updated_states)} game state(s)")
        
        # Get all game states
        print("\n[6] Getting all game states...")
        all_states = monitor.get_all_states()
        print(f"✓ Found {len(all_states)} game state(s)")
        
        # Check for halftime games
        print("\n[7] Checking for halftime games...")
        halftime_games = []
        for game_id, state in all_states.items():
            print(f"  - Game {game_id}: status={state.status}, period={state.period}, time={state.time_remaining}")
            if state.status == "halftime":
                halftime_games.append((game_id, state))
        
        print(f"\n✓ Found {len(halftime_games)} halftime game(s):")
        for game_id, state in halftime_games:
            print(f"  - Game {game_id}: {state.away_team} @ {state.home_team}")
            print(f"    Status: {state.status}, Period: {state.period}, Score: {state.away_score}-{state.home_score}")
        
        # Evaluate triggers
        print("\n[8] Evaluating triggers...")
        fired_events = trigger_engine.evaluate_all(platforms=None)
        print(f"✓ Evaluated triggers, fired {len(fired_events)} trigger(s)")
        
        if fired_events:
            print("\n" + "="*80)
            print("FIRED TRIGGERS:")
            print("="*80)
            for event in fired_events:
                print(f"  - Game: {event.game_id}")
                print(f"    Type: {event.trigger_type}")
                print(f"    Time: {event.fired_at}")
                if event.prediction:
                    print(f"    Prediction status: {event.prediction.get('status')}")
                    print(f"    Prediction error: {event.prediction.get('error')}")
        else:
            print("\n⚠️  No triggers fired!")
        
        # Get trigger engine stats
        print("\n[9] Trigger engine stats...")
        print(f"  - Fired triggers: {trigger_engine.fired_triggers}")
        print(f"  - Number of fired triggers: {len(trigger_engine.fired_triggers)}")
        
        # Check if fired_triggers contains the halftime games
        if halftime_games:
            print("\n[10] Checking if halftime triggers already fired...")
            for game_id, state in halftime_games:
                from src.automation.trigger_engine import TriggerType
                trigger_key = trigger_engine._make_trigger_key(game_id, TriggerType.HALFTIME)
                already_fired = trigger_key in trigger_engine.fired_triggers
                print(f"  - Game {game_id}: {'ALREADY FIRED' if already_fired else 'NOT YET FIRED'}")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
