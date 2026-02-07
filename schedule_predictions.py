#!/usr/bin/env python3
"""Schedule and Run All Predictions

This script runs all three prediction models (pregame, halftime, Q3)
for a given date. Designed to be called from cron or scheduler.

Usage:
    python schedule_predictions.py
    python schedule_predictions.py --date 2026-02-05
    python schedule_predictions.py --models pregame halftime
"""

import argparse
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_command(command: list) -> bool:
    """Run a command and return True if successful."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"ERROR: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Schedule and run all prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Run all models for today
  python schedule_predictions.py
  
  # Run all models for specific date
  python schedule_predictions.py --date 2026-02-05
  
  # Run specific models only
  python schedule_predictions.py --models pregame halftime
  
  # Run for specific games
  python schedule_predictions.py --games 0022500733 0022500734

Cron Examples:
  # Run pregame at 6:00 PM (for 7:30 PM games)
  0 18 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1
  
  # Check halftime every 5 minutes during games
  */5 19-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models halftime >> logs/halftime.log 2>&1
  
  # Check Q3 every 5 minutes during games
  */5 20-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models q3 >> logs/q3.log 2>&1
"""
    )
    
    parser.add_argument(
        '--date', '-d',
        default=None,
        help='Date in YYYY-MM-DD format (default: today)',
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['pregame', 'halftime', 'q3'],
        default=['pregame', 'halftime', 'q3'],
        help='Models to run (default: all)',
    )
    
    parser.add_argument(
        '--games', '-g',
        nargs='+',
        help='Specific game IDs to predict (overrides date)',
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show commands without executing',
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    print()
    print("=" * 80)
    print("PERRY PICKS V3 - PREDICTION SCHEDULER")
    print("=" * 80)
    print(f"Date: {date_str}")
    print(f"Models: {', '.join(args.models).upper()}")
    if args.games:
        print(f"Games: {', '.join(args.games)}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 80)
    print()
    
    # Build commands
    commands = []
    
    for model in args.models:
        if model == 'pregame':
            cmd = ['uv', 'run', 'python', 'run_pregame_predictions.py']
        elif model == 'halftime':
            cmd = ['uv', 'run', 'python', 'run_halftime_predictions.py']
        elif model == 'q3':
            cmd = ['uv', 'run', 'python', 'run_q3_predictions.py']
        else:
            continue
        
        # Add date
        cmd.append(date_str)
        
        # Add games if specified
        if args.games:
            cmd.extend(['--games'] + args.games)
        
        commands.append((model.upper(), cmd))
    
    # Show commands or run them
    if args.dry_run:
        print("COMMANDS TO BE RUN:")
        print()
        for model_name, cmd in commands:
            print(f"{model_name}:")
            print(f"  {' '.join(cmd)}")
            print()
    else:
        # Run commands
        results = {}
        
        for model_name, cmd in commands:
            print(f"Running {model_name} predictions...")
            print("-" * 80)
            
            success = run_command(cmd)
            results[model_name] = 'SUCCESS' if success else 'FAILED'
            
            print()
            print("-" * 80)
            print()
            
            # Delay between models to avoid API rate limiting
            if model != args.models[-1]:
                print("Waiting 10 seconds before next model...")
                time.sleep(10)
                print()
        
        # Print summary
        print("=" * 80)
        print("SCHEDULER SUMMARY")
        print("=" * 80)
        for model_name, result in results.items():
            status_symbol = "✓" if result == 'SUCCESS' else "✗"
            print(f"{status_symbol} {model_name}: {result}")
        print("=" * 80)
        
        # Exit with error if any failed
        if any(result != 'SUCCESS' for result in results.values()):
            sys.exit(1)


if __name__ == '__main__':
    main()
