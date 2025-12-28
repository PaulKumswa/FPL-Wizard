"""
src/check_deadline.py
Description: A utility script used by GitHub Actions (or other schedulers) to determine if the pipeline should run.
It checks the FPL API for the next gameweek deadline and compares it with the current time.
It sets a GitHub Output variable `should_update` to 'true' if the update window is open (e.g., < 24h before deadline) 
and predictions haven't been generated yet.
"""

import os
import sys
import json
import requests
import datetime
from pathlib import Path

# Configuration
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HISTORY_FILE = Path("data/history/predictions_log.json")
UPDATE_WINDOW_HOURS_START = 24
UPDATE_WINDOW_HOURS_END = 0 # Update anytime from 24h before up to the deadline

def fetch_bootstrap():
    """Fetches FPL bootstrap data."""
    try:
        response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching bootstrap data: {e}")
        sys.exit(1)

def get_next_gameweek(bootstrap_data):
    """Finds the next upcoming gameweek."""
    for event in bootstrap_data.get('events', []):
        if event.get('is_next'):
            return event
    return None

def has_updated_for_gameweek(gameweek_id):
    """Checks if we have already generated predictions for the given gameweek."""
    if not HISTORY_FILE.exists():
        return False
    
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            # Handle empty list or unexpected format
            if not isinstance(history, list):
                return False
            
            # Check if any entry matches the gameweek_id
            for entry in history:
                if entry.get('gameweek') == gameweek_id:
                    return True
    except Exception as e:
        print(f"Warning: Could not read history file: {e}")
        return False
    
    return False

def set_github_output(key, value):
    """Sets a GitHub Actions output variable."""
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"{key}={value}\n")
    else:
        # Fallback for local testing
        print(f"GITHUB_OUTPUT: {key}={value}")

def main():
    print("Checking Gameweek Deadlines...")
    
    data = fetch_bootstrap()
    next_gw = get_next_gameweek(data)
    
    if not next_gw:
        print("No next gameweek found. Season might be over.")
        set_github_output("should_update", "false")
        return

    gw_id = next_gw['id']
    deadline_str = next_gw['deadline_time']
    # FPL dates are UTC
    deadline_dt = datetime.datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    
    time_diff = deadline_dt - now_dt
    hours_until_deadline = time_diff.total_seconds() / 3600
    
    print(f"Next Gameweek: {gw_id}")
    print(f"Deadline: {deadline_dt} (UTC)")
    print(f"Current Time: {now_dt} (UTC)")
    print(f"Hours until deadline: {hours_until_deadline:.2f}")

    # Logic:
    # 1. Check if we are within the update window (e.g., 24 hours before deadline).
    #    We also want to ensure we don't update *too* late if we want to (though user said 12 or 24h).
    #    Let's say anything less than 24 hours is "time to update".
    #    Unless the deadline has passed (hours < 0).
    
    is_time_to_update = 0 < hours_until_deadline <= UPDATE_WINDOW_HOURS_START
    
    if not is_time_to_update:
        if hours_until_deadline > UPDATE_WINDOW_HOURS_START:
            print(f"Deadline is too far away (> {UPDATE_WINDOW_HOURS_START} hours). No update needed yet.")
        else:
            print("Deadline has passed.")
        set_github_output("should_update", "false")
        return

    # 2. Check if we already have predictions for this GW
    if has_updated_for_gameweek(gw_id):
        print(f"Predictions for Gameweek {gw_id} already exist. No update needed.")
        set_github_output("should_update", "false")
        return

    print("Update Window Open AND No predictions found for this Gameweek.")
    print(">>> TRIGGERING UPDATE <<<")
    set_github_output("should_update", "true")

if __name__ == "__main__":
    main()
