import json
import pandas as pd
from datetime import datetime

# 1. Check Bassey's Status
try:
    with open('data/raw/fpl_bootstrap.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    bassey = next((p for p in data['elements'] if 'Bassey' in p['web_name']), None)
    if bassey:
        print(f"\n--- Player Status: {bassey['web_name']} ---")
        print(f"Status: {bassey['status']}") # a=available, u=unavailable, i=international
        print(f"Chance of playing next round: {bassey['chance_of_playing_next_round']}")
        print(f"News: {bassey['news']}")
    else:
        print("\nPlayer 'Bassey' not found.")
        
    # Check current gameweek in bootstrap
    current_gw = next((e for e in data['events'] if e['is_current']), None)
    next_gw = next((e for e in data['events'] if e['is_next']), None)
    print(f"\n--- Gameweek Info ---")
    print(f"Current GW: {current_gw['id'] if current_gw else 'None'}")
    print(f"Next GW: {next_gw['id'] if next_gw else 'None'}")

except Exception as e:
    print(f"Error checking FPL data: {e}")

# 2. Check Understat Freshness
try:
    with open('data/raw/understat_matches_2024.json', 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    # Matches is a list of dicts. Each has 'date' (usually "2024-08-16 19:00:00")
    # Filter for played matches (goals usually populated, or just check all dates)
    played_matches = [m for m in matches if m.get('isResult', False)]
    
    if played_matches:
        last_match = max(played_matches, key=lambda x: x['datetime'])
        print(f"\n--- Understat Data Freshness ---")
        print(f"Total Matches: {len(matches)}")
        print(f"Played Matches: {len(played_matches)}")
        print(f"Most Recent Match: {last_match['h']['title']} vs {last_match['a']['title']} on {last_match['datetime']}")
    else:
        print("\nNo played matches found in Understat data.")

except Exception as e:
    print(f"Error checking Understat data: {e}")
