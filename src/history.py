"""
src/history.py
Description: Manages the storage and retrieval of prediction history and actual points.
Key functions:
- `log_predictions`: Saves generated predictions (top 5) for a gameweek to a JSON log.
- `update_actuals`: Backfills the 'actual_points' for past predictions by checking against historical data.
It ensures that we can track performance over time.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

HISTORY_FILE = Path('data/history/predictions_log.json')

def ensure_history_dir():
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)

def load_history():
    ensure_history_dir()
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_history(history):
    ensure_history_dir()
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def log_predictions(predictions_df, gameweek_info):
    """
    Logs the top 5 predictions for the current gameweek.
    """
    history = load_history()
    
    current_gw = gameweek_info.get('next_gameweek')
    if not current_gw:
        print("Warning: Could not determine next gameweek. Skipping log.")
        return

    # Check if we already logged this GW
    existing_entry = next((item for item in history if item['gameweek'] == current_gw), None)
    if existing_entry:
        print(f"Predictions for GW {current_gw} already logged. Overwriting.")
        history.remove(existing_entry)

    # Prepare log entry
    timestamp = datetime.now().isoformat()
    
    # predictions_df is expected to be the top 5 rows
    picks = []
    for _, row in predictions_df.iterrows():
        picks.append({
            'player_id': int(row['element']) if 'element' in row else int(row.get('id', 0)),
            'web_name': row['web_name'],
            'position': row.get('position', 'UNK'),
            'predicted_points': float(row['predicted_points']),
            'actual_points': None  # To be filled later
        })

    entry = {
        'gameweek': current_gw,
        'timestamp': timestamp,
        'picks': picks
    }
    
    history.append(entry)
    save_history(history)
    print(f"Logged predictions for GW {current_gw}")

def update_actuals():
    """
    Updates actual points for past gameweeks using fpl_histories.
    """
    history = load_history()
    updated = False
    
    # Load latest histories
    try:
        histories_path = Path('data/raw/fpl_histories.parquet')
        if not histories_path.exists():
            print("No history data found to update actuals.")
            return
        
        df_hist = pd.read_parquet(histories_path)
    except Exception as e:
        print(f"Error loading histories: {e}")
        return

    for entry in history:
        gw = entry['gameweek']
        
        # Check if any pick needs updating
        needs_update = any(p['actual_points'] is None for p in entry['picks'])
        if not needs_update:
            continue
            
        print(f"Checking actuals for GW {gw}...")
        
        for pick in entry['picks']:
            if pick['actual_points'] is not None:
                continue
                
            # Find match in histories
            # Filter by player_id (element) and round (gameweek)
            match = df_hist[
                (df_hist['element'] == pick['player_id']) & 
                (df_hist['round'] == gw)
            ]
            
            if not match.empty:
                actual = float(match.iloc[0]['total_points'])
                pick['actual_points'] = actual
                updated = True
                print(f"  Updated {pick['web_name']}: {actual} pts")
    
    if updated:
        save_history(history)
        print("History log updated with actual points.")
    else:
        print("No new actual points found.")
