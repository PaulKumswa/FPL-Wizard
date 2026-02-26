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
from src.config import MODEL_VERSION, MODEL_ERAS

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
            'confidence_score': float(row.get('confidence_score', 50.0)),
            'is_wildcard': bool(row.get('is_wildcard', False)),
            'actual_points': None  # To be filled later
        })

    entry = {
        'gameweek': current_gw,
        'timestamp': timestamp,
        'model_version': MODEL_VERSION['version'],
        'model_name': MODEL_VERSION['name'],
        'model_type': MODEL_VERSION['type'],
        'picks': picks
    }
    
    history.append(entry)
    save_history(history)
    print(f"Logged predictions for GW {current_gw}")


FULL_PREDICTIONS_FILE = Path('data/history/full_predictions_log.json')

def log_full_predictions(all_predictions_df, gameweek_info, top_n=30):
    """Save top N predictions (all positions) for retrospective analysis."""
    ensure_history_dir()
    
    current_gw = gameweek_info.get('next_gameweek')
    if not current_gw:
        print("Warning: Could not determine next gameweek. Skipping full log.")
        return
    
    # Load existing full log
    full_log = []
    if FULL_PREDICTIONS_FILE.exists():
        try:
            with open(FULL_PREDICTIONS_FILE, 'r') as f:
                full_log = json.load(f)
        except json.JSONDecodeError:
            full_log = []
    
    # Remove existing entry for this GW (overwrite)
    full_log = [e for e in full_log if e['gameweek'] != current_gw]
    
    # Filter to high+medium confidence (>=40%) then take top N by predicted points
    confident = all_predictions_df[
        pd.to_numeric(all_predictions_df.get('confidence_score', 50.0), errors='coerce').fillna(50.0) >= 40.0
    ]
    top = confident.nlargest(top_n, 'predicted_points')
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    players = []
    for _, row in top.iterrows():
        players.append({
            'player_id': int(row['element']),
            'web_name': row['web_name'],
            'position': pos_map.get(int(row.get('element_type', 0)), 'UNK'),
            'predicted_points': round(float(row['predicted_points']), 2),
            'confidence_score': round(float(row.get('confidence_score', 50.0)), 1),
            'now_cost': float(row.get('now_cost', 0)),
            'selected_by_percent': float(row.get('selected_by_percent', 0)),
            'team_name': str(row.get('team_name', '')),
            'actual_points': None
        })
    
    full_log.append({
        'gameweek': current_gw,
        'timestamp': datetime.now().isoformat(),
        'predictions': players
    })
    
    with open(FULL_PREDICTIONS_FILE, 'w') as f:
        json.dump(full_log, f, indent=2)
    
    print(f"Logged top {top_n} full predictions for GW {current_gw}")


COMPLETE_PREDICTIONS_DIR = Path('data/history/complete_predictions')

def save_complete_predictions(all_predictions_df, gameweek_info):
    """Save ALL unfiltered predictions for a gameweek to a separate archive.
    
    Stored as individual files: data/history/complete_predictions/gw_XX.json
    This is for long-term analysis, independent of the Scout tab display.
    """
    ensure_history_dir()
    COMPLETE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    current_gw = gameweek_info.get('next_gameweek')
    if not current_gw:
        print("Warning: Could not determine next gameweek. Skipping complete log.")
        return
    
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    players = []
    for _, row in all_predictions_df.iterrows():
        players.append({
            'player_id': int(row['element']),
            'web_name': row['web_name'],
            'position': pos_map.get(int(row.get('element_type', 0)), 'UNK'),
            'team_name': str(row.get('team_name', '')),
            'team': int(row.get('team', 0)),
            'predicted_points': round(float(row['predicted_points']), 2),
            'confidence_score': round(float(row.get('confidence_score', 50.0)), 1),
            'now_cost': float(row.get('now_cost', 0)),
            'selected_by_percent': float(row.get('selected_by_percent', 0)),
            'status': str(row.get('status', '')),
            'chance_of_playing': float(row.get('chance_of_playing_next_round', 100)),
        })
    
    # Sort by predicted points descending
    players.sort(key=lambda p: p['predicted_points'], reverse=True)
    
    archive = {
        'gameweek': current_gw,
        'timestamp': datetime.now().isoformat(),
        'total_players': len(players),
        'predictions': players
    }
    
    filepath = COMPLETE_PREDICTIONS_DIR / f'gw_{current_gw:02d}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
    
    print(f"Archived {len(players)} complete predictions for GW {current_gw} → {filepath}")

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


def backfill_model_versions():
    """
    Adds model_version metadata to historical entries based on gameweek.
    Run once to populate existing data.
    """
    history = load_history()
    updated = False
    
    # Build gameweek -> era lookup from MODEL_ERAS
    gw_to_era = {}
    for era in MODEL_ERAS:
        for gw in era['gameweeks']:
            gw_to_era[gw] = era
    
    for entry in history:
        gw = entry['gameweek']
        era = gw_to_era.get(gw)
        
        # Check if this is a skipped week (empty picks)
        is_skipped = len(entry.get('picks', [])) == 0
        
        if era:
            # Explicit era assignment
            if entry.get('model_version') != era['version']:
                entry['model_version'] = era['version']
                entry['model_name'] = era['name']
                entry['model_type'] = era['type']
                updated = True
                print(f"  GW {gw} -> {era['version']} ({era['name']})")
        elif is_skipped:
            # Skipped weeks - mark appropriately based on surrounding GWs
            # GW14 was between v1 GWs (13, 15) -> v1
            # GW18 was between v1 GW17 and v2 GW19 -> mark as v1 (last active version)
            if gw == 14:
                if entry.get('model_version') != 'v1':
                    entry['model_version'] = 'v1'
                    entry['model_name'] = '4 Position Regressors'
                    entry['model_type'] = 'regressor'
                    updated = True
                    print(f"  GW {gw} -> v1 (skipped week)")
            elif gw == 18:
                if entry.get('model_version') != 'v1':
                    entry['model_version'] = 'v1'
                    entry['model_name'] = '4 Position Regressors'
                    entry['model_type'] = 'regressor'
                    updated = True
                    print(f"  GW {gw} -> v1 (skipped week)")
            else:
                # Future skipped weeks default to current version
                if entry.get('model_version') != 'v3':
                    entry['model_version'] = 'v3'
                    entry['model_name'] = MODEL_VERSION['name']
                    entry['model_type'] = MODEL_VERSION['type']
                    updated = True
                    print(f"  GW {gw} -> v3 (skipped, default)")
        else:
            # Non-skipped week not in any era list -> v3 (future weeks)
            if entry.get('model_version') != 'v3':
                entry['model_version'] = 'v3'
                entry['model_name'] = MODEL_VERSION['name']
                entry['model_type'] = MODEL_VERSION['type']
                updated = True
                print(f"  GW {gw} -> v3 (default/current)")
    
    if updated:
        save_history(history)
        print("Backfilled model versions for historical entries.")
    else:
        print("All entries already have correct model versions.")
