"""
src/id_map.py
Description: Handles the mapping of player IDs between FPL (Fantasy Premier League) and Understat data sources.
Since FPL and Understat use different naming conventions and IDs, this script uses fuzzy matching (fuzzywuzzy) 
to link players. It maintains a persistent mapping file `known_id_mapping.json` to store verified mappings, 
improving accuracy and performance over subsequent runs.
"""
import json
import pandas as pd
import os
from pathlib import Path
from fuzzywuzzy import process, fuzz
import unidecode

MAPPING_FILE = Path('data/config/known_id_mapping.json')
OUTPUT_FILE = Path('data/processed/id_mapping.csv')

def normalize_name(name):
    """Normalize names: lowercase, remove accents, strip whitespace."""
    if not isinstance(name, str):
        return ""
    return unidecode.unidecode(name).lower().strip()

def load_data():
    """Load FPL and Understat raw data."""
    try:
        with open('data/raw/fpl_bootstrap.json', 'r', encoding='utf-8') as f:
            fpl_data = json.load(f)
        fpl_players = pd.DataFrame(fpl_data['elements'])
        fpl_teams = pd.DataFrame(fpl_data['teams'])
        
        # Add Team Name to FPL Players
        fpl_teams_map = fpl_teams.set_index('id')['name'].to_dict()
        fpl_players['team_name'] = fpl_players['team'].map(fpl_teams_map)
        
        # Load Understat
        files = os.listdir('data/raw')
        understat_files = [f for f in files if 'understat_players_' in f and f.endswith('.json')]
        if not understat_files:
            raise FileNotFoundError("No Understat player data found in data/raw/")
        
        # Pick the one with highest year
        latest_file = sorted(understat_files)[-1]
        print(f"Using Understat file: {latest_file}")
        
        with open(f"data/raw/{latest_file}", 'r', encoding='utf-8') as f:
            understat_data = json.load(f)
        understat_players = pd.DataFrame(understat_data)
        
        return fpl_players, understat_players
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def load_known_mappings():
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_known_mappings(mappings):
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"Updated known mappings: {MAPPING_FILE}")

def map_ids():
    fpl_players, understat_players = load_data()
    if fpl_players is None or understat_players is None:
        return

    known_mappings = load_known_mappings()
    final_mapping_list = []
    
    # Pre-process names
    fpl_players['norm_name'] = (fpl_players['first_name'] + " " + fpl_players['second_name']).apply(normalize_name)
    understat_players['norm_name'] = understat_players['player_name'].apply(normalize_name)
    
    # Create Understat Lookup for fast access
    us_lookup = understat_players.set_index('norm_name')[['id', 'player_name']].to_dict('index')
    
    # List of names for fuzzy matching
    us_names = understat_players['norm_name'].tolist()
    
    new_mappings_found = False
    
    print(f"Mapping {len(fpl_players)} FPL players...")

    for _, fpl_p in fpl_players.iterrows():
        fpl_id = str(fpl_p['id']) # Use string keys for JSON
        fpl_full_name = f"{fpl_p['first_name']} {fpl_p['second_name']}"
        fpl_norm = fpl_p['norm_name']
        
        # 1. Check Known Mappings
        if fpl_id in known_mappings:
            entry = known_mappings[fpl_id]
            final_mapping_list.append({
                'fpl_id': int(fpl_id),
                'understat_id': entry['understat_id'],
                'fpl_name': fpl_full_name,
                'understat_name': entry['understat_name'],
                'score': 100 # Trusted
            })
            continue

        # 2. Try Exact Match
        match_id = None
        match_name = None
        score = 0
        
        if fpl_norm in us_lookup:
            match = us_lookup[fpl_norm]
            match_id = match['id']
            match_name = match['player_name']
            score = 100
        else:
            # 3. Fuzzy Match
            best_match = process.extractOne(fpl_norm, us_names, scorer=fuzz.token_sort_ratio)
            if best_match and best_match[1] > 85: # Threshold
                score = best_match[1]
                # Find ID
                matched_row = understat_players[understat_players['norm_name'] == best_match[0]].iloc[0]
                match_id = matched_row['id']
                match_name = matched_row['player_name']
        
        # 4. Result
        if match_id:
            # Add to Final List
            final_mapping_list.append({
                'fpl_id': int(fpl_id),
                'understat_id': match_id,
                'fpl_name': fpl_full_name,
                'understat_name': match_name,
                'score': score
            })
            
            # Add to Persistent Storage (Learning)
            known_mappings[fpl_id] = {
                'understat_id': match_id,
                'fpl_name': fpl_full_name,
                'understat_name': match_name
            }
            new_mappings_found = True
        else:
            # Log failure (optional: add to a "missing" list?)
            # print(f"Could not map: {fpl_full_name}")
            pass
            
    # Save Persistent
    if new_mappings_found:
        save_known_mappings(known_mappings)
        
    # Save CSV for Pipeline
    mapping_df = pd.DataFrame(final_mapping_list)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Mapping complete. Mapped {len(mapping_df)}/{len(fpl_players)} players.")
    print(f"Saved pipeline map to {OUTPUT_FILE}")

if __name__ == "__main__":
    map_ids()
