import json
import pandas as pd
import os
from pathlib import Path
from fuzzywuzzy import process, fuzz
import unidecode

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
        # Find the latest understat file or specific one. For now assume 2024.
        # Check files in dir for latest
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

def map_ids():
    fpl_players, understat_players = load_data()
    if fpl_players is None or understat_players is None:
        return

    # Load Overrides
    overrides = {}
    if os.path.exists('data/config/id_mapping_overrides.json'):
        with open('data/config/id_mapping_overrides.json', 'r') as f:
            overrides = json.load(f)

    # Prepare Mappings
    mapping = []
    
    # Pre-process names
    fpl_players['norm_name'] = (fpl_players['first_name'] + " " + fpl_players['second_name']).apply(normalize_name)
    fpl_players['web_name_norm'] = fpl_players['web_name'].apply(normalize_name)
    understat_players['norm_name'] = understat_players['player_name'].apply(normalize_name)
    
    # Team Mapping (FPL -> Understat)
    # Understat team names might differ slightly (e.g. "Man Utd" vs "Manchester United")
    # We might need a small map here too if they differ drastically.
    # For now, let's try fuzzy matching on player + team.
    
    matched_count = 0
    
    print(f"Mapping {len(fpl_players)} FPL players...")

    for _, fpl_p in fpl_players.iterrows():
        fpl_name = fpl_p['norm_name']
        fpl_web_name = fpl_p['web_name_norm']
        fpl_id = fpl_p['id']
        fpl_team = normalize_name(fpl_p['team_name'])
        
        match_id = None
        match_name = None
        score = 0
        
        # 1. Check Overrides (using full raw name)
        full_name_raw = f"{fpl_p['first_name']} {fpl_p['second_name']}"
        if full_name_raw in overrides:
            target_name = overrides[full_name_raw]
            # Find ID for this target name
            u_p = understat_players[understat_players['player_name'] == target_name]
            if not u_p.empty:
                match_id = u_p.iloc[0]['id']
                match_name = u_p.iloc[0]['player_name']
                score = 100
        
        # 2. Exact/Fuzzy Match
        if match_id is None:
            # Filter Understat players by Team if possible? 
            # Team names might not match exactly, so we default to searching global list if we can't match team.
            # But searching global list is risky for common names.
            # Let's try to find potential candidates.
            
            # Simple Exact Match first
            exact = understat_players[understat_players['norm_name'] == fpl_name]
            if not exact.empty:
                match_id = exact.iloc[0]['id']
                match_name = exact.iloc[0]['player_name']
                score = 100
            else:
                # Fuzzy Match
                # We prioritize matching last name or web name
                best_match = process.extractOne(fpl_name, understat_players['norm_name'].tolist(), scorer=fuzz.token_sort_ratio)
                if best_match and best_match[1] > 85:
                    idx = understat_players[understat_players['norm_name'] == best_match[0]].index[0]
                    match_id = understat_players.loc[idx, 'id']
                    match_name = understat_players.loc[idx, 'player_name']
                    score = best_match[1]
                
        # Append Result
        if match_id:
            mapping.append({
                'fpl_id': fpl_id,
                'understat_id': match_id,
                'fpl_name': full_name_raw,
                'understat_name': match_name,
                'score': score
            })
            matched_count += 1
        
    # Save
    mapping_df = pd.DataFrame(mapping)
    os.makedirs('data/processed', exist_ok=True)
    mapping_df.to_csv('data/processed/id_mapping.csv', index=False)
    
    print(f"Mapping complete. Mapped {matched_count}/{len(fpl_players)} players.")
    print(f"Saved to data/processed/id_mapping.csv")

if __name__ == "__main__":
    map_ids()
