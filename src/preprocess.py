"""
src/preprocess.py
Description: The core data processing module for the FPL pipeline.
It merges raw data from multiple sources (FPL Bootstrap, FPL Histories, FPL Fixtures, Understat Matches) to create:
1. `train_data.csv`: Historical data with features (rolling averages of form, xG, xGA, etc.) for model training.
2. `inference_data.csv`: A snapshot of players and their features for the upcoming gameweek to generate predictions.
It handles feature engineering, team strength mapping, and data cleaning.
"""
import pandas as pd
import json
import os
import numpy as np
import datetime
import re
from fuzzywuzzy import process
from pathlib import Path

def load_data():
    # Load FPL Bootstrap
    with open('data/raw/fpl_bootstrap.json', 'r', encoding='utf-8') as f:
        bootstrap = json.load(f)
    
    players = pd.DataFrame(bootstrap['elements'])
    teams = pd.DataFrame(bootstrap['teams'])
    events = pd.DataFrame(bootstrap['events'])
    
    # Load FPL Histories
    histories = pd.read_parquet('data/raw/fpl_histories.parquet')
    
    # Load Fixtures
    with open('data/raw/fpl_fixtures.json', 'r', encoding='utf-8') as f:
        fixtures = json.load(f)
    fixtures_df = pd.DataFrame(fixtures)
    
    # Load Understat Matches
    # Find latest understat matches file
    raw_dir = Path('data/raw')
    match_files = list(raw_dir.glob('understat_matches_*.json'))
    if match_files:
        # Sort by season (year)
        latest_match_file = sorted(match_files, key=lambda f: int(re.search(r'\d+', f.name).group()))[-1]
        with open(latest_match_file, 'r', encoding='utf-8') as f:
            understat_matches = json.load(f)
        understat_matches_df = pd.json_normalize(understat_matches)
    else:
        understat_matches_df = pd.DataFrame()
    
    # Load Understat Player Data (for player-level xG/xA)
    player_files = list(raw_dir.glob('understat_players_*.json'))
    if player_files:
        latest_player_file = sorted(player_files, key=lambda f: int(re.search(r'\d+', f.name).group()))[-1]
        with open(latest_player_file, 'r', encoding='utf-8') as f:
            understat_players = json.load(f)
        understat_players_df = pd.DataFrame(understat_players)
    else:
        understat_players_df = pd.DataFrame()
    
    # Load FPL-to-Understat ID Mapping
    id_mapping_path = Path('data/processed/id_mapping.csv')
    if id_mapping_path.exists():
        id_mapping_df = pd.read_csv(id_mapping_path)
    else:
        id_mapping_df = pd.DataFrame()
        
    return players, teams, events, histories, fixtures_df, understat_matches_df, understat_players_df, id_mapping_df


def get_gameweek_info(events):
    # Find current and next gameweek
    # 'is_current' = True for the active GW. 'is_next' = True for the upcoming one.
    # Sometimes no GW is current (between weeks), so we look for next.
    
    current_gw = events[events['is_current'] == True]
    next_gw = events[events['is_next'] == True]
    
    current_gw_id = current_gw['id'].iloc[0] if not current_gw.empty else None
    next_gw_id = next_gw['id'].iloc[0] if not next_gw.empty else None
    
    # If no current GW, assume we are pre-season or between weeks, use next - 1 or something?
    # For inference, we strictly need next_gw_id.
    
    if next_gw_id is None:
        # End of season or data issue?
        # Fallback: max ID + 1?
        pass
        
    return current_gw_id, next_gw_id

def map_understat_teams(fpl_teams, us_matches):
    """Map Understat team names to FPL team IDs using persistent JSON mapping."""
    if us_matches.empty:
        return {}
    
    TEAM_MAPPING_FILE = Path('data/config/known_team_mapping.json')
    
    # Load known mappings (Understat name → FPL name)
    known_mappings = {}
    if TEAM_MAPPING_FILE.exists():
        with open(TEAM_MAPPING_FILE, 'r', encoding='utf-8') as f:
            known_mappings = json.load(f)
    
    # Build FPL name → ID lookup
    fpl_name_to_id = fpl_teams.set_index('name')['id'].to_dict()
    
    # Get unique Understat team names (from home and away columns)
    us_teams = pd.concat([us_matches['h.title'], us_matches['a.title']]).unique()
    
    mapping = {}
    unmapped = []
    
    for us_name in us_teams:
        # Use known mapping or assume exact match
        fpl_name = known_mappings.get(us_name, us_name)
        if fpl_name in fpl_name_to_id:
            mapping[us_name] = fpl_name_to_id[fpl_name]
        else:
            unmapped.append(us_name)
    
    if unmapped:
        print(f"Warning: Could not map Understat teams: {unmapped}")
        print("Add mappings to data/config/known_team_mapping.json")
    
    return mapping

def preprocess_data(players, teams, events, histories, fixtures, understat_matches, understat_players=None, id_mapping=None):
    current_gw_id, next_gw_id = get_gameweek_info(events)
    print(f"Current GW: {current_gw_id}, Next GW: {next_gw_id}")
    
    # Handle optional parameters
    if understat_players is None:
        understat_players = pd.DataFrame()
    if id_mapping is None:
        id_mapping = pd.DataFrame()
    
    # --- Prepare Team Strength Map ---
    team_strength = teams[['id', 'name', 'short_name', 'strength', 'code']].set_index('id')
    
    # --- Process Understat Data ---
    # We want to attach xG and xGA to every FIXTURE to compute rolling means
    
    if not understat_matches.empty:
        # Map Teams
        team_map = map_understat_teams(teams, understat_matches)
        
        # Prepare FPL Fixtures for merging
        # We need clean dates
        fixtures['kickoff_time'] = pd.to_datetime(fixtures['kickoff_time']).dt.date
        understat_matches['datetime'] = pd.to_datetime(understat_matches['datetime']).dt.date
        
        # We need to inject understat stats into 'fixtures' dataframe
        # Create a lookup: (Date, HomeTeamID) -> {xG_h, xG_a}
        
        us_lookup = {}
        for _, row in understat_matches.iterrows():
            date = row['datetime']
            h_name = row['h.title']
            a_name = row['a.title']
            
            h_id = team_map.get(h_name)
            a_id = team_map.get(a_name)
            
            if h_id and a_id:
                # Store tuple (Date, HomeID, AwayID)
                # xG values are strings in JSON
                try:
                    # Handle possibility of None (null in JSON) or missing keys
                    raw_h = row.get('xG.h')
                    raw_a = row.get('xG.a')
                    
                    xg_h = float(raw_h) if raw_h is not None else 0.0
                    xg_a = float(raw_a) if raw_a is not None else 0.0
                    
                    # Store for Home Team
                    us_lookup[(date, h_id)] = {'team_xg': xg_h, 'team_xga': xg_a} 
                    # Store for Away Team (they are 'away' but their xG is xg_a)
                    us_lookup[(date, a_id)] = {'team_xg': xg_a, 'team_xga': xg_h}
                    
                except (ValueError, KeyError, TypeError):
                    continue

        # Now enrich 'histories' via 'fixture' link? 
        # Easier: Enrich 'histories' which has 'kickoff_time' and 'team'? 
        # Actually 'histories' doesn't have kickoff_time usually, it has 'fixture' ID.
        # Let's merge kickoff_time from fixtures to histories
        
        fixtures_subset = fixtures[['id', 'kickoff_time', 'team_h', 'team_a']]
        histories = histories.merge(fixtures_subset, left_on='fixture', right_on='id', suffixes=('', '_fix'))
        
        # Function to apply lookup
        def get_us_stats(row):
            d = row['kickoff_time']
            # histories usually has kickoff_time as string or datetime
            if isinstance(d, pd.Timestamp):
                 d = d.date()
                 
            # Note: Histories rows are PER PLAYER.
            # We need the player's team.
            # 'was_home' tells us if they were home or away.
            # But 'histories' doesn't explicitly store 'my_team_id' easily 
            # without looking up 'element' -> 'team' separately OR using 'fixture' teams.
            # However, preprocess joins player_info later.
            pass 
            
    # Ensure 'is_home' exists (histories usually has 'was_home')
    if 'was_home' in histories.columns:
        histories.rename(columns={'was_home': 'is_home'}, inplace=True)
    
    # --- Enrich Histories with Match Date & Understat ---
    # Histories needs match_date for Understat lookup
    fixtures_date_map = fixtures.set_index('id')['kickoff_time'].to_dict()
    # Ensure fixtures_date_map values are dates
    fixtures_date_map = {k: pd.to_datetime(v).date() for k, v in fixtures_date_map.items()}
    
    histories['match_date'] = histories['fixture'].map(fixtures_date_map)
    
    # Merge Understat into HISTORIES (so it's available for both Train and Inference lookups)
    if not understat_matches.empty:
        # Note: 'team' in histories might be missing?
        # Actually histories usually doesn't have 'team' column, it has 'element'.
        # We need to map element -> team.
        # But players can change teams? For now assume current team (limitation).
        # Or better: use 'fixture' and 'was_home' to determine team?
        # Histories has 'was_home'. Fixture has home_team and away_team.
        # This is more robust.
        
        fixture_teams = fixtures[['id', 'team_h', 'team_a']].set_index('id')
        
        def get_team_id(row):
            fix_id = row['fixture']
            if fix_id not in fixture_teams.index: return None
            if row['was_home']:
                return fixture_teams.loc[fix_id, 'team_h']
            else:
                return fixture_teams.loc[fix_id, 'team_a']
        
        # This apply is slow. Better to merge.
        # Use suffixes to avoid collision if histories already has team_h (e.g. score)
        histories = histories.merge(fixtures[['id', 'team_h', 'team_a']], left_on='fixture', right_on='id', how='left', suffixes=('', '_fix'))
        
        # Determine team ID based on was_home
        # If histories already has team_h (score?), we want the ID from fixtures (team_h_fix or team_h)
        col_h = 'team_h_fix' if 'team_h_fix' in histories.columns else 'team_h'
        col_a = 'team_a_fix' if 'team_a_fix' in histories.columns else 'team_a'
        
        histories['team'] = np.where(histories['is_home'], histories[col_h], histories[col_a])
        
        # Apply lookup
        def lookup_xg(row):
            key = (row['match_date'], row['team'])
            return us_lookup.get(key, {'team_xg': np.nan, 'team_xga': np.nan})
            
        us_cols = histories.apply(lookup_xg, axis=1, result_type='expand')
        histories = pd.concat([histories, us_cols], axis=1)
        
        histories['team_xg'] = histories['team_xg'].fillna(0)
        histories['team_xga'] = histories['team_xga'].fillna(0)
        
        # Cleanup temp columns
        histories.drop(columns=['team_h', 'team_a', 'id'], inplace=True, errors='ignore') # id from fixture merge
    else:
        histories['team_xg'] = 0.0
        histories['team_xga'] = 0.0

    # Clean numeric columns in histories data (global fix)
    metrics_to_clean = [
        'total_points', 'clean_sheets', 'saves', 'goals_conceded', 'goals_scored', 
        'assists', 'threat', 'influence', 'creativity', 'penalties_saved', 
        'team_xg', 'team_xga', 'expected_goals', 'expected_assists'
    ]
    for col in metrics_to_clean:
        if col in histories.columns:
            histories[col] = pd.to_numeric(histories[col], errors='coerce').fillna(0)

    # --- Prepare Training Data (Historical) ---
    # Merge player info to histories
    player_info = players[['id', 'web_name', 'team', 'element_type', 'now_cost', 'selected_by_percent', 'code', 'ict_index']]
    # Note: 'team' from player_info is CURRENT team. histories['team'] is HISTORICAL team.
    # train_df should probably use histories['team'] for accuracy, but player_info has other metadata.
    # efficient merge
    train_df = histories.merge(player_info, left_on='element', right_on='id', suffixes=('', '_player'))

    
    # Add opponent strength
    train_df = train_df.merge(team_strength[['strength']], left_on='opponent_team', right_index=True)
    train_df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
    
    # --- Merge Understat Player-Level Data ---
    # Uses ID mapping from id_map.py to link FPL players to Understat stats
    if not understat_players.empty and not id_mapping.empty:
        print("Merging Understat player-level data...")
        
        # Prepare Understat player stats: calculate per-90 metrics
        us_cols = ['id', 'xG', 'xA', 'npxG', 'time']
        us_available = [c for c in us_cols if c in understat_players.columns]
        us_subset = understat_players[us_available].copy()
        
        # Convert to numeric
        for col in ['xG', 'xA', 'npxG', 'time']:
            if col in us_subset.columns:
                us_subset[col] = pd.to_numeric(us_subset[col], errors='coerce').fillna(0)
        
        # Calculate per-90 metrics (avoid division by zero)
        if 'time' in us_subset.columns:
            us_subset['minutes_played'] = us_subset['time'].clip(lower=1)  # At least 1 minute
            if 'npxG' in us_subset.columns:
                us_subset['us_npxG_per90'] = (us_subset['npxG'] / us_subset['minutes_played']) * 90
            if 'xA' in us_subset.columns:
                us_subset['us_xA_per90'] = (us_subset['xA'] / us_subset['minutes_played']) * 90
        
        # Merge via ID mapping: fpl_id -> understat_id
        id_map_subset = id_mapping[['fpl_id', 'understat_id']].drop_duplicates()
        us_subset = us_subset.rename(columns={'id': 'understat_id'})
        us_subset['understat_id'] = pd.to_numeric(us_subset['understat_id'], errors='coerce')
        
        # Join: id_mapping -> understat_players
        player_us_stats = id_map_subset.merge(us_subset, on='understat_id', how='left')
        
        # Join to train_df via fpl_id (element)
        train_df = train_df.merge(
            player_us_stats[['fpl_id', 'us_npxG_per90', 'us_xA_per90']].drop_duplicates(),
            left_on='element', right_on='fpl_id', how='left'
        )
        train_df.drop(columns=['fpl_id'], inplace=True, errors='ignore')
        
        # Fill NaN for players without Understat mapping
        train_df['us_npxG_per90'] = train_df['us_npxG_per90'].fillna(0)
        train_df['us_xA_per90'] = train_df['us_xA_per90'].fillna(0)
        
        print(f"Understat player data merged. Players with stats: {(train_df['us_npxG_per90'] > 0).sum()}")
    else:
        train_df['us_npxG_per90'] = 0.0
        train_df['us_xA_per90'] = 0.0
        print("No Understat player data available. Using defaults.")
    
    # --- Feature Engineering: Rolling Averages ---
    # Ensure sorted by element and round
    train_df = train_df.sort_values(['element', 'round'])
    
    # Metrics to calculate rolling averages for
    metrics = [
        'total_points',
        'clean_sheets', 
        'saves', 
        'goals_conceded', 
        'goals_scored', 
        'assists', 
        'threat', 
        'influence', 
        'creativity', 
        'penalties_saved',
        'team_xg',
        'team_xga',
        'expected_goals',
        'expected_assists',
        'us_npxG_per90',
        'us_xA_per90'
    ]
    
    # Calculate rolling averages (lagged by 1 to represent "form entering the match")
    # We shift by 1 first, so we don't use the current match's stats to predict the current match's points
    
    # Ensure all metrics are numeric first
    print(f"DEBUG: train_df columns: {train_df.columns.tolist()}")
    if len(train_df.columns) != len(set(train_df.columns)):
        print("DEBUG: Duplicate columns detected!")
        print(train_df.columns[train_df.columns.duplicated()])
        # Deduplicate by keeping first
        train_df = train_df.loc[:, ~train_df.columns.duplicated()]

    for col in metrics:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

    for metric in metrics:
        if metric not in train_df.columns:
            train_df[metric] = 0.0
            
        col_name = f'recent_{metric}'
        try:
            # Group by element, shift 1, then rolling mean
            train_df[f'prev_{metric}'] = train_df.groupby('element')[metric].shift(1)
            train_df[col_name] = train_df.groupby('element')[f'prev_{metric}'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            train_df[col_name] = train_df[col_name].fillna(0)
        except Exception as e:
            print(f"ERROR calculating rolling mean for {metric}: {e}")
            train_df[col_name] = 0.0
        
        # Drop the temporary prev column
        train_df.drop(columns=[f'prev_{metric}'], inplace=True)
    
    # Rename recent_total_points to recent_form for consistency with existing code
    train_df.rename(columns={'recent_total_points': 'recent_form'}, inplace=True)

    # --- Binary Target Columns for Component-Based Prediction ---
    # These are used by the component classifiers to predict individual outcomes
    train_df['target_goal'] = (train_df['goals_scored'] > 0).astype(int)
    train_df['target_assist'] = (train_df['assists'] > 0).astype(int)
    train_df['target_clean_sheet'] = (train_df['clean_sheets'] > 0).astype(int)

    # Clean up
    train_df.rename(columns={'was_home': 'is_home'}, inplace=True)
    train_df['is_home'] = train_df['is_home'].astype(int)
    train_df['selected_by_percent'] = train_df['selected_by_percent'].astype(float)
    
    # --- Prepare Inference Data (Next GW) ---
    if next_gw_id is not None:
        # Get fixtures for next GW
        next_fixtures = fixtures[fixtures['event'] == next_gw_id]
        
        # Prepare Understat player stats lookup for inference
        us_player_lookup = {}
        if not understat_players.empty and not id_mapping.empty:
            us_cols = ['id', 'npxG', 'xA', 'time']
            us_available = [c for c in us_cols if c in understat_players.columns]
            us_data = understat_players[us_available].copy()
            for col in ['npxG', 'xA', 'time']:
                if col in us_data.columns:
                    us_data[col] = pd.to_numeric(us_data[col], errors='coerce').fillna(0)
            if 'time' in us_data.columns:
                us_data['minutes_played'] = us_data['time'].clip(lower=1)
                if 'npxG' in us_data.columns:
                    us_data['us_npxG_per90'] = (us_data['npxG'] / us_data['minutes_played']) * 90
                if 'xA' in us_data.columns:
                    us_data['us_xA_per90'] = (us_data['xA'] / us_data['minutes_played']) * 90
            us_data['understat_id'] = pd.to_numeric(us_data['id'], errors='coerce')
            id_map_dict = id_mapping.set_index('fpl_id')['understat_id'].to_dict()
            us_id_to_stats = us_data.set_index('understat_id')[['us_npxG_per90', 'us_xA_per90']].to_dict('index')
            for fpl_id, us_id in id_map_dict.items():
                if us_id in us_id_to_stats:
                    us_player_lookup[fpl_id] = us_id_to_stats[us_id]
        
        inference_rows = []
        
        for _, fixture in next_fixtures.iterrows():
            home_team = fixture['team_h']
            away_team = fixture['team_a']
            
            # Helper to build row
            def build_row(p, is_home, opponent_id):
                opp_strength = team_strength.loc[opponent_id, 'strength']
                
                # Get recent stats from history
                p_hist = histories[histories['element'] == p['id']]
                
                # Metrics that come from histories (not Understat player data)
                history_metrics = [m for m in metrics if m not in ['us_npxG_per90', 'us_xA_per90']]
                
                stats = {}
                if not p_hist.empty:
                    p_hist = p_hist.sort_values('round')
                    # Calculate rolling mean of last 5 actual matches
                    for metric in history_metrics:
                        if metric in p_hist.columns:
                            val = p_hist[metric].tail(5).mean()
                        else:
                            val = 0.0
                        key = 'recent_form' if metric == 'total_points' else f'recent_{metric}'
                        stats[key] = val
                else:
                    for metric in history_metrics:
                        key = 'recent_form' if metric == 'total_points' else f'recent_{metric}'
                        stats[key] = 0.0

                row = {
                    'element': p['id'],
                    'web_name': p['web_name'],
                    'team': p['team'],
                    'team_name': team_strength.loc[p['team'], 'name'],
                    'next_opponent_id': opponent_id,
                    'next_opponent_name': team_strength.loc[opponent_id, 'name'],
                    'is_home': 1 if is_home else 0,
                    'opponent_strength': opp_strength,
                    'now_cost': p['now_cost'],
                    'selected_by_percent': float(p['selected_by_percent']),
                    'element_type': p['element_type'],
                    'code': p['code'],
                    'team_code': team_strength.loc[p['team'], 'code'],
                    'opponent_team_code': team_strength.loc[opponent_id, 'code'],
                    'ict_index': float(p['ict_index']),
                    'status': p['status'],
                    'chance_of_playing_next_round': p['chance_of_playing_next_round'],
                    'news': p['news']
                }
                # Add rolling stats
                row.update(stats)
                
                # Add Understat player-level stats (season aggregate per-90)
                us_stats = us_player_lookup.get(p['id'], {'us_npxG_per90': 0.0, 'us_xA_per90': 0.0})
                row['recent_us_npxG_per90'] = us_stats.get('us_npxG_per90', 0.0)
                row['recent_us_xA_per90'] = us_stats.get('us_xA_per90', 0.0)
                
                return row

            # Home Players
            home_players = players[players['team'] == home_team]
            for _, p in home_players.iterrows():
                inference_rows.append(build_row(p, True, away_team))
                
            # Away Players
            away_players = players[players['team'] == away_team]
            for _, p in away_players.iterrows():
                inference_rows.append(build_row(p, False, home_team))
                
        inference_df = pd.DataFrame(inference_rows)
    else:
        inference_df = pd.DataFrame()
        print("Warning: No next gameweek found.")

    return train_df, inference_df, current_gw_id, next_gw_id

def main():
    try:
        players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping = load_data()
        print("Data loaded.")
        
        train_df, inference_df, current_gw, next_gw = preprocess_data(
            players, teams, events, histories, fixtures, understat_matches, 
            understat_players, id_mapping
        )
        
        os.makedirs('data/processed', exist_ok=True)
        
        # Save Training Data
        print(f"Train DF Columns: {train_df.columns.tolist()}")
        train_df.to_csv('data/processed/train_data.csv', index=False)
        print(f"Saved train_data.csv ({len(train_df)} rows)")
        
        # Save Inference Data
        inference_df.to_csv('data/processed/inference_data.csv', index=False)
        print(f"Saved inference_data.csv ({len(inference_df)} rows)")
        
        # Save Metadata
        metadata = {
            'current_gameweek': int(current_gw) if current_gw else None,
            'next_gameweek': int(next_gw) if next_gw else None
        }
        with open('data/processed/metadata.json', 'w') as f:
            json.dump(metadata, f)
        print("Saved metadata.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    import re # Needed for regex in load_data
    main()
