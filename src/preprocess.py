import pandas as pd
import json
import os
import numpy as np

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
    
    return players, teams, events, histories, fixtures_df

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

def preprocess_data(players, teams, events, histories, fixtures):
    current_gw_id, next_gw_id = get_gameweek_info(events)
    print(f"Current GW: {current_gw_id}, Next GW: {next_gw_id}")
    
    # --- Prepare Team Strength Map ---
    team_strength = teams[['id', 'name', 'short_name', 'strength', 'code']].set_index('id')
    
    # --- Prepare Training Data (Historical) ---
    # Merge player info to histories
    player_info = players[['id', 'web_name', 'team', 'element_type', 'now_cost', 'selected_by_percent', 'code', 'ict_index']]
    train_df = histories.merge(player_info, left_on='element', right_on='id', suffixes=('', '_player'))
    
    # Add opponent strength
    train_df = train_df.merge(team_strength[['strength']], left_on='opponent_team', right_index=True)
    train_df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
    
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
        'penalties_saved'
    ]
    
    # Calculate rolling averages (lagged by 1 to represent "form entering the match")
    # We shift by 1 first, so we don't use the current match's stats to predict the current match's points
    for metric in metrics:
        col_name = f'recent_{metric}'
        # Group by element, shift 1, then rolling mean
        train_df[f'prev_{metric}'] = train_df.groupby('element')[metric].shift(1)
        train_df[col_name] = train_df.groupby('element')[f'prev_{metric}'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        train_df[col_name] = train_df[col_name].fillna(0)
        
        # Drop the temporary prev column
        train_df.drop(columns=[f'prev_{metric}'], inplace=True)
    
    # Rename recent_total_points to recent_form for consistency with existing code
    train_df.rename(columns={'recent_total_points': 'recent_form'}, inplace=True)

    # Clean up
    train_df.rename(columns={'was_home': 'is_home'}, inplace=True)
    train_df['is_home'] = train_df['is_home'].astype(int)
    train_df['selected_by_percent'] = train_df['selected_by_percent'].astype(float)
    
    # --- Prepare Inference Data (Next GW) ---
    if next_gw_id is not None:
        # Get fixtures for next GW
        next_fixtures = fixtures[fixtures['event'] == next_gw_id]
        
        inference_rows = []
        
        for _, fixture in next_fixtures.iterrows():
            home_team = fixture['team_h']
            away_team = fixture['team_a']
            
            # Helper to build row
            def build_row(p, is_home, opponent_id):
                opp_strength = team_strength.loc[opponent_id, 'strength']
                
                # Get recent stats from history
                p_hist = histories[histories['element'] == p['id']]
                
                stats = {}
                if not p_hist.empty:
                    p_hist = p_hist.sort_values('round')
                    # Calculate rolling mean of last 5 actual matches
                    for metric in metrics:
                        val = p_hist[metric].tail(5).mean()
                        key = 'recent_form' if metric == 'total_points' else f'recent_{metric}'
                        stats[key] = val
                else:
                    for metric in metrics:
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
                    'ict_index': float(p['ict_index'])
                }
                # Add stats
                row.update(stats)
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
        players, teams, events, histories, fixtures = load_data()
        print("Data loaded.")
        
        train_df, inference_df, current_gw, next_gw = preprocess_data(players, teams, events, histories, fixtures)
        
        os.makedirs('data/processed', exist_ok=True)
        
        # Save Training Data
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

if __name__ == "__main__":
    main()
