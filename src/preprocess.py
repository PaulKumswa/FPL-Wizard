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
    # We want 'strength_overall_home' and 'strength_overall_away' or just 'strength'
    # Let's use 'strength' combined.
    # teams columns: id, name, strength, strength_overall_home, strength_overall_away, etc.
    team_strength = teams[['id', 'name', 'short_name', 'strength', 'code']].set_index('id')
    
    # --- Prepare Training Data (Historical) ---
    # histories has: element, round, opponent_team, was_home, total_points, etc.
    
    # Merge player info to histories
    # players cols: id, web_name, team, element_type, now_cost
    player_info = players[['id', 'web_name', 'team', 'element_type', 'now_cost', 'selected_by_percent', 'code', 'ict_index']]
    train_df = histories.merge(player_info, left_on='element', right_on='id', suffixes=('', '_player'))
    
    # Add opponent strength
    # opponent_team is the ID of the opponent
    train_df = train_df.merge(team_strength[['strength']], left_on='opponent_team', right_index=True)
    train_df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
    
    # Calculate Form (Rolling average of points)
    # Ensure sorted by element and round
    train_df = train_df.sort_values(['element', 'round'])
    
    # We want "form entering the match". So shift the points and calculate rolling mean.
    # Group by element, shift 1 (to exclude current match), then rolling mean.
    train_df['prev_points'] = train_df.groupby('element')['total_points'].shift(1)
    train_df['recent_form'] = train_df.groupby('element')['prev_points'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # Fill NaN form (first matches) with 0 or global average? 0 is safer for now.
    train_df['recent_form'] = train_df['recent_form'].fillna(0)
    
    # Features for training
    # Target: total_points
    # Features: now_cost (proxy for quality), selected_by_percent, recent_form, opponent_strength, was_home
    
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
            
            # Get players for home team
            home_players = players[players['team'] == home_team]
            for _, p in home_players.iterrows():
                # Opponent is away_team
                opp_strength = team_strength.loc[away_team, 'strength']
                
                # Get recent form (from last available history)
                # We can take the last 'recent_form' from train_df for this player
                # OR re-calculate including the very last match.
                # Let's take the last calculated form from train_df + update with last match?
                # Simpler: Just take the mean of last 5 matches in histories.
                p_hist = histories[histories['element'] == p['id']]
                if not p_hist.empty:
                    p_hist = p_hist.sort_values('round')
                    form = p_hist['total_points'].tail(5).mean()
                else:
                    form = 0
                
                inference_rows.append({
                    'element': p['id'],
                    'web_name': p['web_name'],
                    'team': home_team,
                    'team_name': team_strength.loc[home_team, 'name'],
                    'next_opponent_id': away_team,\
                    'next_opponent_name': team_strength.loc[away_team, 'name'],
                    'is_home': 1,
                    'opponent_strength': opp_strength,
                    'recent_form': form,
                    'now_cost': p['now_cost'],
                    'selected_by_percent': float(p['selected_by_percent']),
                    'element_type': p['element_type'],
                    'code': p['code'],
                    'team_code': team_strength.loc[home_team, 'code'],
                    'opponent_team_code': team_strength.loc[away_team, 'code'],
                    'ict_index': float(p['ict_index'])
                })
                
            # Get players for away team
            away_players = players[players['team'] == away_team]
            for _, p in away_players.iterrows():
                # Opponent is home_team
                opp_strength = team_strength.loc[home_team, 'strength']
                
                p_hist = histories[histories['element'] == p['id']]
                if not p_hist.empty:
                    p_hist = p_hist.sort_values('round')
                    form = p_hist['total_points'].tail(5).mean()
                else:
                    form = 0
                
                inference_rows.append({
                    'element': p['id'],
                    'web_name': p['web_name'],
                    'team': away_team,
                    'team_name': team_strength.loc[away_team, 'name'],
                    'next_opponent_id': home_team,
                    'next_opponent_name': team_strength.loc[home_team, 'name'],
                    'is_home': 0,
                    'opponent_strength': opp_strength,
                    'recent_form': form,
                    'now_cost': p['now_cost'],
                    'selected_by_percent': float(p['selected_by_percent']),
                    'element_type': p['element_type'],
                    'code': p['code'],
                    'team_code': team_strength.loc[away_team, 'code'],
                    'opponent_team_code': team_strength.loc[home_team, 'code'],
                    'ict_index': float(p['ict_index'])
                })
                
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
