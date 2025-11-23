import pandas as pd
import json
import os

def load_data():
    # Load FPL Bootstrap
    with open('data/raw/fpl_bootstrap.json', 'r', encoding='utf-8') as f:
        bootstrap = json.load(f)
    
    players = pd.DataFrame(bootstrap['elements'])
    teams = pd.DataFrame(bootstrap['teams'])
    
    # Load FPL Histories
    histories = pd.read_parquet('data/raw/fpl_histories.parquet')
    
    # Load Understat
    try:
        understat = pd.read_csv('data/raw/understat_players_2023.csv')
    except Exception:
        print("Warning: Could not load Understat data. Proceeding without it.")
        understat = None
    
    return players, teams, histories, understat

def preprocess_data(players, teams, histories, understat):
    # Basic cleaning
    players = players[['id', 'web_name', 'team', 'element_type', 'now_cost', 'selected_by_percent', 'total_points']]
    players['selected_by_percent'] = players['selected_by_percent'].astype(float)
    
    # Merge team names
    teams = teams[['id', 'name', 'short_name']]
    players = players.merge(teams, left_on='team', right_on='id', suffixes=('', '_team'))
    
    # Calculate recent form from histories (last 5 games)
    # This is a simplified version. In a real scenario, we'd need to handle gameweeks properly.
    # For now, we'll just take the average of the last 5 entries per player in histories
    
    # Ensure histories is sorted by round
    if 'round' in histories.columns:
        histories = histories.sort_values('round')
        
    recent_form = histories.groupby('element').tail(5).groupby('element')['total_points'].mean().reset_index()
    recent_form.rename(columns={'total_points': 'recent_form_points'}, inplace=True)
    
    players = players.merge(recent_form, left_on='id', right_on='element', how='left')
    
    # Merge Understat data if available
    if understat is not None:
        # Placeholder for merge logic if we had a good key
        pass
    
    return players

def main():
    try:
        players, teams, histories, understat = load_data()
        print("Data loaded successfully.")
        
        processed_df = preprocess_data(players, teams, histories, understat)
        print(f"Processed data shape: {processed_df.shape}")
        
        # Filter for underdogs
        underdogs = processed_df[processed_df['selected_by_percent'] < 10.0]
        print(f"Found {len(underdogs)} underdogs.")
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        processed_df.to_csv('data/processed/fpl_data.csv', index=False)
        print("Saved processed data to data/processed/fpl_data.csv")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()
