from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)

def load_model_and_data():
    try:
        models = {}
        features_map = {}
        
        # Load models for each position
        positions = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        for name, _ in positions.items():
            model_path = f'models/fpl_model_{name}.pkl'
            feature_path = f'models/model_features_{name}.pkl'
            
            if not os.path.exists(model_path) or not os.path.exists(feature_path):
                print(f"Warning: Model or features for {name} not found.")
                continue
                
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
            
            with open(feature_path, 'rb') as f:
                features_map[name] = pickle.load(f)
            
        df = pd.read_csv('data/processed/inference_data.csv')
        
        with open('data/processed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return models, features_map, df, metadata
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    models, features_map, df, metadata = load_model_and_data()
    
    if not models:
        return jsonify({'error': 'Models or data not found'}), 500
    
    # Advanced Underdog Logic
    # 1. Ownership < 10% (Deep Differential)
    # 2. Price < £8.0m (Budget Gem)
    # 3. Form > 2.0 OR ICT Index > 3.0 (Active/Good Underlying Stats)
    
    # Ensure columns are numeric
    df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'])
    df['now_cost'] = pd.to_numeric(df['now_cost'])
    df['recent_form'] = pd.to_numeric(df['recent_form'])
    df['ict_index'] = pd.to_numeric(df['ict_index'])

    df = df[
        (df['selected_by_percent'] < 10) & 
        (df['now_cost'] < 80) & 
        ((df['recent_form'] > 2.0) | (df['ict_index'] > 3.0))
    ]
    
    if df.empty:
        # Fallback if too strict: just < 10% ownership
        df = pd.read_csv('data/processed/inference_data.csv')
        df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'])
        df = df[df['selected_by_percent'] < 10]

    # Make predictions per position
    df['predicted_points'] = 0.0
    
    # Position Mapping: 1=GKP, 2=DEF, 3=MID, 4=FWD
    pos_map_rev = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    for pos_id, pos_name in pos_map_rev.items():
        if pos_name not in models:
            continue
            
        model = models[pos_name]
        features = features_map[pos_name]
        
        # Filter for this position
        pos_mask = df['element_type'] == pos_id
        if not pos_mask.any():
            continue
            
        # Ensure features exist
        # If a feature is missing (e.g. recent_saves), fill with 0
        X = df.loc[pos_mask, features].copy()
        for f in features:
            if f not in X.columns:
                X[f] = 0
                
        preds = model.predict(X)
        df.loc[pos_mask, 'predicted_points'] = preds

    # Selection Logic: Top 1 per position + 1 Wildcard
    final_picks = []
    
    # Sort by predicted points descending
    df_sorted = df.sort_values('predicted_points', ascending=False)
    
    # Pick Top 1 for each position
    for pos_id in [1, 2, 3, 4]:
        pos_candidates = df_sorted[df_sorted['element_type'] == pos_id]
        if not pos_candidates.empty:
            pick = pos_candidates.iloc[0]
            final_picks.append(pick)
            # Remove from pool to avoid duplicates (though unlikely with 1 per pos)
            df_sorted = df_sorted[df_sorted['element'] != pick['element']]
            
    # Pick 1 Wildcard (highest remaining)
    if len(final_picks) < 5 and not df_sorted.empty:
        wildcard = df_sorted.iloc[0]
        final_picks.append(wildcard)
        
    # Convert back to DataFrame for easier handling
    result_df = pd.DataFrame(final_picks)
    
    # Sort by predicted points for display
    result_df = result_df.sort_values('predicted_points', ascending=False)
    
    # Select columns to display
    display_cols = ['web_name', 'team_name', 'next_opponent_name', 'now_cost', 'selected_by_percent', 'predicted_points', 'code', 'team_code', 'opponent_team_code', 'element_type']
    result = result_df[display_cols].to_dict(orient='records')
    
    # Position Mapping
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    # Add image URLs and Position
    for player in result:
        player['photo_url'] = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(player['code'])}.png"
        player['team_logo_url'] = f"https://resources.premierleague.com/premierleague/badges/t{int(player['team_code'])}.png"
        player['opponent_logo_url'] = f"https://resources.premierleague.com/premierleague/badges/t{int(player['opponent_team_code'])}.png"
        player['position'] = pos_map.get(player['element_type'], 'UNK')
        player['profile_url'] = f"https://www.premierleague.com/players/{int(player['code'])}/{player['web_name']}/overview"

    response = {
        'gameweek_info': metadata,
        'predictions': result
    }
    
    return jsonify(response)

@app.route('/api/history')
def get_history():
    try:
        with open('data/history/predictions_log.json', 'r') as f:
            history = json.load(f)
        # Sort by gameweek descending
        history.sort(key=lambda x: x['gameweek'], reverse=True)
        return jsonify(history)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
