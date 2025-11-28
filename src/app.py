from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)

def load_model_and_data():
    try:
        with open('models/fpl_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/model_features.pkl', 'rb') as f:
            features = pickle.load(f)
            
        df = pd.read_csv('data/processed/inference_data.csv')
        
        with open('data/processed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, features, df, metadata
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    model, features, df, metadata = load_model_and_data()
    
    if model is None:
        return jsonify({'error': 'Model or data not found'}), 500
    
    # Make predictions
    # Ensure features exist
    for feature in features:
        if feature not in df.columns:
            return jsonify({'error': f'Missing feature: {feature}'}), 500
            
    # Drop rows with missing values in features
    df = df.dropna(subset=features)

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

    predictions = model.predict(df[features])
    df['predicted_points'] = predictions
    
    # Sort by predicted points descending and take Top 5
    top_players = df.sort_values('predicted_points', ascending=False).head(5)
    
    # Select columns to display
    # Select columns to display
    display_cols = ['web_name', 'team_name', 'next_opponent_name', 'now_cost', 'selected_by_percent', 'predicted_points', 'code', 'team_code', 'opponent_team_code', 'element_type']
    result = top_players[display_cols].to_dict(orient='records')
    
    # Position Mapping
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    # Add image URLs and Position
    for player in result:
        player['photo_url'] = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(player['code'])}.png"
        player['team_logo_url'] = f"https://resources.premierleague.com/premierleague/badges/t{int(player['team_code'])}.png"
        player['opponent_logo_url'] = f"https://resources.premierleague.com/premierleague/badges/t{int(player['opponent_team_code'])}.png"
        player['position'] = pos_map.get(player['element_type'], 'UNK')

    response = {
        'gameweek_info': metadata,
        'predictions': result
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
