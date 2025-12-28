"""
src/app.py
Description: The main Flask application entry point for the FPL Predictor website.
It defines the web routes (/, /api/predictions, /api/history, /api/stats) and handles the application logic.
Key features include:
- Serving the main HTML template.
- Providing a JSON API for predictions, defaulting to historical data for consistency but falling back to live inference.
- Logging and serving simple usage statistics (visit counts).
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import pickle
import json
import os
import sqlite3
from datetime import datetime
import src.inference as inference

app = Flask(__name__)

# --- Usage Statistics Setup ---
DB_PATH = 'data/stats.db'

def init_db():
    """Initialize the stats database if it doesn't exist."""
    print("Initializing database...") # Debug print
    try:
        if not os.path.exists('data'):
             os.makedirs('data')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                user_agent TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def log_visit(endpoint):
    """Log a visit to a specific endpoint."""
    try:
        # Initialize strictly if needed, but ideally we call it on startup
        # For simplicity in this script, checking exists is enough or robust error handling
        if not os.path.exists(DB_PATH):
            init_db()

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        user_agent = request.headers.get('User-Agent')
        c.execute('INSERT INTO visits (timestamp, endpoint, user_agent) VALUES (?, ?, ?)',
                  (timestamp, endpoint, user_agent))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log visit: {e}")

# Initialize DB on import/startup
init_db()
# -----------------------------

def load_data():
    try:
        if not os.path.exists('data/processed/inference_data.csv') or not os.path.exists('data/processed/metadata.json'):
             return None, None

        df = pd.read_csv('data/processed/inference_data.csv')
        
        with open('data/processed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return df, metadata
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None



@app.route('/')
def index():
    log_visit('index')
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    log_visit('predictions')

    # Load data only initially (Fast)
    df, metadata = load_data()
    
    if df is None:
        return jsonify({'error': 'Data not found'}), 500

    # Try to load from history first (Consistency with Pipeline)
    try:
        current_gw = metadata.get('next_gameweek')
        history_path = 'data/history/predictions_log.json'
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Find entry for current gameweek
            gw_entry = next((item for item in history if item['gameweek'] == current_gw), None)
            
            if gw_entry:
                print(f"Loading predictions from history for GW {current_gw}")
                # Get player IDs from history
                hist_ids = [p['player_id'] for p in gw_entry['picks']]
                
                # Filter DataFrame for these players
                final_df = df[df['element'].isin(hist_ids)].copy()
                
                # Ensure we strictly follow the history order/content if possible, 
                # or just return these players. 
                # We need to map predicted_points from history if we want to be exact matches
                # but using the re-calculated ones from model is fine too as they should be identical.
                # Let's map strict points from history to be 100% sure.
                id_to_points = {p['player_id']: p['predicted_points'] for p in gw_entry['picks']}
                final_df['predicted_points'] = final_df['element'].map(id_to_points)
                
                # Prepare result immediately
                return format_predictions_response(final_df, metadata)

    except Exception as e:
        print(f"Warning: Failed to load from history: {e}")
        # Fallthrough to calculation
    
    # Load models only if calculation is needed (Slow)
    print("History not found or failed, loading models for inference...")
    models = inference.load_models()

    if not models:
        return jsonify({'error': 'Models not found'}), 500

    # Predict
    df = inference.predict_points(df, models)
    
    # Select Best Team
    # app.py previously had manual "Advanced Underdog Logic" hardcoded here.
    # Now that logic is in inference.select_best_team via src.config
    final_picks_df = inference.select_best_team(df)
    
    if final_picks_df.empty:
         return jsonify({'error': 'No valid predictions found'}), 500

    return format_predictions_response(final_picks_df, metadata)
        
    # Convert back to DataFrame for easier handling
    result_df = pd.DataFrame(final_picks)
    
    return format_predictions_response(result_df, metadata)

def format_predictions_response(result_df, metadata):
    # Sort by predicted points for display
    result_df = result_df.sort_values('predicted_points', ascending=False)
    
    # Select columns to display
    # Select columns to display
    display_cols = ['web_name', 'team_name', 'next_opponent_name', 'now_cost', 'selected_by_percent', 'predicted_points', 'code', 'team_code', 'opponent_team_code', 'element_type', 'recent_expected_goals', 'recent_expected_assists', 'recent_team_xga']
    result = result_df[display_cols].to_dict(orient='records')
    
    # Position Mapping
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    # Add image URLs and Position
    for player in result:
        player['photo_url'] = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(player['code'])}.png"
        player['team_logo_url'] = f"https://resources.premierleague.com/premierleague/badges/t{int(player['team_code'])}.png"
        player['opponent_logo_url'] = f"https://resources.premierleague.com/premierleague/badges/t{int(player['opponent_team_code'])}.png"
        player['position'] = pos_map.get(player['element_type'], 'UNK')
        player['profile_url'] = f"https://www.premierleague.com/en/players/{int(player['code'])}/{player['web_name']}/overview"

    response = {
        'gameweek_info': metadata,
        'predictions': result
    }
    
    return jsonify(response)
    


@app.route('/api/history')
def get_history():
    log_visit('history')
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

@app.route('/api/stats')
def get_stats():
    try:
        if not os.path.exists(DB_PATH):
             return jsonify({'error': 'No stats database found'}), 404
             
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total visits per endpoint
        c.execute('SELECT endpoint, COUNT(*) FROM visits GROUP BY endpoint')
        total_counts = dict(c.fetchall())
        
        # Visits today
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute('SELECT endpoint, COUNT(*) FROM visits WHERE timestamp LIKE ? GROUP BY endpoint', (f'{today}%',))
        today_counts = dict(c.fetchall())
        
        conn.close()
        
        return jsonify({
            'total_visits': total_counts,
            'todays_visits': today_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
