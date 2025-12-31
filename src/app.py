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
import src.data_fetch as data_fetch
import time

app = Flask(__name__)

# --- Cache Setup ---
LIVE_DATA_CACHE = {
    'last_updated': 0,
    'data': None,
    'gameweek': None,
    'window_start': None,
    'window_end': None
}
CACHE_DURATION = 300  # 5 minutes
WINDOW_BUFFER = 2.5 * 3600 # 2.5 hours in seconds

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
    display_cols = ['element', 'web_name', 'team_name', 'next_opponent_name', 'now_cost', 'selected_by_percent', 'predicted_points', 'code', 'team_code', 'opponent_team_code', 'element_type', 'recent_expected_goals', 'recent_expected_assists', 'recent_team_xga']
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

@app.route('/api/live')
def get_live_scores():
    global LIVE_DATA_CACHE
    
    current_time = time.time()
    
    # 0. Check Window optimizations (if we have window data)
    # If we are strictly OUTSIDE the window (and have data), we can extend cache duration
    # or just return early.
    
    if LIVE_DATA_CACHE['gameweek']:
        # If before start, return empty (save API call)
        if LIVE_DATA_CACHE['window_start'] and current_time < LIVE_DATA_CACHE['window_start']:
             print("Optimization: Before Gameweek Start. Returning empty.")
             return jsonify({})
             
        # If after end, Cache Duration can be longer (e.g. 1 hour)
        if LIVE_DATA_CACHE['window_end'] and current_time > LIVE_DATA_CACHE['window_end']:
             # If we have data and it's fresh-ish (1 hour), return it
             if LIVE_DATA_CACHE['data'] and (current_time - LIVE_DATA_CACHE['last_updated'] < 3600):
                 print("Optimization: Gameweek Over (Cached).")
                 return jsonify(LIVE_DATA_CACHE['data'])

    # Standard Cache Check
    if LIVE_DATA_CACHE['data'] and (current_time - LIVE_DATA_CACHE['last_updated'] < CACHE_DURATION):
        print("Returning live data from cache")
        return jsonify(LIVE_DATA_CACHE['data'])
        
    print("Fetching fresh live data from FPL...")
    try:
        bootstrap = data_fetch.fetch_fpl_bootstrap()
        current_event = next((e for e in bootstrap['events'] if e['is_current']), None)
        
        if not current_event:
            return jsonify({})
            
        gw_id = current_event['id']
        
        # Calculate Window (First fetch or update)
        fixtures = data_fetch.fetch_fpl_fixtures()
        gw_fixtures = [f for f in fixtures if f['event'] == gw_id]
        
        if gw_fixtures:
            # Parse kickoffs
            kickoffs = [datetime.fromisoformat(f['kickoff_time'].replace('Z', '+00:00')).timestamp() for f in gw_fixtures]
            if kickoffs:
                min_ko = min(kickoffs)
                max_ko = max(kickoffs)
                
                window_start = min_ko
                window_end = max_ko + WINDOW_BUFFER
                
                # Check Optimization AGAIN after fetching metadata
                # (Handle case where we didn't have cache yet)
                if current_time < window_start:
                    print(f"Optimization: GW{gw_id} starts at {datetime.fromtimestamp(window_start)}. Now: {datetime.fromtimestamp(current_time)}. Skipping live fetch.")
                    # Update cache metadata only
                    LIVE_DATA_CACHE['gameweek'] = gw_id
                    LIVE_DATA_CACHE['window_start'] = window_start
                    LIVE_DATA_CACHE['window_end'] = window_end
                    return jsonify({})
        
        # Proceed to fetch live data...
        live_json = data_fetch.get_gameweek_live_data(gw_id)
        
        if not live_json or 'elements' not in live_json:
            return jsonify({})
            
        live_map = {}
        for element in live_json['elements']:
            stats = element['stats']
            live_map[element['id']] = {
                'points': stats['total_points'],
                'minutes': stats['minutes'],
                'finished': False 
            }
        
        team_fixture_status = {} 
        
        for f in gw_fixtures:
            is_finished = f['finished']
            team_fixture_status[f['team_h']] = is_finished
            team_fixture_status[f['team_a']] = is_finished
            
        id_to_team = {e['id']: e['team'] for e in bootstrap['elements']}
        
        for pid, data in live_map.items():
            tid = id_to_team.get(pid)
            if tid:
                 data['finished'] = team_fixture_status.get(tid, False)
                 
        LIVE_DATA_CACHE = {
            'last_updated': current_time,
            'data': live_map,
            'gameweek': gw_id,
            'window_start': locals().get('window_start'), # Safe get
            'window_end': locals().get('window_end')
        }
        
        return jsonify(live_map)
        
    except Exception as e:
        print(f"Error in /api/live: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
