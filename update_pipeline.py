import subprocess
import sys
import time
import os
import webbrowser
from threading import Timer
import argparse
import json
import pandas as pd
import pickle
import src.history as history

def run_command(command, description):
    print(f"--- {description} ---")
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: {description} failed with return code {process.returncode}")
            sys.exit(1)
            
        print(f"--- {description} Completed ---\n")
        
    except Exception as e:
        print(f"Error executing {description}: {e}")
        sys.exit(1)

def open_browser():
    print("Opening web browser...")
    webbrowser.open('http://127.0.0.1:5000')

def main():
    parser = argparse.ArgumentParser(description='FPL Update Pipeline')
    parser.add_argument('--no-serve', action='store_true', help='Run pipeline only, do not launch the app (for CI/CD)')
    parser.add_argument('--quick', action='store_true', help='Skip data fetching and training, just launch the app')
    args = parser.parse_args()

    print("Starting FPL Pipeline...\n")
    
    if not args.quick:
        # 1. Fetch Data
        # Fetch Bootstrap
        run_command(
            f"{sys.executable} -m src.data_fetch --resource fpl_bootstrap --out data/raw/fpl_bootstrap.json",
            "Fetching FPL Bootstrap Data"
        )
        
        # Fetch Fixtures
        run_command(
            f"{sys.executable} -m src.data_fetch --resource fpl_fixtures --out data/raw/fpl_fixtures.json",
            "Fetching FPL Fixtures"
        )
        
        # Fetch Histories
        run_command(
            f"{sys.executable} -m src.data_fetch --resource fpl_histories --out data/raw/fpl_histories.parquet",
            "Fetching FPL Player Histories (This may take a few minutes)"
        )
        
        # Update Actuals for past predictions
        print("--- Updating History Actuals ---")
        try:
            history.update_actuals()
        except Exception as e:
            print(f"Warning: Failed to update history actuals: {e}")
        print("--- History Update Complete ---\n")
        
        # 2. Preprocess
        run_command(
            f"{sys.executable} src/preprocess.py",
            "Preprocessing Data"
        )
        
        # 3. Train Model
        run_command(
            f"{sys.executable} src/train_model.py",
            "Training Model"
        )
        
    # 4. Log New Predictions
    print("--- Logging New Predictions ---")
    try:
        # Load metadata
        with open('data/processed/metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Load inference data
        df = pd.read_csv('data/processed/inference_data.csv')
        
        # Load models and features
        models = {}
        features_map = {}
        positions = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        
        for name, _ in positions.items():
            model_path = f'models/fpl_model_{name}.pkl'
            feature_path = f'models/model_features_{name}.pkl'
            if os.path.exists(model_path) and os.path.exists(feature_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                with open(feature_path, 'rb') as f:
                    features_map[name] = pickle.load(f)
            
        # Apply filters (Same as app.py)
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
                df = pd.read_csv('data/processed/inference_data.csv')
                df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'])
                df = df[df['selected_by_percent'] < 10]
        
        # Predict per position
        df['predicted_points'] = 0.0
        pos_map_rev = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for pos_id, pos_name in pos_map_rev.items():
            if pos_name not in models:
                continue
            
            model = models[pos_name]
            features = features_map[pos_name]
            
            pos_mask = df['element_type'] == pos_id
            if not pos_mask.any():
                continue
                
            X = df.loc[pos_mask, features].copy()
            for f in features:
                if f not in X.columns:
                    X[f] = 0
                    
            preds = model.predict(X)
            df.loc[pos_mask, 'predicted_points'] = preds
        
        # Selection Logic: Top 1 per position + 1 Wildcard
        final_picks = []
        df_sorted = df.sort_values('predicted_points', ascending=False)
        
        # Track (team, position) pairs to prevent "Same Team + Same Position" duplicates
        selected_combinations = set()

        for pos_id in [1, 2, 3, 4]:
            pos_candidates = df_sorted[df_sorted['element_type'] == pos_id]
            if not pos_candidates.empty:
                pick = pos_candidates.iloc[0]
                final_picks.append(pick)
                selected_combinations.add((pick['team'], pick['element_type']))
                df_sorted = df_sorted[df_sorted['element'] != pick['element']]
                
        if len(final_picks) < 5 and not df_sorted.empty:
            wildcard = None
            # Find first candidate that isn't (Same Team AND Same Position) as an existing pick
            for idx, row in df_sorted.iterrows():
                if (row['team'], row['element_type']) not in selected_combinations:
                    wildcard = row
                    break
            
            # Fallback: If for some reason we filtered everyone out (unlikely), take top remaining
            if wildcard is None and not df_sorted.empty:
                wildcard = df_sorted.iloc[0]

            if wildcard is not None:
                final_picks.append(wildcard)
        
        top_5 = pd.DataFrame(final_picks)
        top_5['position'] = top_5['element_type'].map(pos_map_rev)
        
        # Log
        history.log_predictions(top_5, metadata)
        
    except Exception as e:
        print(f"Warning: Failed to log predictions: {e}")
        import traceback
        traceback.print_exc()
    print("--- Logging Complete ---\n")

    if args.quick:
        print("Quick mode enabled: Skipping data fetch and training.\n")
    
    if args.no_serve:
        print("Pipeline finished. Exiting (No Serve Mode).")
        return

    # 4. Launch App
    print("--- Launching Application ---")
    
    # Schedule browser open after 2 seconds
    Timer(2, open_browser).start()
    
    # Run Flask app
    # We use call directly here because we want it to block and serve
    try:
        subprocess.run([sys.executable, "src/app.py"], check=True)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
    except Exception as e:
        print(f"Error launching app: {e}")

if __name__ == "__main__":
    main()
