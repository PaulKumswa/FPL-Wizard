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
            
            # Load model features
            with open('models/model_features.pkl', 'rb') as f:
                features = pickle.load(f)
            
            # Load model
            with open('models/fpl_model.pkl', 'rb') as f:
                model = pickle.load(f)
                
            # Apply filters (Same as app.py)
            df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'])
            df['now_cost'] = pd.to_numeric(df['now_cost'])
            df['recent_form'] = pd.to_numeric(df['recent_form'])
            df['ict_index'] = pd.to_numeric(df['ict_index'])

            filtered_df = df[
                (df['selected_by_percent'] < 10) & 
                (df['now_cost'] < 80) & 
                ((df['recent_form'] > 2.0) | (df['ict_index'] > 3.0))
            ]
            
            if filtered_df.empty:
                 filtered_df = df[df['selected_by_percent'] < 10]
            
            # Predict
            # Ensure features match
            filtered_df = filtered_df.dropna(subset=features)
            predictions = model.predict(filtered_df[features])
            filtered_df['predicted_points'] = predictions
            
            # Top 5
            top_5 = filtered_df.sort_values('predicted_points', ascending=False).head(5)
            
            # Log
            history.log_predictions(top_5, metadata)
            
        except Exception as e:
            print(f"Warning: Failed to log predictions: {e}")
            import traceback
            traceback.print_exc()
        print("--- Logging Complete ---\n")

    else:
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
