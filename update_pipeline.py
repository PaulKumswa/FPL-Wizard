"""
update_pipeline.py
Description: This is the main orchestration script for the FPL Predictor application.
It handles the entire end-to-end flow:
1. Fetching raw data from FPL and Understat APIs (via src.data_fetch).
2. Updating historical actual points (via src.history).
3. Preprocessing data and mapping IDs (via src.preprocess and src.id_map).
4. Training machine learning models (via src.train_model).
5. Generating and logging new predictions (via src.inference and src.history).
6. Launching the Flask web application.
It supports flags for quick runs (skipping data/training) and CI/CD modes.
"""
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
import src.inference as inference

def run_command(command, description):
    print(f"--- {description} ---")
    if isinstance(command, list):
        print(f"Running: {' '.join(command)}")
        shell_mode = False
    else:
        print(f"Running: {command}")
        shell_mode = True
        
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            shell=shell_mode,
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
    parser.add_argument('--no-fetch', action='store_true', help='Skip data fetching, but run training and app')
    args = parser.parse_args()

    print("Starting FPL Pipeline...\n")
    print(f"Python Executable: {sys.executable}")
    
    if not args.quick:
        # 1. Fetch Data
        if not args.no_fetch:
            # Fetch Bootstrap
            run_command(
                [sys.executable, '-m', 'src.data_fetch', '--resource', 'fpl_bootstrap', '--out', 'data/raw/fpl_bootstrap.json'],
                "Fetching FPL Bootstrap Data"
            )
            
            # Fetch Fixtures
            run_command(
                [sys.executable, '-m', 'src.data_fetch', '--resource', 'fpl_fixtures', '--out', 'data/raw/fpl_fixtures.json'],
                "Fetching FPL Fixtures"
            )
            
            # Fetch Histories
            run_command(
                [sys.executable, '-m', 'src.data_fetch', '--resource', 'fpl_histories', '--out', 'data/raw/fpl_histories.parquet'],
                "Fetching FPL Histories"
            )
    
            # Fetch Understat (All) - using new consolidated resource
            run_command(
                [sys.executable, '-m', 'src.data_fetch', '--resource', 'understat_all', '--season', '2025'],
                "Fetching Understat Data"
            )
            
            # Update Actuals for past predictions
            print("--- Updating History Actuals ---")
            try:
                history.update_actuals()
            except Exception as e:
                print(f"Warning: Failed to update history actuals: {e}")
            print("--- History Update Complete ---\n")
        else:
            print("Skipping Data Fetch (--no-fetch enabled)\n")
        
        # 2. Preprocess
        run_command([sys.executable, '-m', 'src.preprocess'], "Preprocessing Data")

        # 3. ID Mapping
        run_command([sys.executable, '-m', 'src.id_map'], "Mapping IDs")

        # 4. Train Models
        run_command([sys.executable, '-m', 'src.train_model'], "Training Models")
        
    # 4. Log New Predictions
    print("--- Logging New Predictions ---")
    try:
        # Load metadata
        with open('data/processed/metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Load inference data
        df = pd.read_csv('data/processed/inference_data.csv')
        
        # Load models
        models = inference.load_models()
        
        # Predict
        df = inference.predict_points(df, models)
        
        # Select Best Team
        top_5 = inference.select_best_team(df)
        
        if top_5.empty:
            print("Warning: No valid team selected.")
        else:
            # Map position IDs to names for logging
            top_5['position'] = top_5['element_type'].map(inference.POSITION_MAP)
            
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
        subprocess.run([sys.executable, "-m", "src.app"], check=True)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
    except Exception as e:
        print(f"Error launching app: {e}")

if __name__ == "__main__":
    main()
