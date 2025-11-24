import subprocess
import sys
import time
import os
import webbrowser
from threading import Timer

import argparse

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
    else:
        print("Quick mode enabled: Skipping data fetch and training.\n")
    
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
