"""
test_app_performance.py
Description: This script tests the consistency and response time of the Flask application's API.
It spins up the server in a separate thread and makes requests to the predictions endpoint to measure latency.
Useful for verifying that the application is serving requests efficiently.
"""
import time
import json
import requests
import threading
from src.app import app

def run_server():
    app.run(port=5001, debug=False, use_reloader=False)

def test_performance():
    # Start server in a separate thread
    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()
    
    # Give it a moment to define routes
    time.sleep(2)
    
    try:
        print("Testing API Response Time...")
        
        # 1. First Request (Simulate Cold Start/Cache Hit)
        start = time.time()
        response = requests.get('http://127.0.0.1:5001/api/predictions')
        end = time.time()
        
        if response.status_code == 200:
            print(f"Request 1 (Likely History Hit): {end - start:.4f} seconds")
            data = response.json()
            if 'gameweek_info' in data:
                print(f"Served predictions for GW: {data['gameweek_info'].get('next_gameweek')}")
        else:
            print(f"Request 1 Failed: {response.status_code}")

    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == '__main__':
    test_performance()
