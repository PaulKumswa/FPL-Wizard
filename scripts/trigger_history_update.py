"""
scripts/trigger_history_update.py
Description: A utility script to manually trigger the `update_actuals` function from the history module.
This updates the 'actual_points' in the prediction history log based on the latest data.
Useful for ad-hoc updates or testing the history update logic in isolation.
"""
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src import history

print("Updating history actuals...")
history.update_actuals()
print("Done.")
