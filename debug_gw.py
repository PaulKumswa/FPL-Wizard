"""
debug_gw.py
Description: This script reads the FPL bootstrap data and prints a formatted table of gameweek (event) information.
It displays the ID, Name, Deadline, Current Status, Next Status, and Finished Status for a range of gameweeks.
Useful for verifying the current state of the FPL season as seen by the application.
"""
import json
import pandas as pd
from datetime import datetime

with open('data/raw/fpl_bootstrap.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

events = data['events']
print(f"{'ID':<4} {'Name':<15} {'Deadline':<25} {'Current':<8} {'Next':<8} {'Finished':<8}")
print("-" * 70)
for e in events:
    if e['id'] >= 14 and e['id'] <= 22:
        print(f"{e['id']:<4} {e['name']:<15} {e['deadline_time']:<25} {str(e['is_current']):<8} {str(e['is_next']):<8} {str(e['finished']):<8}")
