"""
tests/test_preprocess.py
Description: Unit tests for the src.preprocess module.
It focuses on:
- Validating the calculation of rolling averages (recent_form, etc.)
- Ensuring that inference data (upcoming matches) correctly pulls the most recent historical form (last 5 games).
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data

# Mock Data Fixtures
@pytest.fixture
def mock_data():
    # Players
    players = pd.DataFrame({
        'id': [1],
        'web_name': ['TestPlayer'],
        'team': [1],
        'element_type': [3], # MID
        'now_cost': [50],
        'selected_by_percent': ['10.0'],
        'code': [123],
        'ict_index': ['5.0'],
        'status': ['a'],
        'chance_of_playing_next_round': [100],
        'news': ['']
    })
    
    # Teams
    teams = pd.DataFrame({
        'id': [1, 2],
        'name': ['Team A', 'Team B'],
        'short_name': ['TEA', 'TEB'],
        'strength': [3, 4],
        'code': [1, 2]
    })
    
    # Events (Gameweeks)
    events = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'is_current': [False, False, False, False, False, True], # GW 6 current
        'is_next': [False, False, False, False, False, False]
    })
    # Add GW 7 as next? Logic in preprocess is minimal for events, mostly for inference rows.
    # We are testing TRAIN df (historical).
    
    # Fixtures
    fixtures = pd.DataFrame({
        'id': [101, 102, 103, 104, 105, 106],
        'event': [1, 2, 3, 4, 5, 6],
        'kickoff_time': pd.to_datetime(['2023-08-01', '2023-08-08', '2023-08-15', '2023-08-22', '2023-08-29', '2023-09-05']),
        'team_h': [1, 1, 1, 1, 1, 1], # Always Team 1 home for simplicity
        'team_a': [2, 2, 2, 2, 2, 2]
    })
    
    # Histories (The critical part)
    # Player 1 played in GW 1-5.
    histories = pd.DataFrame({
        'element': [1, 1, 1, 1, 1],
        'fixture': [101, 102, 103, 104, 105],
        'round': [1, 2, 3, 4, 5],
        'total_points': [10, 2, 6, 4, 8], # Avg = 6.0
        'minutes': [90, 90, 90, 90, 90],
        'was_home': [True, True, True, True, True],
        'opponent_team': [2, 2, 2, 2, 2],
        # Add other req cols
        'goals_scored': [0]*5, 'assists': [0]*5, 'clean_sheets': [0]*5,
        'goals_conceded': [0]*5, 'own_goals': [0]*5, 'penalties_saved': [0]*5,
        'penalties_missed': [0]*5, 'yellow_cards': [0]*5, 'red_cards': [0]*5,
        'saves': [0]*5, 'bonus': [0]*5, 'bps': [0]*5, 'influence': [0]*5,
        'creativity': [0]*5, 'threat': [0]*5, 'ict_index': [0]*5,
        'expected_goals': [0.1]*5, 'expected_assists': [0.1]*5,
        'team_xg': [1.0]*5, 'team_xga': [1.0]*5,
        'us_npxG_per90': [0.0]*5, 'us_xA_per90': [0.0]*5  # New Understat features
    })
    
    # Understat Matches (Empty for basic test)
    understat_matches = pd.DataFrame()
    
    # Understat Players (Empty for basic test)
    understat_players = pd.DataFrame()
    
    # ID Mapping (Empty for basic test)
    id_mapping = pd.DataFrame()
    
    return players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping

def test_rolling_averages(mock_data):
    players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping = mock_data
    
    train_df, inference_df, _, _ = preprocess_data(players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping)
    
    # Check Player 1
    p1 = train_df[train_df['element'] == 1].sort_values('round')
    
    # Row for Round 1:
    # previous points: NaN (shift 1). Rolling: 0.0 (fillna) or NaN?
    # Logic: groupby shift(1). Round 1 shift is NaN. rolling min_periods=1.
    # If purely NaN, mean is NaN -> fillna(0).
    # So Round 1 recent_form should be 0.
    assert p1.iloc[0]['recent_form'] == 0.0
    
    # Row for Round 2:
    # Uses Round 1 points (10). Avg(10) = 10.
    assert p1.iloc[1]['recent_form'] == 10.0
    
    # Row for Round 3:
    # Uses Round 1, 2 (10, 2). Avg = 6.0.
    assert p1.iloc[2]['recent_form'] == 6.0
    
    # Row for Round 6? (Wait, histories only went to 5).
    # train_df only contains rows provided in histories (actual matches played).
    # So we check the LAST row (Round 5).
    # Uses R1, R2, R3, R4 (10, 2, 6, 4). Avg = 22/4 = 5.5.
    assert p1.iloc[4]['recent_form'] == 5.5

def test_inference_features(mock_data):
    # Verify that INFERENCE rows get the rolling average of the last 5 ACTUAL rounds
    players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping = mock_data
    
    train_df, inference_df, _, _ = preprocess_data(players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping)
    
    # Inference is for Next GW (which we need to set in events)
    # Mock data sets GW 6 as current. So logic looks for GW 6? Or 7?
    # Logic: next_gw = events[is_next == True].
    # In mock, no is_next is True.
    # Let's fix mock in this test function by modifying it.
    events.loc[5, 'is_next'] = True # GW 6 is next
    
    # Re-run
    train_df, inference_df, _, _ = preprocess_data(players, teams, events, histories, fixtures, understat_matches, understat_players, id_mapping)
    
    assert not inference_df.empty
    
    # Inference for Player 1
    # Should use last 5 rounds: 1, 2, 3, 4, 5.
    # Points: 10, 2, 6, 4, 8.
    # Avg: 30 / 5 = 6.0.
    
    row = inference_df[inference_df['element'] == 1].iloc[0]
    assert row['recent_form'] == 6.0
