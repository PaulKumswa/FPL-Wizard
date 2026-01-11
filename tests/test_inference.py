"""
tests/test_inference.py
Description: Unit tests for the src.inference module.
It verifies:
- `predict_points` correctly tracks form (via a MockModel)
- `select_best_team` respects rules (1 per position, cost limits, no same-team duplicates in certain roles)
"""

import pytest
import pandas as pd
import numpy as np
from src.inference import select_best_team, predict_points, calculate_confidence
from src.config import MAX_COST, MAX_OWNERSHIP


def test_calculate_confidence_high():
    """Test that decisive probabilities (near 0 or 1) produce high confidence."""
    # p=0.15 (d=0.7), p=0.10 (d=0.8), p=0.80 (d=0.6) -> mean = 0.7 -> 70%
    confidence = calculate_confidence(0.15, 0.10, 0.80)
    assert confidence >= 65 and confidence <= 75, f"Expected ~70, got {confidence}"


def test_calculate_confidence_low():
    """Test that uncertain probabilities (near 0.5) produce low confidence."""
    # p=0.45 (d=0.1), p=0.48 (d=0.04), p=0.52 (d=0.04) -> mean = 0.06 -> 6%
    confidence = calculate_confidence(0.45, 0.48, 0.52)
    assert confidence < 15, f"Expected <15%, got {confidence}"


def test_calculate_confidence_arrays():
    """Test that calculate_confidence works with numpy arrays."""
    p_goal = np.array([0.1, 0.5, 0.9])
    p_assist = np.array([0.1, 0.5, 0.9])
    p_cs = np.array([0.1, 0.5, 0.9])
    
    confidence = calculate_confidence(p_goal, p_assist, p_cs)
    
    assert len(confidence) == 3
    assert confidence[0] > 70  # All decisive (near 0)
    assert confidence[1] < 10  # All uncertain (at 0.5)
    assert confidence[2] > 70  # All decisive (near 1)

# Mock Model
class MockModel:
    def predict(self, X):
        # Predict based on 'recent_form' column if exists, else random
        if 'recent_form' in X.columns:
            return X['recent_form'] * 2
        return np.ones(len(X)) * 5.0

@pytest.fixture
def mock_models():
    return {
        'GKP': MockModel(),
        'DEF': MockModel(),
        'MID': MockModel(),
        'FWD': MockModel()
    }

@pytest.fixture
def sample_df():
    # Create a DataFrame with candidates for all positions
    data = {
        'element': range(1, 21),
        'web_name': [f'Player{i}' for i in range(1, 21)],
        'team': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], # Teams
        'element_type': [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4], # Pos
        'now_cost': [50] * 20,
        'selected_by_percent': [5.0] * 20,
        'recent_form': [1.0, 5.0, 3.0, 8.0, 2.0, 6.0, 4.0, 9.0, 1.0, 1.0, 1.0, 1.0, 7.0, 2.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        'ict_index': [10.0] * 20,
        'status': ['a'] * 20,
        'chance_of_playing_next_round': [100] * 20
    }
    return pd.DataFrame(data)

def test_predict_points(sample_df, mock_models):
    df = predict_points(sample_df, mock_models)
    assert 'predicted_points' in df.columns
    # Check that highest form got highest points (since our mock model uses form*2)
    # Player 17 has form 10.0 -> pred 20.0
    # Player 1 has form 1.0 -> pred 2.0
    p17 = df[df['element'] == 17].iloc[0]
    p1 = df[df['element'] == 1].iloc[0]
    assert p17['predicted_points'] > p1['predicted_points']

def test_select_best_team_basic(sample_df, mock_models):
    df = predict_points(sample_df, mock_models)
    team = select_best_team(df)
    
    # Should select 5 players
    assert len(team) == 5
    
    # Should have 1 GKP, 1 DEF, 1 MID, 1 FWD + 1 Wildcard
    counts = team['element_type'].value_counts()
    assert counts[1] >= 1
    assert counts[2] >= 1
    assert counts[3] >= 1
    assert counts[4] >= 1

def test_select_best_team_constraints(sample_df, mock_models):
    # Set one player to high cost
    sample_df.loc[sample_df['element'] == 17, 'now_cost'] = MAX_COST + 1
    
    df = predict_points(sample_df, mock_models)
    team = select_best_team(df)
    
    # Player 17 should NOT be typically selected despite high form (unless fallback triggers, which it shouldn't here)
    assert 17 not in team['element'].values

def test_select_best_team_no_duplicates(sample_df, mock_models):
    # Force a scenario where top GKP and top DEF are same team?
    # Actually, logic prevents "Same Team AND Same Position".
    
    # Let's clean test:
    # Top GKP: Team 1
    # Top DEF: Team 1
    # Top MID: Team 1
    # Top FWD: Team 1
    
    data = {
        'element': [1, 2, 3, 4],
        'team': [1, 1, 1, 1],
        'element_type': [1, 2, 3, 4], # GKP, DEF, MID, FWD
        'now_cost': [50, 50, 50, 50],
        'selected_by_percent': [1, 1, 1, 1],
        'recent_form': [10, 10, 10, 10], # High points
        'ict_index': [10, 10, 10, 10],
        'status': ['a', 'a', 'a', 'a'],
        'chance_of_playing_next_round': [100, 100, 100, 100]
    }
    # Add a wildcard candidate who is SAME TEAM + SAME POS as GKP
    # Player 5: Team 1, GKP. Form 9.
    
    data['element'].append(5)
    data['team'].append(1)
    data['element_type'].append(1) # GKP
    data['now_cost'].append(50)
    data['selected_by_percent'].append(1)
    data['recent_form'].append(9)
    data['ict_index'].append(10)
    data['status'].append('a')
    data['chance_of_playing_next_round'].append(100)
    
    # Add a valid wildcard (diff team)
    # Player 6: Team 2, GKP. Form 8.
    data['element'].append(6)
    data['team'].append(2)
    data['element_type'].append(1) # GKP
    data['now_cost'].append(50)
    data['selected_by_percent'].append(1)
    data['recent_form'].append(8)
    data['ict_index'].append(10)
    data['status'].append('a')
    data['chance_of_playing_next_round'].append(100)
    
    df = pd.DataFrame(data)
    df = predict_points(df, mock_models)
    
    team = select_best_team(df)
    
    # Logic:
    # 1. Selects Top GKP (Player 1, Team 1)
    # 2. Selects Top DEF (Player 2, Team 1) -> Allowed (Diff Pos)
    # 3. Selects Top MID (Player 3, Team 1) -> Allowed
    # 4. Selects Top FWD (Player 4, Team 1) -> Allowed
    # 5. Wildcard:
    #    - Candidate: Player 5 (Team 1, GKP). BUT we already have (Team 1, GKP) via Player 1.
    #    - Result: REJECT Player 5.
    #    - Candidate: Player 6 (Team 2, GKP).
    #    - Result: ACCEPT Player 6.
    
    assert 1 in team['element'].values
    assert 5 not in team['element'].values
    assert 6 in team['element'].values
