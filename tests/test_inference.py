"""
tests/test_inference.py
Description: Unit tests for the src.inference module.
It verifies:
- `predict_points` correctly tracks form (via a MockModel)
- `calculate_confidence` with position-specific weights
- `calculate_p_six_plus` probability derivation per position
- `select_best_team` picks top 5 by P(>=6), respects cost cap and max 3 per team
- `select_best_team` deduplicates DGW players (multiple rows per fixture) to prevent duplicate picks
"""

import pytest
import pandas as pd
import numpy as np
from src.inference import select_best_team, predict_points, calculate_confidence, calculate_p_six_plus
from src.config import MAX_COST_HARD, MAX_PER_POSITION


def test_calculate_confidence_high():
    """Test that decisive probabilities (near 0 or 1) produce high confidence."""
    # pos_id=3 (MID): weights [0.40, 0.40, 0.20]
    # p=0.15 (d=0.7), p=0.10 (d=0.8), p=0.80 (d=0.6)
    # = 0.7*0.4 + 0.8*0.4 + 0.6*0.2 = 0.28 + 0.32 + 0.12 = 0.72 -> 72%
    confidence = calculate_confidence(0.15, 0.10, 0.80, pos_id=3)
    assert confidence >= 65 and confidence <= 80, f"Expected ~72, got {confidence}"


def test_calculate_confidence_low():
    """Test that uncertain probabilities (near 0.5) produce low confidence."""
    confidence = calculate_confidence(0.45, 0.48, 0.52, pos_id=3)
    assert confidence < 15, f"Expected <15%, got {confidence}"


def test_calculate_confidence_arrays():
    """Test that calculate_confidence works with numpy arrays."""
    p_goal = np.array([0.1, 0.5, 0.9])
    p_assist = np.array([0.1, 0.5, 0.9])
    p_cs = np.array([0.1, 0.5, 0.9])
    
    confidence = calculate_confidence(p_goal, p_assist, p_cs, pos_id=3)
    
    assert len(confidence) == 3
    assert confidence[0] > 60  # All decisive (near 0)
    assert confidence[1] < 10  # All uncertain (at 0.5)
    assert confidence[2] > 60  # All decisive (near 1)


# --- P(>=6) tests ---

def test_p_six_plus_def_clean_sheet():
    """DEF with high CS probability should have high P(>=6)."""
    # DEF: P(>=6) = 1 - (1-P(cs)) * (1-P(goal))
    p6 = calculate_p_six_plus(0.05, 0.10, 0.60, pos_id=2)
    # = 1 - 0.95 * 0.40 = 1 - 0.38 = 0.62
    assert abs(p6 - 0.62) < 0.01


def test_p_six_plus_fwd_goal():
    """FWD P(>=6) is just P(goal)."""
    p6 = calculate_p_six_plus(0.30, 0.15, 0.0, pos_id=4)
    assert abs(p6 - 0.30) < 0.01


def test_p_six_plus_mid():
    """MID: P(>=6) = 1 - (1-P(goal)) * (1 - P(assist)*P(cs))."""
    p6 = calculate_p_six_plus(0.20, 0.25, 0.50, pos_id=3)
    # = 1 - (1-0.20) * (1 - 0.25*0.50) = 1 - 0.80 * 0.875 = 1 - 0.70 = 0.30
    assert abs(p6 - 0.30) < 0.01


def test_p_six_plus_arrays():
    """P(>=6) should work with numpy arrays."""
    p_goal = np.array([0.0, 0.5, 1.0])
    p_cs = np.array([0.0, 0.5, 1.0])
    p6 = calculate_p_six_plus(p_goal, np.zeros(3), p_cs, pos_id=2)
    # [0, 1-(0.5*0.5), 1-(0*0)] = [0, 0.75, 1.0]
    assert abs(p6[0] - 0.0) < 0.01
    assert abs(p6[1] - 0.75) < 0.01
    assert abs(p6[2] - 1.0) < 0.01

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
    # Legacy-only predictions default to 50% confidence; override to test selection logic
    df['confidence_score'] = 80.0
    team = select_best_team(df)
    
    # Should select exactly 5 players
    assert len(team) == 5
    
    # No position constraints — just verify we get the highest reliability picks
    # Players 17-20 have form=10 -> pred=20, they should be top picks
    top_elements = set(team['element'].values)
    for high_form_player in [17, 18, 19, 20]:
        assert high_form_player in top_elements, f"Player {high_form_player} (form=10) should be selected"

def test_select_best_team_cost_cap(sample_df, mock_models):
    """Player above £7.5m cost cap should be excluded."""
    sample_df.loc[sample_df['element'] == 17, 'now_cost'] = MAX_COST_HARD + 1
    
    df = predict_points(sample_df, mock_models)
    # Legacy-only predictions default to 50% confidence; override to test selection logic
    df['confidence_score'] = 80.0
    team = select_best_team(df)
    
    # Player 17 should NOT be selected despite high form
    assert 17 not in team['element'].values

def test_select_best_team_max_per_team():
    """Max 3 players from the same team."""
    data = {
        'element': [1, 2, 3, 4, 5, 6],
        'web_name': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'team': [1, 1, 1, 1, 2, 2],  # 4 from team 1, 2 from team 2
        'element_type': [2, 3, 4, 1, 2, 3],  # Mixed positions
        'now_cost': [50] * 6,
        'selected_by_percent': [5.0] * 6,
        'predicted_points': [9, 8.5, 8, 7.5, 7, 6.5],
        'p_six_plus': [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        'confidence_score': [80, 80, 80, 80, 80, 80],
        'status': ['a'] * 6,
        'chance_of_playing_next_round': [100] * 6
    }
    df = pd.DataFrame(data)
    team = select_best_team(df)
    
    # Should select 5 players
    assert len(team) == 5
    
    # Max 3 from team 1
    team1_count = len(team[team['team'] == 1])
    assert team1_count <= 3, f"Expected max 3 from team 1, got {team1_count}"
    
    # Player 4 (team 1, lowest of the 4) should be skipped
    assert 4 not in team['element'].values
    
    # Both team 2 players should be picked to fill the 5 slots
    assert 5 in team['element'].values
    assert 6 in team['element'].values


def test_select_best_team_max_per_position():
    """Max 2 players from the same position."""
    data = {
        'element': [1, 2, 3, 4, 5, 6, 7],
        'web_name': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
        'team': [1, 2, 3, 4, 5, 6, 7],  # All different teams
        'element_type': [2, 2, 2, 2, 3, 4, 1],  # 4 DEF, 1 MID, 1 FWD, 1 GKP
        'now_cost': [50] * 7,
        'selected_by_percent': [5.0] * 7,
        'predicted_points': [9, 8.5, 8, 7.5, 7, 6.5, 6],
        'p_six_plus': [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
        'confidence_score': [80, 80, 80, 80, 80, 80, 80],
        'status': ['a'] * 7,
        'chance_of_playing_next_round': [100] * 7
    }
    df = pd.DataFrame(data)
    team = select_best_team(df)
    
    assert len(team) == 5
    
    # Max 2 DEFs
    def_count = len(team[team['element_type'] == 2])
    assert def_count <= MAX_PER_POSITION, f"Expected max {MAX_PER_POSITION} DEFs, got {def_count}"
    
    # Non-DEF players should fill remaining slots
    assert 5 in team['element'].values  # MID
    assert 6 in team['element'].values  # FWD
    assert 7 in team['element'].values  # GKP


def test_select_best_team_dgw_dedup():
    """DGW players with multiple rows (one per fixture) should be deduplicated.
    
    In a Double Gameweek, preprocess.py creates one row per fixture per player,
    with different opponent_strength/is_home values. select_best_team must
    collapse these to one pick per player.
    """
    data = {
        # Player 1 appears TWICE (DGW: two fixtures with different opponents)
        'element':        [1,    1,    2,    3,    4,    5,    6],
        'web_name':       ['DGW_Star', 'DGW_Star', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'team':           [1,    1,    2,    3,    4,    5,    6],
        'element_type':   [3,    3,    2,    4,    3,    2,    1],  # MID, DEF, FWD, MID, DEF, GKP
        'now_cost':       [50] * 7,
        'selected_by_percent': [5.0] * 7,
        'predicted_points': [9, 8.5, 8, 7.5, 7, 6.5, 6],  # Player 1 has two different scores
        'p_six_plus': [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
        'confidence_score': [80, 75, 80, 80, 80, 80, 80],
        'opponent_strength': [3, 5, 4, 4, 4, 4, 4],  # Different opponents for DGW player
        'is_home':        [1, 0, 1, 0, 1, 0, 1],     # Home for one, away for other
        'status': ['a'] * 7,
        'chance_of_playing_next_round': [100] * 7
    }
    df = pd.DataFrame(data)
    team = select_best_team(df)
    
    # Player 1 should appear exactly ONCE despite having two rows
    player1_count = len(team[team['element'] == 1])
    assert player1_count == 1, f"DGW player should appear once, got {player1_count}"
    
    # Should still select exactly 5 unique players
    assert len(team) == 5
    assert team['element'].nunique() == 5, "All 5 picks should be unique players"


def test_select_best_team_dgw_no_double_pick():
    """Even if multiple DGW rows pass all filters, a player must never be picked twice.
    
    This tests the picked_ids guard: if deduplication somehow doesn't catch a 
    duplicate, the greedy loop's picked_ids set must prevent double selection.
    """
    data = {
        # Player 1 appears THREE times (extreme DGW edge case)
        'element':        [1,    1,    1,    2,    3,    4,    5],
        'web_name':       ['Triple', 'Triple', 'Triple', 'P2', 'P3', 'P4', 'P5'],
        'team':           [1,    1,    1,    2,    3,    4,    5],
        'element_type':   [3,    3,    3,    2,    4,    1,    2],
        'now_cost':       [50] * 7,
        'selected_by_percent': [5.0] * 7,
        'predicted_points': [9, 9, 9, 8, 7, 6, 5],
        'p_six_plus': [0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5],
        'confidence_score': [80] * 7,
        'status': ['a'] * 7,
        'chance_of_playing_next_round': [100] * 7
    }
    df = pd.DataFrame(data)
    team = select_best_team(df)
    
    # Player 1 must appear at most once
    player1_count = len(team[team['element'] == 1])
    assert player1_count <= 1, f"Player should appear at most once, got {player1_count}"
    
    # All selected players must be unique
    assert team['element'].nunique() == len(team), "No duplicate players allowed in final picks"
