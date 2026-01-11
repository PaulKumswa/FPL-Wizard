"""
tests/test_component_models.py
Description: Unit tests for the component-based prediction system.
Tests:
- Component model outputs are valid probabilities [0, 1]
- Aggregated points are in reasonable range
- Scoring constants are applied correctly per position
"""

import pytest
import pandas as pd
import numpy as np
from src.inference import predict_points, load_component_models
from src.scoring_constants import (
    GOAL_POINTS, ASSIST_POINTS, CLEAN_SHEET_POINTS, 
    APPEARANCE_POINTS, CLEAN_SHEET_POSITIONS
)


class MockClassifier:
    """Mock classifier that returns a fixed probability."""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([1 - np.full(n, self.prob), np.full(n, self.prob)])


class MockRegressor:
    """Mock regressor for legacy model."""
    def predict(self, X):
        return np.ones(len(X)) * 4.0  # 4 points baseline


@pytest.fixture
def mock_models():
    """Legacy regressor models (one per position)."""
    return {
        'GKP': MockRegressor(),
        'DEF': MockRegressor(),
        'MID': MockRegressor(),
        'FWD': MockRegressor()
    }


@pytest.fixture
def mock_component_models():
    """Component classifier models (goal/assist/cleansheet per position)."""
    return {
        'GKP': {
            'goal': MockClassifier(0.01),      # GKPs rarely score
            'assist': MockClassifier(0.02),
            'cleansheet': MockClassifier(0.35)  # ~35% clean sheet rate
        },
        'DEF': {
            'goal': MockClassifier(0.05),
            'assist': MockClassifier(0.08),
            'cleansheet': MockClassifier(0.35)
        },
        'MID': {
            'goal': MockClassifier(0.15),
            'assist': MockClassifier(0.12),
            'cleansheet': MockClassifier(0.30)
        },
        'FWD': {
            'goal': MockClassifier(0.25),      # FWDs score most often
            'assist': MockClassifier(0.10)
            # No cleansheet for FWD
        }
    }


@pytest.fixture
def sample_df():
    """Sample inference dataframe with players from each position."""
    data = {
        'element': [1, 2, 3, 4],
        'web_name': ['GK1', 'DEF1', 'MID1', 'FWD1'],
        'team': [1, 2, 3, 4],
        'element_type': [1, 2, 3, 4],  # GKP, DEF, MID, FWD
        'now_cost': [45, 50, 75, 80],
        'selected_by_percent': [2.0, 5.0, 8.0, 9.0],
        'recent_form': [3.0, 4.0, 5.0, 6.0],
        'opponent_strength': [3, 3, 3, 3],
        'is_home': [1, 0, 1, 0],
        'recent_clean_sheets': [0.5, 0.4, 0.3, 0.0],
        'recent_saves': [3.0, 0.0, 0.0, 0.0],
        'recent_goals_conceded': [1.0, 1.2, 0.5, 0.0],
        'recent_penalties_saved': [0.0, 0.0, 0.0, 0.0],
        'recent_goals_scored': [0.0, 0.1, 0.3, 0.5],
        'recent_assists': [0.0, 0.1, 0.2, 0.15],
        'recent_threat': [0.0, 10.0, 30.0, 50.0],
        'recent_influence': [20.0, 25.0, 35.0, 40.0],
        'recent_creativity': [5.0, 10.0, 40.0, 20.0],
        'recent_team_xg': [1.2, 1.3, 1.5, 1.8],
        'recent_team_xga': [1.1, 1.0, 1.2, 1.5],
        'recent_expected_goals': [0.0, 0.05, 0.15, 0.25],
        'recent_expected_assists': [0.0, 0.05, 0.12, 0.08],
        'ict_index': [10.0, 15.0, 25.0, 35.0],
        'status': ['a', 'a', 'a', 'a'],
        'chance_of_playing_next_round': [100, 100, 100, 100]
    }
    return pd.DataFrame(data)


def test_component_probabilities_in_valid_range(sample_df, mock_models, mock_component_models):
    """Test that component probabilities are between 0 and 1."""
    df = predict_points(sample_df, mock_models, mock_component_models)
    
    assert (df['p_goal'] >= 0).all() and (df['p_goal'] <= 1).all()
    assert (df['p_assist'] >= 0).all() and (df['p_assist'] <= 1).all()
    assert (df['p_cleansheet'] >= 0).all() and (df['p_cleansheet'] <= 1).all()


def test_predicted_points_reasonable_range(sample_df, mock_models, mock_component_models):
    """Test that predicted points are in a reasonable range."""
    df = predict_points(sample_df, mock_models, mock_component_models)
    
    # Minimum: appearance points (2)
    # Maximum: appearance + goal (10) + assist (3) + clean sheet (4) = 19 for GKP
    assert (df['predicted_points'] >= APPEARANCE_POINTS).all()
    assert (df['predicted_points'] <= 20).all()


def test_position_specific_scoring(sample_df, mock_models, mock_component_models):
    """Test that different positions get different point values for same outcomes."""
    df = predict_points(sample_df, mock_models, mock_component_models)
    
    # FWD (pos_id=4) should have 0 clean sheet contribution
    fwd_row = df[df['element_type'] == 4].iloc[0]
    gkp_row = df[df['element_type'] == 1].iloc[0]
    
    # FWD clean sheet prob should be 0 (not modeled)
    assert fwd_row['p_cleansheet'] == 0.0
    
    # GKP should have clean sheet contribution
    assert gkp_row['p_cleansheet'] > 0


def test_expected_points_formula(sample_df, mock_models, mock_component_models):
    """Test that expected points are calculated correctly from components."""
    df = predict_points(sample_df, mock_models, mock_component_models)
    
    # Verify formula for each position
    for pos_id in [1, 2, 3, 4]:
        row = df[df['element_type'] == pos_id].iloc[0]
        
        expected = (
            row['p_goal'] * GOAL_POINTS[pos_id] +
            row['p_assist'] * ASSIST_POINTS +
            row['p_cleansheet'] * CLEAN_SHEET_POINTS[pos_id] +
            APPEARANCE_POINTS
        )
        
        assert abs(row['predicted_points'] - expected) < 0.01, \
            f"Position {pos_id}: expected {expected}, got {row['predicted_points']}"


def test_legacy_fallback(sample_df, mock_models):
    """Test that legacy model works when no component models provided."""
    df = predict_points(sample_df, mock_models, component_models=None)
    
    # Should use legacy predictions (all 4.0 from MockRegressor)
    assert (df['predicted_points'] == 4.0).all()
    assert (df['predicted_points_legacy'] == 4.0).all()


def test_scoring_constants():
    """Test that scoring constants are configured correctly."""
    # Goal points should decrease from GKP to FWD
    assert GOAL_POINTS[1] > GOAL_POINTS[2] > GOAL_POINTS[3] > GOAL_POINTS[4]
    
    # Assist points same for all
    assert ASSIST_POINTS == 3
    
    # Clean sheet: GKP/DEF get 4, MID gets 1, FWD gets 0
    assert CLEAN_SHEET_POINTS[1] == 4
    assert CLEAN_SHEET_POINTS[2] == 4
    assert CLEAN_SHEET_POINTS[3] == 1
    assert CLEAN_SHEET_POINTS[4] == 0
    
    # FWD should NOT be in clean sheet positions
    assert 4 not in CLEAN_SHEET_POSITIONS
