"""
src/config.py
Description: Central configuration file for the FPL Predictor.
It contains:
- Feature definitions per player position (GKP, DEF, MID, FWD) used for model training and inference.
- Mappings between position IDs and names.
- logic/constraints for selecting the "best team" (e.g., max cost, ownership, min form).
"""

# src/config.py

# Feature Configuration per Position
# 1: GKP, 2: DEF, 3: MID, 4: FWD
FEATURE_CONFIGS = {
    1: {
        'name': 'GKP',
        'features': [
            'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
            'recent_clean_sheets', 'recent_saves', 'recent_goals_conceded', 'recent_penalties_saved',
            'recent_team_xga'
        ],
        'min_samples_leaf': 5
    },
    2: {
        'name': 'DEF',
        'features': [
            'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
            'recent_clean_sheets', 'recent_goals_conceded', 'recent_assists', 'recent_goals_scored',
            'recent_threat', 'recent_influence', 'recent_team_xga'
        ],
        'min_samples_leaf': 3
    },
    3: {
        'name': 'MID',
        'features': [
            'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
            'recent_goals_scored', 'recent_assists', 'recent_clean_sheets', 
            'recent_creativity', 'recent_threat', 'recent_influence',
            'recent_team_xg', 'recent_expected_goals', 'recent_expected_assists',
            'recent_us_npxG_per90', 'recent_us_xA_per90'
        ],
        'min_samples_leaf': 3
    },
    4: {
        'name': 'FWD',
        'features': [
            'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
            'recent_goals_scored', 'recent_assists', 
            'recent_threat', 'recent_influence',
            'recent_team_xg', 'recent_expected_goals', 'recent_expected_assists',
            'recent_us_npxG_per90', 'recent_us_xA_per90'
        ],
        'min_samples_leaf': 3
    }
}

# Position ID to Name Mapping
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POSITION_MAP_REV = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}

# Selection Criteria - Percentile-Based (Dynamic)
# These percentiles are applied to the current week's player pool at inference time
# to compute data-driven thresholds that adapt to season dynamics.
#
# OWNERSHIP_PERCENTILE: Select from bottom X% of ownership (lower = more differential)
# COST_PERCENTILE: Select from below X% of cost (higher = more budget options)
# FORM_PERCENTILE: Exclude bottom X% by form (lower = less restrictive)
# ICT_PERCENTILE: Exclude bottom X% by ICT index (lower = less restrictive)
#
OWNERSHIP_PERCENTILE = 25   # Bottom 25% of ownership = true differentials
COST_PERCENTILE = 60        # Below 60th percentile cost = budget-friendly
FORM_PERCENTILE = 30        # Above 30th percentile form = exclude poor form
ICT_PERCENTILE = 30         # Above 30th percentile ICT = exclude inactive players
