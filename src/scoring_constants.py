"""
src/scoring_constants.py
Description: FPL scoring rules for component-based point prediction.
Maps player positions to point values for goals, assists, clean sheets, etc.
Used by inference.py to aggregate component predictions into expected points.
"""

# Goal points by position (1=GKP, 2=DEF, 3=MID, 4=FWD)
GOAL_POINTS = {
    1: 10,  # GKP
    2: 6,   # DEF
    3: 5,   # MID
    4: 4    # FWD
}

# Assist points (same for all positions)
ASSIST_POINTS = 3

# Clean sheet points by position
CLEAN_SHEET_POINTS = {
    1: 4,  # GKP
    2: 4,  # DEF
    3: 1,  # MID
    4: 0   # FWD (no clean sheet points)
}

# Baseline points for playing 60+ minutes
APPEARANCE_POINTS = 2

# Positions that get clean sheet points (used to skip training for FWD)
CLEAN_SHEET_POSITIONS = [1, 2, 3]  # GKP, DEF, MID

# Component model types
COMPONENT_TARGETS = ['goal', 'assist', 'cleansheet']
