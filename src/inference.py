"""
src/inference.py
Description: Generates FPL points predictions using component-based machine learning models.
"""
import pandas as pd
import pickle
import os
import json
import numpy as np
from src.config import (
    FEATURE_CONFIGS, POSITION_MAP_REV, POSITION_MAP,
    MAX_COST_HARD, MIN_CONFIDENCE, MAX_PREDICTED_POINTS, MAX_PER_POSITION
)
from src.scoring_constants import (
    GOAL_POINTS, ASSIST_POINTS, CLEAN_SHEET_POINTS, 
    APPEARANCE_POINTS, CLEAN_SHEET_POSITIONS, COMPONENT_TARGETS
)


def load_models(model_dir='models'):
    """Load legacy regressor models from the specified directory."""
    models = {}
    
    for name, _ in POSITION_MAP_REV.items():
        model_path = os.path.join(model_dir, f'fpl_model_{name}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            print(f"Warning: Legacy model for {name} not found at {model_path}")
            
    return models


def load_component_models(model_dir='models'):
    """
    Load component classifier models for each position.
    """
    component_models = {}
    
    for pos_name in POSITION_MAP_REV.keys():
        component_models[pos_name] = {}
        pos_id = POSITION_MAP_REV[pos_name]
        
        for component in COMPONENT_TARGETS:
            # Skip clean sheet for FWD
            if component == 'cleansheet' and pos_id not in CLEAN_SHEET_POSITIONS:
                continue
                
            model_path = os.path.join(model_dir, f'fpl_{component}_model_{pos_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    component_models[pos_name][component] = pickle.load(f)
            else:
                print(f"Warning: Component model not found: {model_path}")
                
    return component_models


def calculate_confidence(p_goal, p_assist, p_cs, pos_id):
    """
    Calculate prediction confidence score (0-100) based on position-weighted component probabilities.
    """
    # Define Weights [Goal, Assist, CS]
    WEIGHTS = {
        1: [0.10, 0.10, 0.80],  # GKP
        2: [0.15, 0.15, 0.70],  # DEF
        3: [0.40, 0.40, 0.20],  # MID
        4: [0.50, 0.50, 0.00]   # FWD
    }
    
    w = WEIGHTS.get(pos_id, [0.33, 0.33, 0.33])
    
    # Handle array input
    if hasattr(p_goal, '__iter__'):
        d_goal = np.abs(np.array(p_goal) - 0.5) * 2
        d_assist = np.abs(np.array(p_assist) - 0.5) * 2
        d_cs = np.abs(np.array(p_cs) - 0.5) * 2
        
        confidence_val = (d_goal * w[0]) + (d_assist * w[1]) + (d_cs * w[2])
    else:
        d_goal = abs(p_goal - 0.5) * 2
        d_assist = abs(p_assist - 0.5) * 2
        d_cs = abs(p_cs - 0.5) * 2
        
        confidence_val = (d_goal * w[0]) + (d_assist * w[1]) + (d_cs * w[2])
        
    return confidence_val * 100.0


def predict_points(df, models, component_models=None):
    """
    Apply models to the dataframe to generate 'predicted_points'.
    """
    df = df.copy()
    df['predicted_points'] = 0.0
    df['predicted_points_legacy'] = 0.0
    df['p_goal'] = 0.0
    df['p_assist'] = 0.0
    df['p_cleansheet'] = 0.0
    df['confidence_score'] = 50.0  # Default moderate confidence
    
    use_components = component_models is not None and len(component_models) > 0
    
    for pos_name, legacy_model in models.items():
        pos_id = POSITION_MAP_REV[pos_name]
        config = FEATURE_CONFIGS[pos_id]
        features = config['features']
        
        # Filter for this position
        pos_mask = df['element_type'] == pos_id
        if not pos_mask.any():
            continue
            
        # Prepare Features
        X = df.loc[pos_mask].copy()
        
        # Ensure all features exist, fill missing with 0
        for f in features:
            if f not in X.columns:
                X[f] = 0.0
        
        X_features = X[features]
        
        # Legacy prediction
        legacy_preds = legacy_model.predict(X_features)
        df.loc[pos_mask, 'predicted_points_legacy'] = legacy_preds
        
        # Component-based prediction
        if use_components and pos_name in component_models:
            pos_components = component_models[pos_name]
            
            def get_probs(comp_name):
                if comp_name in pos_components:
                    return pos_components[comp_name].predict_proba(X_features)[:, 1]
                return np.zeros(len(X_features))

            p_goal = get_probs('goal')
            p_assist = get_probs('assist')
            p_cs = get_probs('cleansheet') if pos_id in CLEAN_SHEET_POSITIONS else np.zeros(len(X_features))
            
            df.loc[pos_mask, 'p_goal'] = p_goal
            df.loc[pos_mask, 'p_assist'] = p_assist
            df.loc[pos_mask, 'p_cleansheet'] = p_cs
            
            # Calculate WEIGHTED confidence
            confidence = calculate_confidence(p_goal, p_assist, p_cs, pos_id)
            df.loc[pos_mask, 'confidence_score'] = confidence
            
            # Aggregate into expected points
            expected_pts = (
                p_goal * GOAL_POINTS[pos_id] +
                p_assist * ASSIST_POINTS +
                p_cs * CLEAN_SHEET_POINTS[pos_id] +
                APPEARANCE_POINTS
            )
            # Cap predicted points to prevent over-prediction outliers
            expected_pts = np.clip(expected_pts, 0, MAX_PREDICTED_POINTS)
            df.loc[pos_mask, 'predicted_points'] = expected_pts
        else:
            # Fallback to legacy
            df.loc[pos_mask, 'predicted_points'] = legacy_preds
        
    return df


def select_best_team(df):
    """
    Select top 5 players by predicted points within high/medium confidence bands.
    Position-agnostic, £7.5m cost cap, max 3 per team, max 2 per position,
    confidence >= MIN_CONFIDENCE.
    """
    
    # 1. Ensure numeric columns
    cols_to_numeric = ['selected_by_percent', 'now_cost', 'predicted_points', 'confidence_score']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'confidence_score' not in df.columns:
        df['confidence_score'] = 50.0
    
    # 2. Availability filter
    if 'status' in df.columns:
        df = df[~df['status'].isin(['s', 'u', 'n', 'i', 'd'])]
    if 'chance_of_playing_next_round' in df.columns:
        df['chance_of_playing_next_round'] = pd.to_numeric(
            df['chance_of_playing_next_round'], errors='coerce'
        ).fillna(100)
        df = df[df['chance_of_playing_next_round'] >= 75]

    # 3. Filter: cost cap + confidence >= MIN_CONFIDENCE (high + medium bands only)
    pool = df[
        (df['now_cost'] <= MAX_COST_HARD) &
        (df['confidence_score'] >= MIN_CONFIDENCE)
    ].copy()
    
    # 3b. Deduplicate: DGW players appear multiple times (one row per fixture)
    pool = pool.drop_duplicates(subset=['element'])
    
    # 4. Sort by predicted points (highest upside within confident players)
    pool = pool.sort_values('predicted_points', ascending=False)
    
    # 5. Greedy selection: top 5 with max 3 per team, max per position
    final_picks = []
    team_counts = {}
    position_counts = {}
    picked_ids = set()
    
    for _, player in pool.iterrows():
        if len(final_picks) >= 5:
            break
        
        player_id = player.get('element', 0)
        if player_id in picked_ids:
            continue
        
        team_id = player.get('team', 0)
        if team_counts.get(team_id, 0) >= 3:
            continue
        
        pos_id = player.get('element_type', 0)
        if position_counts.get(pos_id, 0) >= MAX_PER_POSITION:
            continue
        
        pick_dict = player.to_dict()
        pick_dict['is_wildcard'] = False
        final_picks.append(pick_dict)
        picked_ids.add(player_id)
        team_counts[team_id] = team_counts.get(team_id, 0) + 1
        position_counts[pos_id] = position_counts.get(pos_id, 0) + 1
    
    return pd.DataFrame(final_picks)

