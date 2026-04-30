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
    MAX_COST_HARD, MIN_CONFIDENCE, MAX_PREDICTED_POINTS, MAX_PREDICTED_POINTS_DGW,
    MAX_PER_POSITION
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


def calculate_p_six_plus(p_goal, p_assist, p_cs, pos_id):
    """
    Calculate P(player scores >= 6 FPL points) from component probabilities.
    
    The main paths to 6+ points per position:
      GKP: Clean sheet (2+4=6) or Goal (2+10=12)
      DEF: Clean sheet (2+4=6) or Goal (2+6=8)
      MID: Goal (2+5=7) or Assist+CS (2+3+1=6)
      FWD: Goal (2+4=6)
    
    Assumes independence between goal, assist, and clean sheet events.
    """
    p_goal = np.asarray(p_goal, dtype=float)
    p_assist = np.asarray(p_assist, dtype=float)
    p_cs = np.asarray(p_cs, dtype=float)

    if pos_id == 1:  # GKP: CS alone = 6, goal alone = 12
        p6 = 1 - (1 - p_cs) * (1 - p_goal)
    elif pos_id == 2:  # DEF: CS alone = 6, goal alone = 8
        p6 = 1 - (1 - p_cs) * (1 - p_goal)
    elif pos_id == 3:  # MID: goal alone = 7, assist+CS = 6
        p6 = 1 - (1 - p_goal) * (1 - p_assist * p_cs)
    elif pos_id == 4:  # FWD: goal alone = 6
        p6 = p_goal
    else:
        p6 = np.zeros_like(p_goal)

    return np.clip(p6, 0.0, 1.0)


def aggregate_dgw_predictions(df):
    """
    Aggregate per-fixture predictions for DGW players into one row per player.
    
    For players with multiple rows (one per fixture in a Double Gameweek):
      - predicted_points: summed across fixtures, capped at MAX_PREDICTED_POINTS_DGW
      - p_six_plus: hybrid of haul path and accumulation path (see below)
      - confidence_score: averaged across fixtures
      - p_goal, p_assist, p_cleansheet: max across fixtures (for display)
      - is_dgw: True for DGW players, False for SGW
      - dgw_fixture_count: number of fixtures (1 or 2)
    
    P(≥6) hybrid formula:
      p_haul  = 1 - ∏(1 - p_six_plus_i)  — chance of 6+ in at least one fixture
      p_accum = clamp(sum_predicted_pts / 12, 0, 1)  — accumulation path
      p_six_plus = max(p_haul, p_accum)
    
    Single-GW players pass through with is_dgw=False, dgw_fixture_count=1.
    """
    df = df.copy()
    
    # Count fixtures per player
    fixture_counts = df.groupby('element').size()
    dgw_elements = fixture_counts[fixture_counts > 1].index
    
    if len(dgw_elements) == 0:
        # No DGW players — just add flags and return
        df['is_dgw'] = False
        df['dgw_fixture_count'] = 1
        return df
    
    # Split into SGW and DGW
    sgw_mask = ~df['element'].isin(dgw_elements)
    sgw_df = df[sgw_mask].copy()
    sgw_df['is_dgw'] = False
    sgw_df['dgw_fixture_count'] = 1
    
    dgw_df = df[df['element'].isin(dgw_elements)].copy()
    
    # Aggregate DGW players
    # Columns to sum
    sum_cols = ['predicted_points', 'predicted_points_legacy']
    # Columns to average
    avg_cols = ['confidence_score']
    # Columns to max (component probabilities for display)
    max_cols = ['p_goal', 'p_assist', 'p_cleansheet']
    # Columns to keep first (non-fixture-specific metadata)
    first_cols = [c for c in dgw_df.columns 
                  if c not in sum_cols + avg_cols + max_cols + ['p_six_plus']]
    
    # Build aggregation dict
    agg_dict = {}
    for col in first_cols:
        if col in dgw_df.columns:
            agg_dict[col] = 'first'
    for col in sum_cols:
        if col in dgw_df.columns:
            agg_dict[col] = 'sum'
    for col in avg_cols:
        if col in dgw_df.columns:
            agg_dict[col] = 'mean'
    for col in max_cols:
        if col in dgw_df.columns:
            agg_dict[col] = 'max'
    
    # For p_six_plus, we need custom aggregation — collect as list first
    if 'p_six_plus' in dgw_df.columns:
        agg_dict['p_six_plus'] = list
    
    dgw_agg = dgw_df.groupby('element', as_index=False).agg(agg_dict)
    
    # Apply hybrid P(≥6) formula
    if 'p_six_plus' in dgw_agg.columns:
        hybrid_p6 = []
        for idx, row in dgw_agg.iterrows():
            p6_values = row['p_six_plus']
            # Haul path: P(≥6 in at least one fixture)
            p_haul = 1.0 - np.prod([1.0 - p for p in p6_values])
            # Accumulation path: based on summed expected points
            sum_pts = row.get('predicted_points', 0)
            p_accum = np.clip(sum_pts / 12.0, 0.0, 1.0)
            # Take the higher of the two paths
            hybrid_p6.append(max(p_haul, p_accum))
        dgw_agg['p_six_plus'] = hybrid_p6
    
    # Cap DGW summed predictions
    if 'predicted_points' in dgw_agg.columns:
        dgw_agg['predicted_points'] = np.clip(
            dgw_agg['predicted_points'], 0, MAX_PREDICTED_POINTS_DGW
        )
    
    # Add DGW flags
    dgw_agg['is_dgw'] = True
    dgw_agg['dgw_fixture_count'] = dgw_agg['element'].map(
        fixture_counts[dgw_elements].to_dict()
    )
    
    # Combine back
    result = pd.concat([sgw_df, dgw_agg], ignore_index=True)
    
    dgw_count = len(dgw_elements)
    print(f"DGW aggregation: {dgw_count} players had multiple fixtures -> summed predictions")
    
    return result


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
    df['p_six_plus'] = 0.0  # P(player scores >= 6 pts)
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
            
            # Aggregate into expected points (component model)
            component_pts = (
                p_goal * GOAL_POINTS[pos_id] +
                p_assist * ASSIST_POINTS +
                p_cs * CLEAN_SHEET_POINTS[pos_id] +
                APPEARANCE_POINTS
            )
            # Blend component (60%) with legacy regressor (40%) to reduce over-prediction.
            # Component model ranks well but over-predicts; legacy is more conservative.
            blended_pts = (0.6 * component_pts) + (0.4 * legacy_preds)
            # Cap to prevent extreme outliers
            blended_pts = np.clip(blended_pts, 0, MAX_PREDICTED_POINTS)
            df.loc[pos_mask, 'predicted_points'] = blended_pts
            
            # Calculate P(>=6 points) — the actual selection metric
            p6 = calculate_p_six_plus(p_goal, p_assist, p_cs, pos_id)
            df.loc[pos_mask, 'p_six_plus'] = p6
        else:
            # Fallback to legacy
            df.loc[pos_mask, 'predicted_points'] = legacy_preds
    
    # Aggregate DGW players: sum points, combine P(≥6), average confidence
    df = aggregate_dgw_predictions(df)
    
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
        df = df[~df['status'].isin(['s', 'u', 'n', 'i', 'd'])].copy()
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
    
    # 3b. DGW aggregation already collapsed multi-fixture rows in predict_points().
    #     drop_duplicates is kept as a safety net only.
    pool = pool.drop_duplicates(subset=['element'])
    
    # 4. Sort by P(>=6 points) — directly optimizes for the 6+ hit target
    #    Falls back to predicted_points if p_six_plus is not available
    sort_col = 'p_six_plus' if 'p_six_plus' in pool.columns and pool['p_six_plus'].sum() > 0 else 'predicted_points'
    pool = pool.sort_values(sort_col, ascending=False)
    
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

