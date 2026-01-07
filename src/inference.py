"""
src/inference.py
Description: Generates FPL points predictions using component-based machine learning models.

This module implements COMPONENT-BASED prediction:
1. Loads component classifiers (goals, assists, clean sheets) for each position
2. Predicts probability of each outcome occurring
3. Aggregates probabilities into expected points using FPL scoring rules

Key Functions:
- `load_models`: Loads legacy regressor models (backward compatible)
- `load_component_models`: Loads component classifiers
- `predict_points`: Applies models to generate predictions
- `select_best_team`: Selects top players based on predictions and constraints
"""
import pandas as pd
import pickle
import os
import json
import numpy as np
from src.config import (
    FEATURE_CONFIGS, POSITION_MAP_REV, POSITION_MAP,
    OWNERSHIP_PERCENTILE, COST_PERCENTILE, FORM_PERCENTILE, ICT_PERCENTILE
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
    Returns nested dict: {position: {component: model}}
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


def predict_points(df, models, component_models=None):
    """
    Apply models to the dataframe to generate 'predicted_points'.
    
    If component_models are available, uses component-based prediction:
    - P(goal) * goal_points + P(assist) * assist_points + P(clean_sheet) * cs_points + appearance
    
    Falls back to legacy regressor if component models unavailable.
    
    Returns the modified dataframe with additional columns:
    - predicted_points: Final prediction (component-based if available)
    - predicted_points_legacy: Legacy regressor prediction
    - p_goal, p_assist, p_cleansheet: Component probabilities
    """
    df = df.copy()
    df['predicted_points'] = 0.0
    df['predicted_points_legacy'] = 0.0
    df['p_goal'] = 0.0
    df['p_assist'] = 0.0
    df['p_cleansheet'] = 0.0
    
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
        
        # Legacy prediction (always compute for comparison)
        legacy_preds = legacy_model.predict(X_features)
        df.loc[pos_mask, 'predicted_points_legacy'] = legacy_preds
        
        # Component-based prediction
        if use_components and pos_name in component_models:
            pos_components = component_models[pos_name]
            
            # Goal probability
            if 'goal' in pos_components:
                p_goal = pos_components['goal'].predict_proba(X_features)[:, 1]
                df.loc[pos_mask, 'p_goal'] = p_goal
            else:
                p_goal = 0.0
            
            # Assist probability
            if 'assist' in pos_components:
                p_assist = pos_components['assist'].predict_proba(X_features)[:, 1]
                df.loc[pos_mask, 'p_assist'] = p_assist
            else:
                p_assist = 0.0
            
            # Clean sheet probability
            if 'cleansheet' in pos_components and pos_id in CLEAN_SHEET_POSITIONS:
                p_cs = pos_components['cleansheet'].predict_proba(X_features)[:, 1]
                df.loc[pos_mask, 'p_cleansheet'] = p_cs
            else:
                p_cs = 0.0
            
            # Aggregate into expected points
            expected_pts = (
                p_goal * GOAL_POINTS[pos_id] +
                p_assist * ASSIST_POINTS +
                p_cs * CLEAN_SHEET_POINTS[pos_id] +
                APPEARANCE_POINTS  # Baseline for 60+ minutes
            )
            df.loc[pos_mask, 'predicted_points'] = expected_pts
        else:
            # Fallback to legacy
            df.loc[pos_mask, 'predicted_points'] = legacy_preds
        
    return df


def calculate_dynamic_thresholds(df):
    """
    Calculate data-driven selection thresholds based on current week's player pool.
    
    Uses percentile-based approach to adapt to season dynamics:
    - OWNERSHIP: Bottom X percentile (true differentials)
    - COST: Below X percentile (budget-friendly)
    - FORM: Above X percentile (exclude poor form)
    - ICT: Above X percentile (exclude inactive players)
    
    Returns dict with computed threshold values.
    """
    thresholds = {}
    
    # Ownership: Bottom X percentile (e.g., 25th percentile = pick from bottom 25%)
    if 'selected_by_percent' in df.columns:
        ownership_vals = pd.to_numeric(df['selected_by_percent'], errors='coerce').dropna()
        if len(ownership_vals) > 0:
            thresholds['max_ownership'] = np.percentile(ownership_vals, OWNERSHIP_PERCENTILE)
        else:
            thresholds['max_ownership'] = 10.0  # Fallback
    else:
        thresholds['max_ownership'] = 10.0
    
    # Cost: Below X percentile (e.g., 60th percentile = exclude top 40% expensive players)
    if 'now_cost' in df.columns:
        cost_vals = pd.to_numeric(df['now_cost'], errors='coerce').dropna()
        if len(cost_vals) > 0:
            thresholds['max_cost'] = np.percentile(cost_vals, COST_PERCENTILE)
        else:
            thresholds['max_cost'] = 80.0  # Fallback
    else:
        thresholds['max_cost'] = 80.0
    
    # Form: Above X percentile (e.g., 30th percentile = exclude bottom 30%)
    if 'recent_form' in df.columns:
        form_vals = pd.to_numeric(df['recent_form'], errors='coerce').dropna()
        if len(form_vals) > 0:
            thresholds['min_form'] = np.percentile(form_vals, FORM_PERCENTILE)
        else:
            thresholds['min_form'] = 2.0  # Fallback
    else:
        thresholds['min_form'] = 2.0
    
    # ICT: Above X percentile (e.g., 30th percentile = exclude bottom 30%)
    if 'ict_index' in df.columns:
        ict_vals = pd.to_numeric(df['ict_index'], errors='coerce').dropna()
        if len(ict_vals) > 0:
            thresholds['min_ict'] = np.percentile(ict_vals, ICT_PERCENTILE)
        else:
            thresholds['min_ict'] = 3.0  # Fallback
    else:
        thresholds['min_ict'] = 3.0
    
    print(f"[Dynamic Thresholds] Ownership < {thresholds['max_ownership']:.1f}%, "
          f"Cost < £{thresholds['max_cost']/10:.1f}m, "
          f"Form > {thresholds['min_form']:.2f}, ICT > {thresholds['min_ict']:.2f}")
    
    return thresholds


def select_best_team(df):
    """
    Select the top candidate for each position + 1 wildcard.
    Uses data-driven thresholds computed from current week's player pool.
    Applies filters (Cost, Ownership, Status) dynamically.
    """
    
    # 1. Apply Filtering Logic (if not pre-filtered)
    # Ensure numeric
    cols_to_numeric = ['selected_by_percent', 'now_cost', 'recent_form', 'ict_index']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Availability Filter
    if 'status' in df.columns:
         df = df[~df['status'].isin(['s', 'u', 'n', 'i', 'd'])]
    if 'chance_of_playing_next_round' in df.columns:
        df['chance_of_playing_next_round'] = pd.to_numeric(df['chance_of_playing_next_round'], errors='coerce').fillna(100)
        df = df[df['chance_of_playing_next_round'] >= 75]

    # Calculate dynamic thresholds from current week's player pool
    thresholds = calculate_dynamic_thresholds(df)
    max_ownership = thresholds['max_ownership']
    max_cost = thresholds['max_cost']
    min_form = thresholds['min_form']
    min_ict = thresholds['min_ict']

    # Criteria Filter (using dynamic thresholds)
    df_filtered = df[
        (df['selected_by_percent'] < max_ownership) & 
        (df['now_cost'] < max_cost) & 
        ((df['recent_form'] > min_form) | (df['ict_index'] > min_ict))
    ].copy()
    
    # Fallback if empty: Just ownership filter
    if df_filtered.empty:
        print("[Warning] Primary filters too restrictive, relaxing to ownership-only.")
        df_filtered = df[df['selected_by_percent'] < max_ownership].copy()
        
    if df_filtered.empty:
        return pd.DataFrame() # No valid candidates

    # 2. Selection Logic
    final_picks = []
    df_sorted = df_filtered.sort_values('predicted_points', ascending=False)
    selected_combinations = set()

    # Select Top 1 for each position
    for pos_id in [1, 2, 3, 4]:
        pos_candidates = df_sorted[df_sorted['element_type'] == pos_id]
        if not pos_candidates.empty:
            pick = pos_candidates.iloc[0]
            final_picks.append(pick)
            selected_combinations.add((pick['team'], pick['element_type']))
            df_sorted = df_sorted[df_sorted['element'] != pick['element']]
            
    # Select Wildcard (Best Remaining who is NOT same Team+Position as existing pick)
    if len(final_picks) < 5 and not df_sorted.empty:
        wildcard = None
        for _, row in df_sorted.iterrows():
            if (row['team'], row['element_type']) not in selected_combinations:
                wildcard = row
                break
        
        # Absolute Fallback
        if wildcard is None and not df_sorted.empty:
            wildcard = df_sorted.iloc[0]

        if wildcard is not None:
            final_picks.append(wildcard)
            
    return pd.DataFrame(final_picks)
