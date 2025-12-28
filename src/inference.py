"""
src/inference.py
Description: Provides logic for generating FPL points predictions using trained machine learning models.
Key functionality:
- `load_models`: Loads the serialized RandomForest models for each position.
- `predict_points`: Applies the models to a DataFrame of player features to predict points.
- `select_best_team`: Implements the selection algorithm to pick the top 5 players (1 GK, 1 DEF, 1 MID, 1 FWD, 1 Wildcard)
  based on predicted points, cost constraints, ownership limits, and form/ICT filters.
"""
import pandas as pd
import pickle
import os
import json
from src.config import FEATURE_CONFIGS, POSITION_MAP_REV, POSITION_MAP, MAX_COST, MAX_OWNERSHIP, MIN_FORM, MIN_ICT

def load_models(model_dir='models'):
    """Load trained models from the specified directory."""
    models = {}
    
    for name, _ in POSITION_MAP_REV.items():
        model_path = os.path.join(model_dir, f'fpl_model_{name}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            print(f"Warning: Model for {name} not found at {model_path}")
            
    return models

def predict_points(df, models):
    """
    Apply models to the dataframe to generate 'predicted_points'.
    Returns the modified dataframe.
    """
    df['predicted_points'] = 0.0
    
    for pos_name, model in models.items():
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
        
        X = X[features] # Ensure correct order and selection
        
        preds = model.predict(X)
        df.loc[pos_mask, 'predicted_points'] = preds
        
    return df

def select_best_team(df):
    """
    Select the top candidate for each position + 1 wildcard.
    Applies filters (Cost, Ownership, Status) if they haven't been applied already.
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

    # Criteria Filter
    df_filtered = df[
        (df['selected_by_percent'] < MAX_OWNERSHIP) & 
        (df['now_cost'] < MAX_COST) & 
        ((df['recent_form'] > MIN_FORM) | (df['ict_index'] > MIN_ICT))
    ].copy()
    
    # Fallback if empty: Just ownership
    if df_filtered.empty:
        df_filtered = df[df['selected_by_percent'] < MAX_OWNERSHIP].copy()
        
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
