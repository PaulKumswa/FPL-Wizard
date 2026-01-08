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
    MAX_COST, MAX_OWNERSHIP, MIN_FORM, MIN_ICT,
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


def calculate_confidence(p_goal, p_assist, p_cs):
    """
    Calculate prediction confidence score (0-100) based on component probabilities.
    
    Confidence is based on how "decisive" probabilities are:
    - Probabilities near 0 or 1 = high confidence (model is certain)
    - Probabilities near 0.5 = low confidence (model is uncertain)
    
    Returns confidence as a percentage (0-100).
    """
    # Calculate decisiveness for each component: |p - 0.5| * 2
    # This gives 0 when p=0.5 (uncertain) and 1 when p=0 or p=1 (certain)
    decisiveness = []
    
    # Handle scalar and array inputs
    if hasattr(p_goal, '__iter__'):
        # Array input (vectorized)
        decisiveness_goal = np.abs(np.array(p_goal) - 0.5) * 2
        decisiveness_assist = np.abs(np.array(p_assist) - 0.5) * 2
        decisiveness_cs = np.abs(np.array(p_cs) - 0.5) * 2
        
        # Average across components, multiply by 100 for percentage
        confidence = (decisiveness_goal + decisiveness_assist + decisiveness_cs) / 3 * 100
    else:
        # Scalar input
        d_goal = abs(p_goal - 0.5) * 2
        d_assist = abs(p_assist - 0.5) * 2
        d_cs = abs(p_cs - 0.5) * 2
        
        confidence = (d_goal + d_assist + d_cs) / 3 * 100
    
    return confidence


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
    - confidence_score: Model confidence (0-100%)
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
                p_goal = np.zeros(len(X_features))
            
            # Assist probability
            if 'assist' in pos_components:
                p_assist = pos_components['assist'].predict_proba(X_features)[:, 1]
                df.loc[pos_mask, 'p_assist'] = p_assist
            else:
                p_assist = np.zeros(len(X_features))
            
            # Clean sheet probability
            if 'cleansheet' in pos_components and pos_id in CLEAN_SHEET_POSITIONS:
                p_cs = pos_components['cleansheet'].predict_proba(X_features)[:, 1]
                df.loc[pos_mask, 'p_cleansheet'] = p_cs
            else:
                p_cs = np.zeros(len(X_features))
            
            # Calculate confidence score
            confidence = calculate_confidence(p_goal, p_assist, p_cs)
            df.loc[pos_mask, 'confidence_score'] = confidence
            
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
    
    # Ownership: Bottom X percentile
    # SAFETY: Ensure at least 10% ownership is allowed (User feedback)
    if 'selected_by_percent' in df.columns:
        ownership_vals = pd.to_numeric(df['selected_by_percent'], errors='coerce').dropna()
        if len(ownership_vals) > 0:
            p_val = np.percentile(ownership_vals, OWNERSHIP_PERCENTILE)
            thresholds['max_ownership'] = max(p_val, 10.0) 
        else:
            thresholds['max_ownership'] = 10.0
    else:
        thresholds['max_ownership'] = 10.0
    
    # Cost: Below X percentile
    # SAFETY: Ensure at least £7.0m is allowed (User feedback)
    if 'now_cost' in df.columns:
        cost_vals = pd.to_numeric(df['now_cost'], errors='coerce').dropna()
        if len(cost_vals) > 0:
            c_val = np.percentile(cost_vals, COST_PERCENTILE)
            thresholds['max_cost'] = max(c_val, 70.0) 
        else:
            thresholds['max_cost'] = 80.0
    else:
        thresholds['max_cost'] = 80.0
    
    # Form: Above X percentile
    if 'recent_form' in df.columns:
        form_vals = pd.to_numeric(df['recent_form'], errors='coerce').dropna()
        if len(form_vals) > 0:
            thresholds['min_form'] = np.percentile(form_vals, FORM_PERCENTILE)
        else:
            thresholds['min_form'] = 2.0
    else:
        thresholds['min_form'] = 2.0
    
    # ICT: Above X percentile
    if 'ict_index' in df.columns:
        ict_vals = pd.to_numeric(df['ict_index'], errors='coerce').dropna()
        if len(ict_vals) > 0:
            thresholds['min_ict'] = np.percentile(ict_vals, ICT_PERCENTILE)
        else:
            thresholds['min_ict'] = 3.0
    else:
        thresholds['min_ict'] = 3.0
        
    # Predicted Points: Ensure "worthwhile" picks
    # User feedback: Target ~6pts, but allow higher cap
    if 'predicted_points' in df.columns:
        pred_vals = pd.to_numeric(df['predicted_points'], errors='coerce').dropna()
        if len(pred_vals) > 0:
            # P92 should be closer to 6.0 pts based on analysis
            p92 = np.percentile(pred_vals, 92)
            # Floor 5.5, Cap 8.0 (raised from 6.0)
            thresholds['min_predicted'] = min(max(p92, 5.5), 8.0)
        else:
            thresholds['min_predicted'] = 4.0
    else:
        thresholds['min_predicted'] = 4.0

    print(f"[Dynamic Thresholds] Ownership < {thresholds['max_ownership']:.1f}%, "
          f"Cost < £{thresholds['max_cost']/10:.1f}m, "
          f"Form > {thresholds['min_form']:.2f}, ICT > {thresholds['min_ict']:.2f}, "
          f"Pred > {thresholds['min_predicted']:.1f}")
    
    return thresholds


def select_best_team(df):
    """
    Select the top candidate for each position + 1 wildcard.
    Uses data-driven thresholds computed from current week's player pool.
    
    CONFIDENCE-BASED SELECTION (Jan 2026):
    Prioritizes reliable predictions by filtering on confidence score first,
    then relaxing confidence thresholds before ownership/cost constraints.
    
    Ensures a valid pick for every position by relaxing filters locally if needed.
    """
    
    # 1. Ensure numeric columns
    cols_to_numeric = ['selected_by_percent', 'now_cost', 'recent_form', 'ict_index', 'predicted_points', 'confidence_score']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Default confidence if not present
    if 'confidence_score' not in df.columns:
        df['confidence_score'] = 50.0
    
    # 2. Availability Filter (Hard Constraint)
    # We never want to pick injured players, even if we have to fallback
    if 'status' in df.columns:
         df = df[~df['status'].isin(['s', 'u', 'n', 'i', 'd'])]
    if 'chance_of_playing_next_round' in df.columns:
        df.loc[:, 'chance_of_playing_next_round'] = pd.to_numeric(df['chance_of_playing_next_round'], errors='coerce').fillna(100)
        df = df[df['chance_of_playing_next_round'] >= 75]

    # 3. Calculate Global Dynamic Thresholds
    thresholds = calculate_dynamic_thresholds(df)
    max_own_global = thresholds['max_ownership']
    max_cost_global = thresholds['max_cost']
    min_form_global = thresholds['min_form']
    min_ict_global = thresholds['min_ict']
    min_pred_global = thresholds['min_predicted']
    
    # Confidence thresholds for fail upward logic
    HIGH_CONFIDENCE = 60.0    # Very reliable predictions
    MEDIUM_CONFIDENCE = 40.0  # Acceptable uncertainty
    MIN_POINTS_TARGET = 6.0   # Target "worthwhile" picks

    final_picks = []
    selected_ids = set()
    selected_team_pos = set() # Track (team, pos) to avoid duplicates

    # 4. Select Top 1 for each position (1:GKP, 2:DEF, 3:MID, 4:FWD)
    for pos_id in [1, 2, 3, 4]:
        # Get all available players for this position
        pos_pool = df[df['element_type'] == pos_id].copy()
        
        if pos_pool.empty:
            print(f"[Warning] No players found for Position {pos_id}")
            continue

        # === CONFIDENCE-BASED FAIL UPWARD LOGIC ===
        # Priority: High confidence + 6pts > Medium confidence + 6pts > Any confidence + 6pts
        # Then relax ownership/cost constraints as before
        
        # Level 1A: Perfect Underdog with HIGH Confidence (≥60%)
        # The ideal pick: reliable prediction, low ownership, affordable, high points
        candidates = pos_pool[
            (pos_pool['confidence_score'] >= HIGH_CONFIDENCE) &
            (pos_pool['selected_by_percent'] < max_own_global) & 
            (pos_pool['now_cost'] < max_cost_global) & 
            ((pos_pool['recent_form'] > min_form_global) | (pos_pool['ict_index'] > min_ict_global)) &
            (pos_pool['predicted_points'] >= MIN_POINTS_TARGET)
        ]

        # Level 1B: Perfect Underdog with MEDIUM Confidence (≥40%)
        if candidates.empty:
            candidates = pos_pool[
                (pos_pool['confidence_score'] >= MEDIUM_CONFIDENCE) &
                (pos_pool['selected_by_percent'] < max_own_global) & 
                (pos_pool['now_cost'] < max_cost_global) & 
                ((pos_pool['recent_form'] > min_form_global) | (pos_pool['ict_index'] > min_ict_global)) &
                (pos_pool['predicted_points'] >= MIN_POINTS_TARGET)
            ]
        
        # Level 1C: Perfect Underdog ANY Confidence (relax confidence, keep all other filters)
        if candidates.empty:
            candidates = pos_pool[
                (pos_pool['selected_by_percent'] < max_own_global) & 
                (pos_pool['now_cost'] < max_cost_global) & 
                ((pos_pool['recent_form'] > min_form_global) | (pos_pool['ict_index'] > min_ict_global)) &
                (pos_pool['predicted_points'] >= MIN_POINTS_TARGET)
            ]

        # Level 2: Relax Ownership (Value Pick) - FAIL UPWARD
        # Prioritize points + budget over ownership, prefer high confidence
        if candidates.empty:
            candidates = pos_pool[
                (pos_pool['now_cost'] < max_cost_global) & 
                (pos_pool['predicted_points'] >= MIN_POINTS_TARGET)
            ]
            if not candidates.empty:
                # Sort by confidence (descending), then points, then ownership
                candidates = candidates.sort_values(
                    ['confidence_score', 'predicted_points', 'selected_by_percent'], 
                    ascending=[False, False, True]
                )

        # Level 3: Relax Cost (Premium Differential) - FAIL UPWARD
        # Prioritize points + differential status over budget
        if candidates.empty:
            candidates = pos_pool[
                (pos_pool['selected_by_percent'] < max_own_global) & 
                (pos_pool['predicted_points'] >= MIN_POINTS_TARGET)
            ]
        
        # Level 4: Relax Both Ownership & Cost (Just Points) - FAIL UPWARD
        # Prioritize points above all else, prefer high confidence
        if candidates.empty:
            candidates = pos_pool[
                (pos_pool['predicted_points'] >= MIN_POINTS_TARGET)
            ]

        # Level 5: Relax Points Target (Last Resort) - FAIL DOWNWARD
        # If no one meets the 6 point target, use dynamic threshold
        if candidates.empty:
            candidates = pos_pool[
                (pos_pool['selected_by_percent'] < 15.0) & 
                (pos_pool['predicted_points'] >= min_pred_global)
            ]
            
            if candidates.empty:
                # Try 4.0 pts floor, then 3.0
                candidates = pos_pool[(pos_pool['predicted_points'] >= 4.0)]
            
            if candidates.empty:
                candidates = pos_pool[(pos_pool['predicted_points'] >= 3.0)]

        # Pick Best: Sort by Confidence (desc), then Points (desc)
        if not candidates.empty:
            best_pick = candidates.sort_values(
                ['confidence_score', 'predicted_points'], 
                ascending=[False, False]
            ).iloc[0]
            final_picks.append(best_pick)
            selected_ids.add(best_pick['element'])
            selected_team_pos.add((best_pick['team'], best_pick['element_type']))
            
    # 5. Wildcard Selection
    # Pick best remaining player from ANY position who fits strict filters
    # Uses same confidence-based logic as position selection
    
    # Exclude already selected
    remaining_pool = df[~df['element'].isin(selected_ids)].copy()
    
    # Filter Remaining Pool (Strict): High Confidence + 6 pts + Underdog
    wildcard_candidates = remaining_pool[
        (remaining_pool['confidence_score'] >= HIGH_CONFIDENCE) &
        (remaining_pool['selected_by_percent'] < max_own_global) & 
        (remaining_pool['now_cost'] < max_cost_global) & 
        (remaining_pool['predicted_points'] >= MIN_POINTS_TARGET)
    ]
    
    # Relax Confidence to Medium
    if wildcard_candidates.empty:
        wildcard_candidates = remaining_pool[
            (remaining_pool['confidence_score'] >= MEDIUM_CONFIDENCE) &
            (remaining_pool['selected_by_percent'] < max_own_global) & 
            (remaining_pool['now_cost'] < max_cost_global) & 
            (remaining_pool['predicted_points'] >= MIN_POINTS_TARGET)
        ]
    
    # Relax Confidence completely
    if wildcard_candidates.empty:
        wildcard_candidates = remaining_pool[
            (remaining_pool['selected_by_percent'] < max_own_global) & 
            (remaining_pool['now_cost'] < max_cost_global) & 
            (remaining_pool['predicted_points'] >= MIN_POINTS_TARGET)
        ]
    
    # Relax Ownership/Cost (Just high confidence + points)
    if wildcard_candidates.empty:
        wildcard_candidates = remaining_pool[
            (remaining_pool['predicted_points'] >= MIN_POINTS_TARGET)
        ]
        
    # Final Fallback (lower points threshold)
    if wildcard_candidates.empty:
        wildcard_candidates = remaining_pool[
             (remaining_pool['predicted_points'] >= min_pred_global)
        ]
        
    if wildcard_candidates.empty:
        wildcard_candidates = remaining_pool[
             (remaining_pool['predicted_points'] >= 4.0)
        ]
        
    # Sort by confidence (desc), then points (desc)
    wildcard_candidates = wildcard_candidates.sort_values(
        ['confidence_score', 'predicted_points'], 
        ascending=[False, False]
    )
    
    # Try to find one with different Team+Pos to add variety (optional but good)
    wildcard_pick = None
    for _, row in wildcard_candidates.iterrows():
        if (row['team'], row['element_type']) not in selected_team_pos:
            wildcard_pick = row
            break
            
    # Fallback if variation impossible
    if wildcard_pick is None and not wildcard_candidates.empty:
        wildcard_pick = wildcard_candidates.iloc[0]
        
    if wildcard_pick is not None:
        final_picks.append(wildcard_pick)

    return pd.DataFrame(final_picks)
