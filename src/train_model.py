"""
src/train_model.py
Description: Trains machine learning models for FPL points prediction.

This module implements a COMPONENT-BASED prediction approach:
1. Train separate LightGBM classifiers for goals, assists, and clean sheets
2. Each component predicts probability of the outcome occurring
3. Probabilities are aggregated by inference.py using FPL scoring rules

Model Architecture:
- Goal models: LGBMClassifier per position (GKP, DEF, MID, FWD)
- Assist models: LGBMClassifier per position (GKP, DEF, MID, FWD)
- Clean sheet models: LGBMClassifier for GKP, DEF, MID only (FWD gets 0 pts)
- Legacy points model: LGBMRegressor (kept for fallback/comparison)

Validation Strategy:
- Uses TimeSeriesSplit (5 folds) to ensure temporal integrity
- Model is always validated on "future" data it hasn't seen
- Final model is trained on ALL data after cross-validation metrics are computed
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
import lightgbm as lgb
import pickle
import pickle
import os
import json
from datetime import datetime
from src.config import FEATURE_CONFIGS
from src.scoring_constants import CLEAN_SHEET_POSITIONS, COMPONENT_TARGETS

# Number of cross-validation folds
N_SPLITS = 5

# Component target column mapping
COMPONENT_TARGET_COLUMNS = {
    'goal': 'target_goal',
    'assist': 'target_assist',
    'cleansheet': 'target_clean_sheet'
}

METRICS_FILE = 'data/history/model_metrics.json'
FEATURE_IMPORTANCE_FILE = 'data/history/feature_importance.json'

def log_metrics(metrics_data):
    """Log model metrics to a persistent JSON file."""
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    
    current_data = []
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                current_data = json.load(f)
        except json.JSONDecodeError:
            pass  # Start fresh if corrupt
            
    # Add timestamp
    metrics_data['timestamp'] = datetime.now().isoformat()
    current_data.append(metrics_data)
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(current_data, f, indent=2)

def save_feature_importance(importance_data):
    """Save feature importance to a JSON file (overwrites with latest)."""
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_FILE), exist_ok=True)
    
    with open(FEATURE_IMPORTANCE_FILE, 'w') as f:
        json.dump(importance_data, f, indent=2)
    print(f"Feature importance saved to {FEATURE_IMPORTANCE_FILE}")


def train_component_models(all_importances=None):
    """Train component classifiers (goals, assists, clean sheets) for each position."""
    
    # Load processed data
    try:
        df = pd.read_csv('data/processed/train_data.csv')
    except FileNotFoundError:
        print("Error: Training data not found. Run src/preprocess.py first.")
        return

    feature_configs = FEATURE_CONFIGS
    os.makedirs('models', exist_ok=True)
    
    # Train component models for each position
    for pos_id, config in feature_configs.items():
        pos_name = config['name']
        print(f"\n{'='*60}")
        print(f"Training Component Models for {pos_name}")
        print(f"{'='*60}")
        
        # Initialize position in importance dict
        if all_importances is not None:
            if pos_name not in all_importances['models']:
                all_importances['models'][pos_name] = {}
        
        # Filter data for this position
        pos_df = df[df['element_type'] == pos_id].copy()
        
        if pos_df.empty:
            print(f"Warning: No data found for {pos_name}. Skipping.")
            continue
        
        features = config['features']
        
        # Drop rows with missing values in relevant features
        target_cols = list(COMPONENT_TARGET_COLUMNS.values())
        pos_df = pos_df.dropna(subset=features + target_cols)
        
        # CRITICAL: Sort by round (gameweek) for proper temporal ordering
        if 'round' in pos_df.columns:
            pos_df = pos_df.sort_values('round')
        
        X = pos_df[features]
        
        # Train each component model
        for component in COMPONENT_TARGETS:
            # Skip clean sheet for FWD (they get 0 points anyway)
            if component == 'cleansheet' and pos_id not in CLEAN_SHEET_POSITIONS:
                print(f"  Skipping {component} for {pos_name} (no points awarded)")
                continue
            
            target_col = COMPONENT_TARGET_COLUMNS[component]
            y = pos_df[target_col]
            
            # Check class distribution
            pos_rate = y.mean()
            print(f"\n--- Training {component.upper()} model for {pos_name} ---")
            print(f"  Positive rate: {pos_rate:.2%} ({int(y.sum())}/{len(y)} samples)")
            
            # Skip if too few positive samples
            if y.sum() < 10:
                print(f"  Skipping: insufficient positive samples")
                continue
            
            # TimeSeriesSplit Cross-Validation
            tscv = TimeSeriesSplit(n_splits=N_SPLITS)
            fold_aucs = []
            fold_accs = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Skip fold if validation set has no positive samples
                if y_val.sum() == 0:
                    continue
                
                model = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.05,
                    min_child_samples=max(5, config['min_samples_leaf']),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    class_weight='balanced'  # Handle class imbalance
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                preds_proba = model.predict_proba(X_val)[:, 1]
                preds = model.predict(X_val)
                
                try:
                    auc = roc_auc_score(y_val, preds_proba)
                    fold_aucs.append(auc)
                except ValueError:
                    pass  # Single class in fold
                
                acc = accuracy_score(y_val, preds)
                fold_accs.append(acc)
            
            # Report CV results
            if fold_aucs:
                print(f"  CV AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
            if fold_accs:
                print(f"  CV Accuracy: {np.mean(fold_accs):.4f}")
            
            # Log metrics
            log_metrics({
                'model_type': 'component',
                'position': pos_name,
                'component': component,
                'cv_auc_mean': float(np.mean(fold_aucs)) if fold_aucs else None,
                'cv_auc_std': float(np.std(fold_aucs)) if fold_aucs else None,
                'cv_accuracy': float(np.mean(fold_accs)) if fold_accs else None,
                'samples': len(X)
            })
            
            # Train final model on ALL data
            print(f"  Training final model on {len(X)} samples...")
            final_model = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                min_child_samples=max(5, config['min_samples_leaf']),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                class_weight='balanced'
            )
            final_model.fit(X, y)
            
            # Capture feature importance
            if all_importances is not None:
                importance = final_model.feature_importances_
                # Normalize importance to sum to 100 for better display
                total_importance = importance.sum()
                if total_importance > 0:
                    importance = (importance / total_importance) * 100
                    
                feat_imp = {feat: float(imp) for feat, imp in zip(features, importance)}
                # Sort descending
                feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
                all_importances['models'][pos_name][component] = feat_imp
            
            # Save model
            model_path = f'models/fpl_{component}_model_{pos_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
            print(f"  Saved: {model_path}")


def train_legacy_model(all_importances=None):
    """Train the legacy total_points regressor (kept for fallback/comparison)."""
    
    try:
        df = pd.read_csv('data/processed/train_data.csv')
    except FileNotFoundError:
        print("Error: Training data not found. Run src/preprocess.py first.")
        return

    feature_configs = FEATURE_CONFIGS
    target = 'total_points'
    os.makedirs('models', exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Training Legacy Points Models (Fallback)")
    print(f"{'='*60}")
    
    for pos_id, config in feature_configs.items():
        pos_name = config['name']
        print(f"\n--- Training {pos_name} Legacy Model ---")
        
        # Initialize position in importance dict if not exists
        if all_importances is not None:
            if pos_name not in all_importances['models']:
                all_importances['models'][pos_name] = {}
        
        pos_df = df[df['element_type'] == pos_id].copy()
        
        if pos_df.empty:
            print(f"Warning: No data found for {pos_name}. Skipping.")
            continue
            
        features = config['features']
        pos_df = pos_df.dropna(subset=features + [target])
        
        if 'round' in pos_df.columns:
            pos_df = pos_df.sort_values('round')
        
        X = pos_df[features]
        y = pos_df[target]
        
        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                min_child_samples=config['min_samples_leaf'],
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            fold_mae = mean_absolute_error(y_val, preds)
            fold_maes.append(fold_mae)
        
        print(f"  CV MAE: {np.mean(fold_maes):.4f} (+/- {np.std(fold_maes):.4f})")
        
        # Log metrics
        log_metrics({
            'model_type': 'legacy',
            'position': pos_name,
            'component': None,
            'cv_mae_mean': float(np.mean(fold_maes)) if fold_maes else None,
            'cv_mae_std': float(np.std(fold_maes)) if fold_maes else None,
            'samples': len(X)
        })
        
        # Train final model
        final_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            min_child_samples=config['min_samples_leaf'],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        final_model.fit(X, y)
        
        # Capture feature importance
        if all_importances is not None:
            importance = final_model.feature_importances_
            # Normalize to 100
            total_importance = importance.sum()
            if total_importance > 0:
                importance = (importance / total_importance) * 100
            
            feat_imp = {feat: float(imp) for feat, imp in zip(features, importance)}
            # Sort descending
            feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            all_importances['models'][pos_name]['legacy'] = feat_imp
        
        # Save model (legacy naming for backward compatibility)
        model_path = f'models/fpl_model_{pos_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
            
        # Save features list
        feature_path = f'models/model_features_{pos_name}.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
            
        print(f"  Saved: {model_path}")


def train_model():
    """Main training function - trains both component and legacy models."""
    print("\n" + "="*70)
    print("FPL Model Training - Component-Based Approach")
    print("="*70)
    
    # Initialize dictionary to hold feature importance
    all_importances = {
        'timestamp': datetime.now().isoformat(),
        'models': {} 
    }
    
    # Train component classifiers (primary approach)
    train_component_models(all_importances)
    
    # Train legacy regressor (fallback)
    train_legacy_model(all_importances)
    
    # Save feature importance to file
    save_feature_importance(all_importances)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    train_model()
