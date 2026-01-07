"""
src/train_model.py
Description: Responsible for training the machine learning models used for points prediction.
It loads the processed `train_data.csv`, splits it by player position (GKP, DEF, MID, FWD), and trains a 
LightGBM gradient boosting model for each position. The trained models and their feature lists are saved to the `models/` directory.

LightGBM is used instead of RandomForest because:
- Better at capturing weak, additive signals (important for noisy FPL points)
- Faster training (5-10x) for equivalent or better accuracy
- Smaller model files
- Sequential tree building corrects errors iteratively

Validation Strategy:
- Uses TimeSeriesSplit (5 folds) to ensure temporal integrity
- Model is always validated on "future" data it hasn't seen
- Prevents data leakage that can occur with random splits on time-series data
- Final model is trained on ALL data after cross-validation metrics are computed
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import pickle
import os
from src.config import FEATURE_CONFIGS

# Number of cross-validation folds
N_SPLITS = 5

def train_model():
    # Load processed data
    try:
        df = pd.read_csv('data/processed/train_data.csv')
    except FileNotFoundError:
        print("Error: Training data not found. Run src/preprocess.py first.")
        return

    # Use Feature Configs from src.config
    feature_configs = FEATURE_CONFIGS
    
    target = 'total_points'
    os.makedirs('models', exist_ok=True)
    
    # Train a model for each position
    for pos_id, config in feature_configs.items():
        print(f"\n--- Training {config['name']} Model (LightGBM + TimeSeriesCV) ---")
        
        # Filter data for this position
        pos_df = df[df['element_type'] == pos_id].copy()
        
        if pos_df.empty:
            print(f"Warning: No data found for {config['name']}. Skipping.")
            continue
            
        features = config['features']
        
        # Drop rows with missing values in relevant features
        pos_df = pos_df.dropna(subset=features + [target])
        
        # CRITICAL: Sort by round (gameweek) for proper temporal ordering
        # TimeSeriesSplit assumes data is ordered chronologically
        if 'round' in pos_df.columns:
            pos_df = pos_df.sort_values('round')
        
        X = pos_df[features]
        y = pos_df[target]
        
        # TimeSeriesSplit Cross-Validation
        # This ensures we always train on past and validate on future
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        
        fold_maes = []
        print(f"  Running {N_SPLITS}-fold TimeSeriesSplit CV...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
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
            
            # Evaluate on validation fold
            preds = model.predict(X_val)
            fold_mae = mean_absolute_error(y_val, preds)
            fold_maes.append(fold_mae)
        
        # Report CV results
        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        print(f"  CV MAE: {mean_mae:.4f} (+/- {std_mae:.4f})")
        print(f"  Fold MAEs: {[round(m, 4) for m in fold_maes]}")
        
        # Train final model on ALL data
        # This gives the most robust model for production inference
        print(f"  Training final model on all {len(X)} samples...")
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
        
        # Save model
        model_path = f'models/fpl_model_{config["name"]}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
            
        # Save features list for this model
        feature_path = f'models/model_features_{config["name"]}.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
            
        print(f"  Saved to {model_path}")

if __name__ == "__main__":
    train_model()

