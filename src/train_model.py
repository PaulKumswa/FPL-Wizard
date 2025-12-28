"""
src/train_model.py
Description: Responsible for training the machine learning models used for points prediction.
It loads the processed `train_data.csv`, splits it by player position (GKP, DEF, MID, FWD), and trains a 
RandomForestRegressor for each position. The trained models and their feature lists are saved to the `models/` directory.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import os
from src.config import FEATURE_CONFIGS

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
        print(f"\n--- Training {config['name']} Model ---")
        
        # Filter data for this position
        pos_df = df[df['element_type'] == pos_id].copy()
        
        if pos_df.empty:
            print(f"Warning: No data found for {config['name']}. Skipping.")
            continue
            
        features = config['features']
        
        # Drop rows with missing values in relevant features
        pos_df = pos_df.dropna(subset=features + [target])
        
        X = pos_df[features]
        y = pos_df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with specific regularization
        model = RandomForestRegressor(
            n_estimators=200, 
            min_samples_leaf=config['min_samples_leaf'],
            max_depth=15,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        if not X_test.empty:
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            print(f"{config['name']} Model MAE: {mae:.4f}")
        else:
            print(f"{config['name']} Model: Not enough data for test set.")
        
        # Save model
        model_path = f'models/fpl_model_{config["name"]}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save features list for this model
        feature_path = f'models/model_features_{config["name"]}.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
            
        print(f"Saved to {model_path}")

if __name__ == "__main__":
    train_model()
