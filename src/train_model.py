import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import os

def train_model():
    # Load processed data
    try:
        df = pd.read_csv('data/processed/train_data.csv')
    except FileNotFoundError:
        print("Error: Training data not found. Run src/preprocess.py first.")
        return

    # Define Feature Sets per Position
    # 1: GKP, 2: DEF, 3: MID, 4: FWD
    feature_configs = {
        1: {
            'name': 'GKP',
            'features': [
                'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
                'recent_clean_sheets', 'recent_saves', 'recent_goals_conceded', 'recent_penalties_saved'
            ],
            'min_samples_leaf': 5  # Strict regularization for sparse data
        },
        2: {
            'name': 'DEF',
            'features': [
                'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
                'recent_clean_sheets', 'recent_goals_conceded', 'recent_assists', 'recent_goals_scored',
                'recent_threat', 'recent_influence'
            ],
            'min_samples_leaf': 1  # Standard
        },
        3: {
            'name': 'MID',
            'features': [
                'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
                'recent_goals_scored', 'recent_assists', 'recent_clean_sheets', 
                'recent_creativity', 'recent_threat', 'recent_influence'
            ],
            'min_samples_leaf': 1  # Standard
        },
        4: {
            'name': 'FWD',
            'features': [
                'now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home',
                'recent_goals_scored', 'recent_assists', 
                'recent_threat', 'recent_influence'
            ],
            'min_samples_leaf': 3  # Moderate regularization
        }
    }
    
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
            n_estimators=100, 
            min_samples_leaf=config['min_samples_leaf'],
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
