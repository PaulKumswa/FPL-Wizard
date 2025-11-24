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

    # Define features and target
    # Features:
    features = ['now_cost', 'selected_by_percent', 'recent_form', 'opponent_strength', 'is_home']
    target = 'total_points'
    
    # Drop rows with missing values
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model MAE: {mae}")
    
    # Save model and features
    os.makedirs('models', exist_ok=True)
    with open('models/fpl_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/model_features.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    print("Model saved to models/fpl_model.pkl")

if __name__ == "__main__":
    train_model()
