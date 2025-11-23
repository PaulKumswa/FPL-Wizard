from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

def load_model_and_data():
    try:
        with open('models/fpl_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/model_features.pkl', 'rb') as f:
            features = pickle.load(f)
            
        df = pd.read_csv('data/processed/fpl_data.csv')
        
        return model, features, df
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    model, features, df = load_model_and_data()
    
    if model is None:
        return jsonify({'error': 'Model or data not found'}), 500
    
    # Filter for underdogs (ownership < 10%)
    # We already filtered in preprocess, but let's be safe or if we want to adjust
    underdogs = df[df['selected_by_percent'] < 10.0].copy()
    
    # Make predictions
    # Ensure features exist
    for feature in features:
        if feature not in underdogs.columns:
            return jsonify({'error': f'Missing feature: {feature}'}), 500
            
    # Drop rows with missing values in features
    underdogs = underdogs.dropna(subset=features)
    
    predictions = model.predict(underdogs[features])
    underdogs['predicted_points'] = predictions
    
    # Sort by predicted points descending
    top_underdogs = underdogs.sort_values('predicted_points', ascending=False).head(20)
    
    # Select columns to display
    display_cols = ['web_name', 'name', 'now_cost', 'selected_by_percent', 'predicted_points']
    result = top_underdogs[display_cols].to_dict(orient='records')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
