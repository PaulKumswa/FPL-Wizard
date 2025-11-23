# FPL Underdog Predictor Walkthrough

## Overview
We have built a machine learning system to predict "underdog" high scorers in the Fantasy Premier League.
"Underdogs" are defined as players with < 10% ownership.

## Components

### 1. Data Collection & Processing
- **Source**: FPL Official API.
- **Script**: `src/data_fetch.py` (existing) and `src/preprocess.py` (new).
- **Process**:
    - Fetched player metadata and historical performance.
    - Calculated "recent form" (average points in last 5 games).
    - Merged with team data.
    - Filtered for players with < 10% ownership.

### 2. Machine Learning Model
- **Script**: `src/train_model.py`.
- **Model**: Random Forest Regressor (`scikit-learn`).
- **Features**: `now_cost`, `selected_by_percent`, `recent_form_points`.
- **Target**: `total_points` (proxy for future performance in this MVP).
- **Performance**: MAE of ~3.91 points.

### 3. Web Application
- **Script**: `src/app.py`.
- **Framework**: Flask.
- **Interface**: Simple HTML table displaying top 20 predicted underdogs.
- **URL**: http://127.0.0.1:5000

## How to Run

1. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Fetch Data**:
   ```powershell
   python -m src.data_fetch --resource fpl_bootstrap --out data/raw/fpl_bootstrap.json
   python -m src.data_fetch --resource fpl_histories --limit 50 --out data/raw/fpl_histories.parquet
   ```

3. **Process Data**:
   ```powershell
   python -m src.preprocess
   ```

4. **Train Model**:
   ```powershell
   python -m src.train_model
   ```

5. **Run Web App**:
   ```powershell
   python -m src.app
   ```
   Open http://127.0.0.1:5000 in your browser.

## Verification Results
- **Model Training**: Successfully trained and saved to `models/fpl_model.pkl`.
- **API**: `/api/predictions` returns JSON data with predicted points.
- **UI**: Displays the table of predictions correctly.
