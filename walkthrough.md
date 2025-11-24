# FPL Next Gameweek Predictor Walkthrough

## Overview
We have updated the system to predict the **Top 5 Point Scorers** for the **Next Gameweek**.
It uses historical match data to learn the relationship between player form, opponent difficulty, and points.

## Components

### 1. Data Collection & Processing
- **Source**: FPL Official API.
- **Scripts**: `src/data_fetch.py`, `src/preprocess.py`.
- **New Data**: `fpl_fixtures.json` (for opponent difficulty and schedule).
- **Process**:
    - **Training Data**: Historical matches with `recent_form` (avg points last 5 games) and `opponent_strength`.
    - **Inference Data**: All players for the *Next Gameweek* with their upcoming opponent and current form.
    - **Metadata**: Tracks Current and Next Gameweek IDs.

### 2. Machine Learning Model
- **Script**: `src/train_model.py`.
- **Model**: Random Forest Regressor (`scikit-learn`).
- **Features**: `now_cost`, `selected_by_percent`, `recent_form`, `opponent_strength`, `is_home`.
- **Target**: `total_points` (in a specific match).
- **Performance**: MAE of ~1.71 points.

### 3. Web Application
- **Script**: `src/app.py`.
- **Framework**: Flask.
- **Interface**:
    - Displays **Current Gameweek** and **Next Gameweek**.
    - Shows a table of **Top 5 Predicted Scorers**.
    - Includes **Next Opponent** information.
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
   python -m src.data_fetch --resource fpl_fixtures --out data/raw/fpl_fixtures.json
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
- **Model Training**: Successfully trained (MAE: ~1.71).
- **API**: `/api/predictions` returns Top 5 players with next opponent info.
- **UI**: Displays Gameweek info and the predictions table correctly.
