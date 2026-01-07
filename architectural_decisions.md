# FPL Predictor - Architectural & Design Decisions

This document serves as the single source of truth for the architectural decisions, data strategies, and model configurations used in the FPL Predictor project.

## 1. Data Architecture

### 1.1 Data Sources
*   **FPL Official API**: The primary source for player stats, fixtures, and gameweek info.
    *   *Endpoints*: `bootstrap-static`, `fixtures`, `element-summary/{id}`.
*   **Understat**: Secondary source for advanced metrics (Expected Goals - xG, Expected Assists - xA).
    *   *Method*: Scraped/API via `understatapi`.
    *   *Granularity*: Match-level team analytics and player-level data.

### 1.2 Data Pipeline Strategy
The pipeline follows a strict extraction-transformation-loading (ETL) pattern:
1.  **Fetch**: Raw JSON/Parquet files are downloaded to `data/raw/` to ensure reproducibility and offline debugging.
2.  **Preprocess**:
    *   **Merging Strategy**: Understat Team xG/xGA is merged onto FPL Player Histories based on `(Match Date, Team Name)`.
    *   *Rationale*: This allows us to use team-level performance (e.g., "Man City creates a lot of chances") to predict individual returns, even for players who didn't score in a specific game.
3.  **Feature Engineering**:
    *   **Rolling Averages**: Calculated over a **5-Gameweek Window**.
    *   **Lagging**: Metrics are lagged by 1 gameweek (e.g., using GW1-5 stats to predict GW6). This prevents data leakage.

## 2. Model Architecture

### 2.1 Algorithm
*   **Type**: LightGBM Gradient Boosting (`lightgbm.LGBMRegressor`).
*   *Rationale*: LightGBM was chosen over RandomForest (Jan 2026) because:
    *   **Better for weak signals**: FPL points are noisy and composed of many small additive factors. Gradient boosting's sequential error correction captures these subtle patterns better than RF's parallel averaging.
    *   **Faster training**: 5-10x faster than RandomForest, important for CI/CD pipeline efficiency.
    *   **Smaller models**: ~100KB vs ~10MB per position, reducing storage and load times.
    *   **Proven performance**: Gradient boosting dominates tabular data regression tasks.

### 2.2 Model Configuration (Updated Jan 2026)
LightGBM hyperparameters are tuned for generalization on noisy FPL data:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **n_estimators** | `200` | Sufficient trees for convergence with low learning rate. |
| **max_depth** | `10` | Shallower trees work better with LightGBM's leaf-wise growth strategy. |
| **learning_rate** | `0.05` | Slow learning for better generalization; prevents overfitting. |
| **subsample** | `0.8` | Row sampling adds stochasticity to reduce variance. |
| **colsample_bytree** | `0.8` | Feature sampling per tree for regularization. |
| **min_child_samples** | `3-5` | Position-specific (from config); ensures leaf nodes represent player clusters, not individuals. |

### 2.3 Feature Selection
Models are trained independently for each position (`element_type`) to capture unique positional requirements.

#### Goalkeepers (GKP) & Defenders (DEF)
*   **Focus**: Defensive solidity and accumulation.
*   **Key Features**:
    *   `recent_team_xga`: Team Expected Goals Against (Proxy for detailed defensive strength).
    *   `recent_clean_sheets`, `recent_saves`.
    *   `opponent_strength`: Difficulty of the upcoming fixture.

#### Midfielders (MID) & Forwards (FWD)
*   **Focus**: Attacking returns (Goals/Assists) and involvement.
*   **Key Features**:
    *   `recent_team_xg`: Team Expected Goals (Proxy for team service/dominance).
    *   `recent_expected_goals` (xG): Individual quality of chances.
    *   `recent_expected_assists` (xA): Individual creativity.
    *   `recent_form`: Generic FPL form.

## 3. Training & Inference

### 3.1 Validation Strategy (Updated Jan 2026)
*   **Method**: TimeSeriesSplit with 5 folds (replaces random 80/20 split).
*   *Rationale*: FPL data is inherently temporal. Random splits can leak future data patterns into training, giving overly optimistic metrics. TimeSeriesSplit ensures each validation fold only contains data *after* the training period.
*   **Process**:
    1. Data is sorted by `round` (gameweek) for proper chronological order.
    2. 5 rolling splits are made: train on early data, validate on later data.
    3. CV MAE is reported with standard deviation for stability assessment.
    4. Final production model is trained on ALL data after CV metrics are computed.

*   **Target Variable**: `total_points` (Actual FPL points for the specific gameweek).
*   **Inference**:
    *   Generates predictions for the **Next Gameweek** only.
    *   Filters: `status != 'u'` (injured), `chance_of_playing >= 75%`.
    *   Selection: Picks top player per position + 1 Wildcard (highest predicted remaining player).

## 4. Operational Decisions
*   **Updates**: The pipeline (`update_pipeline.py`) is designed to run end-to-end (Fetch -> Train -> Predict) or in `--quick` mode (Predict only).
*   **Visualization**: Web interface serves purely as a display layer for the pre-calculated `inference_data.csv`.
*   **Modular Inference**:
    *   Per the Dec 2025 Refactor, all prediction logic and model loading is centralized in `src/inference.py`.
    *   Feature selection and hyperparameters are defined in `src/config.py`.
    *   Both `update_pipeline.py` (Automation) and `src/app.py` (Web UI) consume these shared modules to ensure 100% consistency in results.

## 5. Code Structure & Modules (New Jan 2026)
*   **Package-First Approach**: The project is structured as a Python package (`src`).
*   **Execution**: All scripts within `src/` MUST be executed as modules (e.g., `python -m src.train_model`) rather than as standalone scripts (`python src/train_model.py`).
    *   *Rationale*: This ensures the `sys.path` is correctly set to the project root (`C:\fpl-ml`), enabling absolute imports (e.g., `from src.config import ...`) to resolve correctly from anywhere in the codebase.

## 6. User Interface & Live Data
*   **Live Points**: During active gameweeks, the "Selection %" column is replaced by "Live Points".
    *   **Visual Indicators**:
        *   **Red**: Match is Live/Ongoing.
        *   **Green**: Match Finished.
    *   **Data Source**: FPL `/event/{id}/live` API (Gameweek Live Data).
    *   **Synchronization**: The frontend explicitly requests live data for the **Predicted Gameweek** (e.g., `?gw=20`).
        *   *Rationale*: This prevents the display of stale data from the *previous* "current" gameweek (e.g., GW19) when valid predictions exist for an upcoming, unstarted gameweek.
    *   **Caching**: Server-side caching (5 minutes) is implemented on the `/api/live` endpoint to minimize calls to the FPL API and prevent rate-limiting.

## 7. Infrastructure & Reliability
*   **Keep Alive Strategy**: To prevent the free-tier hosting (Render) from spinning down due to inactivity, a GitHub Action (`keep_alive.yml`) pings the site.
    *   **Randomization**: The workflow runs every 5 minutes but includes a random sleep delay (0-10 minutes) before the ping. This ensures the site stays active while making the traffic pattern less predictable.
