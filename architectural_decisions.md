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
*   **Type**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`).
*   *Rationale*: Robust against outliers, handles non-linear relationships well, and requires minimal feature scaling compared to Neural Networks.

### 2.2 Model Configuration (Updated Dec 2025)
To accommodate high-precision continuous variables (xG/xGA), the following parameters were enforced to prevent overfitting and "memorization" of specific floating-point values:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **n_estimators** | `200` | Increased from 100 to reduce variance introduced by new noisy features. |
| **max_depth** | `15` | Capped (prev. Unlimited) to force generalization and prevent isolating single player-matches. |
| **min_samples_leaf** | `3` | Increased (prev. 1) for DEF/MID to ensure rules apply to clusters of players, not individuals. |

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
*   **Training Split**: 80% Train / 20% Test.
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

## 5. User Interface & Live Data
*   **Live Points**: During active gameweeks, the "Selection %" column is replaced by "Live Points".
    *   **Visual Indicators**:
        *   **Red**: Match is Live/Ongoing.
        *   **Green**: Match Finished.
    *   **Data Source**: FPL `element-summary` API.
    *   **Caching**: Server-side caching (5 minutes) is implemented on the `/api/live-data` endpoint to minimize calls to the FPL API and prevent rate-limiting.

## 6. Infrastructure & Reliability
*   **Keep Alive Strategy**: To prevent the free-tier hosting (Render) from spinning down due to inactivity, a GitHub Action (`keep_alive.yml`) pings the site.
    *   **Randomization**: The workflow runs every 5 minutes but includes a random sleep delay (0-10 minutes) before the ping. This ensures the site stays active while making the traffic pattern less predictable.
