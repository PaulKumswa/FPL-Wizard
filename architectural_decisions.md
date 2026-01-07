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

### 1.3 Understat Player-Level Data (Added Jan 2026)
*   **Source**: `understat_players_{season}.json` fetched via `understatapi`.
*   **Mapping**: Uses `id_mapping.csv` from `id_map.py` to link FPL player IDs → Understat player IDs.
*   **Features Added**:
    *   `us_npxG_per90`: Non-penalty Expected Goals per 90 minutes (more predictive than raw xG).
    *   `us_xA_per90`: Expected Assists per 90 minutes.
*   *Rationale*: FPL API's `expected_goals` field is less accurate and detailed than Understat's xG model. Using per-90 metrics normalizes for playing time differences.
*   **Position Usage**: MID and FWD positions use `recent_us_npxG_per90` and `recent_us_xA_per90` as features. GKP/DEF do not use these attacking metrics.

### 1.4 Team Name Mapping (Added Jan 2026)
*   **File**: `data/config/known_team_mapping.json`
*   **Purpose**: Maps Understat team names to FPL team names (e.g., "Manchester City" → "Man City").
*   *Rationale*: Replaces unreliable fuzzy matching with explicit, maintainable mappings.

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

### 3.2 Component-Based Prediction (Updated Jan 2026)
*   **Approach**: Decompose `total_points` into predictable sub-components.
*   *Rationale*: FPL points are noisy due to bonus points and random events. Predicting individual outcomes (goals, assists, clean sheets) is more stable than predicting raw points directly.
*   **Architecture**:
    *   **Goal Models**: LGBMClassifier per position, predicts P(player scores ≥1 goal)
    *   **Assist Models**: LGBMClassifier per position, predicts P(player gets ≥1 assist)
    *   **Clean Sheet Models**: LGBMClassifier for GKP, DEF, MID only (FWD gets 0 pts)
    *   **Legacy Model**: LGBMRegressor kept as fallback for comparison
*   **Aggregation Formula**:
    ```
    Expected Points = P(goal) × GOAL_PTS[pos] + P(assist) × 3 + P(cs) × CS_PTS[pos] + 2
    ```
    Where `GOAL_PTS = {GKP: 10, DEF: 6, MID: 5, FWD: 4}` and `CS_PTS = {GKP: 4, DEF: 4, MID: 1, FWD: 0}`.
*   **Phase 2 (Deferred)**: See Section 8.

### 3.3 Inference
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

## 8. Phase 2: Deferred Enhancements (Indefinitely Deferred)

The following features were considered for Phase 2 of the component-based prediction system but have been **indefinitely deferred** due to complexity vs. impact trade-offs:

### 8.1 Bonus Points Prediction
*   **Description**: Predict P(player earns 1/2/3 bonus points) based on the BPS (Bonus Points System).
*   **Why Deferred**:
    *   BPS is calculated using 30+ in-match statistics (passes, tackles, saves, etc.) that are not available pre-match.
    *   Would require training a separate model on post-match BPS data, then using expected stats to infer pre-match probabilities.
    *   High implementation complexity for marginal improvement (~1-3 points).
*   **Alternative**: The legacy regressor implicitly captures some bonus correlation.

### 8.2 Goalkeeper Saves Regression
*   **Description**: Predict expected saves for GKPs (1 point per 3 saves).
*   **Why Deferred**:
    *   Save count is highly dependent on opponent shot volume, which varies unpredictably.
    *   Would require opponent xG as a feature, adding pipeline complexity.
    *   Impact: ~0.3-1.0 points per gameweek for GKPs only.

### 8.3 Penalty Events
*   **Description**: Predict P(penalty save) for GKPs, P(penalty miss) for outfield.
*   **Why Deferred**:
    *   Extremely rare events (~2% of matches have penalties).
    *   Insufficient training data for reliable classification.
    *   Impact: High variance (5 pts for save, -2 for miss), but low expected value.

### 8.4 Goals Conceded Penalty
*   **Description**: Predict expected goals conceded for GKP/DEF (-1 point per 2 goals conceded).
*   **Why Deferred**:
    *   Would require clean sheet model inversion + Poisson regression for goal count.
    *   Complex interaction with team defensive stats.
    *   Already partially captured by opponent strength feature.

### 8.5 Multi-Goal/Assist Prediction
*   **Description**: Predict P(2+ goals) or P(2+ assists) for haul potential.
*   **Why Deferred**:
    *   Current binary approach (0/1+) captures majority of expected value.
    *   Multi-goal games are rare (~5% of goals come from braces+).
    *   Would require ordinal or count regression, adding model complexity.

### 8.6 Decision Rationale
Phase 1 (goals, assists, clean sheets) captures **~80% of FPL point variance** for the typical player. The remaining components (bonus, saves, penalties) have:
*   High prediction uncertainty
*   Low marginal expected value
*   Significant implementation complexity

These will be revisited if Phase 1 performance plateaus and additional accuracy is needed.

## 9. Dynamic Selection Thresholds (Added Jan 2026)

### 9.1 Problem with Static Thresholds
Previously, underdog selection used hardcoded thresholds:
```python
MAX_COST = 80       # £8.0m
MAX_OWNERSHIP = 10  # 10%
MIN_FORM = 2.0
MIN_ICT = 3.0
```

These static values had several issues:
*   **Early Season**: Many quality players still under 10% ownership → thresholds too lenient
*   **Mid-Season**: Template teams form, only 10-15 players above 10% → thresholds too restrictive
*   **Late Season**: Ownership distributions shift as managers wildcard
*   **No Rationale**: Values were arbitrary with no documented justification

### 9.2 Solution: Percentile-Based Dynamic Calculation
Thresholds are now computed at inference time based on the current week's player pool:

| Metric | Percentile | Interpretation |
| :--- | :--- | :--- |
| **Ownership** | 50th percentile (Floor 10%) | Select from bottom ~50% (but allow at least 10% ownership) |
| **Cost** | 75th percentile (Floor £7.0m) | Select from below ~75th percentile cost (budget-friendly) |
| **Predicted Points** | 92nd percentile (Range 5.5-8.0) | Target high scorers (approx > 6.0 pts) |
| **Form/ICT** | 30th percentile | Exclude bottom 30% by form/activity |

### 9.3 Implementation
*   **Config** (`src/config.py`): Defines `OWNERSHIP_PERCENTILE`, `COST_PERCENTILE`, `FORM_PERCENTILE`, `ICT_PERCENTILE`
*   **Inference** (`src/inference.py`): 
    *   Added `calculate_dynamic_thresholds(df)` function which computes thresholds.
    *   Added `select_best_team(df)` with robust fallback logic.
*   **Logging**: Computed thresholds are printed each inference run for transparency

### 9.4 'Fail Upward' Strategy
If no "perfect underdog" (Low Ownership + Low Cost + High Points) matches the criteria:

1.  **Prioritize Points**: The system looks for high-scoring players who might be slightly more expensive or popular.
2.  **Fallback Levels**:
    *   **Level 1 (Ideal)**: Meets all strict criteria.
    *   **Level 2 (Value Pick)**: Relax Ownership constraint.
    *   **Level 3 (Premium Differential)**: Relax Cost constraint.
    *   **Level 4 (Points Only)**: Ignore Ownership/Cost, just find a scorer.
    *   **Level 5 (Last Resort)**: Lower the predicted points expectation.

This ensures the system always returns the best available players rather than forcing low-scoring underdogs.

