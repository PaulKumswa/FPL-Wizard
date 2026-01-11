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

### 3.4 Model Metrics Storage (Added Jan 2026)
*   **File**: `data/history/model_metrics.json`
*   **Content**: Logs validation performance (AUC, Accuracy, MAE) for every training run.
*   **Schema**:
    ```json
    {
      "timestamp": "ISO8601",
      "model_type": "component|legacy",
      "position": "GKP|DEF|MID|FWD",
      "component": "goal|assist|cleansheet|null",
      "metrics": { ... },
      "samples": 1234
    }
    ```
*   **Rationale**: Enables longitudinal tracking of model performance to detect degradation or improvements over time. Replaces ephemeral console output.

### 3.5 Feature Importance Logging & Visualization (Added Jan 2026)
*   **File**: `data/history/feature_importance.json`
*   **Content**: Captures the LightGBM `feature_importances_` (normalized to 0-100%) for every component model.
*   **Visualization**: Exposed via `/feature-importance` UI route.
*   **Rationale**: 
    1.  **Explainability**: Helps users and developers understand *why* the model makes certain predictions (e.g., "GKP prediction driven by Team xGA").
    2.  **Debugging**: Catch data issues (e.g., if `now_cost` becomes the #1 predictor for everything, something might be wrong).
    3.  **Trust**: Transparently showing model logic increases user confidence in the "black box".

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

## 10. Prediction Confidence Scoring (Added Jan 2026)

### 10.1 Problem
Predictions were point estimates with no indication of model certainty. Users couldn't distinguish between confident picks and "coin flip" predictions.

### 10.2 Solution: Weighted Probability Decisiveness
Confidence is calculated from how "decisive" the component probabilities are, **weighted by their importance to the player's position**:

```python
confidence = sum(|p_comp - 0.5| × 2 × weight_comp) × 100
```

*   **Position Weights**:
    *   **GKP/DEF**: Heavy weight on Clean Sheet probability (70-80%).
    *   **MID/FWD**: Heavy weight on Goal/Assist probabilities (80-100%).
*   **Intuition**: A Defender with 90% Clean Sheet probability is "High Confidence" even if their Goal probability is uncertain (50%), whereas a Forward needs high certainty in attacking returns to be reliable.
*   **Example**: Casemiro (MID) might have high points but low confidence if his underlying attacking probabilities are "coin flips" (0.5), effectively penalizing him in the selection logic.

### 10.3 Implementation
*   **Backend** (`inference.py`): `calculate_confidence()` computes the score.
*   **API** (`app.py`): Exposes `confidence_score` in the predictions response.
*   **Frontend** (`index.html`): Color-codes the Min Points column:
    *   **Green (≥70%)**: High confidence
    *   **Yellow (40-69%)**: Medium confidence
    *   **Orange (<40%)**: Low confidence/risky

## 11. Confidence-Based Selection Strategy (Added Jan 2026)

### 11.1 Objective
Maximize expected points **reliably** by prioritizing model confidence when selecting players. The goal is to find high-confidence predictions that meet the 6+ point target, relaxing confidence before other constraints.

### 11.2 Selection Cascade (Fail Upward)
### 11.2 Selection Cascade (Strict Reliability)
The selection logic uses a multi-level cascade to ensure quality. Crucially, **Level 1 and Level 2 now enforce a strict Confidence Threshold (≥60%)** to prevent efficient but uncertain "coin flip" players from being selected as the main pick.

| Level | Confidence | Ownership | Cost | Points | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1A (Main)** | **≥60%** (Strict) | Low | Low | ≥6.0 | Reliable underdog (The Gold Standard) |
| **1B (Safe)** | **≥60%** (Strict) | Low | Low | ≥4.0 | Safe reliable underdog (>4 pts) |
| **2** | ≥40% (Medium) | Low | Low | ≥6.0 | High upside, medium reliability |
| **3** | Any | Any | Low | ≥6.0 | Value pick (relax ownership) |
| **4** | Any | Low | Any | ≥6.0 | Premium differential (relax cost) |
| **5** | Any | Any | Any | Any | Last resort (highest reliability score remaining) |

### 11.3 Wildcard Selection
The 5th player is a designated **Wildcard**:
*   **Criteria**: The unpicked player with the highest **Predicted Points** (Upside) who has at least **60% Confidence**.
*   **Fallback**: If no high-confidence upside players exist, pick the highest Reliability Score remaining.
*   **Badge**: Labeled with a Purple "WILDCARD" badge in the UI.

### 11.3 Tiebreaker Logic
When multiple players pass the same level's filters:
1. **Primary Sort**: Confidence Score (descending) - prefer more certain predictions
2. **Secondary Sort**: Predicted Points (descending) - higher returns

### 11.4 Rationale
*   **Reliability over Variance**: A 5.8 pt pick with 70% confidence is preferable to a 6.2 pt pick with 25% confidence
*   **Points Preservation**: The 6.0 point floor is maintained through Levels 1-4 before relaxing
*   **Underdog DNA**: The system still tries to find low-ownership/cost players first, but won't sacrifice prediction quality for it

## 12. User Interface & Navigation (Added Jan 2026)

### 12.1 Mobile-First Design
The frontend follows a mobile-first responsive design approach:
*   **Primary Target**: Mobile devices (majority of FPL managers check on phones)
*   **Progressive Enhancement**: Desktop view enhances mobile layout with additional details

### 12.2 Tab-Based Navigation
Both mobile and desktop views use a consistent three-tab navigation structure:

| Tab | Content | Purpose |
| :--- | :--- | :--- |
| **Picks** | Player predictions table/cards | Primary use case: weekly picks |
| **History** | Past gameweek predictions | Track accuracy and backfill points |
| **Info** | How it works, accuracy, credits | System explanation and transparency |

### 12.3 Mobile-Specific Features
*   **Player Cards**: Swipeable cards replace tables for better touch interaction
*   **Expandable Details**: Tap to reveal additional player stats (xG, xA, cost)
*   **Bottom Navigation Bar**: Fixed navigation for easy thumb access
*   **Confidence Indicators**: Color-coded points with confidence tooltip

### 12.4 Desktop Enhancements
*   **Full Table View**: All columns visible without scrolling
*   **Hover Effects**: Interactive row highlighting
*   **Architecture Diagram**: System overview visible on desktop Info tab
*   **Player Links**: Direct links to Premier League player profiles

### 12.5 Live Match Integration
*   **Column Toggle**: "Selection %" becomes "Live Pts" during active matches
*   **Status Indicators**:
    *   🔴 Red text = Match in progress
    *   🟢 Green text = Match finished
*   **Cache Strategy**: 5-minute server-side cache to reduce FPL API calls

## 13. Player Profile URLs (Added Jan 2026)

### 13.1 URL Format
Player profile links point to the official Premier League website:
```
https://www.premierleague.com/en/players/{code}/{name-slug}/overview
```

### 13.2 Implementation
*   **Code**: Uses FPL's `code` field (not `element` ID) for stable URLs
*   **Name Slug Generation** (in `app.py`):
    1. Normalize Unicode characters (e.g., é → e)
    2. Convert to lowercase
    3. Replace spaces and apostrophes with hyphens
    4. Example: "Salah" → `salah`, "De Bruyne" → `de-bruyne`

### 13.3 Rationale
*   **Stability**: The `code` remains constant even if player changes teams or FPL ID changes
*   **SEO-Friendly**: Slugified names improve URL readability
*   **Cross-Platform**: Links work on both desktop and mobile site

## 14. Deployment & Infrastructure (Updated Jan 2026)

### 14.1 Platform Migration
*   **Previous**: Render (free tier)
*   **Current**: Northflank with custom domain

### 14.2 Northflank Configuration

| Setting | Value | Notes |
| :--- | :--- | :--- |
| **Service Type** | Combined Service | Build + Run in one service |
| **Build Type** | Buildpack | Auto-detects Python |
| **Run Command** | `gunicorn src.app:app` | WSGI server |
| **Port** | 8080 | Default Northflank port |
| **Auto-Deploy** | Enabled | Deploys on push to main branch |

### 14.3 Custom Domain (fplbangers.com)

DNS configuration via Cloudflare:

| Record Type | Name | Target |
| :--- | :--- | :--- |
| **CNAME** | `@` (root) | Northflank endpoint |
| **CNAME** | `www` | Northflank endpoint |

### 14.4 SSL/TLS Configuration
*   **Cloudflare SSL Mode**: Full (Strict)
*   **Certificate**: Auto-provisioned by Northflank
*   **Initial Setup Note**: Disable Cloudflare proxy (orange cloud → grey) during initial SSL provisioning to allow Northflank to verify domain ownership

### 14.5 Keep-Alive Strategy
GitHub Actions workflow (`keep_alive.yml`) prevents platform spin-down:
*   **Frequency**: Every 5 minutes
*   **Randomization**: Random 0-10 minute delay before ping
*   **Purpose**: Maintain warm instances for fast response times

## 15. Model Evolution History (Added Jan 2026)

This section documents the iterative development of the prediction system across multiple model versions.

### 15.1 Version Tracking
*   **Config**: `MODEL_VERSION` and `MODEL_ERAS` in `src/config.py`
*   **Logging**: Each prediction entry in `predictions_log.json` includes `model_version`, `model_name`, and `model_type`
*   **API**: `/api/model-stats` returns per-era performance statistics
*   **UI**: History page shows Model Evolution cards and version badges on gameweek dropdowns

### 15.2 Era Timeline

| Era | Version | Dates | Gameweeks | Model Type |
| :--- | :--- | :--- | :--- | :--- |
| **Era 1** | v1 | Nov 28 - Dec 20 | GW 13, 15, 16, 17 | 4 Position Regressors |
| **Era 2** | v2 | Dec 26 - Jan 6 | GW 19, 20, 21 | Regressors + Team Understat |
| **Era 3** | v3 | Jan 11+ | GW 22+ | Component-Based |

*Note: GW 14 and GW 18 were skipped due to active development.*

### 15.3 Era Details

#### v1: 4 Position Regressors (Nov 28, 2025)
*   **Change**: Split single global regressor into 4 position-specific RandomForest models
*   **Features**: FPL API stats only (form, cost, ownership, clean sheets, etc.)
*   **Rationale**: Different positions have fundamentally different scoring patterns
*   **Limitations**: No external data sources, regressor predicts noisy total_points directly

#### v2: Regressors + Team Understat (Dec 26, 2025)
*   **Change**: Added team-level Understat data (xG, xGA)
*   **New Features**: `recent_team_xg`, `recent_team_xga`
*   **Rationale**: Team attacking/defensive strength provides context for individual predictions
*   **Data Pipeline**: Merged Understat team data on (Match Date, Team Name)
*   **Limitations**: Still using regressor approach, no player-level advanced stats

#### v3: Component-Based + Player Understat (Jan 11, 2026)
*   **Major Refactor**: Decomposed prediction into probability components
*   **Architecture**: 
    *   ~10 LightGBM Classifiers (Goal/Assist/CS per position)
    *   P(goal) × GOAL_PTS + P(assist) × 3 + P(cs) × CS_PTS + 2
*   **New Features**: `us_npxG_per90`, `us_xA_per90` for MID/FWD
*   **Confidence Scoring**: Added uncertainty quantification based on probability decisiveness
*   **Selection Logic**: Confidence-first cascade prioritizes reliable predictions
*   **Rationale**: 
    *   Binary outcomes (goal: yes/no) are more stable to predict than noisy point totals
    *   Player-level xG/xA more predictive than team aggregates for attacking players

### 15.4 Performance Comparison
Performance statistics are calculated and displayed in the UI via `/api/model-stats`:
*   **Hit Rate**: % of picks where Actual ≥ 90% of floor(Predicted)
*   **Avg Predicted**: Mean predicted points per pick
*   **Avg Actual**: Mean actual points per pick

This allows users to see how model improvements have affected prediction accuracy over time.
