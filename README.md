# FPL Underdog Predictor

**Live Site:** [fplbangers.com](https://fplbangers.com)

Built with ChatGPT Codex and Google Antigravity.

A machine learning-powered Fantasy Premier League (FPL) assistant that identifies high-potential "underdog" players—those with less than 10% ownership and under £8.0m—who have strong form or underlying stats. Predictions are updated weekly using custom-trained models.

## Key Features

- **Component-Based Predictions**: Predicts goals, assists, and clean sheets separately using LightGBM classifiers
- **Confidence Scoring**: Color-coded predictions showing model certainty (green = high, yellow = medium, orange = low)
- **Dynamic Thresholds**: Percentile-based player selection adapts to seasonal ownership/cost shifts
- **Live Match Scores**: Real-time points updates during active gameweeks
- **Feature Importance Dashboard**: Transparent view of which stats drive predictions
- **Prediction History**: Track past picks with actual points backfill

## Data Sources

- **Fantasy Premier League Official API** – Player metadata, fixtures, and per-gameweek points
- **Understat API** – Advanced metrics (xG, xA) and team defensive strength (xGA)

## Quick Start

See [run_instructions.md](run_instructions.md) for detailed setup and deployment instructions.

```powershell
# Install dependencies
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run full pipeline
python update_pipeline.py

# Or quick launch (app only)
python update_pipeline.py --quick
```

## Documentation

- **[run_instructions.md](run_instructions.md)** – Setup, running, and deployment guide
- **[architectural_decisions.md](architectural_decisions.md)** – Technical design decisions and model architecture
- **[claude_analysis.md](claude_analysis.md)** – System audit and improvement tracking

## Tech Stack

- **Backend**: Flask (Python)
- **ML Models**: LightGBM with TimeSeriesSplit cross-validation
- **Frontend**: Responsive HTML/CSS/JS with tab-based navigation
- **Deployment**: Northflank with Cloudflare DNS
- **CI/CD**: GitHub Actions for automated weekly updates

## License

MIT
