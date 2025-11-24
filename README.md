# FPL ML (weekly team picker)
Built with ChatGPT Codex and Google Antigravity

It provides predictions based on a custom trained machine learning model, updated weekly. The model identifies high-potential "underdog" players (less than 10% ownership, under £8.0m) with strong form or underlying stats.

This repository pulls its raw data from two stable sources:

- **Fantasy Premier League official API** – provides player metadata, fixtures, and per-gameweek points.
- **Understat** – scraped to add xG/xA and other advanced metrics for each season.

## Quick start

1. Create/activate the virtual environment (example on PowerShell):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
2. **Run the Automation Pipeline**:
   To fetch the latest data, train the model, and launch the app:
   ```bash
   python update_pipeline.py
   ```

   **Quick Launch (App Only)**:
   If you already have data and just want to launch the website:
   ```bash
   python update_pipeline.py --quick
   ```

   *Note: The full pipeline fetches data from the FPL API and Understat. The first run may take a few minutes.*

`src/test_api.py` contains a lightweight smoke test that hits each data source and prints samples (disable or trim network calls if running in a restricted environment).

Next steps will build feature engineering, modelling, and optimisation layers on top of these new data sources.
