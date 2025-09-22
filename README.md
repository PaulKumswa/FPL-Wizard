# FPL ML (weekly team picker)

This repository pulls its raw data from two stable sources:

- **Fantasy Premier League official API** – provides player metadata, fixtures, and per-gameweek points.
- **Understat** – scraped to add xG/xA and other advanced metrics for each season.

## Quick start

1. Create/activate the virtual environment (example on PowerShell):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt  # generate this when ready
   ```
2. Update `configs/.env` if you want to override sleep or user-agent values (optional).
3. Use `python -m src.data_fetch` to retrieve datasets:
   ```powershell
   # FPL bootstrap-static payload (JSON)
   python -m src.data_fetch --resource fpl_bootstrap --out data/raw/fpl_bootstrap.json

   # First 25 player gameweeks (Parquet)
   python -m src.data_fetch --resource fpl_histories --limit 25 --out data/raw/fpl_histories_SAMPLE.parquet

   # Understat Premier League players for 2023 season (CSV)
   python -m src.data_fetch --resource understat_players --season 2023 --out data/raw/understat_players_2023.csv
   ```

`src/test_api.py` contains a lightweight test that hits each data source and prints samples.

Next steps will build feature engineering, modelling, and optimisation layers on top of these new data sources.
