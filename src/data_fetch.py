"""
data_fetch.py
--------------
Client utilities for pulling data from the official Fantasy Premier League API
and enriching it with Understat advanced metrics.

Primary sources:
  * Fantasy Premier League (FPL) public API
  * Understat league pages (scraped JSON embedded in HTML)

Typical usage (PowerShell):
  # Save the latest bootstrap-static payload to data/raw/
  python -m src.data_fetch --resource fpl_bootstrap --out data/raw/fpl_bootstrap.json

  # Build a compact per-player gameweek history sample (first 25 players)
  python -m src.data_fetch --resource fpl_histories --limit 25 --out data/raw/fpl_histories_SAMPLE.parquet

  # Fetch Understat player stats for the 2023 Premier League season
  python -m src.data_fetch --resource understat_players --season 2023 --out data/raw/understat_players_2023.csv
"""

from __future__ import annotations

import argparse
import sys
import json
import os
import time
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path('configs/.env'), override=False)

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
UNDERSTAT_BASE_URL = "https://understat.com"

DEFAULT_USER_AGENT = os.getenv(
    "UNDERSTAT_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be a float, got {raw!r}"
        ) from exc


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(obj: Any, path: Path) -> Path:
    path = _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    print(f"[ok] wrote {path}")
    return path


def _save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path = _ensure_parent(path)
    if path.suffix.lower() == '.csv':
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {'.parquet', '.pq'}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"[ok] wrote {path} ({len(df):,} rows)")
    return path


DEFAULT_FPL_SLEEP = _float_env('FPL_SLEEP_SEC', 0.35)
DEFAULT_UNDERSTAT_SLEEP = _float_env('UNDERSTAT_SLEEP_SEC', 2.5)

# ---------------------------------------------------------------------------
# FPL Official API
# ---------------------------------------------------------------------------


def fetch_fpl_bootstrap(*, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    url = f"{FPL_BASE_URL}/bootstrap-static/"
    sess = session or requests.Session()
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_fixtures(*, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    url = f"{FPL_BASE_URL}/fixtures/"
    sess = session or requests.Session()
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_player_history(
    element_id: int,
    *,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    url = f"{FPL_BASE_URL}/element-summary/{element_id}/"
    sess = session or requests.Session()
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def build_fpl_player_gameweeks(
    *,
    limit: Optional[int] = None,
    sleep_sec: float = DEFAULT_FPL_SLEEP,
) -> pd.DataFrame:
    sess = requests.Session()
    bootstrap = fetch_fpl_bootstrap(session=sess)
    elements = pd.DataFrame(bootstrap['elements'])
    if limit is not None:
        elements = elements.head(limit)

    rows: List[Dict[str, Any]] = []
    for _, row in elements.iterrows():
        element_id = int(row['id'])
        history_payload = fetch_fpl_player_history(element_id, session=sess)
        for fixture in history_payload.get('history', []):
            fixture['element'] = element_id
            rows.append(fixture)
        time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    numeric_cols = [
        'round',
        'total_points',
        'minutes',
        'goals_scored',
        'assists',
        'clean_sheets',
        'goals_conceded',
        'own_goals',
        'penalties_saved',
        'penalties_missed',
        'yellow_cards',
        'red_cards',
        'saves',
        'bonus',
        'bps',
        'influence',
        'creativity',
        'threat',
        'ict_index',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ---------------------------------------------------------------------------
# Understat API (using understatapi library)
# ---------------------------------------------------------------------------

def fetch_understat_data(
    league: str = 'EPL',
    season: int = 2025,
    out_dir: Path = Path('data/raw')
) -> None:
    """
    Fetches both Player and Team data from Understat using `understatapi`.
    Saves to:
      - data/raw/understat_players_{season}.json
      - data/raw/understat_teams_{season}.json
    """
    try:
        from understatapi import UnderstatClient
    except ImportError:
        print("Error: understatapi not installed. Run `pip install understatapi`")
        return

    print(f"Fetching Understat data for {league} {season}...")
    
    understat = UnderstatClient()
    
    # 1. Fetch Player Data
    # Note: understatapi might raise errors if season is invalid or connection fails
    # We use a try-except block for soft-fail as per plan
    try:
        players_payload = understat.league(league=league).get_player_data(season=str(season))
        # The payload is usually a list of dicts directly
        out_players = out_dir / f"understat_players_{season}.json"
        _save_json(players_payload, out_players)
    except Exception as e:
        print(f"Failed to fetch Understat Players: {e}")
        # Create empty file to prevent downstream crash? Or just leave it missing.
        # Plan says "Soft Fail", so we continue.

    # 2. Fetch Team Data (for xGA)
    try:
        teams_payload = understat.league(league=league).get_team_data(season=str(season))
        out_teams = out_dir / f"understat_teams_{season}.json"
        _save_json(teams_payload, out_teams)
    except Exception as e:
        print(f"Failed to fetch Understat Teams: {e}")

    # 3. Fetch Match Data (for Time-Series)
    try:
        matches_payload = understat.league(league=league).get_match_data(season=str(season))
        # This returns a list of matches. Each match has 'h' (home) and 'a' (away) stats, including players.
        out_matches = out_dir / f"understat_matches_{season}.json"
        _save_json(matches_payload, out_matches)
    except Exception as e:
        print(f"Failed to fetch Understat Matches: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Fetch data from FPL API and Understat')
    ap.add_argument(
        '--resource',
        choices=[
            'fpl_bootstrap',
            'fpl_fixtures',
            'fpl_histories',
            'understat_all', # Consolidated command
        ],
        required=True,
        help='Which dataset to fetch',
    )
    ap.add_argument('--out', type=str, required=False, help='Output file path (DEPRECATED for understat)')
    ap.add_argument('--season', type=int, default=2025, help='Season identifier for Understat (e.g. 2025)')
    ap.add_argument('--league', type=str, default='EPL', help='Understat league code (default: EPL)')
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional limit when building FPL player histories',
    )
    ap.add_argument(
        '--sleep',
        type=float,
        default=None,
        help='Override sleep between FPL history requests',
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    # Handle optional out path for FPL
    out_path = Path(args.out) if args.out else None

    if args.resource == 'fpl_bootstrap':
        if not out_path: raise ValueError("--out required for fpl_bootstrap")
        payload = fetch_fpl_bootstrap()
        _save_json(payload, out_path)
        return

    if args.resource == 'fpl_fixtures':
        if not out_path: raise ValueError("--out required for fpl_fixtures")
        fixtures = fetch_fpl_fixtures()
        _save_json(fixtures, out_path)
        return

    if args.resource == 'fpl_histories':
        if not out_path: raise ValueError("--out required for fpl_histories")
        sleep_sec = args.sleep if args.sleep is not None else DEFAULT_FPL_SLEEP
        df = build_fpl_player_gameweeks(limit=args.limit, sleep_sec=sleep_sec)
        _save_dataframe(df, out_path)
        return

    if args.resource == 'understat_all':
        fetch_understat_data(league=args.league, season=args.season)
        return

    raise ValueError(f'Unsupported resource: {args.resource}')


if __name__ == '__main__':
    main()

