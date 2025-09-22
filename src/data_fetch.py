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
# Understat scraping utilities
# ---------------------------------------------------------------------------


def _understat_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({'User-Agent': DEFAULT_USER_AGENT})
    return sess


def _extract_understat_json(script_text: str, var_name: str) -> Any:
    pattern = re.compile(rf"var\s+{re.escape(var_name)}\s*=\s*JSON.parse\('([^']*)'\)")
    match = pattern.search(script_text)
    if match:
        raw = match.group(1)
        decoded = bytes(raw, 'utf-8').decode('unicode_escape')
        return json.loads(decoded)
    fallback = re.compile(rf"var\s+{re.escape(var_name)}\s*=\s*(\[.*?\]);", re.DOTALL)
    match = fallback.search(script_text)
    if match:
        return json.loads(match.group(1))
    return None


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fetch_understat_league_page(
    league: str = 'EPL',
    season: int = 2023,
    *,
    session: Optional[requests.Session] = None,
) -> BeautifulSoup:
    url = f"{UNDERSTAT_BASE_URL}/league/{league}/{season}"
    sess = session or _understat_session()
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, 'html.parser')


def fetch_understat_players(
    league: str = 'EPL',
    season: int = 2023,
    *,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    soup = fetch_understat_league_page(league=league, season=season, session=session)
    script = next((s for s in soup.find_all('script') if 'playersData' in s.text), None)
    if script is None:
        raise RuntimeError('Could not locate playersData in Understat payload')
    data = _extract_understat_json(script.text, 'playersData')
    if data is None:
        raise RuntimeError('Failed to parse playersData JSON')
    df = pd.DataFrame(data)
    numeric_cols = [
        'games',
        'time',
        'goals',
        'xG',
        'assists',
        'xA',
        'shots',
        'key_passes',
        'xGChain',
        'xGBuildup',
    ]
    return _coerce_numeric(df, numeric_cols)


def fetch_understat_matches(
    league: str = 'EPL',
    season: int = 2023,
    *,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    soup = fetch_understat_league_page(league=league, season=season, session=session)
    script = next((s for s in soup.find_all('script') if 'matchesData' in s.text), None)
    if script is None:
        raise RuntimeError('Could not locate matchesData in Understat payload')
    data = _extract_understat_json(script.text, 'matchesData')
    if data is None:
        raise RuntimeError('Failed to parse matchesData JSON')
    df = pd.DataFrame(data)
    numeric_cols = [
        'xG_home',
        'xG_away',
        'forecast_win',
        'forecast_draw',
        'forecast_lose',
    ]
    return _coerce_numeric(df, numeric_cols)

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
            'understat_players',
            'understat_matches',
        ],
        required=True,
        help='Which dataset to fetch',
    )
    ap.add_argument('--out', type=str, required=True, help='Output file path (csv/parquet/json)')
    ap.add_argument('--season', type=int, default=2023, help='Season identifier for Understat (e.g. 2023)')
    ap.add_argument('--league', type=str, default='EPL', help='Understat league code (default: EPL)')
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional limit when building FPL player histories (useful for quick samples)',
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
    out_path = Path(args.out)

    if args.resource == 'fpl_bootstrap':
        payload = fetch_fpl_bootstrap()
        _save_json(payload, out_path)
        return

    if args.resource == 'fpl_fixtures':
        fixtures = fetch_fpl_fixtures()
        _save_json(fixtures, out_path)
        return

    if args.resource == 'fpl_histories':
        sleep_sec = args.sleep if args.sleep is not None else DEFAULT_FPL_SLEEP
        df = build_fpl_player_gameweeks(limit=args.limit, sleep_sec=sleep_sec)
        _save_dataframe(df, out_path)
        return

    session = _understat_session()
    time.sleep(DEFAULT_UNDERSTAT_SLEEP)

    if args.resource == 'understat_players':
        df = fetch_understat_players(league=args.league, season=args.season, session=session)
        _save_dataframe(df, out_path)
        return

    if args.resource == 'understat_matches':
        df = fetch_understat_matches(league=args.league, season=args.season, session=session)
        _save_dataframe(df, out_path)
        return

    raise ValueError(f'Unsupported resource: {args.resource}')


if __name__ == '__main__':
    main()

