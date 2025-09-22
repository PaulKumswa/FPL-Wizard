"""Simple spot-checks for the data_fetch module."""

from __future__ import annotations

from pprint import pprint

from src.data_fetch import (
    build_fpl_player_gameweeks,
    fetch_fpl_bootstrap,
    fetch_fpl_fixtures,
    fetch_understat_players,
)


def main() -> None:
    bootstrap = fetch_fpl_bootstrap()
    print("bootstrap keys:", list(bootstrap.keys())[:5])
    print("teams count:", len(bootstrap["teams"]))

    fixtures = fetch_fpl_fixtures()[:3]
    print("sample fixtures:")
    pprint(fixtures)

    history_df = build_fpl_player_gameweeks(limit=5, sleep_sec=0.1)
    print("history sample rows:")
    print(history_df.head())

    understat_df = fetch_understat_players(season=2023).head()
    print("understat players sample:")
    print(understat_df[["player_name", "team_title", "xG", "xA"]])


if __name__ == "__main__":
    main()
