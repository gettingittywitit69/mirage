"""Minimal CLI to inspect Ken French daily factors."""
from __future__ import annotations

import argparse
from pathlib import Path

from criticality.data_ff import get_ff_factors_daily_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Ken French daily Mkt-RF and RF factors")
    parser.add_argument("--cache-dir", type=Path, default=Path(".ff_cache"), help="Directory to cache downloads")
    parser.add_argument("--force", action="store_true", help="Force re-download of the raw zip")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = get_ff_factors_daily_df(args.cache_dir, force=args.force)
    start, end = df["date"].min(), df["date"].max()
    print(f"Loaded {len(df)} rows spanning {start.date()} to {end.date()}.")
    print()
    print(df[["mkt_rf", "rf"]].describe())
    print()
    print("First 3 rows:")
    print(df.head(3))
    print()
    print("Last 3 rows:")
    print(df.tail(3))


if __name__ == "__main__":
    main()
