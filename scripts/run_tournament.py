"""CLI to run the model tournament."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from criticality.data_ff import add_features
from criticality.events import simulate_poisson
from criticality.tournament import available_models, run_tournament


def _load_input(path: Path | None) -> pd.DataFrame:
    if path is None:
        return simulate_poisson(rate=1.0, T=200, rng=123)
    df = pd.read_csv(path)
    return add_features(df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the criticality model tournament")
    parser.add_argument("--input", type=Path, help="CSV file with time,count columns", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Subset of models to evaluate ({', '.join(available_models())})",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train split fraction")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    df = _load_input(args.input)
    scores = run_tournament(df, models=args.models, train_frac=args.train_frac)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
