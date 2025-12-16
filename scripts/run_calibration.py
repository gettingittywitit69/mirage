"""CLI to run mirage/selection calibration across synthetic DGPs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from criticality.calibration import DEFAULT_DGPS, DEFAULT_MODEL_NAMES, run_mirage_calibration
from criticality.tournament import ModelSpec, available_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mirage calibration and selection tests.")
    parser.add_argument(
        "--rho-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75],
        help="Grid of rho values controlling rate contrast / excitation strength.",
    )
    parser.add_argument(
        "--persistence-grid",
        nargs="+",
        type=float,
        default=[0.6, 0.8, 0.9],
        help="Grid of persistence parameters for regimes or drift.",
    )
    parser.add_argument(
        "--horizons",
        "--T",
        dest="horizons",
        nargs="+",
        type=int,
        default=[2500, 5000],
        help="Series lengths to simulate.",
    )
    parser.add_argument("--runs-per-cell", type=int, default=3, help="Replications per (dgp, rho, persistence, T).")
    parser.add_argument(
        "--dgps",
        nargs="+",
        default=DEFAULT_DGPS,
        choices=DEFAULT_DGPS,
        help="Subset of DGPs to simulate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=available_models(),
        help=f"Subset of models to fit (default: {', '.join(DEFAULT_MODEL_NAMES)}).",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train split fraction inside synthetic series.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=Path("calibration_results.parquet"), help="Parquet output path.")
    parser.add_argument("--confusion", type=Path, default=Path("confusion_matrix.csv"), help="Confusion matrix path.")
    parser.add_argument("--mirage-map", type=Path, default=Path("mirage_map.csv"), help="Mirage map CSV path.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_names = args.models or DEFAULT_MODEL_NAMES
    model_specs = [ModelSpec(name) for name in model_names]

    results_df, confusion_df, mirage_map_df = run_mirage_calibration(
        rho_grid=args.rho_grid,
        persistence_grid=args.persistence_grid,
        horizons=args.horizons,
        runs_per_cell=args.runs_per_cell,
        dgps=args.dgps,
        models=model_specs,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    results_df.to_parquet(args.output, index=False)
    confusion_df.to_csv(args.confusion, index=False)
    mirage_map_df.to_csv(args.mirage_map, index=False)

    print(f"Wrote calibration results to {args.output} ({len(results_df)} rows).")
    print(f"Wrote confusion matrix to {args.confusion}.")
    print(f"Wrote mirage map to {args.mirage_map}.")
    if not confusion_df.empty:
        print("Confusion (win_rate):")
        print(confusion_df[["dgp", "model", "win_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
