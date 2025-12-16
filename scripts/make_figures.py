"""Generate tournament and calibration figures under figures/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from criticality.calibration import run_mirage_calibration
from criticality.plots import (
    plot_confusion_heatmap,
    plot_cumulative_scores,
    plot_mirage_heatmap,
    plot_n_proxy_distribution,
    plot_winner_frequency,
)
from criticality.tournament import ExperimentConfig, ModelSpec, run_tournament


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create tournament and calibration figures.")
    parser.add_argument("--results", type=Path, default=Path("results.csv"), help="Tournament results CSV path.")
    parser.add_argument("--summary", type=Path, default=Path("summary.csv"), help="Tournament summary CSV path.")
    parser.add_argument(
        "--mirage-map",
        type=Path,
        default=Path("mirage_map.csv"),
        help="Mirage map CSV (E[n_proxy_hat | true n=0]).",
    )
    parser.add_argument(
        "--confusion",
        type=Path,
        default=Path("confusion_matrix.csv"),
        help="Confusion matrix CSV (true DGP vs selected model).",
    )
    parser.add_argument("--figdir", type=Path, default=Path("figures"), help="Output directory for figures.")
    parser.add_argument("--config-id", type=int, default=None, help="Config id to use for cumulative score plot.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic fallbacks.")
    return parser


def _ensure_results(results_path: Path, summary_path: Path, seed: int) -> pd.DataFrame:
    if results_path.exists():
        return pd.read_csv(results_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=4000, freq="D")
    returns = pd.Series(rng.normal(0.0, 1.0, len(idx)), index=idx)

    configs = [
        ExperimentConfig(vol_filter="none", event_rule="quantile", q=0.8),
        ExperimentConfig(vol_filter="ewma", event_rule="quantile", q=0.85, vol_kwargs={"span": 20}),
    ]
    models = [ModelSpec("poisson_iid"), ModelSpec("hawkes_dt"), ModelSpec("rs_poisson_dt")]
    run_tournament(
        returns,
        configs=configs,
        models=models,
        train_years=6,
        test_years=2,
        step_years=2,
        seed=seed,
        results_path=str(results_path),
        summary_path=str(summary_path),
    )
    return pd.read_csv(results_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])


def _ensure_calibration(mirage_map_path: Path, confusion_path: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    mirage_df = pd.DataFrame()
    confusion_df = pd.DataFrame()

    if mirage_map_path.exists():
        mirage_df = pd.read_csv(mirage_map_path)
    if confusion_path.exists():
        confusion_df = pd.read_csv(confusion_path)

    if not mirage_df.empty and not confusion_df.empty:
        return mirage_df, confusion_df

    # Lightweight calibration fallback to produce necessary tables.
    calib_models = [ModelSpec("poisson_iid"), ModelSpec("hawkes_dt"), ModelSpec("rs_poisson_dt")]
    _, confusion_df, mirage_df = run_mirage_calibration(
        rho_grid=[0.0, 0.5],
        persistence_grid=[0.6, 0.85],
        horizons=[300],
        runs_per_cell=1,
        dgps=["poisson_drift", "rs_poisson"],
        models=calib_models,
        train_frac=0.7,
        seed=seed,
    )
    mirage_df.to_csv(mirage_map_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    return mirage_df, confusion_df


def _select_config_id(results_df: pd.DataFrame, requested_id: int | None) -> int:
    """Pick a baseline config id, defaulting to the top out-of-sample log-likelihood."""

    for required in ("config_id", "test_loglik"):
        if required not in results_df.columns:
            raise ValueError(f"results dataframe missing required column '{required}'")
    if requested_id is not None:
        return requested_id
    totals = results_df.groupby("config_id")["test_loglik"].sum()
    if totals.empty:
        raise ValueError("Unable to choose a baseline config_id from empty results.")
    return int(totals.idxmax())


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.figdir.mkdir(parents=True, exist_ok=True)

    results_df = _ensure_results(args.results, args.summary, args.seed)
    mirage_df, confusion_df = _ensure_calibration(args.mirage_map, args.confusion, args.seed)

    # 1) Cumulative out-of-sample score
    fig, ax = plt.subplots(figsize=(7, 4))
    cfg_id = _select_config_id(results_df, args.config_id)
    plot_cumulative_scores(results_df, config_id=cfg_id, ax=ax)
    cum_path = args.figdir / "cumulative_scores.png"
    fig.tight_layout()
    fig.savefig(cum_path, dpi=150)
    plt.close(fig)

    # 2) Winner frequency across configs
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_winner_frequency(results_df, ax=ax)
    win_path = args.figdir / "winner_frequency.png"
    fig.tight_layout()
    fig.savefig(win_path, dpi=150)
    plt.close(fig)

    # 3) n_proxy distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_n_proxy_distribution(results_df, ax=ax)
    nproxy_path = args.figdir / "n_proxy_distribution.png"
    fig.tight_layout()
    fig.savefig(nproxy_path, dpi=150)
    plt.close(fig)

    # 4) Mirage heatmap (true n=0)
    mirage_fig = plot_mirage_heatmap(mirage_df)
    mirage_path = args.figdir / "mirage_heatmap.png"
    mirage_fig.savefig(mirage_path, dpi=150)
    plt.close(mirage_fig)

    # 5) Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_confusion_heatmap(confusion_df, ax=ax)
    confusion_fig_path = args.figdir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(confusion_fig_path, dpi=150)
    plt.close(fig)

    outputs = {
        "results": str(args.results),
        "summary": str(args.summary),
        "mirage_map": str(args.mirage_map),
        "confusion_matrix": str(args.confusion),
        "figures": [
            str(cum_path),
            str(win_path),
            str(nproxy_path),
            str(mirage_path),
            str(confusion_fig_path),
        ],
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
