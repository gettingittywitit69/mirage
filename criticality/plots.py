"""Plotting helpers built on matplotlib."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tournament import load_model
from .utils import FitResult


def plot_counts(df: pd.DataFrame) -> None:
    """Plot raw counts over time."""

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.step(df["time"], df["count"], where="post", label="counts")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()


def plot_cumulative_scores(results: pd.DataFrame, config_id: Optional[int] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Line plot of cumulative out-of-sample score by model for a single config."""

    if results.empty:
        raise ValueError("results dataframe is empty")
    cfg_id = results["config_id"].min() if config_id is None else config_id
    subset = results.loc[results["config_id"] == cfg_id]
    if subset.empty:
        raise ValueError(f"No rows for config_id={cfg_id}")

    pivot = (
        subset.sort_values("split_id")
        .pivot_table(index="split_id", columns="model", values="test_loglik", aggfunc="sum")
        .fillna(0.0)
    )
    cumsums = pivot.cumsum()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for model in cumsums.columns:
        ax.plot(cumsums.index, cumsums[model], marker="o", label=str(model))
    ax.set_title(f"Cumulative test log-lik by model (config {cfg_id})")
    ax.set_xlabel("split id")
    ax.set_ylabel("cumulative test log-lik")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_winner_frequency(results: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Bar plot of winner frequency across configs (robustness)."""

    if results.empty:
        raise ValueError("results dataframe is empty")

    scores = (
        results.groupby(["config_id", "model"], dropna=False)["test_loglik"]
        .sum()
        .reset_index()
    )
    winners = (
        scores.sort_values(["config_id", "test_loglik", "model"], ascending=[True, False, True])
        .groupby("config_id", group_keys=False)
        .head(1)
    )
    counts = winners["model"].value_counts().sort_index()
    total_configs = winners["config_id"].nunique()
    freqs = counts / max(total_configs, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(freqs.index, freqs.values, color="steelblue", alpha=0.85)
    ax.set_ylabel("win rate across configs")
    ax.set_ylim(0, 1.05)
    ax.set_title("Winner frequency (robustness)")
    for bar, val in zip(bars, freqs.values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_n_proxy_distribution(results: pd.DataFrame, models: Optional[Sequence[str]] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Show distribution of fitted n_proxy across configs/splits."""

    data = results.copy()
    data = data[np.isfinite(data["n_proxy"])]
    if models:
        data = data[data["model"].isin(models)]
    if data.empty:
        raise ValueError("No finite n_proxy values to plot")

    order = sorted(data["model"].unique())
    grouped = [data.loc[data["model"] == m, "n_proxy"] for m in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(grouped, labels=order, patch_artist=True, medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")
    ax.set_ylabel("n_proxy")
    ax.set_title("Distribution of n_proxy across configs")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_mirage_heatmap(mirage_df: pd.DataFrame, hawkes_models: Optional[Sequence[str]] = None) -> plt.Figure:
    """Heatmap of E[n_proxy_hat | true n=0] over (rho, persistence) for Hawkes-family fits."""

    if mirage_df.empty:
        raise ValueError("mirage_df is empty")
    models = list(hawkes_models or sorted(mirage_df["model"].unique()))
    grouped = (
        mirage_df.groupby(["persistence", "rho", "model"], dropna=False)["n_proxy_mean"]
        .mean()
        .reset_index()
    )
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[0, idx]
        pivot = (
            grouped.loc[grouped["model"] == model]
            .pivot(index="persistence", columns="rho", values="n_proxy_mean")
            .sort_index()
        )
        if pivot.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(pivot, origin="lower", aspect="auto", cmap="magma")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
        ax.set_xlabel("rho")
        ax.set_ylabel("persistence")
        ax.set_title(f"Mirage E[n_proxy] ({model})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i, row_val in enumerate(pivot.index):
            for j, col_val in enumerate(pivot.columns):
                val = pivot.loc[row_val, col_val]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion_heatmap(confusion_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Heatmap of selection accuracy (true DGP vs selected model)."""

    if confusion_df.empty:
        raise ValueError("confusion_df is empty")

    pivot = (
        confusion_df.pivot(index="dgp", columns="model", values="win_rate")
        .fillna(0.0)
        .sort_index()
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot, origin="lower", aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("selected model")
    ax.set_ylabel("true DGP")
    ax.set_title("Selection win rate")
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[row, col]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_forecast(df: pd.DataFrame, model_name: str, fit: Optional[FitResult] = None, seed: Optional[int] = None) -> None:
    """Overlay realized counts with a simulated forecast path."""

    module = load_model(model_name)
    if fit is None:
        fit = module.fit(df)
    forecast = module.simulate(len(df), fit, np.random.default_rng(seed))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.step(df["time"], df["count"], where="post", alpha=0.5, label="realized")
    ax.step(forecast["time"], forecast["count"], where="post", alpha=0.8, label="forecast")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
