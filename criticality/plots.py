"""Plotting helpers built on matplotlib."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
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
    ax.set_axisbelow(True)
    fig.tight_layout()


def plot_cumulative_scores(results: pd.DataFrame, config_id: Optional[int] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Line plot of cumulative out-of-sample score by model for a single config."""

    if results.empty:
        raise ValueError("results dataframe is empty")
    for required in ("config_id", "test_loglik", "split_id", "model"):
        if required not in results.columns:
            raise ValueError(f"results dataframe missing required column '{required}'")

    if config_id is None:
        totals = results.groupby("config_id", dropna=False)["test_loglik"].sum()
        if totals.empty:
            raise ValueError("results dataframe has no config_id groups")
        cfg_id = int(totals.idxmax())
    else:
        cfg_id = config_id
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
        _, ax = plt.subplots(figsize=(8, 4.5))
    styles = ["solid", "dashed", "dotted", (0, (3, 1, 1, 1))]
    colors = cm.get_cmap("tab10").colors
    for idx, model in enumerate(cumsums.columns):
        color = colors[idx % len(colors)]
        ax.plot(
            cumsums.index,
            cumsums[model],
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            linestyle=styles[idx % len(styles)],
            color=color,
            label=str(model),
        )
        final_y = cumsums[model].iloc[-1]
        ax.annotate(
            f"{model}: {final_y:.1f}",
            xy=(cumsums.index[-1], final_y),
            xytext=(5, (idx - len(cumsums.columns) / 2) * 4),
            textcoords="offset points",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(f"Cumulative test log-lik by model (config {cfg_id})")
    ax.set_xlabel("split id")
    ax.set_ylabel("cumulative test log-lik")
    ax.legend(frameon=False, ncol=min(3, len(cumsums.columns)), loc="upper left")
    ax.margins(x=0.02)
    ax.set_axisbelow(True)
    return ax


def plot_winner_frequency(results: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Bar plot of winner frequency across configs (robustness)."""

    if results.empty:
        raise ValueError("results dataframe is empty")
    for required in ("config_id", "model", "test_loglik"):
        if required not in results.columns:
            raise ValueError(f"results dataframe missing required column '{required}'")

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
        _, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(freqs.index, freqs.values, color="#4c72b0", alpha=0.9, width=0.6)
    ax.set_ylabel("win rate across configs")
    ax.set_ylim(0, 1.08)
    ax.set_title("Winner frequency (robustness)")
    for bar, val in zip(bars, freqs.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.025,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_axisbelow(True)
    return ax


def plot_n_proxy_distribution(results: pd.DataFrame, models: Optional[Sequence[str]] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Show distribution of fitted n_proxy across configs/splits."""

    if "n_proxy" not in results.columns:
        raise ValueError("results dataframe missing required column 'n_proxy'")
    data = results.copy()
    data = data[np.isfinite(data["n_proxy"])]
    if models:
        data = data[data["model"].isin(models)]
    if data.empty:
        raise ValueError("No finite n_proxy values to plot")

    order = sorted(data["model"].unique())
    grouped = [data.loc[data["model"] == m, "n_proxy"] for m in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(
        grouped,
        labels=order,
        patch_artist=True,
        medianprops={"color": "black"},
        whiskerprops={"color": "#555"},
        capprops={"color": "#555"},
        boxprops={"linewidth": 1.2, "color": "#555"},
        widths=0.55,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9d9d9")
    ax.set_ylabel("n_proxy")
    ax.set_title("Distribution of n_proxy across configs")
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    return ax


def plot_mirage_heatmap(mirage_df: pd.DataFrame, hawkes_models: Optional[Sequence[str]] = None) -> plt.Figure:
    """Heatmap of E[n_proxy_hat | true n=0] over (rho, persistence) for Hawkes-family fits."""

    if mirage_df.empty:
        raise ValueError("mirage_df is empty")

    df = mirage_df.copy()
    if "true_n" in df.columns:
        df = df[np.isclose(df["true_n"], 0.0)]
    if df.empty:
        raise ValueError("No rows where true_n is 0 for mirage heatmap")

    preferred_models = list(hawkes_models or ["hawkes_dt", "rs_hawkes_dt"])
    available = [m for m in preferred_models if m in set(df["model"])]
    if not available:
        available = sorted(df["model"].unique())
    df = df[df["model"].isin(available)]
    if df.empty:
        raise ValueError("No Hawkes-family models available for mirage heatmap")

    value_col = "n_proxy_mean" if "n_proxy_mean" in df.columns else "n_proxy"
    if value_col not in df.columns:
        raise ValueError("mirage_df missing n_proxy/n_proxy_mean column")
    grouped = (
        df.groupby(["dgp", "persistence", "rho", "model"], dropna=False)[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "n_proxy_mean"})
    )
    grouped = grouped[np.isfinite(grouped["n_proxy_mean"])]
    if grouped.empty:
        raise ValueError("mirage_df has no finite n_proxy_mean values after filtering")
    dgp_levels = sorted(grouped["dgp"].unique()) if "dgp" in grouped.columns else ["all"]
    n_models, n_dgps = len(available), len(dgp_levels)
    fig_width = max(7.0, 4.8 * n_models)
    fig_height = max(4.0, 3.5 * n_dgps)
    fig, axes = plt.subplots(
        n_dgps,
        n_models,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    vmin, vmax = grouped["n_proxy_mean"].min(), grouped["n_proxy_mean"].max()
    mid = (vmin + vmax) / 2.0
    cmap = plt.get_cmap("cividis")
    im = None

    for row_idx, dgp in enumerate(dgp_levels):
        for col_idx, model in enumerate(available):
            ax = axes[row_idx, col_idx]
            pivot = (
                grouped.loc[(grouped["model"] == model) & (grouped["dgp"] == dgp)]
                .pivot(index="persistence", columns="rho", values="n_proxy_mean")
                .sort_index()
            )
            if pivot.empty:
                ax.set_visible(False)
                continue

            im = ax.imshow(pivot.values, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=11)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{r:.2f}" for r in pivot.index], fontsize=11)
            ax.set_xlabel("rho", fontsize=12)
            ax.set_ylabel("persistence", fontsize=12)
            title = f"{model} | DGP={dgp}" if dgp != "all" else f"{model}"
            ax.set_title(title, fontsize=13, fontweight="bold")

            # Light borders to separate cells
            for x in np.arange(-0.5, len(pivot.columns), 1):
                ax.axvline(x, color="white", linewidth=0.8, alpha=0.7)
            for y in np.arange(-0.5, len(pivot.index), 1):
                ax.axhline(y, color="white", linewidth=0.8, alpha=0.7)
            ax.tick_params(axis="both", which="both", length=0)

            for i, row_val in enumerate(pivot.index):
                for j, col_val in enumerate(pivot.columns):
                    val = pivot.loc[row_val, col_val]
                    text_color = "white" if val > mid else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=11)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
        cbar.set_label("E[n_proxy_hat | true n=0]", fontsize=12)
        cbar.ax.tick_params(labelsize=11)
    return fig


def plot_confusion_heatmap(confusion_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Heatmap of selection accuracy (true DGP vs selected model)."""

    if confusion_df.empty:
        raise ValueError("confusion_df is empty")
    for required in ("dgp", "model", "win_rate"):
        if required not in confusion_df.columns:
            raise ValueError(f"confusion_df missing required column '{required}'")

    pivot = (
        confusion_df.pivot(index="dgp", columns="model", values="win_rate")
        .fillna(0.0)
        .sort_index()
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
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
            text_color = "black" if val < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="win rate")
    ax.set_axisbelow(True)
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
