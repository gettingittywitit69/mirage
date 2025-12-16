"""Model orchestration utilities for rolling walk-forward tournaments."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .events import build_event_df, make_events, threshold_fixed, threshold_quantile
from .utils import FitResult, validate_event_df
from .vol_filter import apply_vol_filter, fit_vol_filter


MODEL_REGISTRY: Dict[str, str] = {
    "hawkes_dt": "criticality.models.hawkes_dt",
    "hawkes_dt_flexbaseline": "criticality.models.hawkes_dt_flexbaseline",
    "rs_poisson_dt": "criticality.models.rs_poisson_dt",
    "rs_hawkes_dt": "criticality.models.rs_hawkes_dt",
    "hawkes_dt_mixture": "criticality.models.hawkes_dt_mixture",
    "poisson_iid": "criticality.models.poisson_iid",
    "poisson_flexbaseline": "criticality.models.poisson_flexbaseline",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Grid element capturing volatility filter and event threshold choices."""

    vol_filter: str
    event_rule: str = "quantile"
    q: float | None = None
    u_fixed: float | None = None
    vol_kwargs: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vol_filter": self.vol_filter,
            "event_rule": self.event_rule,
            "q": self.q,
            "u_fixed": self.u_fixed,
            "label": self.label,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Wrapper for model name and optional fit kwargs."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class SplitWindow:
    """Train/test window for rolling splits."""

    split_id: int
    train_index: pd.Index
    test_index: pd.Index
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def available_models() -> List[str]:
    """Return the list of supported model names."""

    return list(MODEL_REGISTRY.keys())


def load_model(name: str):
    """Dynamically import a model module by registry name."""

    path = MODEL_REGISTRY.get(name)
    if path is None:
        raise ValueError(f"Unknown model '{name}'")
    return import_module(path)


def split_events(events: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a simple chronological train/test split (legacy helper)."""

    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    split_idx = int(len(events) * train_frac)
    if split_idx == 0 or split_idx == len(events):
        raise ValueError("train_frac leads to empty split")
    return events.iloc[:split_idx], events.iloc[split_idx:]


def _prepare_returns(r: pd.Series) -> pd.Series:
    series = pd.Series(r, copy=True)
    if series.empty:
        raise ValueError("Return series is empty")
    series.index = pd.to_datetime(series.index)
    if series.index.isna().any():
        raise ValueError("Return index must be datetime-like")
    return series.sort_index()


def rolling_splits(
    r: pd.Series,
    train_years: int = 10,
    test_years: int = 2,
    step_years: int = 2,
) -> List[SplitWindow]:
    """Return a list of rolling train/test windows using year-based offsets."""

    if train_years <= 0 or test_years <= 0 or step_years <= 0:
        raise ValueError("train_years, test_years, and step_years must be positive")

    returns = _prepare_returns(r)
    idx = returns.index
    first, last = idx.min(), idx.max()

    train_offset = pd.DateOffset(years=int(train_years))
    test_offset = pd.DateOffset(years=int(test_years))
    step_offset = pd.DateOffset(years=int(step_years))

    windows: List[SplitWindow] = []
    start = first
    split_id = 0

    while start < last:
        train_end_boundary = start + train_offset
        test_end_boundary = train_end_boundary + test_offset

        train_mask = (idx >= start) & (idx < train_end_boundary)
        test_mask = (idx >= train_end_boundary) & (idx < test_end_boundary)

        train_idx = idx[train_mask]
        test_idx = idx[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            windows.append(
                SplitWindow(
                    split_id=split_id,
                    train_index=train_idx,
                    test_index=test_idx,
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                )
            )
            split_id += 1

        start = start + step_offset
        if start + train_offset >= last and start >= last:
            break

    if not windows:
        raise ValueError("No viable rolling splits found; check date coverage and window sizes.")
    return windows


def _threshold_for_config(z_train: pd.Series, cfg: ExperimentConfig) -> float:
    rule = cfg.event_rule.lower()
    if rule == "quantile":
        if cfg.q is None:
            raise ValueError("Quantile rule requires q to be set on ExperimentConfig")
        return threshold_quantile(z_train.abs().to_numpy(), float(cfg.q))
    if rule == "fixed":
        if cfg.u_fixed is None:
            raise ValueError("Fixed rule requires u_fixed to be set on ExperimentConfig")
        return threshold_fixed(float(cfg.u_fixed))
    raise ValueError("event_rule must be either 'quantile' or 'fixed'")


def _event_frame(
    r_window: pd.Series,
    z_all: pd.Series,
    y_all: pd.Series,
    cfg: ExperimentConfig,
    split: SplitWindow,
    u_used: float,
) -> pd.DataFrame:
    meta = {
        "event_rule": cfg.event_rule,
        "q": cfg.q,
        "u_fixed": cfg.u_fixed,
        "u_used": u_used,
        "train_index": split.train_index,
        "test_index": split.test_index,
    }
    df = build_event_df(r_window, z_all, y_all, meta)
    df = df.copy()
    df["time"] = np.arange(1, len(df) + 1, dtype=float)
    df["count"] = df["y"].astype(float)
    return df


def _n_proxy_from_fit(fit: FitResult) -> float:
    params = dict(fit.params) if fit.params is not None else {}
    if "n" in params:
        return float(params["n"])
    if "alpha" in params and "beta" in params:
        beta = float(params.get("beta", 0.0))
        alpha = float(params["alpha"])
        denom = max(1.0 - beta, 1e-9)
        return float(alpha / denom)
    if "alpha" in params and "decay" in params:
        alpha = float(params.get("alpha", 0.0))
        decay = float(params.get("decay", 0.0))
        decay_factor = float(np.exp(-decay))
        denom = max(1.0 - decay_factor, 1e-9)
        return float(alpha / denom)
    return float("nan")


def _fit_kwargs(module: Any, base_params: Mapping[str, Any], rng: np.random.Generator | None) -> Dict[str, Any]:
    kwargs = dict(base_params or {})
    sig = inspect.signature(module.fit)
    if "rng" in sig.parameters and "rng" not in kwargs and rng is not None:
        kwargs["rng"] = rng
    return kwargs


def _spawn_rng(master: np.random.Generator) -> np.random.Generator:
    seed = int(master.integers(0, np.iinfo(np.int32).max))
    return np.random.default_rng(seed)


def run_tournament(
    r: pd.Series,
    configs: Sequence[ExperimentConfig],
    models: Sequence[ModelSpec],
    train_years: int = 10,
    test_years: int = 2,
    step_years: int = 2,
    seed: int | None = 0,
    results_path: str = "results.csv",
    summary_path: str = "summary.csv",
) -> pd.DataFrame:
    """Walk-forward evaluation over a grid of volatility/event configurations."""

    if not configs:
        raise ValueError("configs must be a non-empty sequence of ExperimentConfig")
    if not models:
        raise ValueError("models must be a non-empty sequence of ModelSpec")

    returns = _prepare_returns(r)
    splits = rolling_splits(returns, train_years=train_years, test_years=test_years, step_years=step_years)
    master_rng = np.random.default_rng(seed)

    records: List[Dict[str, Any]] = []
    config_lookup = list(configs)
    model_lookup = list(models)

    for split in splits:
        window_idx = split.train_index.union(split.test_index).sort_values()
        r_window = returns.loc[window_idx]

        for config_id, cfg in enumerate(config_lookup):
            vol_fit = fit_vol_filter(r_window.loc[split.train_index], method=cfg.vol_filter, **dict(cfg.vol_kwargs or {}))
            _, _, z_all = apply_vol_filter(r_window, vol_fit)
            z_train = z_all.loc[split.train_index]
            u_used = _threshold_for_config(z_train, cfg)
            y_all = make_events(z_all, u_used)
            event_df = _event_frame(r_window, z_all, y_all, cfg, split, u_used)
            train_df = event_df.loc[split.train_index]
            test_df = event_df.loc[split.test_index]

            for model in model_lookup:
                module = load_model(model.name)
                rng = _spawn_rng(master_rng)
                fit_kwargs = _fit_kwargs(module, model.params, rng)
                fit_result: FitResult = module.fit(train_df, **fit_kwargs)
                ll_test = module.loglik(test_df, fit_result)

                records.append(
                    {
                        "split_id": split.split_id,
                        "train_start": split.train_start,
                        "train_end": split.train_end,
                        "test_start": split.test_start,
                        "test_end": split.test_end,
                        "config_id": config_id,
                        "vol_filter": cfg.vol_filter,
                        "event_rule": cfg.event_rule,
                        "q": cfg.q,
                        "u_fixed": cfg.u_fixed,
                        "u_used": u_used,
                        "model": model.display_name,
                        "test_loglik": float(ll_test),
                        "n_proxy": _n_proxy_from_fit(fit_result),
                    }
                )

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(["split_id", "config_id", "model"]).reset_index(drop=True)
    results_df.to_csv(results_path, index=False)

    scores = (
        results_df.groupby(["config_id", "vol_filter", "event_rule", "q", "u_fixed", "model"], dropna=False)["test_loglik"]
        .sum()
        .reset_index(name="total_loglik")
    )
    winners = (
        scores.sort_values(["config_id", "total_loglik", "model"], ascending=[True, False, True])
        .groupby("config_id", group_keys=False)
        .head(1)
        .rename(columns={"model": "winner_model"})
    )

    total_configs = results_df["config_id"].nunique()
    win_counts = winners["winner_model"].value_counts().reindex(results_df["model"].unique(), fill_value=0)
    robustness = win_counts.reset_index().rename(columns={"index": "model", "winner_model": "wins"})
    robustness["wins"] = pd.to_numeric(robustness["wins"], errors="coerce").fillna(0)
    robustness["total_configs"] = total_configs
    robustness["win_rate"] = robustness["wins"] / total_configs if total_configs else np.nan

    scores["section"] = "total_score"
    winners["section"] = "winner"
    robustness["section"] = "robustness"

    summary_df = pd.concat([scores, winners, robustness], ignore_index=True, sort=False)
    summary_df.to_csv(summary_path, index=False)
    return results_df
