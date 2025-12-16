"""Poisson model with a flexible segmented baseline (no excitation)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from ..utils import FitResult, ensure_rng, poisson_loglik, prepare_exog, validate_event_df

_DEFAULT_SEGMENT_COL = "year"
_MIN_RATE = 1e-9


def _normalize_segment(value: Any) -> str:
    return str(value)


def _mu_key(seg: Any) -> str:
    return f"mu_{_normalize_segment(seg)}"


def _weight_key(seg: Any) -> str:
    return f"weight_{_normalize_segment(seg)}"


def _prepare(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    events = validate_event_df(df)
    if segment_col not in events:
        events = events.copy()
        events[segment_col] = 0
    return events


def _solve_eta(count_sum: float, n_obs: int, ridge: float, max_iter: int = 50, tol: float = 1e-8) -> float:
    """Return the penalized MLE for eta where mu = exp(eta)."""

    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    eta = float(np.log(max(count_sum / max(n_obs, 1), _MIN_RATE)))
    for _ in range(max_iter):
        grad = -count_sum + n_obs * np.exp(eta) + ridge * eta
        hess = n_obs * np.exp(eta) + ridge
        if hess <= 0:
            break
        step = grad / hess
        eta_new = eta - step
        if abs(eta_new - eta) < tol:
            eta = eta_new
            break
        eta = eta_new
    return float(eta)


def fit(train_df: pd.DataFrame, segment_col: str = _DEFAULT_SEGMENT_COL, ridge: float = 0.1) -> FitResult:
    """Estimate a segment-wise Poisson baseline with ridge shrinkage on log-rate."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    events = _prepare(train_df, segment_col)
    counts = events["count"].to_numpy()
    base_rate = float(np.maximum(counts.mean(), _MIN_RATE))

    params: Dict[str, float] = {"base_rate": base_rate, "ridge": float(ridge)}
    total_obs = len(events)
    for seg, seg_counts in events.groupby(segment_col)["count"]:
        eta = _solve_eta(float(seg_counts.sum()), len(seg_counts), ridge)
        mu = float(np.exp(eta))
        params[_mu_key(seg)] = max(mu, _MIN_RATE)
        params[_weight_key(seg)] = float(len(seg_counts) / total_obs)

    metadata = {"segment_col": segment_col}
    return FitResult("poisson_flexbaseline", params, metadata=metadata)


def _segment_rate(seg_value: Any, params: Dict[str, float]) -> float:
    return float(params.get(_mu_key(seg_value), params["base_rate"]))


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    """Poisson log-likelihood using the fitted segmented baseline."""

    params = dict(fit.params)
    segment_col = (fit.metadata or {}).get("segment_col", _DEFAULT_SEGMENT_COL)
    events = _prepare(test_df, segment_col)
    counts = events["count"].to_numpy()
    seg_series = events[segment_col]
    lams = np.array([_segment_rate(seg, params) for seg in seg_series], dtype=float)
    lams = np.maximum(lams, _MIN_RATE)
    return poisson_loglik(counts, lams)


def _segment_values(params: Dict[str, float]) -> List[str]:
    return [key.removeprefix("mu_") for key in params if key.startswith("mu_")]


def _segment_weights(values: Iterable[str], params: Dict[str, float]) -> np.ndarray:
    weights = np.array([params.get(_weight_key(seg), 1.0) for seg in values], dtype=float)
    if weights.sum() <= 0 or not np.all(np.isfinite(weights)):
        weights = np.ones_like(weights, dtype=float)
    weights = weights / weights.sum()
    return weights


def simulate(
    T: int,
    fit: FitResult,
    rng: np.random.Generator | int | None = None,
    exog: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate counts under the segmented baseline."""

    if T <= 0:
        raise ValueError("T must be positive")
    params = dict(fit.params)
    segment_col = (fit.metadata or {}).get("segment_col", _DEFAULT_SEGMENT_COL)
    generator = ensure_rng(rng)

    seg_values = _segment_values(params) or ["0"]
    weights = _segment_weights(seg_values, params)
    chosen_segments = generator.choice(seg_values, size=T, p=weights)
    exog_arr = prepare_exog(exog, T)
    lams = np.array([_segment_rate(seg, params) for seg in chosen_segments], dtype=float) + exog_arr
    lams = np.maximum(lams, _MIN_RATE)
    counts = generator.poisson(lams)

    times = np.arange(1, T + 1, dtype=float)
    df = pd.DataFrame({"time": times, "count": counts, segment_col: chosen_segments})
    return df
