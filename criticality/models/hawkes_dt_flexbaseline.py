"""Hawkes model with a segmented baseline drift."""
from __future__ import annotations

from typing import Any, Dict

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


def _prepare_events(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    events = validate_event_df(df)
    if segment_col in events:
        return events
    events = events.copy()
    if segment_col in {"year", "month"} and "date" in events:
        dates = pd.to_datetime(events["date"], errors="coerce")
        derived = dates.dt.year if segment_col == "year" else dates.dt.month
        events[segment_col] = derived.fillna(0)
    else:
        events[segment_col] = 0
    return events


def _state_sequence(counts: np.ndarray, decay: float) -> np.ndarray:
    state = 0.0
    decay_factor = np.exp(-decay)
    states = np.zeros_like(counts, dtype=float)
    for idx, c in enumerate(counts):
        state = decay_factor * state + c
        states[idx] = state
    return states


def _solve_eta(
    counts: np.ndarray,
    offsets: np.ndarray,
    ridge: float,
    init_eta: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    if counts.size == 0:
        raise ValueError("counts must be non-empty")
    mu_init = max(float(counts.mean() - offsets.mean()), _MIN_RATE)
    eta = float(np.log(mu_init)) if init_eta is None else float(init_eta)
    for _ in range(max_iter):
        mu = float(np.exp(eta))
        lam = np.maximum(mu + offsets, _MIN_RATE)
        grad = np.sum(counts * (mu / lam) - mu) - ridge * eta
        hess = mu * (np.sum(counts / lam) - len(counts)) - (mu**2) * np.sum(counts / (lam * lam)) - ridge
        if not np.isfinite(grad):
            break
        if not np.isfinite(hess) or abs(hess) < 1e-12:
            hess = -1e-6
        step = grad / hess
        if not np.isfinite(step):
            break
        eta_new = eta - np.clip(step, -5.0, 5.0)
        if abs(eta_new - eta) < tol:
            eta = eta_new
            break
        eta = eta_new
    return float(eta)


def _segment_penalized_ll(counts: np.ndarray, offsets: np.ndarray, eta: float, ridge: float) -> float:
    mu = float(np.exp(eta))
    lam = np.maximum(mu + offsets, _MIN_RATE)
    ll = float(np.sum(counts * np.log(lam) - lam))
    return ll - 0.5 * ridge * eta * eta


def _segment_rate(seg_value: Any, params: Dict[str, float]) -> float:
    return float(params.get(_mu_key(seg_value), params.get("base_rate", _MIN_RATE)))


def _intensity(
    counts: np.ndarray,
    segments: np.ndarray,
    params: Dict[str, float],
    decay: float,
    exog: np.ndarray | None = None,
) -> np.ndarray:
    state = 0.0
    decay_factor = np.exp(-decay)
    lams = np.zeros_like(counts, dtype=float)
    exog_arr = prepare_exog(exog, len(counts))
    for idx, c in enumerate(counts):
        state = decay_factor * state + c
        base = _segment_rate(segments[idx], params)
        lams[idx] = np.maximum(_MIN_RATE, base + params["alpha"] * state + exog_arr[idx])
    return lams


def _alpha_candidates(counts: np.ndarray, num: int = 5) -> np.ndarray:
    max_alpha = float(np.clip(np.var(counts) / (np.mean(counts) + 1e-6), 0.0, 0.95))
    if not np.isfinite(max_alpha):
        max_alpha = 0.0
    hi = max(max_alpha, 0.05)
    grid = np.linspace(0.0, hi, num=num)
    return np.unique(np.clip(grid, 0.0, 0.95))


def fit(train_df: pd.DataFrame, segment_col: str = _DEFAULT_SEGMENT_COL, ridge: float = 0.1) -> FitResult:
    """Penalized MLE for a Hawkes process with segmented baseline drift."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    events = _prepare_events(train_df, segment_col)
    counts = events["count"].to_numpy()
    segments = events[segment_col].to_numpy()
    base_rate = float(np.maximum(counts.mean(), _MIN_RATE))
    decay = float(0.5)

    states = _state_sequence(counts, decay)
    unique_segments = np.unique(segments)
    total_obs = len(events)

    best_params: Dict[str, float] | None = None
    best_pen_ll = -np.inf

    for alpha in _alpha_candidates(counts):
        offsets = alpha * states
        params: Dict[str, float] = {
            "alpha": float(alpha),
            "decay": decay,
            "base_rate": base_rate,
            "ridge": float(ridge),
        }
        pen_ll = 0.0
        for seg in unique_segments:
            mask = segments == seg
            seg_counts = counts[mask]
            seg_offsets = offsets[mask]
            init_eta = np.log(max(seg_counts.mean() - seg_offsets.mean(), _MIN_RATE))
            eta = _solve_eta(seg_counts, seg_offsets, ridge, init_eta=init_eta)
            mu = float(np.maximum(np.exp(eta), _MIN_RATE))
            params[_mu_key(seg)] = mu
            params[_weight_key(seg)] = float(mask.sum() / total_obs)
            pen_ll += _segment_penalized_ll(seg_counts, seg_offsets, eta, ridge)
        if pen_ll > best_pen_ll:
            best_pen_ll = pen_ll
            best_params = params

    if best_params is None:
        best_params = {
            "alpha": 0.0,
            "decay": decay,
            "base_rate": base_rate,
            "ridge": float(ridge),
            _mu_key(0): base_rate,
            _weight_key(0): 1.0,
        }
    metadata = {"segment_col": segment_col}
    return FitResult("hawkes_dt_flexbaseline", best_params, metadata=metadata)


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    params = dict(fit.params)
    segment_col = (fit.metadata or {}).get("segment_col", _DEFAULT_SEGMENT_COL)
    events = _prepare_events(test_df, segment_col)
    counts = events["count"].to_numpy()
    segments = events[segment_col].to_numpy()
    intensity = _intensity(counts, segments, params, params["decay"])
    return poisson_loglik(counts, intensity)


def _segment_values(params: Dict[str, float]) -> list[str]:
    return [key.removeprefix("mu_") for key in params if key.startswith("mu_")]


def _segment_weights(values: list[str], params: Dict[str, float]) -> np.ndarray:
    weights = np.array([params.get(_weight_key(seg), 1.0) for seg in values], dtype=float)
    if weights.sum() <= 0 or not np.all(np.isfinite(weights)):
        weights = np.ones_like(weights)
    return weights / weights.sum()


def simulate(
    T: int,
    fit: FitResult,
    rng: np.random.Generator | int | None = None,
    exog: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    if T <= 0:
        raise ValueError("T must be positive")
    params = dict(fit.params)
    segment_col = (fit.metadata or {}).get("segment_col", _DEFAULT_SEGMENT_COL)
    generator = ensure_rng(rng)

    seg_values = _segment_values(params) or ["0"]
    weights = _segment_weights(seg_values, params)
    chosen_segments = generator.choice(seg_values, size=T, p=weights)

    counts = np.zeros(T)
    state = 0.0
    decay_factor = np.exp(-params["decay"])
    exog_arr = prepare_exog(exog, T)
    for idx in range(T):
        state = decay_factor * state + (counts[idx - 1] if idx > 0 else 0.0)
        base = _segment_rate(chosen_segments[idx], params)
        lam = np.maximum(_MIN_RATE, base + params["alpha"] * state + exog_arr[idx])
        counts[idx] = generator.poisson(lam)

    times = np.arange(1, T + 1, dtype=float)
    df = pd.DataFrame({"time": times, "count": counts, segment_col: chosen_segments})
    return df
