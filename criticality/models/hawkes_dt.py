"""Discrete-time Hawkes model with exponential kernel."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..utils import FitResult, ensure_rng, poisson_loglik, prepare_exog, validate_event_df


_DEF_DECAY = 0.5


def _hawkes_intensity(
    counts: np.ndarray,
    baseline: float,
    alpha: float,
    decay: float,
    exog: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-bin intensities for a Hawkes process."""

    state = 0.0
    lams = np.zeros_like(counts, dtype=float)
    decay_factor = np.exp(-decay)
    exog_arr = prepare_exog(exog, len(counts))
    for idx, c in enumerate(counts):
        state = decay_factor * state + c
        lams[idx] = max(1e-9, baseline + alpha * state + exog_arr[idx])
    return lams


def fit(train_df: pd.DataFrame) -> FitResult:
    """Estimate Hawkes parameters via simple heuristics."""

    events = validate_event_df(train_df)
    counts = events["count"].to_numpy()
    baseline = float(np.maximum(1e-6, counts.mean()))
    if len(counts) > 1:
        corr = np.corrcoef(counts[1:], counts[:-1])[0, 1]
    else:
        corr = 0.0
    alpha = float(np.clip(np.nan_to_num(corr), 0.0, 0.95))
    decay = float(_DEF_DECAY + np.maximum(0.0, counts.var()) / (1.0 + counts.var()))
    params: Dict[str, float] = {"baseline": baseline, "alpha": alpha, "decay": decay}
    return FitResult("hawkes_dt", params)


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    """Return the Poisson log-likelihood of the Hawkes intensities."""

    events = validate_event_df(test_df)
    counts = events["count"].to_numpy()
    params = dict(fit.params)
    intensity = _hawkes_intensity(counts, params["baseline"], params["alpha"], params["decay"])
    return poisson_loglik(counts, intensity)


def simulate(
    T: int,
    fit: FitResult,
    rng: np.random.Generator | int | None = None,
    exog: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate counts from the fitted Hawkes model for T bins."""

    if T <= 0:
        raise ValueError("T must be positive")
    params = dict(fit.params)
    generator = ensure_rng(rng)
    counts = np.zeros(T)
    state = 0.0
    decay_factor = np.exp(-params["decay"])
    exog_arr = prepare_exog(exog, T)
    for idx in range(T):
        state = decay_factor * state + (counts[idx - 1] if idx > 0 else 0.0)
        lam = max(1e-9, params["baseline"] + params["alpha"] * state + exog_arr[idx])
        counts[idx] = generator.poisson(lam)
    times = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": counts})
