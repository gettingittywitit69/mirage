"""IID Poisson baseline model."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..utils import FitResult, ensure_rng, poisson_loglik, prepare_exog, validate_event_df

_MIN_RATE = 1e-9


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return validated events for Poisson fitting."""

    return validate_event_df(df)


def _intensity(length: int, mu: float, exog: np.ndarray | pd.Series | None = None) -> np.ndarray:
    exog_arr = prepare_exog(exog, length)
    lam = np.maximum(mu + exog_arr, _MIN_RATE)
    return lam


def fit(train_df: pd.DataFrame) -> FitResult:
    """Fit by maximum likelihood: mu = mean(count)."""

    events = _prepare(train_df)
    mu = float(np.maximum(events["count"].mean(), _MIN_RATE))
    params: Dict[str, float] = {"mu": mu}
    return FitResult("poisson_iid", params)


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    """Poisson log-likelihood under a constant baseline."""

    events = _prepare(test_df)
    counts = events["count"].to_numpy()
    params = dict(fit.params)
    lam = _intensity(len(counts), params["mu"])
    return poisson_loglik(counts, lam)


def simulate(
    T: int,
    fit: FitResult,
    rng: np.random.Generator | int | None = None,
    exog: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate IID Poisson counts."""

    if T <= 0:
        raise ValueError("T must be positive")
    params = dict(fit.params)
    generator = ensure_rng(rng)
    lam = _intensity(T, params["mu"], exog=exog)
    counts = generator.poisson(lam)
    times = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": counts})
