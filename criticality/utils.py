"""Utility helpers shared across modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import gammaln


@dataclass
class FitResult:
    """Container representing the outcome of a model fit."""

    model_name: str
    params: Mapping[str, float]
    metadata: Optional[Mapping[str, Union[float, str]]] = None
    mu: Optional[pd.Series] = None
    sigma: Optional[pd.Series] = None

    def to_series(self) -> pd.Series:
        """Return parameters as a pandas Series for convenience."""

        return pd.Series(self.params, name=self.model_name)


def ensure_rng(rng: Optional[Union[np.random.Generator, int]]) -> np.random.Generator:
    """Return a numpy Generator based on a seed or generator."""

    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def prepare_exog(exog: Optional[Union[np.ndarray, pd.Series, Iterable[float]]], length: int) -> np.ndarray:
    """Return an exogenous array aligned to the requested length."""

    if exog is None:
        return np.zeros(length, dtype=float)
    arr = np.asarray(list(exog), dtype=float)
    if arr.shape[0] != length:
        raise ValueError(f"Expected exog of length {length}, got {arr.shape[0]}")
    return arr


def validate_event_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the event DataFrame has the required structure."""

    required = {"time", "count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ordered = df.sort_values("time").reset_index(drop=True)
    ordered["time"] = ordered["time"].astype(float)
    ordered["count"] = ordered["count"].astype(float)
    return ordered


def poisson_loglik(counts: Iterable[float], rates: Iterable[float]) -> float:
    """Compute the Poisson log-likelihood for count/rate pairs."""

    counts_arr = np.asarray(list(counts), dtype=float)
    rates_arr = np.asarray(list(rates), dtype=float)
    if counts_arr.shape != rates_arr.shape:
        raise ValueError("Counts and rates must align.")
    if np.any(rates_arr <= 0):
        raise ValueError("Rates must be positive.")
    log_fact = gammaln(counts_arr + 1.0)
    return float(np.sum(counts_arr * np.log(rates_arr) - rates_arr - log_fact))
