"""Event-centric helper routines."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_rng


def simulate_poisson(rate: float, T: float, rng: np.random.Generator | int | None = None) -> pd.DataFrame:
    """Simulate a homogeneous Poisson event count process."""

    if rate <= 0:
        raise ValueError("rate must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    generator = ensure_rng(rng)
    n_bins = int(T)
    counts = generator.poisson(rate, size=n_bins)
    times = np.arange(1, n_bins + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": counts})


def interarrival_times(timestamps: Sequence[float]) -> np.ndarray:
    """Return inter-arrival times of a timestamp series."""

    arr = np.sort(np.asarray(timestamps, dtype=float))
    if arr.size < 2:
        return np.array([], dtype=float)
    return np.diff(arr)


def thinning_mask(intensity: Iterable[float], baseline: float) -> np.ndarray:
    """Return mask representing thinning probability for Ogata's algorithm."""

    lam = np.asarray(list(intensity), dtype=float)
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    prob = np.clip(lam / (lam.max(initial=baseline) + baseline), 0.0, 1.0)
    return prob


def threshold_quantile(abs_z_train: np.ndarray, q: float) -> float:
    """Return the quantile-based threshold using training-only absolute z-scores."""

    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    arr = np.asarray(abs_z_train, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("abs_z_train must contain at least one finite value")
    threshold = np.quantile(finite, q)
    return float(threshold)


def threshold_fixed(u_fixed: float) -> float:
    """Return a validated fixed event threshold."""

    u_val = float(u_fixed)
    if not np.isfinite(u_val):
        raise ValueError("u_fixed must be finite")
    if u_val <= 0:
        raise ValueError("u_fixed must be positive")
    return u_val


def make_events(z: pd.Series, u: float) -> pd.Series:
    """Return the binary event indicator y_t = 1{|z_t| > u}."""

    u_val = float(u)
    if u_val <= 0:
        raise ValueError("u must be positive")
    z_series = pd.Series(z, copy=True).astype(float)
    events = (z_series.abs() > u_val).astype(int)
    events.loc[z_series.isna()] = 0
    events.name = "y"
    return events


def _enforce_train_test_split(
    meta: Mapping[str, Any],
    abs_z: pd.Series,
    index: pd.Index,
    event_rule: str,
    u_used: float,
) -> None:
    train_idx_meta = meta.get("train_index")
    test_idx_meta = meta.get("test_index")

    if train_idx_meta is None and test_idx_meta is None:
        if event_rule == "quantile":
            raise AssertionError("train_index and test_index are required for quantile thresholds")
        return
    if train_idx_meta is None or test_idx_meta is None:
        raise AssertionError("train_index and test_index must be provided together")

    train_idx = pd.Index(train_idx_meta)
    test_idx = pd.Index(test_idx_meta)
    overlap = train_idx.intersection(test_idx)
    if not overlap.empty:
        raise AssertionError("train_index and test_index must be disjoint")

    index = pd.Index(index)
    missing_train = train_idx.difference(index)
    missing_test = test_idx.difference(index)
    if not missing_train.empty or not missing_test.empty:
        raise AssertionError("train_index/test_index must align to the provided data index")

    if event_rule == "quantile":
        q_val = meta.get("q")
        if q_val is None:
            raise AssertionError("q must be supplied for quantile thresholds")
        if not np.isfinite(u_used):
            raise AssertionError("u_used must be finite for quantile thresholds")
        abs_train = abs_z.loc[train_idx]
        expected_u = threshold_quantile(abs_train.to_numpy(), float(q_val))
        if not np.isclose(u_used, expected_u):
            raise AssertionError("u_used must be computed from the training indices only")


def build_event_df(
    r: pd.Series,
    z: pd.Series,
    y: pd.Series,
    meta: Mapping[str, Any],
) -> pd.DataFrame:
    """Assemble a tidy event dataframe with metadata columns."""

    r_series = pd.Series(r, copy=True)
    z_series = pd.Series(z, copy=True).astype(float)
    y_series = pd.Series(y, copy=True)

    if len(r_series) != len(z_series) or len(z_series) != len(y_series):
        raise ValueError("r, z, and y must have the same length")
    if not r_series.index.equals(z_series.index) or not z_series.index.equals(y_series.index):
        raise ValueError("r, z, and y must share the same index")

    event_rule = str(meta.get("event_rule", "")).lower()
    if event_rule not in {"quantile", "fixed"}:
        raise ValueError("event_rule must be either 'quantile' or 'fixed'")

    q_val = meta.get("q")
    u_fixed_val = meta.get("u_fixed")
    u_used_val = meta.get("u_used")
    u_used_float = float(u_used_val) if u_used_val is not None else np.nan

    date_index = pd.to_datetime(z_series.index)
    if date_index.isna().any():
        raise ValueError("index must be coercible to datetime to build event dataframe")

    abs_z = z_series.abs()
    _enforce_train_test_split(meta, abs_z, z_series.index, event_rule, u_used_float)

    df = pd.DataFrame(
        {
            "date": date_index,
            "r": r_series.astype(float).to_numpy(),
            "z": z_series,
            "abs_z": abs_z,
            "y": y_series.fillna(0).astype(int),
        },
        index=z_series.index,
    )
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["event_rule"] = event_rule
    df["q"] = float(q_val) if q_val is not None else np.nan
    df["u_fixed"] = float(u_fixed_val) if u_fixed_val is not None else np.nan
    df["u_used"] = u_used_float
    return df
