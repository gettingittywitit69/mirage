"""Volatility filters for daily return series."""
from __future__ import annotations

import logging
from typing import Mapping, Tuple

import numpy as np
import pandas as pd

from .utils import FitResult

_MIN_SIGMA = 1e-8
_DEFAULT_SPAN = 20.0
logger = logging.getLogger(__name__)


def _prepare_returns(r: pd.Series) -> pd.Series:
    series = pd.Series(r, copy=True).astype(float)
    return series.replace([np.inf, -np.inf], np.nan)


def _ensure_positive(series: pd.Series, min_value: float) -> pd.Series:
    clipped = series.copy()
    clipped = clipped.clip(lower=min_value)
    if clipped.isna().any():
        clipped = clipped.fillna(method="ffill").fillna(method="bfill")
    return clipped.clip(lower=min_value)


def _span_to_alpha(span: float) -> float:
    if span <= 0:
        raise ValueError("span must be positive")
    return 2.0 / (span + 1.0)


def _initial_variance(value: float | None, min_sigma: float) -> float:
    if value is None:
        return float(min_sigma**2)
    return float(max(min_sigma**2, value))


def _ewma_sigma(
    returns: pd.Series,
    mu: float,
    alpha: float,
    min_sigma: float,
    initial_var: float,
) -> Tuple[pd.Series, float]:
    resid = _prepare_returns(returns) - mu
    var = float(initial_var)
    sigmas = []
    for r_t in resid:
        sigma_t = max(min_sigma, float(np.sqrt(var)))
        sigmas.append(sigma_t)
        if not np.isfinite(r_t):
            continue
        var = (1.0 - alpha) * var + alpha * float(r_t) ** 2
        if not np.isfinite(var) or var < min_sigma**2:
            var = float(min_sigma**2)
    sigma_series = pd.Series(sigmas, index=returns.index)
    return sigma_series, var


def _compute_z(r: pd.Series, mu: pd.Series, sigma: pd.Series, min_sigma: float) -> pd.Series:
    returns = _prepare_returns(r)
    mu_aligned = mu.reindex(returns.index)
    sigma_aligned = _ensure_positive(sigma.reindex(returns.index), min_sigma)
    z = (returns - mu_aligned) / sigma_aligned
    z.loc[returns.isna()] = np.nan
    return z


def _print_train_stats(label: str, z_train: pd.Series) -> None:
    clean = z_train.dropna()
    mean_val = float(clean.mean()) if not clean.empty else np.nan
    std_val = float(clean.std(ddof=0)) if not clean.empty else np.nan
    print(f"[{label}] train z.mean={mean_val:.6f}, z.std={std_val:.6f}")


def _fit_ewma(
    r_train: pd.Series,
    min_sigma: float,
    span: float,
    alpha: float | None = None,
    initial_variance: float | None = None,
    model_name: str = "ewma",
    backend: str = "ewma",
) -> FitResult:
    returns = _prepare_returns(r_train)
    mu_val = float(returns.dropna().mean()) if not returns.dropna().empty else 0.0
    alpha_val = float(alpha) if alpha is not None else _span_to_alpha(span)
    init_var = _initial_variance(initial_variance, min_sigma)
    sigma_train, terminal_var = _ewma_sigma(returns, mu_val, alpha_val, min_sigma, init_var)
    mu_series = pd.Series(mu_val, index=returns.index)
    z_train = _compute_z(returns, mu_series, sigma_train, min_sigma)
    _print_train_stats(model_name, z_train)
    params: Mapping[str, float] = {
        "mu": mu_val,
        "span": float(span),
        "alpha": alpha_val,
        "initial_variance": init_var,
        "min_sigma": min_sigma,
        "terminal_var": float(terminal_var),
    }
    metadata: Mapping[str, float | str] = {"method": model_name, "backend": backend, "train_len": len(r_train)}
    return FitResult(model_name, params, metadata=metadata, mu=mu_series, sigma=sigma_train)


def _fit_garch_if_available(r_train: pd.Series, min_sigma: float, **kw) -> FitResult:
    span = float(kw.pop("span", _DEFAULT_SPAN))
    alpha = kw.pop("alpha", None)
    initial_variance = kw.pop("initial_variance", None)
    try:
        import arch  # type: ignore
    except ImportError:
        logger.info("arch is not installed; falling back to EWMA.")
        return _fit_ewma(
            r_train,
            min_sigma=min_sigma,
            span=span,
            alpha=alpha,
            initial_variance=initial_variance,
            model_name="garch_if_available",
            backend="ewma_fallback",
        )
    returns = _prepare_returns(r_train)
    mu_val = float(returns.dropna().mean()) if not returns.dropna().empty else 0.0
    resid = returns - mu_val
    clean_resid = resid.dropna()
    if clean_resid.empty:
        return _fit_ewma(
            r_train,
            min_sigma=min_sigma,
            span=span,
            alpha=alpha,
            initial_variance=initial_variance,
            model_name="garch_if_available",
            backend="ewma_fallback",
        )
    model = arch.arch_model(clean_resid, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res = model.fit(disp="off", show_warning=False)
    sigma_clean = pd.Series(res.conditional_volatility, index=clean_resid.index)
    sigma_full = sigma_clean.reindex(returns.index)
    sigma_full = sigma_full.ffill().bfill()
    sigma_full = _ensure_positive(sigma_full, min_sigma)
    mu_series = pd.Series(mu_val, index=returns.index)
    z_train = _compute_z(returns, mu_series, sigma_full, min_sigma)
    _print_train_stats("garch_if_available", z_train)
    param_lookup = {name.lower(): float(val) for name, val in res.params.items()}
    omega = param_lookup.get("omega", 0.0)
    alpha1 = param_lookup.get("alpha[1]", param_lookup.get("alpha1", 0.0))
    beta1 = param_lookup.get("beta[1]", param_lookup.get("beta1", 0.0))
    params: Mapping[str, float] = {
        "mu": mu_val,
        "omega": omega,
        "alpha": alpha1,
        "beta": beta1,
        "min_sigma": min_sigma,
        "initial_variance": float(res.conditional_volatility.iloc[0] ** 2),
    }
    metadata: Mapping[str, float | str] = {
        "method": "garch_if_available",
        "backend": "arch",
        "train_len": len(r_train),
        "dist": getattr(res, "distribution", None).name if hasattr(res, "distribution") else "normal",
    }
    return FitResult("garch_if_available", params, metadata=metadata, mu=mu_series, sigma=sigma_full)


def fit_vol_filter(r_train: pd.Series, method: str, **kw) -> FitResult:
    """Fit a volatility filter to a training return series."""

    method_key = method.lower()
    min_sigma = float(kw.pop("min_sigma", _MIN_SIGMA))
    if method_key == "none":
        returns = _prepare_returns(r_train)
        mu_series = pd.Series(0.0, index=returns.index)
        sigma_series = pd.Series(1.0, index=returns.index)
        z_train = _compute_z(returns, mu_series, sigma_series, min_sigma)
        _print_train_stats("none", z_train)
        params: Mapping[str, float] = {"mu": 0.0, "sigma": 1.0, "min_sigma": min_sigma}
        metadata: Mapping[str, float | str] = {"method": "none", "train_len": len(r_train)}
        return FitResult("none", params, metadata=metadata, mu=mu_series, sigma=sigma_series)
    if method_key == "ewma":
        span = float(kw.pop("span", _DEFAULT_SPAN))
        alpha = kw.pop("alpha", None)
        initial_variance = kw.pop("initial_variance", None)
        return _fit_ewma(
            r_train,
            min_sigma=min_sigma,
            span=span,
            alpha=alpha,
            initial_variance=initial_variance,
        )
    if method_key == "garch_if_available":
        return _fit_garch_if_available(r_train, min_sigma=min_sigma, **kw)
    raise ValueError(f"Unknown volatility filter method '{method}'.")


def _apply_ewma(r_all: pd.Series, fit: FitResult, min_sigma: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    params = fit.params
    span = float(params.get("span", _DEFAULT_SPAN))
    alpha = float(params.get("alpha", _span_to_alpha(span)))
    init_var = _initial_variance(params.get("initial_variance"), min_sigma)
    mu_val = float(params.get("mu", _prepare_returns(r_all).dropna().mean()))
    mu_series = pd.Series(mu_val, index=r_all.index)
    sigma_series, _ = _ewma_sigma(_prepare_returns(r_all), mu_val, alpha, min_sigma, init_var)
    z = _compute_z(r_all, mu_series, sigma_series, min_sigma)
    return mu_series, sigma_series, z


def _apply_garch(
    r_all: pd.Series,
    fit: FitResult,
    min_sigma: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    params = fit.params
    mu_val = float(params.get("mu", 0.0))
    omega = float(params.get("omega", 0.0))
    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta", 0.0))
    init_var = _initial_variance(params.get("initial_variance"), min_sigma)
    returns = _prepare_returns(r_all)
    resid = returns - mu_val
    var = init_var
    sigma_list = []
    for r_t in resid:
        sigma_t = max(min_sigma, float(np.sqrt(var)))
        sigma_list.append(sigma_t)
        if not np.isfinite(r_t):
            continue
        var = omega + alpha * float(r_t) ** 2 + beta * var
        if not np.isfinite(var) or var < min_sigma**2:
            var = float(min_sigma**2)
    sigma_series = pd.Series(sigma_list, index=returns.index)
    mu_series = pd.Series(mu_val, index=returns.index)
    z = _compute_z(returns, mu_series, sigma_series, min_sigma)
    return mu_series, sigma_series, z


def apply_vol_filter(r_all: pd.Series, fit: FitResult) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Apply a fitted filter to full returns, returning (mu, sigma, z)."""

    method = fit.model_name.lower()
    min_sigma = float(fit.params.get("min_sigma", _MIN_SIGMA))
    backend = (fit.metadata or {}).get("backend", method)
    if method == "none":
        returns = _prepare_returns(r_all)
        mu_series = pd.Series(0.0, index=returns.index)
        sigma_series = pd.Series(1.0, index=returns.index)
        z = _compute_z(returns, mu_series, sigma_series, min_sigma)
        return mu_series, sigma_series, z
    if method == "ewma" or backend == "ewma_fallback":
        return _apply_ewma(r_all, fit, min_sigma)
    if method == "garch_if_available":
        if backend == "arch":
            return _apply_garch(r_all, fit, min_sigma)
        return _apply_ewma(r_all, fit, min_sigma)
    raise ValueError(f"Unknown volatility filter method '{fit.model_name}'.")


# Backwards compatible helpers retained for callers expecting legacy API.
def standardized_shocks(r: pd.Series, fit: FitResult, min_sigma: float = _MIN_SIGMA) -> pd.Series:
    """Return standardized shocks z_t = (r_t - mu_t) / sigma_t."""

    _, _, z = apply_vol_filter(r, fit)
    return z


def ewma_vol(counts: pd.Series, span: float) -> pd.Series:
    """Compute an exponential weighted moving average volatility."""

    if span <= 0:
        raise ValueError("span must be positive")
    return counts.ewm(span=span, adjust=False).std().fillna(0.0)


def zscore(counts: pd.Series, window: int) -> pd.Series:
    """Return a rolling z-score for event counts."""

    if window <= 1:
        raise ValueError("window must be > 1")
    rolling = counts.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    z = (counts - mean) / std
    return z.fillna(0.0)


def normalize_counts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Append normalized count metrics to an event DataFrame."""

    if "count" not in df:
        raise ValueError("DataFrame must contain a 'count' column")
    enriched = df.copy()
    enriched["vol"] = ewma_vol(enriched["count"], span=max(2, window))
    enriched["zscore"] = zscore(enriched["count"], window=max(3, window))
    return enriched
