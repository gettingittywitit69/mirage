"""Mixture-kernel Hawkes process with J=2 exponential kernels."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils import FitResult, ensure_rng, poisson_loglik, prepare_exog, validate_event_df

_MIN_RATE = 1e-9
_EPS_N = 1e-4
_J = 2


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-8, 1 - 1e-8))
    return float(np.log(p / (1 - p)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    expx = np.exp(x)
    return expx / expx.sum()


def _recursion(y: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Return g[j, t] for all kernels."""

    J = betas.shape[0]
    T = len(y)
    g = np.zeros((J, T), dtype=float)
    for j in range(J):
        beta = betas[j]
        for t in range(1, T):
            g[j, t] = beta * g[j, t - 1] + y[t - 1]
    return g


def _lambda(y: np.ndarray, mu: float, betas: np.ndarray, weights: np.ndarray, n_val: float) -> np.ndarray:
    alpha = weights * n_val * (1.0 - betas)
    g = _recursion(y, betas)
    lam = mu + np.dot(alpha, g)
    return np.maximum(lam, _MIN_RATE), g


def _neg_loglik(theta: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    mu_raw = theta[0]
    b_raw = theta[1:1 + _J]
    w_raw = theta[1 + _J:1 + 2 * _J]
    n_raw = theta[-1]

    mu = float(np.exp(mu_raw))
    betas = _sigmoid(b_raw)
    weights = _softmax(w_raw)
    n_val = float(_sigmoid(n_raw) * (1.0 - _EPS_N))

    lam, g = _lambda(y, mu, betas, weights, n_val)
    loglik = float(np.sum(y * np.log(lam) - lam))
    params = {
        "mu": mu,
        "betas": betas,
        "weights": weights,
        "n": n_val,
        "alpha": weights * n_val * (1.0 - betas),
    }
    return -loglik, params


def fit(
    train_df: pd.DataFrame,
    n_components: int = _J,
    n_init: int = 3,
    max_iter: int = 100,
    rng: np.random.Generator | int | None = None,
) -> FitResult:
    """Fit a 2-kernel Hawkes model via direct likelihood optimization."""

    if n_components != _J:
        raise ValueError(f"Only n_components={_J} is supported")

    events = validate_event_df(train_df)
    y = events["count"].to_numpy(dtype=float)
    generator = ensure_rng(rng)

    best_val = np.inf
    best_params: Dict[str, float] | None = None

    mu0 = np.maximum(y.mean(), 1e-3)
    base_mu_raw = np.log(mu0)
    base_b = np.array([_logit(0.2), _logit(0.8)])
    base_w = np.zeros(_J)
    base_n_raw = _logit(0.2)

    for init_idx in range(n_init):
        theta0 = np.concatenate(
            [
                np.array([base_mu_raw + generator.normal(scale=0.2)]),
                base_b + generator.normal(scale=0.3, size=_J),
                base_w + generator.normal(scale=0.1, size=_J),
                np.array([base_n_raw + generator.normal(scale=0.2)]),
            ]
        )
        res = minimize(
            lambda th: _neg_loglik(th, y)[0],
            theta0,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-6},
        )
        if not res.success:
            continue
        neg_ll, params = _neg_loglik(res.x, y)
        if neg_ll < best_val:
            best_val = neg_ll
            best_params = params

    if best_params is None:
        raise RuntimeError("Failed to fit hawkes_dt_mixture")

    params_out: Dict[str, float] = {
        "mu": float(best_params["mu"]),
        "n": float(best_params["n"]),
        "loglik": float(-best_val),
    }
    for j in range(_J):
        params_out[f"beta_{j}"] = float(best_params["betas"][j])
        params_out[f"weight_{j}"] = float(best_params["weights"][j])
        params_out[f"alpha_{j}"] = float(best_params["alpha"][j])
    params_out["n_components"] = float(_J)

    return FitResult("hawkes_dt_mixture", params_out)


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    events = validate_event_df(test_df)
    y = events["count"].to_numpy(dtype=float)
    params = dict(fit.params)
    mu = float(params.get("mu", max(y.mean(), 1e-3)))
    betas = np.array([params.get(f"beta_{j}", 0.5) for j in range(_J)], dtype=float)
    weights = np.array([params.get(f"weight_{j}", 1.0 / _J) for j in range(_J)], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / _J
    else:
        weights = weights / weights.sum()
    n_val = float(params.get("n", 0.1))
    lam, _ = _lambda(y, mu, betas, weights, n_val)
    return poisson_loglik(y, lam)


def simulate(
    T: int,
    fit: FitResult,
    rng: np.random.Generator | int | None = None,
    exog: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    if T <= 0:
        raise ValueError("T must be positive")
    generator = ensure_rng(rng)
    params = dict(fit.params)
    mu = float(params.get("mu", 1.0))
    betas = np.array([params.get(f"beta_{j}", 0.5) for j in range(_J)], dtype=float)
    weights = np.array([params.get(f"weight_{j}", 1.0 / _J) for j in range(_J)], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / _J
    else:
        weights = weights / weights.sum()
    n_val = float(params.get("n", 0.1))
    alpha = weights * n_val * (1.0 - betas)

    exog_arr = prepare_exog(exog, T)
    y = np.zeros(T, dtype=float)
    g = np.zeros((_J,), dtype=float)
    for t in range(T):
        lam = mu + float(np.dot(alpha, g)) + exog_arr[t]
        lam = max(lam, _MIN_RATE)
        y[t] = generator.poisson(lam)
        g = betas * g + y[t]

    times = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": y})
