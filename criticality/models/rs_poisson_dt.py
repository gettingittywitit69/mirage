"""Regime-switching Poisson HMM (no excitation)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import gammaln

from ..utils import FitResult, ensure_rng, prepare_exog, validate_event_df

_MIN_RATE = 1e-9
_MIN_PROB = 1e-12


def _mu_key(k: int) -> str:
    return f"mu_{k}"


def _pi_key(k: int) -> str:
    return f"pi_{k}"


def _a_key(i: int, j: int) -> str:
    return f"a_{i}_{j}"


def _logsumexp(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_val = np.max(arr, axis=axis, keepdims=True)
    stable = np.log(np.sum(np.exp(arr - max_val), axis=axis, keepdims=True)) + max_val
    if axis is None:
        return float(stable.squeeze())
    return stable.squeeze(axis)


def _prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    events = validate_event_df(df)
    return events


def _log_pois(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
    rates = np.maximum(rates, _MIN_RATE)
    return counts[:, None] * np.log(rates[None, :]) - rates[None, :] - gammaln(counts[:, None] + 1.0)


def _forward_backward(log_pi: np.ndarray, log_A: np.ndarray, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    T, K = log_b.shape
    log_alpha = np.full((T, K), -np.inf)
    log_beta = np.zeros((T, K))

    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(log_A + log_b[t + 1] + log_beta[t + 1], axis=1)

    loglik = _logsumexp(log_alpha[-1])
    return log_alpha, log_beta, float(loglik)


def _fit_single(
    counts: np.ndarray,
    n_states: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    init_mu: np.ndarray | None = None,
) -> tuple[Dict[str, float], float]:
    K = n_states
    if init_mu is None:
        mean = max(counts.mean(), 0.1)
        init_mu = mean * rng.uniform(0.5, 1.5, size=K)
    mu = np.maximum(init_mu, _MIN_RATE)
    pi = rng.dirichlet(np.ones(K))
    A = np.array([rng.dirichlet(np.ones(K)) for _ in range(K)])

    prev_loglik = -np.inf
    for _ in range(max_iter):
        log_b = _log_pois(counts, mu)
        log_pi = np.log(np.maximum(pi, _MIN_PROB))
        log_A = np.log(np.maximum(A, _MIN_PROB))
        log_alpha, log_beta, loglik = _forward_backward(log_pi, log_A, log_b)

        if loglik - prev_loglik < tol:
            break
        prev_loglik = loglik

        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        xi_log = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + log_b[1:, None, :]
            + log_beta[1:, None, :]
            - loglik
        )
        xi = np.exp(xi_log)

        pi = np.maximum(gamma[0], _MIN_PROB)
        pi = pi / pi.sum()

        xi_sum = xi.sum(axis=0)
        A = np.maximum(xi_sum, _MIN_PROB)
        A = A / A.sum(axis=1, keepdims=True)

        weighted_counts = gamma.T @ counts
        weights = gamma.sum(axis=0)
        mu = np.maximum(weighted_counts / np.maximum(weights, _MIN_PROB), _MIN_RATE)

    params: Dict[str, float] = {"n_states": float(K), "loglik": float(prev_loglik)}
    for k in range(K):
        params[_mu_key(k)] = float(mu[k])
        params[_pi_key(k)] = float(pi[k])
    for i in range(K):
        for j in range(K):
            params[_a_key(i, j)] = float(A[i, j])
    return params, float(prev_loglik)


def _best_init_mus(counts: np.ndarray, n_states: int) -> np.ndarray:
    quantiles = np.linspace(0.1, 0.9, n_states)
    mus = np.maximum(np.quantile(counts, quantiles), _MIN_RATE)
    return mus


def fit(
    train_df: pd.DataFrame,
    n_states: int = 2,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-4,
    rng: np.random.Generator | int | None = None,
) -> FitResult:
    """Fit a regime-switching Poisson HMM using Baum–Welch (log-space)."""

    if n_states < 2 or n_states > 4:
        raise ValueError("n_states must be between 2 and 4")
    if n_init <= 0:
        raise ValueError("n_init must be positive")
    events = _prepare_events(train_df)
    counts = events["count"].to_numpy()
    generator = ensure_rng(rng)

    best_params: Dict[str, float] | None = None
    best_loglik = -np.inf
    init_mu_grid = [_best_init_mus(counts, n_states)]
    while len(init_mu_grid) < n_init:
        init_mu_grid.append(None)

    for init_mu in init_mu_grid[:n_init]:
        params, ll = _fit_single(counts, n_states, max_iter, tol, generator, init_mu=init_mu)
        if ll > best_loglik:
            best_loglik = ll
            best_params = params

    if best_params is None:
        raise RuntimeError("Failed to fit rs_poisson_dt")

    metadata = {"n_states": int(n_states)}
    return FitResult("rs_poisson_dt", best_params, metadata=metadata)


def _params_to_arrays(fit: FitResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = dict(fit.params)
    K = int(params.get("n_states", (fit.metadata or {}).get("n_states", 2)))
    mu = np.array([params.get(_mu_key(k), 1.0) for k in range(K)], dtype=float)
    pi = np.array([params.get(_pi_key(k), 1.0 / K) for k in range(K)], dtype=float)
    A = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            A[i, j] = params.get(_a_key(i, j), 1.0 / K)
    # Normalize defensively
    mu = np.maximum(mu, _MIN_RATE)
    pi = np.maximum(pi, _MIN_PROB)
    pi = pi / pi.sum()
    A = np.maximum(A, _MIN_PROB)
    A = A / A.sum(axis=1, keepdims=True)
    return mu, pi, A


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    """Filtering log-likelihood via forward pass."""

    events = _prepare_events(test_df)
    counts = events["count"].to_numpy()
    mu, pi, A = _params_to_arrays(fit)
    log_b = _log_pois(counts, mu)
    log_pi = np.log(pi)
    log_A = np.log(A)
    _, _, loglik_val = _forward_backward(log_pi, log_A, log_b)
    return float(loglik_val)


def smoothed_probs(df: pd.DataFrame, fit: FitResult) -> pd.DataFrame:
    """Return smoothed state probabilities for each time step."""

    events = _prepare_events(df)
    counts = events["count"].to_numpy()
    times = events["time"].to_numpy()
    mu, pi, A = _params_to_arrays(fit)
    log_b = _log_pois(counts, mu)
    log_pi = np.log(pi)
    log_A = np.log(A)
    log_alpha, log_beta, loglik_val = _forward_backward(log_pi, log_A, log_b)
    log_gamma = log_alpha + log_beta - loglik_val
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    data = {"time": times}
    for k in range(gamma.shape[1]):
        data[f"p_state_{k}"] = gamma[:, k]
    return pd.DataFrame(data)


def simulate(
    T: int,
    fit: FitResult,
    rng: np.random.Generator | int | None = None,
    exog: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate from the fitted RS-Poisson HMM."""

    if T <= 0:
        raise ValueError("T must be positive")
    generator = ensure_rng(rng)
    mu, pi, A = _params_to_arrays(fit)
    exog_arr = prepare_exog(exog, T)

    states = np.zeros(T, dtype=int)
    states[0] = generator.choice(len(mu), p=pi)
    counts = np.zeros(T)
    lam = np.maximum(mu[states[0]] + exog_arr[0], _MIN_RATE)
    counts[0] = generator.poisson(lam)
    for t in range(1, T):
        states[t] = generator.choice(len(mu), p=A[states[t - 1]])
        lam = np.maximum(mu[states[t]] + exog_arr[t], _MIN_RATE)
        counts[t] = generator.poisson(lam)

    times = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": counts, "regime": states})
