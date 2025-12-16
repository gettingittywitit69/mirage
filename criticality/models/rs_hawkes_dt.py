"""Regime-switching Hawkes (discrete time) with generalized EM and free n."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils import FitResult, ensure_rng, prepare_exog, validate_event_df

_MIN_RATE = 1e-10
_MIN_PROB = 1e-12
_EPS_N = 1e-4


def _sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
    return float(np.log(p_clipped / (1 - p_clipped)))


def hawkes_state(y: np.ndarray, beta: float) -> np.ndarray:
    """Compute shared Hawkes recursion g_t = beta*g_{t-1} + y_{t-1}."""

    T = len(y)
    g = np.zeros(T, dtype=float)
    for t in range(1, T):
        g[t] = beta * g[t - 1] + y[t - 1]
    return g


def emission_loglik(y: np.ndarray, g: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Return log p(y_t | state=k) for all t,k (broadcasted)."""

    lam = mu[None, :] + alpha * g[:, None]
    lam = np.maximum(lam, _MIN_RATE)
    return y[:, None] * np.log(lam) - lam


def _logsumexp(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_val = np.max(arr, axis=axis, keepdims=True)
    stable = np.log(np.sum(np.exp(arr - max_val), axis=axis, keepdims=True)) + max_val
    if axis is None:
        return float(stable.squeeze())
    return stable.squeeze(axis)


def forward_backward_log(
    logB: np.ndarray, P: np.ndarray, pi: np.ndarray | None = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Log-space forward-backward returning loglik, gamma, and xi aggregates."""

    T, K = logB.shape
    if pi is None:
        pi = np.full(K, 1.0 / K)
    log_pi = np.log(np.maximum(pi, _MIN_PROB))
    log_P = np.log(np.maximum(P, _MIN_PROB))

    log_alpha = np.full((T, K), -np.inf)
    log_beta = np.zeros((T, K))

    log_alpha[0] = log_pi + logB[0]
    for t in range(1, T):
        log_alpha[t] = logB[t] + _logsumexp(log_alpha[t - 1][:, None] + log_P, axis=0)

    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(log_P + logB[t + 1] + log_beta[t + 1], axis=1)

    loglik = _logsumexp(log_alpha[-1])
    gamma = np.exp(log_alpha + log_beta - loglik)

    xi_sum = np.zeros((K, K), dtype=float)
    for t in range(T - 1):
        log_xi = (
            log_alpha[t][:, None] + log_P + logB[t + 1][None, :] + log_beta[t + 1][None, :] - loglik
        )
        xi_sum += np.exp(log_xi)

    return float(loglik), gamma, xi_sum


def _update_transition(xi_sum: np.ndarray) -> np.ndarray:
    P = np.maximum(xi_sum, _MIN_PROB)
    row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sum, _MIN_PROB)
    return P


def _newton_mu(y: np.ndarray, weights: np.ndarray, offsets: np.ndarray, mu_init: float) -> float:
    mu = float(np.maximum(mu_init, _MIN_RATE))
    for _ in range(20):
        lam = mu + offsets
        lam = np.maximum(lam, _MIN_RATE)
        grad = np.sum(weights * (y / lam - 1.0))
        hess = -np.sum(weights * y / (lam * lam))
        if not np.isfinite(grad) or not np.isfinite(hess) or abs(hess) < 1e-10:
            break
        step = grad / hess
        step = np.clip(step, -5.0, 5.0)
        mu_candidate = mu - step
        if not np.isfinite(mu_candidate) or mu_candidate <= 0:
            mu_candidate = mu * 0.5
        if abs(mu_candidate - mu) / max(mu, 1e-6) < 1e-6:
            mu = mu_candidate
            break
        mu = mu_candidate
    return float(np.maximum(mu, _MIN_RATE))


def _init_poisson(y: np.ndarray, K: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    T = len(y)
    quantiles = np.linspace(0.1, 0.9, K)
    mu = np.maximum(np.quantile(y, quantiles), _MIN_RATE)
    P = np.full((K, K), 1.0 / K)
    np.fill_diagonal(P, 0.9)
    P = _update_transition(P)
    pi = np.full(K, 1.0 / K)
    g_zero = np.zeros_like(y)
    loglik_prev = -np.inf
    for _ in range(15):
        logB = emission_loglik(y, g_zero, mu, alpha=0.0)
        loglik, gamma, xi_sum = forward_backward_log(logB, P, pi)
        if loglik - loglik_prev < 1e-6:
            break
        loglik_prev = loglik
        P = _update_transition(xi_sum)
        mu = np.maximum((gamma * y[:, None]).sum(axis=0) / np.maximum(gamma.sum(axis=0), _MIN_PROB), _MIN_RATE)
        pi = np.maximum(gamma[0], _MIN_PROB)
        pi = pi / pi.sum()
    return mu, P, gamma, float(loglik_prev)


def _q_value(y: np.ndarray, gamma: np.ndarray, mu: np.ndarray, alpha: float, g: np.ndarray, lambda_n: float) -> float:
    lam = mu[None, :] + alpha * g[:, None]
    lam = np.maximum(lam, _MIN_RATE)
    loglam = np.log(lam)
    q = np.sum(gamma * (y[:, None] * loglam - lam))
    return float(q - lambda_n * (alpha / max(1.0 - np.exp(-1.0), 1e-6)) ** 2)


@dataclass
class _GemResult:
    mu: np.ndarray
    P: np.ndarray
    beta: float
    n: float
    alpha: float
    loglik: float
    gamma: np.ndarray
    pi: np.ndarray


def fit_rs_hawkes_dt(
    y: np.ndarray,
    K: int,
    n_init: int = 5,
    max_em_iter: int = 30,
    tol: float = 1e-5,
    rng: np.random.Generator | int | None = None,
    lambda_n: float = 0.8,
) -> Dict[str, object]:
    """Generalized EM fit for RS-Hawkes with free branching ratio."""

    if K < 2 or K > 4:
        raise ValueError("K must be in [2, 4]")
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    T = len(y_arr)
    generator = ensure_rng(rng)

    best_loglik = -np.inf
    best_result: _GemResult | None = None

    penalty_weight = lambda_n * float(np.sqrt(T))

    for init_idx in range(n_init):
        mu, P, gamma, _ = _init_poisson(y_arr, K, generator)
        pi = np.maximum(gamma[0], _MIN_PROB)
        pi = pi / pi.sum()

        beta_init = float(np.clip(generator.uniform(0.7, 0.95), 0.05, 0.99))
        n_init_val = 0.05 + 0.1 * generator.random()
        b_raw = _logit(beta_init)
        n_raw = _logit(n_init_val / (1.0 - _EPS_N))

        loglik_prev = -np.inf
        stop_reason = ""

        for em_iter in range(1, max_em_iter + 1):
            beta = float(_sigmoid(b_raw))
            n_val = float(_sigmoid(n_raw) * (1.0 - _EPS_N))
            alpha = n_val * (1.0 - beta)
            g = hawkes_state(y_arr, beta)

            logB = emission_loglik(y_arr, g, mu, alpha)
            loglik, gamma, xi_sum = forward_backward_log(logB, P, pi)

            print(
                f"[rs_hawkes_dt][init {init_idx} iter {em_iter}] "
                f"loglik={loglik:.3f} n={n_val:.4f} beta={beta:.4f} "
                f"mu_min={mu.min():.4f} mu_max={mu.max():.4f}"
            )

            if loglik_prev > -np.inf:
                rel_improve = (loglik - loglik_prev) / max(abs(loglik_prev), 1.0)
                if rel_improve < tol:
                    stop_reason = f"rel_improve<{tol}"
                    break
            loglik_prev = loglik

            P = _update_transition(xi_sum)

            offsets = alpha * g
            mu_updates = []
            for k in range(K):
                weights = gamma[:, k]
                mu_init_val = (weights @ (y_arr - offsets)) / max(weights.sum(), _MIN_PROB)
                mu_k = _newton_mu(y_arr, weights, offsets, mu_init=mu_init_val)
                mu_updates.append(mu_k)
            mu = np.array(mu_updates, dtype=float)

            def _neg_q(theta: np.ndarray) -> float:
                b_r, n_r = theta
                beta_c = float(_sigmoid(b_r))
                n_c = float(_sigmoid(n_r) * (1.0 - _EPS_N))
                alpha_c = n_c * (1.0 - beta_c)
                g_c = hawkes_state(y_arr, beta_c)
                lam = mu[None, :] + alpha_c * g_c[:, None]
                lam = np.maximum(lam, _MIN_RATE)
                loglam = np.log(lam)
                q_val = np.sum(gamma * (y_arr[:, None] * loglam - lam))
                return -float(q_val - penalty_weight * (n_c**2))

            res = minimize(
                _neg_q,
                x0=np.array([b_raw, n_raw], dtype=float),
                method="L-BFGS-B",
                options={"maxiter": 50, "ftol": 1e-6},
            )
            if res.success and np.all(np.isfinite(res.x)):
                b_raw, n_raw = res.x

        beta = float(_sigmoid(b_raw))
        n_val = float(_sigmoid(n_raw) * (1.0 - _EPS_N))
        alpha = n_val * (1.0 - beta)
        if stop_reason == "":
            stop_reason = "max_iter"
        print(
            f"[rs_hawkes_dt][init {init_idx}] stop={stop_reason} "
            f"loglik={loglik_prev:.3f} n={n_val:.4f} beta={beta:.4f}"
        )

        score = loglik_prev - penalty_weight * (n_val**2)
        if score > best_loglik:
            best_loglik = score
            best_result = _GemResult(
                mu=mu,
                P=P,
                beta=beta,
                n=n_val,
                alpha=alpha,
                loglik=loglik_prev,
                gamma=gamma,
                pi=pi,
            )

    if best_result is None:
        raise RuntimeError("Failed to fit RS-Hawkes")

    return {
        "mu": best_result.mu,
        "P": best_result.P,
        "beta": best_result.beta,
        "n": best_result.n,
        "alpha": best_result.alpha,
        "loglik": best_result.loglik,
        "gamma": best_result.gamma,
        "pi": best_result.pi,
    }


# Compatibility wrappers for existing API -------------------------------

def _mu_key(k: int) -> str:
    return f"mu_{k}"


def _pi_key(k: int) -> str:
    return f"pi_{k}"


def _a_key(i: int, j: int) -> str:
    return f"a_{i}_{j}"


def _params_to_arrays(fit: FitResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    params = dict(fit.params)
    K = int(params.get("n_states", (fit.metadata or {}).get("n_states", 2)))
    mu = np.array([params.get(_mu_key(k), 1.0) for k in range(K)], dtype=float)
    P = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            P[i, j] = params.get(_a_key(i, j), 1.0 / K)
    beta = float(params.get("beta", 0.5))
    n_val = float(params.get("n", params.get("alpha", 0.0)))
    alpha = float(params.get("alpha_eff", n_val * (1 - beta)))
    pi = np.array([params.get(_pi_key(k), 1.0 / K) for k in range(K)], dtype=float)

    mu = np.maximum(mu, _MIN_RATE)
    P = _update_transition(P)
    pi = np.maximum(pi, _MIN_PROB)
    pi = pi / pi.sum()
    return mu, P, pi, alpha, beta, n_val


def _intensity(counts: np.ndarray, regimes: np.ndarray, params: Dict[str, float], beta: float) -> np.ndarray:
    """Utility used in legacy tests: compute lambda_t for given regimes."""

    counts_arr = np.asarray(counts, dtype=float)
    regimes_arr = np.asarray(regimes, dtype=int)
    alpha_shared = params.get("alpha", 0.0)
    lams = np.zeros_like(counts_arr, dtype=float)
    g = 0.0
    for t, c in enumerate(counts_arr):
        if t > 0:
            g = beta * g + counts_arr[t - 1]
        reg = regimes_arr[t]
        mu = float(params.get(_mu_key(reg), params.get(f"baseline_{reg}", params.get("baseline", 0.0))))
        alpha_k = float(params.get(f"alpha_{reg}", alpha_shared))
        lams[t] = np.maximum(_MIN_RATE, mu + alpha_k * g)
    return lams


def fit(
    train_df: pd.DataFrame,
    n_states: int = 2,
    n_init: int = 5,
    max_em_iter: int = 30,
    tol: float = 1e-5,
    rng: np.random.Generator | int | None = None,
    outer_iters: int | None = None,  # compatibility no-op
    lambda_n: float = 0.8,
) -> FitResult:
    """Fit wrapper accepting the dataframe used across the project."""

    events = validate_event_df(train_df)
    y = events["count"].to_numpy(dtype=float)
    res = fit_rs_hawkes_dt(
        y,
        n_states,
        n_init=n_init,
        max_em_iter=max_em_iter,
        tol=tol,
        rng=rng,
        lambda_n=lambda_n,
    )

    params: Dict[str, float] = {
        "n_states": float(n_states),
        "alpha": float(res["n"]),
        "alpha_eff": float(res["alpha"]),
        "beta": float(res["beta"]),
        "n": float(res["n"]),
        "loglik": float(res["loglik"]),
    }
    for k, mu_k in enumerate(res["mu"]):
        params[_mu_key(k)] = float(mu_k)
        params[_pi_key(k)] = float(res["gamma"][0, k]) if "gamma" in res else 1.0 / n_states
    for i in range(n_states):
        for j in range(n_states):
            params[_a_key(i, j)] = float(res["P"][i, j])

    metadata = {"n_states": int(n_states)}
    return FitResult("rs_hawkes_dt", params, metadata=metadata)


def loglik(test_df: pd.DataFrame, fit: FitResult) -> float:
    events = validate_event_df(test_df)
    y = events["count"].to_numpy(dtype=float)
    mu, P, pi, alpha, beta, _ = _params_to_arrays(fit)
    g = hawkes_state(y, beta)
    logB = emission_loglik(y, g, mu, alpha)
    loglik_val, _, _ = forward_backward_log(logB, P, pi=pi)
    return float(loglik_val)


def smoothed_probs(df: pd.DataFrame, fit: FitResult) -> pd.DataFrame:
    events = validate_event_df(df)
    y = events["count"].to_numpy(dtype=float)
    times = events["time"].to_numpy(dtype=float)
    mu, P, pi, alpha, beta, _ = _params_to_arrays(fit)
    g = hawkes_state(y, beta)
    logB = emission_loglik(y, g, mu, alpha)
    loglik_val, gamma, _ = forward_backward_log(logB, P, pi=pi)
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
    if T <= 0:
        raise ValueError("T must be positive")
    generator = ensure_rng(rng)
    mu, P, pi, alpha, beta, _ = _params_to_arrays(fit)
    exog_arr = prepare_exog(exog, T)

    K = len(mu)
    states = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)
    g = 0.0
    states[0] = generator.choice(K, p=pi)
    lam0 = np.maximum(mu[states[0]] + alpha * g + exog_arr[0], _MIN_RATE)
    y[0] = generator.poisson(lam0)
    for t in range(1, T):
        g = beta * g + y[t - 1]
        states[t] = generator.choice(K, p=P[states[t - 1]])
        lam = np.maximum(mu[states[t]] + alpha * g + exog_arr[t], _MIN_RATE)
        y[t] = generator.poisson(lam)

    times = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": y, "regime": states})
