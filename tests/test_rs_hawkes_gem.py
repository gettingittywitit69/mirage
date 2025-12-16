from __future__ import annotations

import time

import numpy as np

from criticality.models.rs_hawkes_dt import fit_rs_hawkes_dt


def _sim_rs_poisson(
    T: int,
    mu: tuple[float, float] = (0.4, 2.5),
    transition: np.ndarray | None = None,
    rng: np.random.Generator | int | None = 0,
) -> np.ndarray:
    generator = np.random.default_rng(rng)
    A = (
        transition
        if transition is not None
        else np.array([[0.97, 0.03], [0.02, 0.98]], dtype=float)
    )
    mu_arr = np.array(mu, dtype=float)
    states = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)
    states[0] = generator.choice(len(mu_arr))
    y[0] = generator.poisson(mu_arr[states[0]])
    for t in range(1, T):
        states[t] = generator.choice(len(mu_arr), p=A[states[t - 1]])
        y[t] = generator.poisson(mu_arr[states[t]])
    return y


def _sim_rs_hawkes(
    T: int,
    mu: tuple[float, float] = (0.4, 1.2),
    n: float = 0.4,
    beta: float = 0.9,
    transition: np.ndarray | None = None,
    rng: np.random.Generator | int | None = 0,
) -> np.ndarray:
    generator = np.random.default_rng(rng)
    A = (
        transition
        if transition is not None
        else np.array([[0.97, 0.03], [0.03, 0.97]], dtype=float)
    )
    mu_arr = np.array(mu, dtype=float)
    alpha = n * (1.0 - beta)
    states = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)
    g = 0.0
    states[0] = generator.choice(len(mu_arr))
    y[0] = generator.poisson(np.maximum(mu_arr[states[0]] + alpha * g, 1e-9))
    for t in range(1, T):
        g = beta * g + y[t - 1]
        states[t] = generator.choice(len(mu_arr), p=A[states[t - 1]])
        lam = np.maximum(mu_arr[states[t]] + alpha * g, 1e-9)
        y[t] = generator.poisson(lam)
    return y


def test_rs_poisson_truth_keeps_n_small() -> None:
    rng = np.random.default_rng(0)
    n_hats = []
    for _ in range(10):
        y = _sim_rs_poisson(T=400, rng=rng)
        res = fit_rs_hawkes_dt(y, K=2, n_init=3, max_em_iter=20, tol=1e-4, rng=rng)
        n_hats.append(res["n"])
    n_hats_arr = np.array(n_hats)
    assert np.median(n_hats_arr) < 0.1
    assert n_hats_arr.max() < 0.4


def test_rs_hawkes_truth_recovers_n() -> None:
    rng = np.random.default_rng(1)
    y = _sim_rs_hawkes(T=700, n=0.4, beta=0.9, rng=rng)
    res = fit_rs_hawkes_dt(y, K=2, n_init=4, max_em_iter=25, tol=1e-5, rng=rng)
    assert 0.2 <= res["n"] <= 0.6


def test_speed_smoke_runs_quickly() -> None:
    rng = np.random.default_rng(2)
    y = _sim_rs_poisson(T=5000, rng=rng)
    start = time.time()
    res = fit_rs_hawkes_dt(y, K=2, n_init=2, max_em_iter=10, tol=1e-4, rng=rng)
    duration = time.time() - start
    print(f"[speed_smoke] duration={duration:.2f}s loglik={res['loglik']:.2f} n={res['n']:.3f}")
    assert np.isfinite(res["loglik"])
