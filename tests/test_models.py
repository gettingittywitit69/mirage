from __future__ import annotations

import numpy as np
import pandas as pd

from criticality.models import (
    hawkes_dt,
    hawkes_dt_flexbaseline,
    hawkes_dt_mixture,
    poisson_flexbaseline,
    poisson_iid,
    rs_hawkes_dt,
    rs_poisson_dt,
)
from criticality.tournament import split_events


def _simple_df(n: int = 60) -> pd.DataFrame:
    times = np.arange(1, n + 1, dtype=float)
    counts = np.where(times % 5 == 0, 2.0, 0.5)
    return pd.DataFrame({"time": times, "count": counts})


def _regime_df() -> pd.DataFrame:
    df = _simple_df(40)
    df["regime"] = (df.index % 2).astype(int)
    return df


def _drifting_poisson_df(n_per_segment: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    low = rng.poisson(0.8, size=n_per_segment)
    high = rng.poisson(3.0, size=n_per_segment)
    counts = np.concatenate([low, high])
    times = np.arange(1, len(counts) + 1, dtype=float)
    years = np.concatenate([np.full(n_per_segment, 2020), np.full(n_per_segment, 2021)])
    return pd.DataFrame({"time": times, "count": counts, "year": years})


def _rs_poisson_sim(
    T: int = 600,
    mu: tuple[float, float] = (0.4, 2.5),
    transition: np.ndarray | None = None,
    rng: np.random.Generator | int | None = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(rng)
    A = (
        transition
        if transition is not None
        else np.array([[0.97, 0.03], [0.02, 0.98]], dtype=float)
    )
    mu_arr = np.array(mu, dtype=float)
    states = np.zeros(T, dtype=int)
    states[0] = generator.choice(len(mu_arr))
    counts = np.zeros(T)
    counts[0] = generator.poisson(mu_arr[states[0]])
    for t in range(1, T):
        states[t] = generator.choice(len(mu_arr), p=A[states[t - 1]])
        counts[t] = generator.poisson(mu_arr[states[t]])
    times = np.arange(1, T + 1, dtype=float)
    df = pd.DataFrame({"time": times, "count": counts})
    return df, states


def _rs_hawkes_sim(
    T: int = 800,
    mu: tuple[float, float] = (0.4, 2.0),
    alpha: float = 0.7,
    decay: float = 0.4,
    transition: np.ndarray | None = None,
    rng: np.random.Generator | int | None = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(rng)
    A = (
        transition
        if transition is not None
        else np.array([[0.97, 0.03], [0.03, 0.97]], dtype=float)
    )
    mu_arr = np.array(mu, dtype=float)
    states = np.zeros(T, dtype=int)
    counts = np.zeros(T)
    g = 0.0
    decay_factor = np.exp(-decay)
    states[0] = generator.choice(len(mu_arr))
    lam0 = np.maximum(mu_arr[states[0]] + alpha * g, 1e-9)
    counts[0] = generator.poisson(lam0)
    for t in range(1, T):
        g = decay_factor * g + (1.0 - decay_factor) * counts[t - 1]
        states[t] = generator.choice(len(mu_arr), p=A[states[t - 1]])
        lam = np.maximum(mu_arr[states[t]] + alpha * g, 1e-9)
        counts[t] = generator.poisson(lam)
    times = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time": times, "count": counts}), states


def test_models_share_uniform_api_and_exog() -> None:
    df = _simple_df()
    for module in [
        hawkes_dt,
        hawkes_dt_flexbaseline,
        hawkes_dt_mixture,
        rs_poisson_dt,
        rs_hawkes_dt,
        poisson_iid,
        poisson_flexbaseline,
    ]:
        fit = module.fit(df)
        ll = module.loglik(df, fit)
        assert np.isfinite(ll)
        base_sim = module.simulate(10, fit, np.random.default_rng(0))
        zero_exog = np.zeros(len(base_sim))
        sim_with_exog = module.simulate(10, fit, np.random.default_rng(0), exog=zero_exog)
        pd.testing.assert_frame_equal(base_sim[["time", "count"]], sim_with_exog[["time", "count"]])


def test_hawkes_recursion_matches_manual_expectation() -> None:
    counts = np.array([1.0, 0.0, 2.0])
    exog = np.array([0.1, 0.0, 0.0])
    baseline, alpha, decay = 0.3, 0.7, 0.5
    decay_factor = np.exp(-decay)
    expected = []
    state = 0.0
    for idx, c in enumerate(counts):
        state = decay_factor * state + c
        expected.append(baseline + alpha * state + exog[idx])
    lams = hawkes_dt._hawkes_intensity(counts, baseline, alpha, decay, exog=exog)
    assert np.allclose(lams, expected)


def test_regime_switching_hawkes_changes_intensity() -> None:
    counts = np.ones(4)
    regimes = np.array([0, 1, 0, 1])
    params = {"baseline": 0.2, "alpha": 0.1, "decay": 0.0, "baseline_1": 1.0, "alpha_1": 0.5}
    lams = rs_hawkes_dt._intensity(counts, regimes, params, params["decay"])
    assert lams[1] > lams[0]
    assert lams[3] > lams[2]


def test_hawkes_dt_mixture_weights_and_likelihood() -> None:
    df = _simple_df(25)
    fit = hawkes_dt_mixture.fit(df, n_components=2)
    weight_sum = fit.params["weight_0"] + fit.params["weight_1"]
    assert np.isclose(weight_sum, 1.0)
    ll = hawkes_dt_mixture.loglik(df, fit)
    assert np.isfinite(ll)


def test_likelihood_prefers_temporal_order() -> None:
    df = _simple_df()
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    fit = hawkes_dt.fit(df)
    in_sample = hawkes_dt.loglik(df, fit)
    out_sample = hawkes_dt.loglik(shuffled, fit)
    assert in_sample >= out_sample or np.isclose(in_sample, out_sample)


def test_flexbaseline_beats_iid_on_drifting_poisson() -> None:
    df = _drifting_poisson_df()
    train, test = split_events(df, train_frac=0.75)
    fit_iid = poisson_iid.fit(train)
    fit_flex = poisson_flexbaseline.fit(train, ridge=0.05)
    ll_iid = poisson_iid.loglik(test, fit_iid)
    ll_flex = poisson_flexbaseline.loglik(test, fit_flex)
    assert ll_flex > ll_iid


def test_hawkes_flex_baseline_beats_stationary_on_drift() -> None:
    df = _drifting_poisson_df()
    train, test = split_events(df, train_frac=0.75)
    fit_stationary = hawkes_dt.fit(train)
    fit_flex = hawkes_dt_flexbaseline.fit(train, segment_col="year", ridge=0.05)
    ll_stationary = hawkes_dt.loglik(test, fit_stationary)
    ll_flex = hawkes_dt_flexbaseline.loglik(test, fit_flex)
    assert ll_flex > ll_stationary


def test_rs_poisson_recovers_regimes_on_simulated() -> None:
    df, true_states = _rs_poisson_sim()
    fit = rs_poisson_dt.fit(df, n_states=2, n_init=5, max_iter=200, rng=0)
    smoothed = rs_poisson_dt.smoothed_probs(df, fit)
    prob_cols = [c for c in smoothed.columns if c.startswith("p_state_")]
    inferred = smoothed[prob_cols].to_numpy().argmax(axis=1)
    direct = (inferred == true_states).mean()
    flipped = (1 - inferred == true_states).mean()
    accuracy = max(direct, flipped)
    assert accuracy > 0.75


def test_rs_hawkes_alpha_near_zero_under_poisson_dgp() -> None:
    df, _ = _rs_poisson_sim(T=700)
    fit = rs_hawkes_dt.fit(df, n_states=2, n_init=3, outer_iters=2, rng=0)
    assert fit.params["alpha"] < 0.2


def test_rs_hawkes_recovers_excitation() -> None:
    df, _ = _rs_hawkes_sim(T=700, alpha=0.8, decay=0.3)
    fit = rs_hawkes_dt.fit(df, n_states=2, n_init=3, outer_iters=3, rng=0)
    assert fit.params["alpha"] > 0.3
