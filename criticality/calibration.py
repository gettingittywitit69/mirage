"""Calibration helpers focused on mirage risk and selection error."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import hawkes_dt, rs_hawkes_dt, rs_poisson_dt
from .tournament import (
    ModelSpec,
    _fit_kwargs,
    _n_proxy_from_fit,
    _spawn_rng,
    available_models,
    load_model,
    split_events,
)
from .utils import FitResult, validate_event_df


def calibrate(df: pd.DataFrame, model_name: str, train_frac: float = 0.8) -> Tuple[FitResult, Dict[str, float]]:
    """Fit a model and return the fit alongside basic log-likelihood diagnostics (legacy helper)."""

    events = validate_event_df(df)
    train, test = split_events(events, train_frac=train_frac)
    module = load_model(model_name)
    fit = module.fit(train)
    metrics = {
        "train_loglik": float(module.loglik(train, fit)),
        "test_loglik": float(module.loglik(test, fit)),
    }
    return fit, metrics


def calibrate_all(df: pd.DataFrame, train_frac: float = 0.8) -> pd.DataFrame:
    """Fit every registered model and return a tidy metrics table (legacy helper)."""

    records = []
    for name in available_models():
        fit, metrics = calibrate(df, name, train_frac=train_frac)
        records.append(
            {
                "model": name,
                "train_loglik": metrics["train_loglik"],
                "test_loglik": metrics["test_loglik"],
                "n_params": len(getattr(fit, "params", {})),
            }
        )
    return pd.DataFrame(records).sort_values("test_loglik", ascending=False).reset_index(drop=True)


# ---- Mirage calibration helpers ------------------------------------------------

DEFAULT_MODEL_NAMES: List[str] = [
    "poisson_iid",
    "poisson_flexbaseline",
    "hawkes_dt",
    "rs_poisson_dt",
    "rs_hawkes_dt",
]
DEFAULT_MODEL_SPECS: List[ModelSpec] = [ModelSpec(name) for name in DEFAULT_MODEL_NAMES]
DEFAULT_DGPS: List[str] = ["hawkes", "rs_poisson", "rs_hawkes", "poisson_drift"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _transition_matrix(persistence: float) -> np.ndarray:
    p = _clamp(persistence, 0.05, 0.995)
    return np.array([[p, 1.0 - p], [1.0 - p, p]], dtype=float)


def _simulate_stationary_hawkes(rho: float, persistence: float, T: int, rng: np.random.Generator) -> pd.DataFrame:
    branching = _clamp(rho, 0.0, 0.98)
    decay_factor = _clamp(persistence, 0.01, 0.99)
    decay = float(-np.log(decay_factor))
    alpha = branching * (1.0 - decay_factor)
    fit = FitResult(
        "hawkes_dt",
        {"baseline": 1.0, "alpha": alpha, "decay": decay},
        metadata={"rho": branching, "persistence": decay_factor, "true_n": branching},
    )
    df = hawkes_dt.simulate(T, fit, rng=rng)
    return validate_event_df(df)


def _simulate_rs_poisson(rho: float, persistence: float, T: int, rng: np.random.Generator) -> pd.DataFrame:
    rate_ratio = max(1.0, 1.0 + float(rho))
    mu_low = 0.8
    mu_high = mu_low * rate_ratio
    A = _transition_matrix(persistence)
    params: Dict[str, float] = {
        "n_states": 2.0,
        "mu_0": mu_low,
        "mu_1": mu_high,
        "pi_0": 0.5,
        "pi_1": 0.5,
    }
    for i in range(2):
        for j in range(2):
            params[f"a_{i}_{j}"] = float(A[i, j])
    fit = FitResult("rs_poisson_dt", params, metadata={"n_states": 2})
    df = rs_poisson_dt.simulate(T, fit, rng=rng)
    return validate_event_df(df)


def _simulate_rs_hawkes(rho: float, persistence: float, T: int, rng: np.random.Generator) -> pd.DataFrame:
    branching = _clamp(rho, 0.0, 0.9)
    beta = _clamp(persistence, 0.05, 0.99)
    alpha = branching * (1.0 - beta)
    A = _transition_matrix(persistence)
    mu_low = 0.7
    mu_high = mu_low * (1.0 + max(branching, 0.2))
    params: Dict[str, float] = {
        "n_states": 2.0,
        "alpha": alpha,
        "alpha_eff": alpha,
        "beta": beta,
        "n": branching,
        "mu_0": mu_low,
        "mu_1": mu_high,
        "pi_0": 0.5,
        "pi_1": 0.5,
    }
    for i in range(2):
        for j in range(2):
            params[f"a_{i}_{j}"] = float(A[i, j])
    fit = FitResult("rs_hawkes_dt", params, metadata={"n_states": 2})
    df = rs_hawkes_dt.simulate(T, fit, rng=rng)
    return validate_event_df(df)


def _simulate_drifting_poisson(rho: float, persistence: float, T: int, rng: np.random.Generator) -> pd.DataFrame:
    base_rate = 1.0
    phi = _clamp(persistence, 0.0, 0.98)
    shock_scale = max(0.05, float(rho))
    drift = 0.0
    counts = np.zeros(T, dtype=float)
    for t in range(T):
        drift = phi * drift + rng.normal(0.0, shock_scale)
        lam = base_rate * max(0.05, 1.0 + drift)
        counts[t] = rng.poisson(lam)
    times = np.arange(1, T + 1, dtype=float)
    df = pd.DataFrame({"time": times, "count": counts})
    return validate_event_df(df)


_DGP_BUILDERS = {
    "hawkes": _simulate_stationary_hawkes,
    "rs_poisson": _simulate_rs_poisson,
    "rs_hawkes": _simulate_rs_hawkes,
    "poisson_drift": _simulate_drifting_poisson,
}


def simulate_dgp(dgp: str, rho: float, persistence: float, T: int, rng: np.random.Generator) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Return simulated events and metadata for the requested DGP."""

    builder = _DGP_BUILDERS.get(dgp)
    if builder is None:
        raise ValueError(f"Unknown DGP '{dgp}'. Options: {sorted(_DGP_BUILDERS)}")
    events = builder(float(rho), float(persistence), int(T), rng)
    if dgp == "hawkes":
        true_n = _clamp(rho, 0.0, 0.98)
    elif dgp == "rs_hawkes":
        true_n = _clamp(rho, 0.0, 0.9)
    else:
        true_n = 0.0
    meta = {
        "dgp": dgp,
        "rho": float(rho),
        "persistence": float(persistence),
        "T": int(T),
        "true_n": float(true_n),
    }
    return events, meta


def evaluate_models_on_events(
    events: pd.DataFrame,
    models: Sequence[ModelSpec],
    train_frac: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fit each model on train split and score out-of-sample log-likelihood."""

    events_df = validate_event_df(events)
    train, test = split_events(events_df, train_frac=train_frac)

    records: List[Dict[str, Any]] = []
    for spec in models:
        module = load_model(spec.name)
        child_rng = _spawn_rng(rng)
        fit_kwargs = _fit_kwargs(module, spec.params, child_rng)
        fit: FitResult = module.fit(train, **fit_kwargs)
        train_ll = module.loglik(train, fit)
        test_ll = module.loglik(test, fit)
        records.append(
            {
                "model": spec.name,
                "train_loglik": float(train_ll),
                "test_loglik": float(test_ll),
                "n_proxy": _n_proxy_from_fit(fit),
            }
        )

    results_df = pd.DataFrame(records)
    if results_df.empty:
        return results_df
    winner_row = results_df.sort_values(["test_loglik", "model"], ascending=[False, True]).iloc[0]
    results_df["winner"] = results_df["model"] == winner_row["model"]
    results_df["winner_model"] = winner_row["model"]
    return results_df


def run_mirage_calibration(
    rho_grid: Iterable[float],
    persistence_grid: Iterable[float],
    horizons: Iterable[int],
    runs_per_cell: int = 3,
    dgps: Sequence[str] | None = None,
    models: Sequence[ModelSpec] | None = None,
    train_frac: float = 0.7,
    seed: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run mirage/selection calibration across DGP grid."""

    if not 0 < train_frac < 1:
        raise ValueError("train_frac must lie in (0, 1)")
    if runs_per_cell <= 0:
        raise ValueError("runs_per_cell must be positive")

    dgp_list = list(dgps or DEFAULT_DGPS)
    model_list = list(models or DEFAULT_MODEL_SPECS)
    master_rng = np.random.default_rng(seed)

    records: List[pd.DataFrame] = []
    run_id = 0

    for T in horizons:
        for rho in rho_grid:
            for persistence in persistence_grid:
                for dgp in dgp_list:
                    for rep in range(runs_per_cell):
                        events, meta = simulate_dgp(dgp, float(rho), float(persistence), int(T), master_rng)
                        eval_df = evaluate_models_on_events(events, model_list, train_frac, master_rng)
                        if eval_df.empty:
                            continue
                        eval_df["dgp"] = dgp
                        eval_df["rho"] = float(rho)
                        eval_df["persistence"] = float(persistence)
                        eval_df["T"] = int(T)
                        eval_df["run_id"] = run_id
                        eval_df["rep"] = rep
                        eval_df["true_n"] = float(meta.get("true_n", 0.0))
                        eval_df["train_frac"] = float(train_frac)
                        records.append(eval_df)
                        run_id += 1

    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results_df = pd.concat(records, ignore_index=True)

    winners = results_df[results_df["winner"]]
    total_runs = winners.groupby("dgp").size().rename("total_runs")
    confusion = winners.groupby(["dgp", "model"]).size().reset_index(name="wins")
    confusion = confusion.merge(total_runs, on="dgp", how="left")
    confusion["win_rate"] = confusion["wins"] / confusion["total_runs"].replace(0, np.nan)
    confusion = confusion.sort_values(["dgp", "win_rate", "model"], ascending=[True, False, True]).reset_index(
        drop=True
    )

    n0_mask = results_df["true_n"].abs() < 1e-8
    hawkes_models = {"hawkes_dt", "rs_hawkes_dt"}
    mirage_map = (
        results_df.loc[n0_mask & results_df["model"].isin(hawkes_models)]
        .groupby(["dgp", "rho", "persistence", "T", "model"], dropna=False)["n_proxy"]
        .agg(n_proxy_mean="mean", n_proxy_std="std", runs="size")
        .reset_index()
    )

    return results_df, confusion, mirage_map
