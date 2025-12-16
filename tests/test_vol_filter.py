from __future__ import annotations

import numpy as np
import pandas as pd

from criticality.vol_filter import apply_vol_filter, fit_vol_filter


def _synthetic_returns(n: int = 300) -> pd.Series:
    rng = np.random.default_rng(42)
    base = rng.standard_normal(n) * 0.01
    return pd.Series(base)


def test_none_filter_passes_through() -> None:
    r = _synthetic_returns(40)
    r.iloc[5] = np.nan
    fit = fit_vol_filter(r.iloc[:20], method="none")
    mu, sigma, z = apply_vol_filter(r, fit)
    assert np.allclose(mu.dropna(), 0.0)
    assert np.allclose(sigma.dropna(), 1.0)
    pd.testing.assert_series_equal(z, r)


def test_ewma_respects_no_lookahead() -> None:
    base = pd.Series(np.linspace(0.0, 0.1, 60))
    fit = fit_vol_filter(base.iloc[:40], method="ewma", span=5, initial_variance=1e-6)
    _, sigma_base, _ = apply_vol_filter(base, fit)
    perturbed = base.copy()
    perturbed.iloc[45:] = perturbed.iloc[45:] + 5.0
    _, sigma_perturbed, _ = apply_vol_filter(perturbed, fit)
    pd.testing.assert_series_equal(sigma_base.iloc[:45], sigma_perturbed.iloc[:45])


def test_garch_method_falls_back_without_arch() -> None:
    r = _synthetic_returns(120)
    fit = fit_vol_filter(r, method="garch_if_available")
    _, sigma, z = apply_vol_filter(r, fit)
    assert float(sigma.min()) > 0
    assert not z.isna().any()
    assert (fit.metadata or {}).get("backend") != "arch"


def test_missing_values_propagate() -> None:
    r = _synthetic_returns(80)
    r.iloc[10:12] = np.nan
    fit = fit_vol_filter(r.iloc[:50], method="ewma", span=10, initial_variance=1e-6)
    _, _, z = apply_vol_filter(r, fit)
    assert z.iloc[10:12].isna().all()
    assert not z.dropna().isna().any()
