from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from criticality.events import build_event_df, make_events, threshold_fixed, threshold_quantile


def test_quantile_threshold_uses_train_only_and_hits_expected_rate() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    r = pd.Series(np.linspace(0.0, 0.09, 10), index=dates)
    z = pd.Series(np.arange(10, dtype=float), index=dates)
    q = 0.75
    train_idx = dates[:8]
    test_idx = dates[8:]
    u = threshold_quantile(z.loc[train_idx].abs().to_numpy(), q)
    y = make_events(z, u)

    df = build_event_df(
        r,
        z,
        y,
        {"event_rule": "quantile", "q": q, "u_fixed": None, "u_used": u, "train_index": train_idx, "test_index": test_idx},
    )

    event_rate_train = df.loc[train_idx, "y"].mean()
    assert event_rate_train == pytest.approx(1 - q, rel=0.05, abs=0.05)


def test_build_event_df_rejects_non_train_quantile_threshold() -> None:
    dates = pd.date_range("2021-01-01", periods=6, freq="D")
    r = pd.Series(np.linspace(0.0, 0.05, 6), index=dates)
    z = pd.Series([1.0, 2.0, 3.0, 4.0, 50.0, 60.0], index=dates)
    q = 0.5
    train_idx = dates[:4]
    test_idx = dates[4:]
    u_wrong = threshold_quantile(z.abs().to_numpy(), q)
    y_wrong = make_events(z, u_wrong)

    with pytest.raises(AssertionError):
        build_event_df(
            r,
            z,
            y_wrong,
            {"event_rule": "quantile", "q": q, "u_fixed": None, "u_used": u_wrong, "train_index": train_idx, "test_index": test_idx},
        )


def test_build_event_df_requires_disjoint_train_test_indices() -> None:
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    r = pd.Series(0.01, index=dates)
    z = pd.Series([0.5, 3.0, -0.1, 4.0], index=dates)
    u = threshold_fixed(2.0)
    y = make_events(z, u)

    with pytest.raises(AssertionError):
        build_event_df(
            r,
            z,
            y,
            {"event_rule": "fixed", "q": None, "u_fixed": u, "u_used": u, "train_index": dates[:3], "test_index": dates[2:]},
        )


def test_build_event_df_fixed_rule_adds_required_columns() -> None:
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    r = pd.Series([0.01, -0.02, 0.0], index=dates)
    z = pd.Series([0.5, -3.1, 1.2], index=dates)
    u = threshold_fixed(3.0)
    y = make_events(z, u)

    df = build_event_df(r, z, y, {"event_rule": "fixed", "q": None, "u_fixed": u, "u_used": u})

    required_cols = {"date", "r", "z", "abs_z", "y", "year", "month", "event_rule", "q", "u_fixed", "u_used"}
    assert required_cols.issubset(df.columns)
    assert (df["event_rule"] == "fixed").all()
    assert df["u_fixed"].iloc[0] == pytest.approx(u)
    assert df["u_used"].iloc[0] == pytest.approx(u)
    assert np.isnan(df["q"].iloc[0])
