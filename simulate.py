"""Simulation helpers for models."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from criticality.events import simulate_poisson
from criticality.utils import FitResult
from tournament import MODEL_NAMES, load_model


def default_training_data(T: int = 50) -> pd.DataFrame:
    """Return a light-weight default training set for bootstrapping fits."""

    return simulate_poisson(rate=1.0, T=float(T), rng=np.random.default_rng(0))


def simulate_model(
    model_name: str,
    T: int,
    seed: Optional[int] = None,
    fit: FitResult | None = None,
) -> pd.DataFrame:
    """Simulate synthetic data from a given model."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'")
    module = load_model(model_name)
    if fit is None:
        fit = module.fit(default_training_data())
    rng = np.random.default_rng(seed)
    return module.simulate(T, fit, rng)
