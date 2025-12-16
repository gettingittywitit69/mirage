"""Compatibility wrappers pointing at the packaged tournament helpers."""
from criticality.tournament import (  # noqa: F401
    MODEL_REGISTRY,
    ExperimentConfig,
    ModelSpec,
    available_models,
    load_model,
    rolling_splits,
    run_tournament,
    split_events,
)

MODEL_NAMES = available_models()

__all__ = [
    "MODEL_REGISTRY",
    "MODEL_NAMES",
    "ExperimentConfig",
    "ModelSpec",
    "available_models",
    "load_model",
    "rolling_splits",
    "run_tournament",
    "split_events",
]
