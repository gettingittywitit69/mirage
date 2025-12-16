# Criticality Sandbox

A lightweight discrete-time Hawkes playground with multiple model variants and simple tournament utilities.

## Quick start

```bash
pip install -e .
python -m scripts.run_tournament --help
python -m scripts.run_calibration hawkes_dt
pytest
```

The package targets Python 3.11+ and depends on numpy, pandas, scipy, statsmodels, and matplotlib.
