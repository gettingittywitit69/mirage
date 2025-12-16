"""Fama-French factor loading utilities and event feature helpers."""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .utils import validate_event_df

logger = logging.getLogger(__name__)

FF_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FF_ZIP_NAME = "ff_factors_daily.zip"
_DATE_PATTERN = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2}|\d{8})")


# ---------------------------------------------------------------------------
# Ken French Data Library loaders

def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _choose_column(columns: Sequence[str], targets: set[str]) -> str | None:
    mapping = {_clean_name(col): col for col in columns}
    for target in targets:
        if target in mapping:
            return mapping[target]
    return None


def _dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("date")
    if ordered["date"].duplicated(keep="last").any():
        logger.warning("Duplicate FF daily dates detected; keeping the last occurrence for each date.")
        ordered = ordered[~ordered["date"].duplicated(keep="last")]
    ordered["mkt_rf"] = ordered["mkt_rf"].astype(float)
    ordered["rf"] = ordered["rf"].astype(float)
    return ordered.sort_values("date").reset_index(drop=True)


def validate_units(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure factor units are in decimals; auto-fix obvious percent inputs."""

    checked = df.copy()
    factor_cols = ["mkt_rf", "rf"]
    max_abs = checked[factor_cols].abs().max().max()
    converted = False
    if max_abs > 5:
        logger.warning("FF daily factors appear to be percent values; converting to decimals.")
        checked[factor_cols] = checked[factor_cols] / 100.0
        converted = True
    max_mkt = checked["mkt_rf"].abs().max()
    max_rf = checked["rf"].abs().max()
    if max_mkt >= 1.0 or max_rf >= 0.5:
        if not converted:
            logger.warning("FF daily factors exceed expected decimal ranges; converting from percent units.")
            checked[factor_cols] = checked[factor_cols] / 100.0
            max_mkt = checked["mkt_rf"].abs().max()
            max_rf = checked["rf"].abs().max()
        if max_mkt >= 1.0 or max_rf >= 0.5:
            raise ValueError(
                f"Fama-French daily factors appear mis-scaled (max mkt_rf={max_mkt:.4f}, rf={max_rf:.4f})."
            )
    return checked


def _standardize_factor_df(df_raw: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df_raw.copy()
    date_col = _choose_column(df.columns, {"date", "yyyymmdd", "caldt"})
    if date_col is None:
        date_col = df.columns[0]
        logger.warning("Using first column %s as date from %s data.", date_col, source)

    mkt_col = _choose_column(df.columns, {"mktrf", "mktminusrf"})
    rf_col = _choose_column(df.columns, {"rf", "riskfree"})
    if mkt_col is None and len(df.columns) > 1:
        mkt_col = df.columns[1]
        logger.warning("Falling back to column %s for Mkt-RF in %s payload.", mkt_col, source)
    if rf_col is None and len(df.columns) > 2:
        rf_col = df.columns[-1]
        logger.warning("Falling back to column %s for RF in %s payload.", rf_col, source)
    if mkt_col is None or rf_col is None:
        raise ValueError(f"Missing required columns in Fama-French data from {source}.")

    df = df[[date_col, mkt_col, rf_col]].rename(columns={date_col: "date", mkt_col: "mkt_rf", rf_col: "rf"})
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    df["mkt_rf"] = pd.to_numeric(df["mkt_rf"], errors="coerce")
    df["rf"] = pd.to_numeric(df["rf"], errors="coerce")
    df = df.dropna(subset=["date", "mkt_rf", "rf"])
    df = validate_units(df)
    return _dedupe_and_sort(df)


def _split_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter == ",":
        return [part.strip() for part in line.split(",")]
    return [part for part in re.split(r"\s+", line.strip()) if part]


def _find_header_fields(lines: list[str], start_idx: int, delimiter: str | None) -> list[str] | None:
    split = lambda l: _split_line(l, delimiter)
    for offset in range(1, 5):
        idx = start_idx - offset
        if idx < 0:
            break
        candidate = lines[idx].strip()
        if not candidate or _DATE_PATTERN.match(candidate):
            continue
        if re.search(r"[A-Za-z]", candidate):
            return split(candidate)
    return None


def _prepare_header(fields: list[str] | None, column_count: int) -> list[str]:
    if not fields:
        return ["date"] + [f"value_{i}" for i in range(1, column_count)]
    cleaned = [field.strip() for field in fields]
    if cleaned and cleaned[0] == "":
        cleaned[0] = "date"
    normalized = [_clean_name(f) for f in cleaned]
    if "date" not in normalized and column_count == len(cleaned) + 1:
        cleaned = ["date"] + cleaned
    while len(cleaned) < column_count:
        cleaned.append(f"value_{len(cleaned)}")
    if len(cleaned) > column_count:
        cleaned = cleaned[:column_count]
    return cleaned


def download_raw_ff_daily_zip(cache_dir: str | Path, force: bool = False) -> Path:
    """Download the Ken French daily factor zip and cache it locally."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / FF_ZIP_NAME
    if zip_path.exists() and not force:
        return zip_path

    req = Request(FF_DAILY_URL, headers={"User-Agent": "python"})
    with urlopen(req) as resp:
        zip_path.write_bytes(resp.read())
    return zip_path


def parse_ff_daily_from_zip(zip_path: Path) -> pd.DataFrame:
    """Parse the raw Ken French daily factor zip into a clean dataframe."""

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        if not members:
            raise ValueError(f"No files found inside {zip_path}.")
        with zf.open(sorted(members)[0]) as fh:
            text = fh.read().decode("latin-1")

    lines = text.splitlines()
    data_start = next((idx for idx, line in enumerate(lines) if _DATE_PATTERN.match(line.strip())), None)
    if data_start is None:
        raise ValueError("Could not locate start of data in Ken French daily zip.")

    delimiter = "," if "," in lines[data_start] else None
    header_fields = _find_header_fields(lines, data_start, delimiter)
    column_count = len(_split_line(lines[data_start], delimiter))
    headers = _prepare_header(header_fields, column_count)

    split = lambda l: _split_line(l, delimiter)
    rows: list[dict[str, str]] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not _DATE_PATTERN.match(stripped):
            break
        parts = split(line)
        if len(parts) < len(headers):
            parts.extend([None] * (len(headers) - len(parts)))
        elif len(parts) > len(headers):
            parts = parts[: len(headers)]
        rows.append(dict(zip(headers, parts)))

    raw_df = pd.DataFrame(rows)
    return _standardize_factor_df(raw_df, source="zip")


def _load_via_datareader() -> pd.DataFrame | None:
    try:
        from pandas_datareader import data as web
    except ImportError:
        logger.info("pandas_datareader not installed; skipping DataReader path.")
        return None

    try:
        payload = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")
    except Exception as exc:  # pragma: no cover - network path may vary
        logger.warning("pandas_datareader famafrench fetch failed: %s", exc)
        return None

    dataset = None
    if isinstance(payload, dict):
        dataset = payload.get(0) or (next(iter(payload.values())) if payload else None)
    elif isinstance(payload, (list, tuple)):
        dataset = payload[0]
    else:
        dataset = payload

    if not isinstance(dataset, pd.DataFrame):
        logger.warning("Unexpected pandas_datareader payload type: %s", type(dataset).__name__)
        return None

    return _standardize_factor_df(dataset.reset_index(), source="pandas_datareader")


def get_ff_factors_daily_df(cache_dir: str | Path, force: bool = False) -> pd.DataFrame:
    """Return daily Mkt-RF and RF factors as a clean dataframe."""

    df = _load_via_datareader()
    if df is not None:
        return df

    zip_path = download_raw_ff_daily_zip(cache_dir, force=force)
    return parse_ff_daily_from_zip(zip_path)


def get_market_excess_daily(cache_dir: str | Path, force: bool = False) -> pd.Series:
    """Return daily market excess returns (Mkt-RF) as a Series indexed by date."""

    df = get_ff_factors_daily_df(cache_dir, force=force)
    series = df.set_index("date")["mkt_rf"].sort_index()
    series.name = "mkt_rf"
    return series


# ---------------------------------------------------------------------------
# Event feature engineering helpers (existing behavior preserved)

def bin_events(timestamps: Sequence[float], bin_width: float) -> pd.DataFrame:
    """Aggregate raw timestamps into discrete count bins."""

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    timestamps_arr = np.asarray(timestamps, dtype=float)
    if timestamps_arr.size == 0:
        return pd.DataFrame({"time": [], "count": []})
    start = np.floor(np.min(timestamps_arr) / bin_width) * bin_width
    bins = ((timestamps_arr - start) / bin_width).astype(int)
    counts = np.bincount(bins)
    times = start + (np.arange(len(counts)) + 1) * bin_width
    return pd.DataFrame({"time": times, "count": counts})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper features such as cumulative counts."""

    events = validate_event_df(df).copy()
    events["cum_count"] = events["count"].cumsum()
    events["rolling_avg"] = events["count"].rolling(window=5, min_periods=1).mean()
    return events


def merge_streams(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Merge two event streams by time using outer join semantics."""

    left = validate_event_df(primary)
    right = validate_event_df(secondary)
    merged = (
        pd.merge(left, right, on="time", how="outer", suffixes=("_a", "_b"))
        .fillna(0.0)
        .sort_values("time")
        .reset_index(drop=True)
    )
    merged.rename(columns={"count_a": "count"}, inplace=True)
    merged["count"] = merged[["count", "count_b"]].sum(axis=1)
    merged.drop(columns=["count_b"], inplace=True)
    return merged
