"""
data_prep.py — Load, clean, aggregate, and resample AQI data.

Key design decisions (see docs/PROJECT_LOG.md Step 3):
- The raw dataset has one row per (station, timestamp, pollutant).
  We collapse to one row per (entity, timestamp) by taking AQI = max across pollutants
  (since AQI is defined as the worst sub-index).
- We support two levels of entity: "station" and "state".
  Station-level data is very sparse (~26 timestamps max), so we primarily use
  state-level aggregation which yields 200-270+ daily points.
- We resample to daily frequency and interpolate small gaps.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BASE_DIR, DATA_DIR


# ────────────────────────────────────────────────────────────
# 1. Load & initial clean
# ────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Load the raw CSV and parse dates."""
    path = os.path.join(DATA_DIR, "enriched_aqi_dataset.csv")
    df = pd.read_csv(path)
    df["last_update"] = pd.to_datetime(df["last_update"], format="mixed", dayfirst=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.1 — Drop exact duplicates; drop rows with invalid AQI.
    AQI should be in [0, 500]. We keep 0 but drop negatives and > 500.
    """
    before = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)

    # Drop invalid AQI
    df = df[(df["AQI"] >= 0) & (df["AQI"] <= 500)].copy()
    after_aqi = len(df)

    print(f"  Clean: {before} -> {after_dedup} (dedup) -> {after_aqi} (valid AQI)")
    return df


# ────────────────────────────────────────────────────────────
# 2. Aggregate: collapse pollutant rows
# ────────────────────────────────────────────────────────────

def aggregate_by_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.2 — Group by (station, last_update).
    AQI = max (worst sub-index), weather columns = mean.
    Returns one row per (station, timestamp).
    """
    agg = df.groupby(["station", "last_update"]).agg(
        AQI=("AQI", "max"),
        Temperature_C=("Temperature_C", "mean"),
        Humidity_pct=("Humidity_%", "mean"),
        Wind_Speed_kmh=("Wind_Speed_kmh", "mean"),
        state=("state", "first"),
        city=("city", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    ).reset_index()
    return agg


def aggregate_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to STATE level: group by (state, last_update).
    AQI = max across all stations in the state at that timestamp,
    weather = mean.  This gives far more data points per entity.
    """
    agg = df.groupby(["state", "last_update"]).agg(
        AQI=("AQI", "max"),
        Temperature_C=("Temperature_C", "mean"),
        Humidity_pct=("Humidity_%", "mean"),
        Wind_Speed_kmh=("Wind_Speed_kmh", "mean"),
    ).reset_index()
    return agg


# ────────────────────────────────────────────────────────────
# 3. Resample to daily frequency
# ────────────────────────────────────────────────────────────

def resample_entity(entity_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Step 3.3-3.4 — For a single entity (station or state):
    - Set datetime index, sort
    - Resample to the chosen frequency (default: daily)
    - Aggregate with mean
    - Interpolate small gaps (<=3 days), leave larger gaps as NaN then drop
    Returns the resampled DataFrame.
    """
    ts = entity_df.set_index("last_update").sort_index()

    # Keep only numeric columns for resampling
    numeric_cols = ["AQI", "Temperature_C", "Humidity_pct", "Wind_Speed_kmh"]
    existing = [c for c in numeric_cols if c in ts.columns]
    ts = ts[existing]

    # Resample
    resampled = ts.resample(freq).mean()

    # Interpolate small gaps (<=3 steps)
    filled_count = 0
    for col in existing:
        before_nulls = resampled[col].isnull().sum()
        resampled[col] = resampled[col].interpolate(method="time", limit=3)
        after_nulls = resampled[col].isnull().sum()
        filled_count += (before_nulls - after_nulls)

    # Drop remaining NaN rows (large gaps)
    resampled = resampled.dropna(subset=["AQI"])

    return resampled, filled_count


# ────────────────────────────────────────────────────────────
# 4. Public API — used by models and app
# ────────────────────────────────────────────────────────────

_CACHE = {}


def _build_cache():
    """Build the processed data cache (run once)."""
    if _CACHE:
        return

    df = load_raw()
    df = clean(df)

    # Station-level
    station_agg = aggregate_by_station(df)
    _CACHE["station_agg"] = station_agg

    # State-level
    state_agg = aggregate_by_state(df)
    _CACHE["state_agg"] = state_agg

    # Resample each state
    state_series = {}
    for state_name in state_agg["state"].unique():
        sdf = state_agg[state_agg["state"] == state_name].copy()
        resampled, filled = resample_entity(sdf, freq="D")
        if len(resampled) >= 10:  # minimum usable
            state_series[state_name] = resampled
    _CACHE["state_series"] = state_series

    # Resample each station (for completeness, though most are too sparse)
    station_series = {}
    for stn in station_agg["station"].unique():
        sdf = station_agg[station_agg["station"] == stn].copy()
        resampled, filled = resample_entity(sdf, freq="D")
        if len(resampled) >= 10:
            station_series[stn] = resampled
    _CACHE["station_series"] = station_series

    # Station metadata (for dropdown)
    meta = station_agg.groupby("station").agg(
        city=("city", "first"),
        state=("state", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        n_timestamps=("last_update", "nunique"),
    ).reset_index()
    _CACHE["station_meta"] = meta


def get_state_series(state: str) -> Optional[pd.DataFrame]:
    """Return the resampled daily AQI series for a state."""
    _build_cache()
    return _CACHE["state_series"].get(state)


def get_station_series(station: str) -> Optional[pd.DataFrame]:
    """Return the resampled daily AQI series for a station."""
    _build_cache()
    return _CACHE["station_series"].get(station)


def list_valid_states(min_points: int = 50) -> List[str]:
    """Return states with at least `min_points` daily data points."""
    _build_cache()
    return sorted([
        s for s, df in _CACHE["state_series"].items()
        if len(df) >= min_points
    ])


def list_valid_stations(min_points: int = 10) -> List[str]:
    """Return stations with at least `min_points` daily data points."""
    _build_cache()
    return sorted([
        s for s, df in _CACHE["station_series"].items()
        if len(df) >= min_points
    ])


def get_station_meta() -> pd.DataFrame:
    """Return station metadata DataFrame."""
    _build_cache()
    return _CACHE["station_meta"]


def get_entity_series(entity_name: str, entity_type: str = "state") -> Optional[pd.DataFrame]:
    """Generic getter: entity_type = 'state' or 'station'."""
    if entity_type == "state":
        return get_state_series(entity_name)
    else:
        return get_station_series(entity_name)


def train_val_test_split(
    series: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 4 — Chronological (time-based) train/validation/test split.

    NEVER shuffle time series! We split in order:
      - First `train_frac` → train
      - Next `val_frac` → validation
      - Remainder → test

    Why no shuffling?  In time series, each observation depends on the past.
    If we randomly shuffled, future data would leak into training — the model
    would 'cheat' by seeing tomorrow's values while learning to predict today.
    A chronological split preserves the natural temporal order.
    """
    n = len(series)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    return train, val, test


# ────────────────────────────────────────────────────────────
# CLI test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building data cache...")
    _build_cache()

    states = list_valid_states(min_points=50)
    print(f"\nValid states (>=50 daily pts): {len(states)}")
    for s in states:
        sr = get_state_series(s)
        print(f"  {s}: {len(sr)} daily points, "
              f"date range {sr.index.min().date()} to {sr.index.max().date()}, "
              f"AQI mean={sr['AQI'].mean():.1f}")

    stations = list_valid_stations(min_points=10)
    print(f"\nValid stations (>=10 daily pts): {len(stations)}")

    # Demo split
    if states:
        sr = get_state_series(states[0])
        train, val, test = train_val_test_split(sr)
        print(f"\nSplit demo ({states[0]}): train={len(train)}, val={len(val)}, test={len(test)}")
