"""
utils.py — Shared utilities for the AQI forecasting project.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)


def aqi_to_bucket(aqi: float) -> str:
    """
    Convert a numeric AQI value to an AQI bucket/category string.

    AQI Buckets (India CPCB standard):
        Good           :   0 -  50
        Satisfactory   :  51 - 100
        Moderate       : 101 - 200
        Poor           : 201 - 300
        Very Poor      : 301 - 400
        Severe         : 401 - 500
    """
    if pd.isna(aqi):
        return "Unknown"
    aqi = float(aqi)
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def bucket_color(bucket: str) -> str:
    """Return a hex colour for each AQI bucket (for Streamlit / Plotly)."""
    colors = {
        "Good": "#009966",
        "Satisfactory": "#58B37D",
        "Moderate": "#FFDE33",
        "Poor": "#FF9933",
        "Very Poor": "#CC0033",
        "Severe": "#660099",
        "Unknown": "#999999",
    }
    return colors.get(bucket, "#999999")


def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))


def mape(actual, predicted, eps=1e-8):
    """Mean Absolute Percentage Error — guarded against zero division."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    mask = np.abs(a) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100


def r_squared(actual, predicted):
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def category_accuracy(actual_aqi, predicted_aqi):
    """Fraction of predictions where the AQI bucket matches."""
    actual_buckets = [aqi_to_bucket(a) for a in actual_aqi]
    pred_buckets = [aqi_to_bucket(p) for p in predicted_aqi]
    matches = sum(a == p for a, p in zip(actual_buckets, pred_buckets))
    return matches / len(actual_buckets) if len(actual_buckets) > 0 else 0.0
