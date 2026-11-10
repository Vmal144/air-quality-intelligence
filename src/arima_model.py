"""
arima_model.py — ARIMA and Seasonal ARIMA forecasting for Air Quality Index (AQI).

This module implements:
1. Stationarity testing using the Augmented Dickey-Fuller (ADF) test.
2. Automated ARIMA parameter selection via pmdarima.auto_arima with weekly seasonality (m=7).
3. Model training, artifact persistence with joblib, and fallback handling.
4. One-step-ahead rolling evaluation on chronological test splits.
5. Flexible prediction supporting historical backtesting and future forecasting with confidence intervals.
"""

import os
import sys
import re
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima.arima import ARIMA

# Ensure imports from local src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BASE_DIR, MODELS_DIR, mae, rmse, mape, r_squared, category_accuracy
from data_prep import get_state_series, get_station_series, list_valid_states, train_val_test_split

# Set global random seed
np.random.seed(42)

# Suppress minor convergence and date index warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def sanitize_name(name: str) -> str:
    """
    Sanitize entity/state names for filesystem-safe storage.
    Replaces spaces and non-alphanumeric characters with underscores, lowercased.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return cleaned


def get_model_path(state: str) -> str:
    """Return the absolute path to the saved ARIMA model pickle."""
    sanitized = sanitize_name(state)
    return os.path.join(MODELS_DIR, f"arima_{sanitized}.pkl")


def check_stationarity(series: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Run Augmented Dickey-Fuller (ADF) test on a time series.

    Null Hypothesis (H0): The series has a unit root (non-stationary).
    Alternative Hypothesis (H1): The series has no unit root (stationary).

    Parameters
    ----------
    series : pd.Series or np.ndarray
        The numeric series to evaluate.

    Returns
    -------
    dict
        Results containing adf_stat, p_value, is_stationary, used_lag,
        critical_values, and recommended differencing order d.
    """
    if isinstance(series, pd.Series):
        clean = series.dropna().values
    else:
        clean = np.asarray(series)
        clean = clean[~np.isnan(clean)]

    if len(clean) < 10:
        return {
            "adf_stat": np.nan,
            "p_value": 1.0,
            "is_stationary": False,
            "used_lag": 0,
            "n_obs": len(clean),
            "critical_values": {},
            "recommended_d": 1,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_res = adfuller(clean, autolag="AIC")

    stat = float(adf_res[0])
    p_val = float(adf_res[1])
    used_lag = int(adf_res[2])
    n_obs = int(adf_res[3])
    crit_vals = {str(k): float(v) for k, v in adf_res[4].items()}
    is_stat = bool(p_val <= 0.05)

    return {
        "adf_stat": stat,
        "p_value": p_val,
        "is_stationary": is_stat,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_vals,
        "recommended_d": 0 if is_stat else 1,
    }


def _fit_model(y_train: np.ndarray, recommended_d: int) -> Tuple[pm.arima.ARIMA, Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Fit auto_arima on training data, falling back to ARIMA(1, 1, 1) on error.
    """
    seasonal = bool(len(y_train) >= 14)
    m = 7 if seasonal else 1

    try:
        model = pm.auto_arima(
            y_train,
            d=recommended_d,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            max_p=3,
            max_q=3,
            max_P=1,
            max_Q=1,
            random_state=42,
        )
        order = model.order
        seasonal_order = getattr(model, "seasonal_order", (0, 0, 0, 0))
    except Exception as exc:
        print(f"    [Warning] auto_arima failed ({exc}). Falling back to ARIMA(1, 1, 1).")
        model = ARIMA(order=(1, 1, 1), suppress_warnings=True)
        model.fit(y_train)
        order = (1, 1, 1)
        seasonal_order = (0, 0, 0, 0)

    return model, order, seasonal_order


def _evaluate_rolling_test(
    model: pm.arima.ARIMA,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Perform 1-step-ahead rolling forecast on the chronological test set
    using the model's Kalman filter state update without refitting parameters.
    """
    full_y = np.concatenate([y_train, y_val, y_test])
    test_start = len(y_train) + len(y_val)

    # Filter forward on the series using statsmodels apply
    res = model.arima_res_.apply(full_y)
    test_preds = np.clip(res.fittedvalues[test_start:], 0.0, 500.0)
    actuals = y_test

    metrics = {
        "mae": float(mae(actuals, test_preds)),
        "rmse": float(rmse(actuals, test_preds)),
        "mape": float(mape(actuals, test_preds)),
        "r2": float(r_squared(actuals, test_preds)),
        "category_accuracy": float(category_accuracy(actuals, test_preds)),
        "MAE": float(mae(actuals, test_preds)),
        "RMSE": float(rmse(actuals, test_preds)),
        "MAPE": float(mape(actuals, test_preds)),
        "R2": float(r_squared(actuals, test_preds)),
        "Cat_Accuracy": float(category_accuracy(actuals, test_preds)),
        "n_test": int(len(actuals)),
    }
    return metrics


def train_arima(state: str, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Train an ARIMA model on a state's daily AQI series.

    Parameters
    ----------
    state : str
        State name (or valid station name).
    force_retrain : bool
        If False and a saved model exists on disk, load and return it.

    Returns
    -------
    dict
        Model metadata, fitted pmdarima model, order, seasonal_order,
        ADF stationarity results, test set metrics, and file path.
    """
    model_path = get_model_path(state)

    if not force_retrain and os.path.exists(model_path):
        try:
            saved = joblib.load(model_path)
            if isinstance(saved, dict) and "model" in saved:
                return saved
        except Exception as e:
            print(f"    [Warning] Could not load cached model for {state} ({e}). Retraining.")

    sr = get_state_series(state)
    if sr is None:
        sr = get_station_series(state)
    if sr is None or len(sr) < 10:
        raise ValueError(f"Insufficient or missing time series for '{state}'.")

    # Chronological train / validation / test split
    train_df, val_df, test_df = train_val_test_split(sr)
    y_train = train_df["AQI"].values
    y_val = val_df["AQI"].values
    y_test = test_df["AQI"].values

    # Step 1: Stationarity check
    adf_info = check_stationarity(train_df["AQI"])
    recommended_d = adf_info["recommended_d"]

    # Step 2: Fit model
    model, order, seasonal_order = _fit_model(y_train, recommended_d)

    # Step 3: Rolling test set evaluation
    metrics = _evaluate_rolling_test(model, y_train, y_val, y_test)

    result_payload = {
        "model": model,
        "state": state,
        "order": order,
        "seasonal_order": seasonal_order,
        "adf_info": adf_info,
        "metrics": metrics,
        "model_path": model_path,
        "train_len": len(train_df),
        "val_len": len(val_df),
        "test_len": len(test_df),
        "last_historical_date": sr.index.max(),
    }

    # Save to models/
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(result_payload, model_path)

    return result_payload


def evaluate_arima(state: str) -> Dict[str, Any]:
    """
    Evaluate the ARIMA model on the state's test set using one-step-ahead rolling forecasts.

    Parameters
    ----------
    state : str
        State name.

    Returns
    -------
    dict
        Evaluation metrics: MAE, RMSE, MAPE, R2, category_accuracy, and model orders.
    """
    model_path = get_model_path(state)
    if not os.path.exists(model_path):
        payload = train_arima(state, force_retrain=False)
    else:
        loaded = joblib.load(model_path)
        payload = loaded if isinstance(loaded, dict) else {"model": loaded}

    model = payload["model"]

    sr = get_state_series(state)
    if sr is None:
        sr = get_station_series(state)
    if sr is None:
        raise ValueError(f"Series for '{state}' not found.")

    train_df, val_df, test_df = train_val_test_split(sr)
    metrics = _evaluate_rolling_test(
        model,
        train_df["AQI"].values,
        val_df["AQI"].values,
        test_df["AQI"].values,
    )

    return {
        "state": state,
        "order": payload.get("order", model.order),
        "seasonal_order": payload.get("seasonal_order", getattr(model, "seasonal_order", (0, 0, 0, 0))),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "r2": metrics["r2"],
        "category_accuracy": metrics["category_accuracy"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "MAPE": metrics["mape"],
        "R2": metrics["r2"],
        "Cat_Accuracy": metrics["category_accuracy"],
        "n_test": metrics["n_test"],
    }


def arima_predict(
    state: str,
    start: Union[str, datetime, pd.Timestamp],
    end: Union[str, datetime, pd.Timestamp],
) -> pd.DataFrame:
    """
    Generate predictions and 95% confidence intervals for a state across a date range.

    Handles:
    - In-sample backtesting within the historical series.
    - Out-of-sample future forecasting beyond historical data.
    - Spanning periods across both historical and future boundaries.

    Parameters
    ----------
    state : str
        State name.
    start : str, datetime, or pd.Timestamp
        Start date (inclusive).
    end : str, datetime, or pd.Timestamp
        End date (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp', 'predicted_AQI', 'lower', 'upper']
    """
    model_path = get_model_path(state)
    if not os.path.exists(model_path):
        payload = train_arima(state, force_retrain=False)
        model = payload["model"]
    else:
        loaded = joblib.load(model_path)
        model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded

    sr = get_state_series(state)
    if sr is None:
        sr = get_station_series(state)
    if sr is None:
        raise ValueError(f"Data series for '{state}' not found.")

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    if start_ts > end_ts:
        raise ValueError(f"start date ({start_ts.date()}) must be <= end date ({end_ts.date()}).")

    hist_dates = sr.index.normalize()
    hist_start = hist_dates.min()
    hist_end = hist_dates.max()

    # Pass entire historical series through the model Kalman filter
    res = model.arima_res_.apply(sr["AQI"].values)

    dfs = []

    # 1. Historical backtest portion
    if start_ts <= hist_end:
        hist_pred_obj = res.get_prediction(start=0, end=len(sr) - 1)
        mean_hist = np.clip(hist_pred_obj.predicted_mean, 0.0, 500.0)
        ci_hist = np.clip(hist_pred_obj.conf_int(), 0.0, 500.0)

        df_hist = pd.DataFrame({
            "timestamp": hist_dates,
            "predicted_AQI": mean_hist,
            "lower": ci_hist[:, 0],
            "upper": ci_hist[:, 1],
        })
        # Filter for requested range
        df_hist = df_hist[(df_hist["timestamp"] >= start_ts) & (df_hist["timestamp"] <= end_ts)]
        dfs.append(df_hist)

    # 2. Future forecast portion
    if end_ts > hist_end:
        total_future_days = (end_ts - hist_end).days
        future_dates = pd.date_range(start=hist_end + pd.Timedelta(days=1), end=end_ts, freq="D")

        fut_pred_obj = res.get_forecast(steps=total_future_days)
        mean_fut = np.clip(fut_pred_obj.predicted_mean, 0.0, 500.0)
        ci_fut = np.clip(fut_pred_obj.conf_int(), 0.0, 500.0)

        df_fut = pd.DataFrame({
            "timestamp": future_dates,
            "predicted_AQI": mean_fut,
            "lower": ci_fut[:, 0],
            "upper": ci_fut[:, 1],
        })
        df_fut = df_fut[(df_fut["timestamp"] >= start_ts) & (df_fut["timestamp"] <= end_ts)]
        dfs.append(df_fut)

    if not dfs:
        return pd.DataFrame(columns=["timestamp", "predicted_AQI", "lower", "upper"])

    result_df = pd.concat(dfs, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Ensure clean non-negative confidence intervals
    result_df["lower"] = np.minimum(result_df["lower"], result_df["predicted_AQI"])
    result_df["upper"] = np.maximum(result_df["upper"], result_df["predicted_AQI"])

    return result_df[["timestamp", "predicted_AQI", "lower", "upper"]]


# ────────────────────────────────────────────────────────────
# CLI execution & verification
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  ARIMA AQI Forecasting Model — Pipeline Verification")
    print("=" * 70)

    # Identify top 5 states by daily data count
    valid_states = list_valid_states(min_points=50)
    counts = []
    for s in valid_states:
        s_df = get_state_series(s)
        if s_df is not None:
            counts.append((s, len(s_df)))

    counts.sort(key=lambda x: x[1], reverse=True)
    top_5_states = [s[0] for s in counts[:5]]

    print(f"\nTop 5 states by available daily observations:")
    for rank, (s, c) in enumerate(counts[:5], 1):
        print(f"  {rank}. {s} ({c} points)")

    results_table = []

    for state in top_5_states:
        print(f"\n--- Training & Evaluating: {state} ---")
        train_res = train_arima(state, force_retrain=True)
        eval_res = evaluate_arima(state)

        adf = train_res["adf_info"]
        order = train_res["order"]
        seasonal_order = train_res["seasonal_order"]

        print(f"  Stationarity: ADF stat={adf['adf_stat']:.3f}, p-val={adf['p_value']:.4f} -> Stationary={adf['is_stationary']} (d={adf['recommended_d']})")
        print(f"  Selected Order: ARIMA{order} x {seasonal_order}")
        print(f"  Test Metrics (1-step rolling):")
        print(f"    MAE:               {eval_res['mae']:.2f}")
        print(f"    RMSE:              {eval_res['rmse']:.2f}")
        print(f"    MAPE:              {eval_res['mape']:.2f}%")
        print(f"    R²:                {eval_res['r2']:.4f}")
        print(f"    Category Accuracy: {eval_res['category_accuracy']:.2%}")

        results_table.append({
            "State": state,
            "Order": f"ARIMA{order}",
            "Seasonal": f"{seasonal_order}",
            "MAE": f"{eval_res['mae']:.2f}",
            "RMSE": f"{eval_res['rmse']:.2f}",
            "MAPE (%)": f"{eval_res['mape']:.2f}%",
            "R²": f"{eval_res['r2']:.4f}",
            "Category Acc": f"{eval_res['category_accuracy']:.2%}",
        })

    # Summary Table
    print("\n" + "=" * 70)
    print("  SUMMARY: Evaluation Results for Top 5 States")
    print("=" * 70)
    summary_df = pd.DataFrame(results_table)
    print(summary_df.to_string(index=False))

    # Demonstrate arima_predict function
    demo_state = top_5_states[0]
    print(f"\n" + "=" * 70)
    print(f"  DEMO: arima_predict for {demo_state}")
    print("=" * 70)

    # 1. Historical Backtest Demo
    pred_backtest = arima_predict(demo_state, start="2026-06-01", end="2026-06-07")
    print("\nBacktest Predictions (2026-06-01 to 2026-06-07):")
    print(pred_backtest.to_string(index=False))

    # 2. Future Forecast Demo
    pred_future = arima_predict(demo_state, start="2028-01-01", end="2028-01-07")
    print("\nFuture Forecast Predictions (2028-01-01 to 2028-01-07):")
    print(pred_future.to_string(index=False))

    print("\nARIMA pipeline completed successfully.")
