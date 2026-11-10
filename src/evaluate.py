"""
evaluate.py — Model comparison and evaluation (Step 8).

Compares Baseline, ARIMA, and LSTM on the same test split for each entity.
Produces:
  - Per-entity metrics table (MAE, RMSE, MAPE, R², category accuracy)
  - Actual vs predicted plots
  - Summary table across all evaluated entities
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    BASE_DIR, DOCS_DIR, mae, rmse, mape, r_squared,
    category_accuracy, aqi_to_bucket
)
from data_prep import (
    get_state_series, list_valid_states, train_val_test_split
)


# ────────────────────────────────────────────────────────────
# Baseline models (Step 5)
# ────────────────────────────────────────────────────────────

def naive_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Naive baseline: predict the last known value for all future steps.

    Why? This is the simplest possible forecast — it says 'tomorrow will be
    the same as today'. Any useful model must beat this, otherwise it's not
    adding value over just repeating the most recent observation.
    """
    last_val = train["AQI"].iloc[-1]
    return np.full(len(test), last_val)


def seasonal_naive_baseline(
    train: pd.DataFrame, test: pd.DataFrame, period: int = 7
) -> np.ndarray:
    """
    Seasonal naive: predict the value from `period` days ago.

    For daily data with weekly seasonality (period=7), this predicts
    'next Monday will be the same as last Monday'. This captures weekly
    patterns that the plain naive baseline misses.
    """
    predictions = []
    for i in range(len(test)):
        # Look back `period` steps from the start of the test set
        lookback_idx = len(train) - period + (i % period)
        if lookback_idx >= 0 and lookback_idx < len(train):
            predictions.append(train["AQI"].iloc[lookback_idx])
        else:
            predictions.append(train["AQI"].iloc[-1])  # fallback
    return np.array(predictions)


# ────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ────────────────────────────────────────────────────────────

def evaluate_all_models(state: str) -> dict:
    """
    Evaluate Baseline, ARIMA, and LSTM on a single state.
    Returns a dict of results.
    """
    series = get_state_series(state)
    if series is None:
        return {"error": f"No data for {state}"}

    train, val, test = train_val_test_split(series)
    actual = test["AQI"].values

    results = {"state": state, "test_size": len(test)}

    # --- Naive baseline ---
    naive_preds = naive_baseline(train, test)
    results["naive"] = {
        "MAE": round(mae(actual, naive_preds), 2),
        "RMSE": round(rmse(actual, naive_preds), 2),
        "MAPE": round(mape(actual, naive_preds), 2),
        "R2": round(r_squared(actual, naive_preds), 4),
        "Cat_Accuracy": round(category_accuracy(actual, naive_preds), 4),
    }

    # --- Seasonal naive baseline ---
    seasonal_preds = seasonal_naive_baseline(train, test, period=7)
    results["seasonal_naive"] = {
        "MAE": round(mae(actual, seasonal_preds), 2),
        "RMSE": round(rmse(actual, seasonal_preds), 2),
        "MAPE": round(mape(actual, seasonal_preds), 2),
        "R2": round(r_squared(actual, seasonal_preds), 4),
        "Cat_Accuracy": round(category_accuracy(actual, seasonal_preds), 4),
    }

    # --- ARIMA ---
    try:
        from arima_model import train_arima, evaluate_arima
        arima_result = train_arima(state)
        arima_eval = evaluate_arima(state)
        results["arima"] = arima_eval
    except Exception as e:
        results["arima"] = {"error": str(e)}

    # --- LSTM ---
    try:
        from rnn_model import train_lstm, evaluate_lstm
        lstm_result = train_lstm(state)
        lstm_eval = evaluate_lstm(state)
        results["lstm"] = lstm_eval
    except Exception as e:
        results["lstm"] = {"error": str(e)}

    return results


def plot_comparison(state: str, save: bool = True) -> plt.Figure:
    """
    Plot actual vs ARIMA vs LSTM on the test period for a state.
    """
    series = get_state_series(state)
    if series is None:
        return None

    train, val, test = train_val_test_split(series)
    actual = test["AQI"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test.index, actual.values, label="Actual", color="black", linewidth=1.5)

    # Naive
    naive_preds = naive_baseline(train, test)
    ax.axhline(y=naive_preds[0], color="gray", linestyle="--", alpha=0.5, label="Naive baseline")

    # ARIMA
    try:
        from arima_model import evaluate_arima
        arima_eval = evaluate_arima(state)
        if "predictions" in arima_eval:
            ax.plot(test.index[:len(arima_eval["predictions"])],
                    arima_eval["predictions"],
                    label=f"ARIMA (MAE={arima_eval['MAE']:.1f})",
                    color="blue", linewidth=1)
    except Exception:
        pass

    # LSTM
    try:
        from rnn_model import evaluate_lstm
        lstm_eval = evaluate_lstm(state)
        if "predictions" in lstm_eval:
            ax.plot(test.index[:len(lstm_eval["predictions"])],
                    lstm_eval["predictions"],
                    label=f"LSTM (MAE={lstm_eval['MAE']:.1f})",
                    color="red", linewidth=1)
    except Exception:
        pass

    ax.set_title(f"AQI Forecast Comparison — {state}")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()

    if save:
        safe_name = state.replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(DOCS_DIR, f"comparison_{safe_name}.png"), dpi=120)

    return fig


def build_summary_table(states: list) -> pd.DataFrame:
    """Evaluate all models on a list of states and return a summary DataFrame."""
    rows = []
    for state in states:
        result = evaluate_all_models(state)
        for model_name in ["naive", "seasonal_naive", "arima", "lstm"]:
            if model_name in result and "error" not in result[model_name]:
                row = {"State": state, "Model": model_name}
                row.update(result[model_name])
                # Remove predictions array from summary row
                row.pop("predictions", None)
                rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 8 — MODEL COMPARISON")
    print("=" * 70)

    # Evaluate top 5 states
    states = list_valid_states(min_points=100)[:5]
    print(f"\nEvaluating states: {states}")

    summary = build_summary_table(states)
    print("\n" + summary.to_string(index=False))

    # Save summary
    summary.to_csv(os.path.join(DOCS_DIR, "model_comparison.csv"), index=False)
    print(f"\nSaved: docs/model_comparison.csv")

    # Plot comparisons
    for state in states:
        plot_comparison(state)
        print(f"Saved comparison plot for {state}")

    plt.close('all')
    print("\nDone!")
