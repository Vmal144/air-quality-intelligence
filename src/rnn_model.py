"""
rnn_model.py — LSTM (RNN) model for AQI forecasting.

Key Design Decisions:
1. Aggregation is by STATE (each valid state has ~550-700+ daily data points).
2. Scaling: MinMaxScaler is fit ONLY on the training split to prevent data leakage.
3. Feature Engineering:
   - AQI (primary target)
   - Day-of-week cyclical features: sin(2*pi*dow/7), cos(2*pi*dow/7)
   - Month cyclical features: sin(2*pi*month/12), cos(2*pi*month/12)
   - Weather features: Temperature_C, Humidity_pct, Wind_Speed_kmh
4. Window size: WINDOW = 14 days of historical observations to predict next-day AQI.
5. Model Architecture:
   - LSTM(64) -> Dropout(0.2) -> Dense(32, activation='relu') -> Dense(1)
   - Loss: Huber loss (robust to extreme AQI outliers/spikes)
   - Optimizer: Adam
   - Callbacks: EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
6. Multi-step recursive forecasting:
   - For future dates beyond historical data, recursive forecasting feeds each
     predicted AQI back into the input window.
   - For weather features in future steps, values are held at seasonal averages
     (or last known values).
   - Forecasting errors compound as the prediction horizon extends.
"""

import os
import sys
import logging
import random
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
import joblib

# Set reproducibility seeds
random.seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rnn_model")

# Suppress verbose TensorFlow C++ logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────────────────────────
# TensorFlow Import & Error Handling
# ─────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    logger.error(
        f"TensorFlow is not available or failed to import correctly: {e}. "
        "Please ensure TensorFlow is installed: pip install tensorflow"
    )

from sklearn.preprocessing import MinMaxScaler

# Add src to path for module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    BASE_DIR, MODELS_DIR, mae, rmse, mape, r_squared,
    category_accuracy, aqi_to_bucket
)
from data_prep import (
    get_state_series, get_station_series, get_entity_series,
    list_valid_states, list_valid_stations, train_val_test_split
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
WINDOW = 14
TARGET_COL = "AQI"
TARGET_IDX = 0
FEATURE_COLS = [
    "AQI",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "Temperature_C",
    "Humidity_pct",
    "Wind_Speed_kmh"
]


# ─────────────────────────────────────────────────────────────
# 1. Scaling Helper Class
# ─────────────────────────────────────────────────────────────
class AQIScaler:
    """
    Dual scaler managing MinMaxScaler for all input features and target AQI.
    Both scalers are fit strictly on training data to prevent data leakage.
    Provides dict-like compatibility and standard scikit-learn transform/inverse_transform methods.
    """
    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "AQIScaler":
        """Fit feature_scaler and target_scaler on training DataFrame only."""
        self.feature_scaler.fit(df[self.feature_cols])
        self.target_scaler.fit(df[[TARGET_COL]])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform feature DataFrame to 2D numpy array."""
        if not self.is_fitted:
            raise RuntimeError("AQIScaler must be fit before transform.")
        return self.feature_scaler.transform(df[self.feature_cols])

    def transform_target(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Transform target values."""
        if not self.is_fitted:
            raise RuntimeError("AQIScaler must be fit before transform.")
        y_arr = np.asarray(y).reshape(-1, 1)
        return self.target_scaler.transform(y_arr).flatten()

    def inverse_transform(self, y: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Inverse-transform target predictions back to original AQI scale.
        Accepts 1D or 2D array of predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("AQIScaler must be fit before inverse_transform.")
        arr = np.asarray(y)
        if arr.ndim == 1:
            return self.target_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()
        elif arr.ndim == 2 and arr.shape[1] == 1:
            return self.target_scaler.inverse_transform(arr).flatten()
        elif arr.ndim == 2 and arr.shape[1] == len(self.feature_cols):
            return self.feature_scaler.inverse_transform(arr)
        else:
            return self.target_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    def __getitem__(self, key: str) -> Any:
        if key in ("scaler_target", "scaler_y", "target_scaler"):
            return self.target_scaler
        if key in ("scaler_features", "scaler_X", "feature_scaler"):
            return self.feature_scaler
        if key in ("features", "feature_cols"):
            return self.feature_cols
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _load_scaler(scaler_path: str) -> AQIScaler:
    """Load scaler object or construct an AQIScaler instance from saved artifact."""
    obj = joblib.load(scaler_path)
    if isinstance(obj, AQIScaler):
        return obj
    elif isinstance(obj, dict):
        s = AQIScaler(feature_cols=obj.get("feature_cols", FEATURE_COLS))
        s.feature_scaler = obj.get("feature_scaler") or obj.get("scaler_features") or obj.get("scaler_X")
        s.target_scaler = obj.get("target_scaler") or obj.get("scaler_target") or obj.get("scaler_y")
        s.is_fitted = True
        return s
    elif isinstance(obj, MinMaxScaler):
        s = AQIScaler()
        s.target_scaler = obj
        s.is_fitted = True
        return s
    return obj


# ─────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal and cyclical features for daily AQI time series.
    Features:
    - AQI (target)
    - dow_sin, dow_cos: cyclical day-of-week encoding
    - month_sin, month_cos: cyclical month encoding
    - Temperature_C, Humidity_pct, Wind_Speed_kmh (imputed if missing)
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Cyclical day-of-week (0=Monday, ..., 6=Sunday)
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    # Cyclical month (1=Jan, ..., 12=Dec)
    month = df.index.month
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    # Impute missing weather features if any
    defaults = {"Temperature_C": 25.0, "Humidity_pct": 50.0, "Wind_Speed_kmh": 10.0}
    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(default_val)
        else:
            df[col] = default_val

    # Ensure AQI has no NaNs
    if "AQI" in df.columns:
        df["AQI"] = df["AQI"].ffill().bfill()

    return df


# ─────────────────────────────────────────────────────────────
# 3. Supervised Sequence Creation
# ─────────────────────────────────────────────────────────────
def create_sequences(
    data: Union[np.ndarray, pd.DataFrame],
    window_size: int = WINDOW,
    target_col_idx: int = TARGET_IDX
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for supervised time-series training.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        2D array of shape (n_samples, n_features).
    window_size : int
        Number of past timesteps in each window (default: 14).
    target_col_idx : int
        Index of the target column (default: 0 for AQI).

    Returns
    -------
    X : np.ndarray
        3D array of shape (samples, window_size, n_features).
    y : np.ndarray
        1D array of shape (samples,) containing next-step target values.
    """
    arr = np.asarray(data, dtype=np.float32)
    n_samples, n_features = arr.shape
    if n_samples <= window_size:
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32)
        )

    X, y = [], []
    for i in range(n_samples - window_size):
        X.append(arr[i : i + window_size])
        y.append(arr[i + window_size, target_col_idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 4. Model Architecture & Path Helpers
# ─────────────────────────────────────────────────────────────
def build_lstm_model(window_size: int = WINDOW, n_features: int = len(FEATURE_COLS)) -> tf.keras.Model:
    """
    Build and compile LSTM model architecture.

    Architecture:
        Sequential([
            LSTM(64, input_shape=(WINDOW, n_features)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    Compiled with Adam optimizer and Huber loss.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build the LSTM model.")

    model = Sequential([
        LSTM(64, input_shape=(window_size, n_features)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def get_model_path(state: str) -> str:
    """Return file path for saved LSTM model."""
    safe_state = state.strip().replace("/", "_").replace("\\", "_")
    p1 = os.path.join(MODELS_DIR, f"lstm_{safe_state}.keras")
    p2 = os.path.join(MODELS_DIR, f"lstm_{safe_state.replace(' ', '_')}.keras")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return p1


def get_scaler_path(state: str) -> str:
    """Return file path for saved scaler."""
    safe_state = state.strip().replace("/", "_").replace("\\", "_")
    p1 = os.path.join(MODELS_DIR, f"scaler_{safe_state}.pkl")
    p2 = os.path.join(MODELS_DIR, f"scaler_{safe_state.replace(' ', '_')}.pkl")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return p1


def _load_entity_series(entity: str) -> pd.DataFrame:
    """Load series by state or fallback to station."""
    series = get_state_series(entity)
    if series is None or len(series) < 14:
        series = get_station_series(entity)
    if series is None or len(series) < 14:
        raise ValueError(
            f"Insufficient data for '{entity}'. "
            f"Need at least 14 points, found {0 if series is None else len(series)}."
        )
    return series


# ─────────────────────────────────────────────────────────────
# 5. Training Function
# ─────────────────────────────────────────────────────────────
def train_lstm(state: str, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Train LSTM model for a given state (or station).

    Parameters
    ----------
    state : str
        State or station name.
    force_retrain : bool
        If False and model + scaler exist, loads existing and returns evaluation.

    Returns
    -------
    dict with training history, test metrics, and paths.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to train the LSTM model.")

    # Reset seeds for reproducible training
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    model_path = get_model_path(state)
    scaler_path = get_scaler_path(state)

    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        logger.info(f"Model and scaler already exist for '{state}'. Skipping re-training.")
        eval_metrics = evaluate_lstm(state)
        return {
            "status": "already_trained",
            "state": state,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "history": {},
            "metrics": eval_metrics,
            **eval_metrics
        }

    logger.info(f"Preparing training data for '{state}'...")
    series = _load_entity_series(state)
    feat_df = add_features(series)

    # Chronological train/val/test split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = train_val_test_split(feat_df, train_frac=0.70, val_frac=0.15)
    if len(train_df) <= WINDOW:
        raise ValueError(f"Train split for '{state}' too small ({len(train_df)} rows) for window size {WINDOW}.")

    # Fit scaler on TRAIN ONLY to avoid data leakage
    scaler = AQIScaler(FEATURE_COLS)
    scaler.fit(train_df)

    # Scale data
    scaled_train = scaler.transform(train_df)
    # Prepend last WINDOW rows of train to val to form complete validation sequences
    val_window_df = pd.concat([train_df.iloc[-WINDOW:], val_df])
    scaled_val = scaler.transform(val_window_df)

    X_train, y_train = create_sequences(scaled_train, window_size=WINDOW, target_col_idx=TARGET_IDX)
    X_val, y_val = create_sequences(scaled_val, window_size=WINDOW, target_col_idx=TARGET_IDX)

    logger.info(
        f"Training shapes for '{state}': X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_val={X_val.shape}, y_val={y_val.shape}"
    )

    # Build model
    model = build_lstm_model(window_size=WINDOW, n_features=len(FEATURE_COLS))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, verbose=0)
    ]

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        logger.error(f"Error during LSTM training for '{state}': {e}")
        raise

    # Save model and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved model to '{model_path}' and scaler to '{scaler_path}'")

    # Evaluate on test set
    eval_metrics = evaluate_lstm(state)

    return {
        "status": "trained",
        "state": state,
        "history": history.history,
        "epochs_trained": len(history.history.get("loss", [])),
        "metrics": eval_metrics,
        "model_path": model_path,
        "scaler_path": scaler_path,
        **eval_metrics
    }


# ─────────────────────────────────────────────────────────────
# 6. Evaluation Function
# ─────────────────────────────────────────────────────────────
def evaluate_lstm(state: str) -> Dict[str, Any]:
    """
    Evaluate trained LSTM model on the test set for the specified state.
    Returns dictionary with MAE, RMSE, MAPE, R2, category_accuracy, etc.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to evaluate the LSTM model.")

    model_path = get_model_path(state)
    scaler_path = get_scaler_path(state)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.info(f"Model or scaler not found for '{state}'. Training first...")
        train_lstm(state, force_retrain=False)
        model_path = get_model_path(state)
        scaler_path = get_scaler_path(state)

    model = load_model(model_path, compile=False)
    scaler = _load_scaler(scaler_path)

    series = _load_entity_series(state)
    feat_df = add_features(series)
    train_df, val_df, test_df = train_val_test_split(feat_df, train_frac=0.70, val_frac=0.15)

    if len(test_df) == 0:
        raise ValueError(f"Test split for '{state}' is empty.")

    # Prepend last WINDOW points of val_df to test_df to predict every test point
    test_window_df = pd.concat([val_df.iloc[-WINDOW:], test_df])
    scaled_test = scaler.transform(test_window_df)

    X_test, _ = create_sequences(scaled_test, window_size=WINDOW, target_col_idx=TARGET_IDX)
    if len(X_test) == 0:
        raise ValueError("Could not construct test sequences.")

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    # Clip predictions to physically realistic range [0, 500]
    y_pred = np.clip(y_pred, 0.0, 500.0)

    # Actual test AQI values (aligned 1-to-1 with test_df)
    y_actual = test_df["AQI"].values[:len(y_pred)]

    m_mae = float(mae(y_actual, y_pred))
    m_rmse = float(rmse(y_actual, y_pred))
    m_mape = float(mape(y_actual, y_pred))
    m_r2 = float(r_squared(y_actual, y_pred))
    m_cat = float(category_accuracy(y_actual, y_pred))

    return {
        "MAE": m_mae,
        "RMSE": m_rmse,
        "MAPE": m_mape,
        "R2": m_r2,
        "Cat_Accuracy": m_cat,
        "category_accuracy": m_cat,
        "n_samples": len(y_actual),
        "state": state
    }


# ─────────────────────────────────────────────────────────────
# 7. Prediction Function (Single-Step & Multi-Step Recursive)
# ─────────────────────────────────────────────────────────────
def lstm_predict(
    state: str,
    start: Union[datetime, str, pd.Timestamp],
    end: Union[datetime, str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Generate AQI predictions for a state over [start, end].

    For historical dates within available data:
        1-step ahead prediction using the preceding WINDOW days.
    For future dates beyond available data:
        Multi-step recursive forecasting: each predicted AQI is fed back
        into the input window for the next day's prediction.
        Weather features are held at seasonal historical averages (or last known).
        Errors compound over time as the horizon grows.

    Parameters
    ----------
    state : str
        State or station name.
    start : datetime or str or Timestamp
        Start date.
    end : datetime or str or Timestamp
        End date (must be >= start).

    Returns
    -------
    pd.DataFrame with columns: [timestamp, predicted_AQI]
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for LSTM predictions.")

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    if end_ts < start_ts:
        raise ValueError(f"End date ({end_ts.date()}) cannot be before start date ({start_ts.date()}).")

    model_path = get_model_path(state)
    scaler_path = get_scaler_path(state)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.info(f"Model or scaler not found for '{state}'. Training first...")
        train_lstm(state, force_retrain=False)
        model_path = get_model_path(state)
        scaler_path = get_scaler_path(state)

    model = load_model(model_path, compile=False)
    scaler = _load_scaler(scaler_path)

    series = _load_entity_series(state)
    feat_df = add_features(series).sort_index()

    data_start = feat_df.index.min().normalize()
    data_end = feat_df.index.max().normalize()

    target_dates = pd.date_range(start_ts, end_ts, freq="D")
    predictions: Dict[pd.Timestamp, float] = {}

    # 1. Historical dates: 1-step ahead predictions using known observations
    hist_target_dates = [d for d in target_dates if d <= data_end]
    for d in hist_target_dates:
        prior_data = feat_df[feat_df.index < d]
        if len(prior_data) >= WINDOW:
            window_df = prior_data.iloc[-WINDOW:]
        else:
            # Pad earliest available row if near the beginning of series
            pad_count = WINDOW - len(prior_data)
            pad_df = pd.concat([feat_df.iloc[[0]]] * pad_count)
            window_df = pd.concat([pad_df, prior_data])

        scaled_window = scaler.transform(window_df)
        X_in = scaled_window.reshape(1, WINDOW, len(FEATURE_COLS))
        y_pred_scaled = model.predict(X_in, verbose=0)
        y_pred = float(scaler.inverse_transform(y_pred_scaled)[0])
        predictions[d] = float(np.clip(y_pred, 0.0, 500.0))

    # 2. Future dates: Recursive multi-step forecasting
    future_needed_dates = [d for d in target_dates if d > data_end]
    if future_needed_dates:
        logger.info(
            f"Forecasting {len(future_needed_dates)} days beyond available data ({data_end.date()}). "
            "Using recursive forecasting: predictions are fed back as inputs. "
            "NOTE: Uncertainty and error variance compound with larger horizons."
        )

        # Working copy to roll predictions into
        working_df = feat_df.copy()

        # Compute seasonal monthly weather averages for realistic imputation in future steps
        monthly_weather = feat_df.groupby(feat_df.index.month)[
            ["Temperature_C", "Humidity_pct", "Wind_Speed_kmh"]
        ].mean()

        # Roll forward day by day from data_end + 1 day up to end_ts
        roll_date = data_end + timedelta(days=1)
        while roll_date <= end_ts:
            window_df = working_df.iloc[-WINDOW:]
            scaled_window = scaler.transform(window_df)
            X_in = scaled_window.reshape(1, WINDOW, len(FEATURE_COLS))

            y_pred_scaled = model.predict(X_in, verbose=0)
            y_pred = float(scaler.inverse_transform(y_pred_scaled)[0])
            pred_aqi = float(np.clip(y_pred, 0.0, 500.0))

            if roll_date in target_dates:
                predictions[roll_date] = pred_aqi

            # Weather features: seasonal average for this month, or last known
            m = roll_date.month
            if m in monthly_weather.index:
                temp_c = float(monthly_weather.loc[m, "Temperature_C"])
                hum_pct = float(monthly_weather.loc[m, "Humidity_pct"])
                wind_kmh = float(monthly_weather.loc[m, "Wind_Speed_kmh"])
            else:
                temp_c = float(working_df["Temperature_C"].iloc[-1])
                hum_pct = float(working_df["Humidity_pct"].iloc[-1])
                wind_kmh = float(working_df["Wind_Speed_kmh"].iloc[-1])

            # Append the newly predicted step to working_df for recursive feedback
            new_row = {
                "AQI": pred_aqi,
                "dow_sin": float(np.sin(2.0 * np.pi * roll_date.dayofweek / 7.0)),
                "dow_cos": float(np.cos(2.0 * np.pi * roll_date.dayofweek / 7.0)),
                "month_sin": float(np.sin(2.0 * np.pi * roll_date.month / 12.0)),
                "month_cos": float(np.cos(2.0 * np.pi * roll_date.month / 12.0)),
                "Temperature_C": temp_c,
                "Humidity_pct": hum_pct,
                "Wind_Speed_kmh": wind_kmh
            }
            working_df.loc[roll_date] = new_row
            roll_date += timedelta(days=1)

    # Format output DataFrame
    preds_list = [predictions.get(d, np.nan) for d in target_dates]
    result_df = pd.DataFrame({
        "timestamp": target_dates,
        "predicted_AQI": preds_list
    })

    return result_df


# ─────────────────────────────────────────────────────────────
# 8. CLI Test & Demonstration
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("AQI Forecasting — LSTM (RNN) Model Training and Evaluation")
    print("=" * 70)

    if not TF_AVAILABLE:
        print("Error: TensorFlow is not available. Please install tensorflow and re-run.")
        sys.exit(1)

    # 1. Discover top 3 states by data count
    states = list_valid_states(min_points=50)
    if not states:
        print("No valid states found with >= 50 daily data points.")
        sys.exit(1)

    state_counts = {}
    for s in states:
        sr = get_state_series(s)
        state_counts[s] = len(sr) if sr is not None else 0

    sorted_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
    top_3 = [s for s, count in sorted_states[:3]]

    print("\nTop 3 states by daily data count:")
    for s, count in sorted_states[:3]:
        print(f"  • {s}: {count} daily points")

    all_results = {}

    # 2. Train and evaluate on top 3 states
    for state_name in top_3:
        print("\n" + "-" * 60)
        print(f"Training LSTM for: {state_name}")
        print("-" * 60)

        try:
            train_res = train_lstm(state_name, force_retrain=True)
            hist = train_res.get("history", {})

            if "loss" in hist and "val_loss" in hist:
                train_loss = hist["loss"]
                val_loss = hist["val_loss"]
                epochs_ran = len(train_loss)
                print(f"\n  [Training Loss Curve]")
                print(f"    Epochs Trained:           {epochs_ran}")
                print(f"    Initial Train Loss:       {train_loss[0]:.4f}")
                print(f"    Final Train Loss (Huber): {train_loss[-1]:.4f}")
                print(f"    Initial Val Loss:         {val_loss[0]:.4f}")
                print(f"    Final Val Loss   (Huber): {val_loss[-1]:.4f}")
                print(f"    Best Val Loss    (Huber): {min(val_loss):.4f}")

            metrics = train_res.get("metrics", {})
            print(f"\n  [Test Set Evaluation Metrics]")
            print(f"    MAE:                      {metrics.get('MAE', 0.0):.2f}")
            print(f"    RMSE:                     {metrics.get('RMSE', 0.0):.2f}")
            print(f"    MAPE:                     {metrics.get('MAPE', 0.0):.1f}%")
            print(f"    R²:                       {metrics.get('R2', 0.0):.4f}")
            print(f"    Category Accuracy:        {metrics.get('Cat_Accuracy', 0.0)*100:.1f}%")

            # 3. Demonstrate lstm_predict: 5 days backtest + 5 days future recursive forecast
            sr = get_state_series(state_name)
            last_date = sr.index.max().date()
            start_date = last_date - timedelta(days=5)
            end_date = last_date + timedelta(days=5)

            preds = lstm_predict(state_name, start_date, end_date)
            print(f"\n  [Sample Predictions: {start_date} to {end_date}]")
            preds_display = preds.copy()
            preds_display["AQI_Bucket"] = preds_display["predicted_AQI"].apply(
                lambda x: aqi_to_bucket(round(x))
            )
            print(preds_display.to_string(index=False))

            all_results[state_name] = metrics

        except Exception as e:
            print(f"Error training/evaluating {state_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("Summary of LSTM Model Evaluation Across Top 3 States")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results).T
    if not summary_df.empty:
        cols_order = ["MAE", "RMSE", "MAPE", "R2", "Cat_Accuracy", "n_samples"]
        display_cols = [c for c in cols_order if c in summary_df.columns]
        print(summary_df[display_cols].to_string())
    print("\nAll tasks completed successfully.")
