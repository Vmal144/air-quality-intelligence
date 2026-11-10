"""
train_model.py
--------------
Trains a Random Forest Regressor pipeline on the enriched AQI dataset
and saves the model + metadata for the Flask app.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── paths ───────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "enriched_aqi_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")

# ── feature definitions (must match notebook exactly) ───────────────────
num_cols = [
    "latitude", "longitude",
    "pollutant_min", "pollutant_max", "pollutant_avg",
    "Temperature_C", "Humidity_%", "Wind_Speed_kmh",
    "Month", "Hour", "Day_of_Week",
]
cat_cols = ["state", "pollutant_id", "Season"]
target = "AQI"

def main():
    # 1. Load dataset
    print(f"Reading dataset from {os.path.abspath(DATASET_PATH)} ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  rows: {len(df)}, columns: {list(df.columns)}")

    # 2. Drop rows where target is missing
    df = df.dropna(subset=[target])

    X = df[num_cols + cat_cols]
    y = df[target]
    print(f"  training samples: {len(X)}")

    # 3. Build preprocessing + model pipeline (mirrors the notebook)
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # 4. Train
    print("Training Random Forest …")
    pipe_rf.fit(X, y)
    print("  done.")

    # 5. Save model
    joblib.dump(pipe_rf, MODEL_PATH)
    print(f"Model saved → {os.path.abspath(MODEL_PATH)}")

    # 6. Save metadata for the frontend (unique sorted lists)
    meta = {
        "states": sorted(df["state"].dropna().unique().tolist()),
        "pollutant_ids": sorted(df["pollutant_id"].dropna().unique().tolist()),
        "seasons": sorted(df["Season"].dropna().unique().tolist()),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved → {os.path.abspath(META_PATH)}")


if __name__ == "__main__":
    main()
