"""
Step 2 — Data Audit / EDA
Performs a thorough exploratory analysis of the enriched AQI dataset.
Outputs statistics to the console and saves plots to docs/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

# --------------- paths ---------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "enriched_aqi_dataset.csv")
DOCS = os.path.join(BASE, "docs")
os.makedirs(DOCS, exist_ok=True)

# --------------- load ---------------
print("=" * 70)
print("STEP 2 — DATA AUDIT")
print("=" * 70)

df = pd.read_csv(DATA)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")

# parse dates
df["last_update"] = pd.to_datetime(df["last_update"], format="mixed", dayfirst=False)
print(f"\nDate range: {df['last_update'].min()} → {df['last_update'].max()}")

# --------------- basic counts ---------------
print(f"\nUnique countries : {df['country'].nunique()}")
print(f"Unique states    : {df['state'].nunique()}")
print(f"Unique cities    : {df['city'].nunique()}")
print(f"Unique stations  : {df['station'].nunique()}")
print(f"Distinct pollutant_ids: {sorted(df['pollutant_id'].dropna().unique())}")
print(f"Total rows       : {len(df)}")

# --------------- missing values ---------------
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(f"\nMissing values:\n{pd.DataFrame({'count': missing, 'pct': missing_pct})}")

# --------------- duplicates ---------------
dup_count = df.duplicated().sum()
print(f"\nExact duplicate rows: {dup_count}")

# --------------- AQI stats ---------------
print(f"\nAQI stats:\n  min  = {df['AQI'].min()}\n  max  = {df['AQI'].max()}\n  mean = {df['AQI'].mean():.2f}\n  median = {df['AQI'].median()}")
print(f"\nAQI_Bucket counts:\n{df['AQI_Bucket'].value_counts()}")

# --------------- rows per station ---------------
station_counts = df.groupby("station").size().sort_values(ascending=False)
print(f"\nRows per station (top 20):\n{station_counts.head(20)}")
print(f"\nRows per station — summary: min={station_counts.min()}, max={station_counts.max()}, mean={station_counts.mean():.1f}, median={station_counts.median()}")

# --------------- unique timestamps per station ---------------
ts_per_station = df.groupby("station")["last_update"].nunique().sort_values(ascending=False)
print(f"\nUnique timestamps per station (top 20):\n{ts_per_station.head(20)}")
print(f"Timestamps per station — summary: min={ts_per_station.min()}, max={ts_per_station.max()}, mean={ts_per_station.mean():.1f}")

# --------------- time gap analysis ---------------
print("\n--- Time gap analysis (top 5 stations by data) ---")
top5 = ts_per_station.head(5).index.tolist()
for stn in top5:
    stn_df = df[df["station"] == stn].sort_values("last_update")
    times = stn_df["last_update"].drop_duplicates().sort_values()
    gaps = times.diff().dropna()
    if len(gaps) > 0:
        print(f"\n  Station: {stn}")
        print(f"    # unique timestamps: {len(times)}")
        print(f"    Date range: {times.min()} → {times.max()}")
        print(f"    Gaps — min: {gaps.min()}, max: {gaps.max()}, median: {gaps.median()}, mode: {gaps.mode().iloc[0] if len(gaps.mode()) > 0 else 'N/A'}")

# --------------- pollutant rows per timestamp per station ---------------
pollutants_per_ts = df.groupby(["station", "last_update"])["pollutant_id"].nunique()
print(f"\nPollutant readings per (station, timestamp) — min: {pollutants_per_ts.min()}, max: {pollutants_per_ts.max()}, mean: {pollutants_per_ts.mean():.1f}")

# --------------- decision: what frequency and which stations ---------------
# After collapsing pollutants → one row per (station, timestamp),
# how many data points per station?
collapsed_counts = df.groupby(["station", "last_update"]).ngroups  # total unique combos
collapsed_per_station = df.groupby("station")["last_update"].nunique().sort_values(ascending=False)

# Stations with enough data (≥200 for LSTM, ≥100 for ARIMA after resampling)
# Since data appears hourly, daily resampling would divide counts by ~24
# Let's estimate both
print("\n--- Data sufficiency assessment ---")
for stn in collapsed_per_station.head(20).index:
    n_ts = collapsed_per_station[stn]
    daily_est = n_ts // 24  # rough estimate after daily resampling
    print(f"  {stn}: {n_ts} hourly pts → ~{daily_est} daily pts")

sufficient_hourly_lstm = collapsed_per_station[collapsed_per_station >= 200].index.tolist()
sufficient_hourly_arima = collapsed_per_station[collapsed_per_station >= 100].index.tolist()
sufficient_daily_lstm = collapsed_per_station[collapsed_per_station >= 200 * 24].index.tolist()
sufficient_daily_arima = collapsed_per_station[collapsed_per_station >= 100 * 24].index.tolist()

print(f"\nStations with ≥200 hourly points (ok for LSTM): {len(sufficient_hourly_lstm)}")
print(f"Stations with ≥100 hourly points (ok for ARIMA): {len(sufficient_hourly_arima)}")
print(f"Stations with ≥200 daily points (ok for daily LSTM): {len(sufficient_daily_lstm)}")
print(f"Stations with ≥100 daily points (ok for daily ARIMA): {len(sufficient_daily_arima)}")

# --------------- PLOTS ---------------
# Plot 1: AQI over time for top 3 stations
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
top3 = collapsed_per_station.head(3).index.tolist()
for i, stn in enumerate(top3):
    stn_df = df[df["station"] == stn].copy()
    agg = stn_df.groupby("last_update")["AQI"].max().sort_index()
    axes[i].plot(agg.index, agg.values, linewidth=0.7)
    axes[i].set_ylabel("AQI")
    axes[i].set_title(f"{stn} ({len(agg)} pts)", fontsize=9)
fig.suptitle("AQI over time — top 3 stations (by data count)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "eda_aqi_timeseries_top3.png"), dpi=120)
print(f"\nSaved: docs/eda_aqi_timeseries_top3.png")

# Plot 2: AQI distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["AQI"].dropna(), bins=60, edgecolor="black", alpha=0.7)
ax.set_xlabel("AQI")
ax.set_ylabel("Count")
ax.set_title("AQI Distribution (all data)")
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "eda_aqi_distribution.png"), dpi=120)
print(f"Saved: docs/eda_aqi_distribution.png")

# Plot 3: AQI by hour of day
fig, ax = plt.subplots(figsize=(8, 5))
hourly_aqi = df.groupby("Hour")["AQI"].mean()
ax.bar(hourly_aqi.index, hourly_aqi.values, color="steelblue")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean AQI")
ax.set_title("Average AQI by Hour of Day")
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "eda_aqi_by_hour.png"), dpi=120)
print(f"Saved: docs/eda_aqi_by_hour.png")

# Plot 4: AQI by month
fig, ax = plt.subplots(figsize=(8, 5))
monthly_aqi = df.groupby("Month")["AQI"].mean()
ax.bar(monthly_aqi.index, monthly_aqi.values, color="coral")
ax.set_xlabel("Month")
ax.set_ylabel("Mean AQI")
ax.set_title("Average AQI by Month")
ax.set_xticks(monthly_aqi.index)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "eda_aqi_by_month.png"), dpi=120)
print(f"Saved: docs/eda_aqi_by_month.png")

plt.close('all')
print("\n✅ Step 2 — Data Audit complete!")
