# 🌍 AeroLens — Neural Air Quality Index (AQI) Intelligence & Forecaster

A modern web application combining **Phase 1 (Time-Series Forecasting with ARIMA & LSTM)** and **Phase 2 (Random Forest Ensemble Bagging)**.

---

## 📋 Input Fields (Synchronized with `AQI_UI`)

The UI keeps **only the essential user fields** from `C:\Users\Vimal\Downloads\AQI_UI`:

1. **Forecast Level**:
   - `State Level`: Regional aggregation (~550–700 daily observations per state).
   - `Station Level`: Specific monitoring station sensors across India (500+ stations).
2. **Entity Selection**:
   - Dynamic glassmorphic dropdown to select any State (`Delhi`, `Maharashtra`, `Karnataka`, `Assam`, etc.) or Station.
   - Live telemetry chip displaying available dataset span, total observations, and baseline stats.
3. **Date Range**:
   - `From Date` & `To Date` date pickers.
   - **Quick Presets**: `Past 14 Days`, `Past 30 Days`, `Next 7 Days (Future)`, `Next 14 Days (Future)`.
4. **Model Selection**:
   - **ARIMA**: Statistical time-series model with 95% Confidence Interval band.
   - **RNN (LSTM)**: Recurrent neural network deep learning model.
   - **Random Forest**: Phase 2 ensemble bagging model ($R^2 = 0.940$, RMSE = 21.7).
   - **Compare Models**: Side-by-side time-series overlay, benchmark test metrics, and winner badge.

---

## ✨ Liquid Glass UI Design

- **Mesh Gradient Canvas**: Multi-layered ambient orbs (`rgba(0, 242, 254)`, `rgba(127, 0, 255)`) animating with continuous liquid refractions.
- **Glassmorphic Panels**: Translucent frosted containers using `backdrop-filter: blur(30px) saturate(160%)` and soft gradient borders.
- **Liquid Glass Action Button**: Interactive sweeping light sheen, specular inset reflections, and active press states.
- **Interactive Time-Series Plot**: Interactive daily chart visualizing:
  - Historical Actual AQI trajectory
  - Model Forecast lines (ARIMA, LSTM, Random Forest)
  - 95% Confidence Interval area
  - Indian CPCB threshold reference levels
- **Model Test Metrics Card**: Compares MAE, RMSE, MAPE, $R^2$, and Category Accuracy.
- **Dynamic Spectrum Gauge & Health Advisory**: 0–500+ spectrum meter and actionable health recommendations.

---

## 🚀 How to Run

### Option 1: Double-Click
- Run `run.bat` for the Flask Liquid Glass UI (`http://127.0.0.1:5000`).
- Run `run_streamlit.bat` for the Streamlit UI (`http://localhost:8501`).

### Option 2: Terminal
```bash
# To run Flask Liquid Glass App:
python app.py

# To run Streamlit App:
streamlit run streamlit_app.py
```

---

## 🌐 Cloud Deployment Guide

### Option 1: Streamlit Community Cloud (100% Free · share.streamlit.io)
1. Initialize Git and push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Deploy AeroLens AQI App"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Select your repository, branch (`main`), and set Main file path to:
   ```
   streamlit_app.py
   ```
5. Click **Deploy!** Your app will be live at `https://<your-app-name>.streamlit.app`.

### Option 2: Render.com (Flask Web Service)
1. Initialize Git and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Deploy AeroLens AQI App"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 2 --threads 4 --timeout 120`
5. Click **Deploy Web Service** to get your public HTTPS URL!

### Option 2: Instant Public Tunnel (For Viva / Demo without account setup)
While running `python app.py`:
```bash
npx localtunnel --port 5000
```
This gives you an instant public web link accessible from any phone or computer.
