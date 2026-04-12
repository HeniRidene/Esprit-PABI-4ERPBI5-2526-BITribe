# Actor 1 — Directeur Écologique

## Models

| Model | Type | Target / Purpose |
|---|---|---|
| **XGBoost Regression** | Gradient Boosting | Predicts `co2_kg` and `energie_kwh` from lag/rolling features |
| **Lasso Regression** | L1-regularised Linear | Sparse baseline comparison model for CO2 prediction |
| **K-Means Clustering** | Centroid-based | Zone pollution profiles — best k chosen via Silhouette score |
| **DBSCAN** | Density-based | Alternative clustering; detects noise/outlier zones automatically |
| **Prophet** | Additive decomposition | AQI / PM2.5 time series forecasting (zone 1) |
| **SARIMA** | Classical statistical | Seasonal time series baseline comparison vs Prophet |

## PKL Files

All artefacts are saved in `outputs/`.

| File | Description |
|---|---|
| `xgboost_co2.pkl` | Best XGBoost regressor (GridSearchCV) trained on `co2_log` |
| `xgboost_nrj.pkl` | XGBoost regressor trained on `energie_kwh` (from legacy pipeline) |
| `xgboost_co2_per_mode.pkl` | Per-mode XGBoost model for CO2 (legacy pipeline) |
| `xgboost_co2_per_mode_features.pkl` | Feature list for per-mode CO2 model |
| `xgboost_charge.pkl` | XGBoost model predicting transport charge (from actor2 pipeline) |
| `xgboost_charge_features.pkl` | Feature list for charge model |
| `xgboost_features.pkl` | Feature list used by the main `xgboost_co2.pkl` model |
| `kmeans_pollution_zones.pkl` | Fitted K-Means model — zone pollution cluster assignments |
| `clustering_scaler.pkl` | StandardScaler fitted on clustering features (pm25, no2, co2_kg, energie_kwh) |
| `pm25_no2_scaler.pkl` | Scaler fitted on PM2.5 and NO2 features (legacy pipeline) |
| `mode_co2_encoding.pkl` | Mean CO2 encoding per transport mode |

## How to Run

```bash
jupyter notebook actor1_ecologique.ipynb
```

> **Requirements:** Ensure the virtual environment is activated and all dependencies are installed before running.

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install statsmodels --quiet
```

Run all cells top-to-bottom (**Kernel → Restart & Run All**).  
Outputs (charts, PKL files) are written to `outputs/`.

## Key Metrics

| Metric | Value | Notes |
|---|---|---|
| R² (energy / `energie_kwh`) | **0.71** | Strong predictive power |
| R² (CO2 / `co2_kg`) | **0.20** | Data ceiling — monthly aggregation masks hourly spikes |
| Prophet MAE (AQI, zone 1) | **1.92** | 3-step holdout forecast |
| K-Means Silhouette (k=2) | **0.32** | Best silhouette across k=2..8 |

## Directory Structure

```
actor1_ecologique/
├── actor1_ecologique.ipynb   ← Main notebook (run this)
├── CLAUDE.md                 ← This file
├── requirements.txt          ← Python dependencies
├── data_loader.py            ← Legacy data loading utilities
├── feature_engineering.py   ← Legacy feature engineering utilities
├── model_regression.py      ← Legacy regression pipeline
├── model_clustering.py      ← Legacy clustering pipeline
├── model_timeseries.py      ← Legacy time-series pipeline
├── evaluate.py               ← Legacy evaluation helpers
├── main.py                   ← Legacy entry-point script
├── export_predictions.py    ← Export helpers
├── outputs/                  ← All artefacts (PKL, PNG, CSV, JSON)
├── fact_impact_territorial.csv
├── dim_temps.csv
├── dim_zone.csv
├── dim_vehicule.csv
├── dim_weather.csv
├── dim_event.csv
├── dim_arret.csv
├── dim_mode.csv
└── dim_lignes.csv
```
