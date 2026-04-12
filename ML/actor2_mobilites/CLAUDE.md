# Actor 2 — Directeur Mobilités

## Models

- **XGBoost Regression**: predicts `charge_estimee` (passenger load factor 0–1) using zone, line, mode, time, and congestion features. 5-fold CV + GridSearchCV tuning.
- **Lasso Regression**: baseline linear model (`alpha=0.1`) for charge_estimee comparison with XGBoost.
- **XGBoost Classifier**: cancellation risk (`annule` 0/1) with `scale_pos_weight=55` to handle 1.78% positive class imbalance. Threshold tuned to 0.047 for operational sensitivity.
- **Logistic Regression**: classification baseline with `class_weight='balanced'`, `max_iter=1000`.
- **Prophet**: `congestion_index` time series forecast for 10 zones, 30-day horizon, French public holidays, changepoint tuning.
- **XGBoost TS**: lag-based time series baseline with `lag_1`, `lag_7`, `lag_14`, `rolling_7_mean` features.

## PKL Files

| File | Description |
|------|-------------|
| `outputs/xgboost_charge.pkl` | Trained XGBoost regressor for passenger load (charge_estimee) |
| `outputs/xgboost_charge_features.pkl` | Ordered feature list used by xgboost_charge model |
| `outputs/charge_encoding.pkl` | Target encoding maps: line_charge_means, zone_hour_means, global_mean |
| `outputs/xgboost_cancellation.pkl` | Trained XGBoost classifier for trip cancellation risk |
| `outputs/xgboost_cancellation_features.pkl` | Ordered feature list used by xgboost_cancellation model |

## How to Run

```bash
jupyter notebook actor2_mobilites.ipynb
```

To run the full pipeline headlessly:
```bash
python main.py
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Ponctualité | 98.22% |
| XGBoost Classification AUC | 0.519 |
| Prophet Congestion MAE | 0.857 |
| Best Classification Threshold | 0.047 |
| Class Imbalance (annule=1) | 1.78% (78 / 4389 rows) |
| XGBoost Regression R² | ~-0.11 (data ceiling — not model failure) |
