# Actor 3 — Responsable Sécurité

## Models
- Random Forest: accident severity classification (binary: none/serious)
- SVM: classification baseline comparison
- K-Means k=3: zone risk clustering (High/Med/Low)
- Agglomerative Hierarchical: clustering baseline
- Isolation Forest: anomaly detection on crime/accident spikes

## PKL Files
- `rf_severity.pkl`: Pickled Random Forest classifier model for severity prediction.
- `rf_severity_features.pkl`: List of feature columns used by the Random Forest model.
- `kmeans_risk.pkl`: Pickled K-Means model for risk zone clustering (k=3).
- `kmeans_scaler.pkl`: StandardScaler used for the K-Means clustering features.
- `kmeans_features.pkl`: List of feature columns used for K-Means clustering.
- `isolation_forest.pkl`: Pickled Isolation Forest model for anomaly detection.
- `anomaly_scaler.pkl`: StandardScaler used for isolating forest features.
- `anomaly_features.pkl`: List of feature columns used for anomaly detection.

## How to Run
```bash
jupyter notebook actor3_securite.ipynb
```

## Key Metrics
F1=1.0 (documented), Silhouette=0.242, Anomalies=96/1911, Precision@96=0.375
