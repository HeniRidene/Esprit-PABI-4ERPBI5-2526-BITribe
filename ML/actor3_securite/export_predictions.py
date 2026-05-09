import os
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GENERATED_AT = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')


def export_predictions(df_ml, df_zone=None, df_cluster=None):
    """
    Generates Power BI-ready CSVs:
      1. severity_predictions.csv  — per-row severity_class prediction + probability
      2. risk_zone_clusters.csv    — already written by model_clustering; verified here
      3. anomaly_flags.csv         — already written by model_anomaly; verified here

    All files include zone_sk + time_sk as joining keys.
    """
    logging.info("=== EXPORT PREDICTIONS — Actor 3 ===")
    os.makedirs('outputs', exist_ok=True)

    # ── 1. severity_predictions.csv ───────────────────────────────────────────
    try:
        rf_model    = joblib.load('outputs/rf_severity.pkl')
        rf_features = joblib.load('outputs/rf_severity_features.pkl')

        avail = [f for f in rf_features if f in df_ml.columns]
        df_pred_in = df_ml[avail].copy()
        for col in df_pred_in.columns:
            if df_pred_in[col].isnull().any():
                df_pred_in[col] = df_pred_in[col].fillna(df_pred_in[col].median())

        proba = rf_model.predict_proba(df_pred_in)
        preds = rf_model.predict(df_pred_in)

        df_sev = df_ml[['time_sk', 'zone_sk', 'mode_sk', 'annee', 'mois',
                         'nb_accidents', 'nb_graves', 'nb_mortels',
                         'volume_crimes', 'taux_criminalite',
                         'severity_class', 'severity_label', 'gravite_index']].copy()
        df_sev['severity_pred']  = preds
        # Probability of most severe available class (class 1 in binary, class 2 in 3-class)
        df_sev['severity_proba'] = proba[:, -1].round(4)
        df_sev['generated_at']   = GENERATED_AT

        out_path = 'outputs/severity_predictions.csv'
        df_sev.to_csv(out_path, index=False)
        logging.info(f"  ✅ {out_path}  ({len(df_sev)} rows)")

    except Exception as e:
        logging.error(f"severity_predictions export failed: {e}")
        raise

    # ── 2. Verify risk_zone_clusters.csv ──────────────────────────────────────
    cluster_path = 'outputs/risk_zone_clusters.csv'
    if os.path.exists(cluster_path):
        n = len(pd.read_csv(cluster_path))
        logging.info(f"  ✅ {cluster_path} ({n} zones)")
    else:
        logging.warning(f"  ⚠️  {cluster_path} not found — run model_clustering.py first.")

    # ── 3. Verify anomaly_flags.csv ───────────────────────────────────────────
    anom_path = 'outputs/anomaly_flags.csv'
    if os.path.exists(anom_path):
        n = len(pd.read_csv(anom_path))
        n_anom = pd.read_csv(anom_path)['is_anomaly'].sum()
        logging.info(f"  ✅ {anom_path} ({n} rows, {n_anom} alerts)")
    else:
        logging.warning(f"  ⚠️  {anom_path} not found — run model_anomaly.py first.")

    logging.info("Export complete.")


if __name__ == "__main__":
    from data_loader import load_security_data
    from feature_engineering import engineer_features
    df_raw = load_security_data()
    df_ml, df_zone = engineer_features(df_raw)
    export_predictions(df_ml, df_zone)
