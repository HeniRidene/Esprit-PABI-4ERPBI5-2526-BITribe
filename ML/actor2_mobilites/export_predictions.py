import os
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GENERATED_AT = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

LOW_THRESHOLD    = 0.20
MEDIUM_THRESHOLD = 0.50


def _risk_label(p):
    if p < LOW_THRESHOLD:
        return 'Low'
    elif p < MEDIUM_THRESHOLD:
        return 'Medium'
    return 'High'


def export_predictions(df_ml, df_ts=None):
    """
    Generates three Power BI-ready CSVs:
      1. load_predictions.csv      — predicted charge_estimee per trip row
      2. cancellation_risk.csv     — annule_pred + annule_proba + risk_label per trip row
      3. congestion_forecast.csv   — already written by model_timeseries; just verified here

    All files include zone_sk + line_sk as join keys for Power BI.
    """
    logging.info("=== EXPORT PREDICTIONS — Actor 2 ===")
    os.makedirs('outputs', exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    reg_model    = joblib.load('outputs/xgboost_charge.pkl')
    reg_features = joblib.load('outputs/xgboost_charge_features.pkl')
    clf_model    = joblib.load('outputs/xgboost_cancellation.pkl')
    clf_features = joblib.load('outputs/xgboost_cancellation_features.pkl')

    # Load charge target encoding maps
    charge_enc = joblib.load('outputs/charge_encoding.pkl') if os.path.exists('outputs/charge_encoding.pkl') else None

    # ── 1. load_predictions.csv ───────────────────────────────────────────────
    # Apply target encoding maps if available
    if charge_enc is not None:
        gm = charge_enc['global_mean']
        df_reg_in = df_ml[[c for c in ['zone_encoded', 'line_encoded', 'mode_encoded', 'hour',
                                        'rush_hour', 'is_weekend', 'congestion_index',
                                        'vitesse_kmh', 'temps_trajet_min'] if c in df_ml.columns]].copy()
        for col in df_reg_in.columns:
            if df_reg_in[col].isnull().any():
                df_reg_in[col] = df_reg_in[col].fillna(df_reg_in[col].median())
        df_reg_in['line_charge_mean']  = df_reg_in['line_encoded'].map(charge_enc['line_charge_means']).fillna(gm)
        zh = charge_enc['zone_hour_means']
        df_reg_in['zone_hour_charge']  = [zh.get((z, h), gm) for z, h in zip(df_reg_in['zone_encoded'], df_reg_in['hour'])]
        df_reg_in['hour_x_congestion'] = df_reg_in['hour'] * df_reg_in['congestion_index']
        df_reg_in['weekend_x_zone']    = df_reg_in['is_weekend'] * df_reg_in['zone_encoded']

    df_load                      = df_ml[['time_sk', 'zone_sk', 'line_sk', 'mode_sk',
                                           'hour', 'annee', 'mois']].copy()
    X_all_reg = df_reg_in[[f for f in reg_features if f in df_reg_in.columns]]
    df_load['charge_predicted']  = reg_model.predict(X_all_reg).round(2)
    df_load['charge_actual']     = df_ml['charge_estimee'].values
    df_load['generated_at']      = GENERATED_AT

    load_path = 'outputs/load_predictions.csv'
    df_load.to_csv(load_path, index=False)
    logging.info(f"  ✅ {load_path}  ({len(df_load)} rows)")

    # ── 2. cancellation_risk.csv ──────────────────────────────────────────────
    df_clf_in = df_ml[[f for f in clf_features if f in df_ml.columns]].copy()
    for col in df_clf_in.columns:
        if df_clf_in[col].isnull().any():
            df_clf_in[col] = df_clf_in[col].fillna(df_clf_in[col].median())

    proba            = clf_model.predict_proba(df_clf_in)[:, 1]
    preds            = (proba >= MEDIUM_THRESHOLD).astype(int)

    df_risk                  = df_ml[['time_sk', 'zone_sk', 'line_sk',
                                       'hour', 'annule']].copy()
    df_risk['annule_pred']   = preds
    df_risk['annule_proba']  = proba.round(4)
    df_risk['risk_label']    = [_risk_label(p) for p in proba]
    df_risk['generated_at']  = GENERATED_AT

    risk_path = 'outputs/cancellation_risk.csv'
    df_risk.to_csv(risk_path, index=False)
    logging.info(f"  ✅ {risk_path}  ({len(df_risk)} rows)")
    logging.info(f"     Risk distribution: {dict(df_risk['risk_label'].value_counts())}")

    # ── 3. Verify congestion_forecast.csv ─────────────────────────────────────
    fc_path = 'outputs/forecast_congestion.csv'
    if os.path.exists(fc_path):
        fc_df = pd.read_csv(fc_path)
        logging.info(f"  ✅ {fc_path} already exists ({len(fc_df)} rows)")
    else:
        logging.warning(f"  ⚠️  {fc_path} not found — run model_timeseries.py first.")

    logging.info("Export complete.")


if __name__ == "__main__":
    from data_loader import load_mobility_data
    from feature_engineering import engineer_features

    df_raw = load_mobility_data()
    df_ml, df_ts = engineer_features(df_raw)
    export_predictions(df_ml, df_ts)
