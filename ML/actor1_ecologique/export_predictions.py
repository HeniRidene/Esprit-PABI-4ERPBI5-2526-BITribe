import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# Database config — set to None to skip the DB push entirely
# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    'user':     'root',
    'password': '',
    'host':     '127.0.0.1',
    'port':     '3306',
    'database': 'outt',
}
ENABLE_DB_PUSH = False   # Set True when DB is ready


def _push_to_mysql(df, table_name, if_exists='replace'):
    try:
        from sqlalchemy import create_engine
        u, pw, h, p, db = (
            DB_CONFIG['user'], DB_CONFIG['password'],
            DB_CONFIG['host'], DB_CONFIG['port'],
            DB_CONFIG['database'],
        )
        engine = create_engine(f"mysql+mysqlconnector://{u}:{pw}@{h}:{p}/{db}")
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        logging.info(f"  DB push OK → table `{table_name}`")
    except Exception as e:
        logging.warning(f"  DB push skipped: {e}")


def generate_power_bi_export():
    """
    Produces all CSVs for Power BI from the trained models.

    Outputs
    -------
    outputs/regression_predictions.csv   — full dataset predictions (historical)
    outputs/future_predictions.csv       — synthetic future (2-year) scenarios
    outputs/forecast_aqi.csv             — Prophet AQI/PM2.5/CO2/Energy (copy)
    outputs/cluster_labels.csv           — zone cluster profile
    """
    logging.info("Initiating Power BI export v2...")
    os.makedirs('outputs', exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Load models ──────────────────────────────────────────────────────────
    feature_path = 'outputs/xgboost_features.pkl'
    if not os.path.exists(feature_path):
        raise FileNotFoundError("xgboost_features.pkl missing — run model_regression.py first.")

    features = joblib.load(feature_path)
    xgb_co2 = joblib.load('outputs/xgboost_co2.pkl')
    xgb_nrj = joblib.load('outputs/xgboost_nrj.pkl')

    # Load mode target encoding (needed to compute mode_co2_mean feature)
    mode_enc_path = 'outputs/mode_co2_encoding.pkl'
    mode_enc = joblib.load(mode_enc_path) if os.path.exists(mode_enc_path) else None

    # ── Load clean data ──────────────────────────────────────────────────────
    from data_loader import load_and_clean_data
    from feature_engineering import engineer_features

    df_clean = load_and_clean_data()
    df_reg, _, _ = engineer_features(df_clean)

    # ── 1. Historical predictions ────────────────────────────────────────────
    # mode_co2_mean is already in df_reg (added by feature_engineering).
    # Override with train-only encoding for consistency with model training.
    if 'mode_co2_mean' in features and mode_enc is not None:
        df_reg['mode_co2_mean'] = df_reg['mode_sk'].map(mode_enc['mode_means']).fillna(mode_enc['global_mean'])
    X_all = df_reg[[f for f in features if f in df_reg.columns]]

    df_reg['co2_predicted']    = np.expm1(xgb_co2.predict(X_all)).round(2)
    df_reg['energie_predicted'] = np.expm1(xgb_nrj.predict(X_all)).round(2)
    df_reg['carbon_intensity'] = np.where(
        df_reg['energie_predicted'] > 0,
        (df_reg['co2_predicted'] / df_reg['energie_predicted']).round(4),
        np.nan,
    )
    df_reg['generated_at'] = generated_at
    df_reg['prediction_type'] = 'historical'

    hist_cols = ['time_sk', 'zone_sk', 'mode_sk', 'annee', 'mois',
                 'co2_predicted', 'energie_predicted', 'carbon_intensity',
                 'prediction_type', 'generated_at']
    hist_df = df_reg[[c for c in hist_cols if c in df_reg.columns]].copy()
    hist_df.to_csv('outputs/regression_predictions.csv', index=False)
    logging.info(f"  ✅ regression_predictions.csv  ({len(hist_df)} rows)")

    # ── 2. Future-year synthetic predictions ─────────────────────────────────
    # Build a synthetic grid for the next 24 months
    logging.info("  Generating future-year predictions (next 24 months)...")

    max_year  = int(df_reg['annee'].max())
    max_month = int(df_reg[df_reg['annee'] == max_year]['mois'].max())

    future_dates = pd.date_range(
        start=f"{max_year}-{max_month:02d}-01",
        periods=25, freq='MS'
    )[1:]   # skip the current last month

    # Use zone × mode combinations from training data
    zone_mode_combos = df_reg[['zone_sk', 'mode_sk', 'zone_encoded', 'mode_encoded']].drop_duplicates()

    future_rows = []
    for _, combo in zone_mode_combos.iterrows():
        for dt in future_dates:
            # Use 12-month-back median as a proxy for lag/rolling features
            hist_zone = df_reg[
                (df_reg['zone_sk'] == combo['zone_sk']) &
                (df_reg['mode_sk'] == combo['mode_sk'])
            ].sort_values('annee_mois_dt').tail(12)

            row = {
                'zone_sk':       combo['zone_sk'],
                'mode_sk':       combo['mode_sk'],
                'zone_encoded':  combo['zone_encoded'],
                'mode_encoded':  combo['mode_encoded'],
                'annee':         dt.year,
                'mois':          dt.month,
                'mois_sin':      np.sin(2 * np.pi * dt.month / 12),
                'mois_cos':      np.cos(2 * np.pi * dt.month / 12),
                'annee_mois_dt': dt,
                'prediction_type': 'future',
            }

            # Fill numeric features from historical median of that zone/mode
            for col in ['pm25', 'no2', 'aqi_index',
                        'co2_lag1', 'co2_lag3', 'energie_lag1', 'energie_lag3',
                        'aqi_lag1', 'pm25_lag1', 'co2_roll3', 'energie_roll3', 'pm25_roll3']:
                if col in hist_zone.columns:
                    row[col] = float(hist_zone[col].median()) if len(hist_zone) > 0 else 0.0
                else:
                    row[col] = 0.0

            future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    # Add mode_co2_mean feature to future rows (same encoding as training)
    if 'mode_co2_mean' in features and mode_enc is not None:
        future_df['mode_co2_mean'] = future_df['mode_sk'].map(mode_enc['mode_means']).fillna(mode_enc['global_mean'])
    future_features = [f for f in features if f in future_df.columns]
    missing_f = set(features) - set(future_features)
    if missing_f:
        for mf in missing_f:
            future_df[mf] = 0.0
        future_features = features

    future_df['co2_predicted']    = np.expm1(xgb_co2.predict(future_df[future_features])).round(2)
    future_df['energie_predicted'] = np.expm1(xgb_nrj.predict(future_df[future_features])).round(2)
    future_df['carbon_intensity']  = np.where(
        future_df['energie_predicted'] > 0,
        (future_df['co2_predicted'] / future_df['energie_predicted']).round(4),
        np.nan,
    )
    future_df['generated_at'] = generated_at

    future_out_cols = ['zone_sk', 'mode_sk', 'annee', 'mois', 'annee_mois_dt',
                       'co2_predicted', 'energie_predicted', 'carbon_intensity',
                       'prediction_type', 'generated_at']
    future_df[future_out_cols].to_csv('outputs/future_predictions.csv', index=False)
    logging.info(f"  ✅ future_predictions.csv  ({len(future_df)} rows — {future_df['annee'].unique().tolist()})")

    # ── 3. Copy Prophet forecasts (already written by model_timeseries.py) ──
    prophet_src = 'outputs/prophet_forecasts.csv'
    if os.path.exists(prophet_src):
        prophet_df = pd.read_csv(prophet_src)
        prophet_df.to_csv('outputs/forecast_aqi.csv', index=False)
        logging.info(f"  ✅ forecast_aqi.csv  ({len(prophet_df)} rows)")
    else:
        logging.warning("  ⚠️  prophet_forecasts.csv not found — skipping forecast_aqi.csv")

    # ── 4. Cluster labels (already written by model_clustering.py) ──────────
    cluster_src = 'outputs/cluster_labels.csv'
    if os.path.exists(cluster_src):
        logging.info(f"  ✅ cluster_labels.csv already exists.")
    else:
        logging.warning("  ⚠️  cluster_labels.csv not found — run model_clustering.py first.")

    # ── 5. Optional DB push ──────────────────────────────────────────────────
    if ENABLE_DB_PUSH:
        _push_to_mysql(hist_df,   'ml_predictions_actor1')
        _push_to_mysql(future_df[future_out_cols], 'ml_future_predictions_actor1')

    logging.info("Export complete.")


if __name__ == "__main__":
    generate_power_bi_export()