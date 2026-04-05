import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def engineer_features(df_clean):
    """
    Transforms clean data into ML-ready features.

    Strategy
    --------
    - Cyclical (sin/cos) month encoding → removes the 1<12 ordinal bias.
    - Lag features (1-month and 3-month) per (zone_sk, mode_sk).
    - Rolling 3-month mean of past values (shifted by 1 to avoid leakage).
    - carbon_intensity is a KPI OUTPUT column only — never used as a predictor.
    - Log1p transform on both regression targets to reduce right skew.
    - Returns df_reg (for XGBoost), df_cluster (for K-Means), df_prophet (raw aggs for Prophet).
    """
    logging.info("=== FEATURE ENGINEERING v3 ===")

    df = df_clean.copy()

    # ── 0. Type safety ────────────────────────────────────────────────────────
    df['annee'] = pd.to_numeric(df['annee'], errors='coerce').astype('Int64')
    df['mois']  = pd.to_numeric(df['mois'],  errors='coerce').astype('Int64')
    df.dropna(subset=['annee', 'mois'], inplace=True)
    df['annee'] = df['annee'].astype(int)
    df['mois']  = df['mois'].astype(int)

    # ── 1. Build datetime column ──────────────────────────────────────────────
    df['annee_mois_dt'] = pd.to_datetime(
        df['annee'].astype(str) + '-' + df['mois'].astype(str).str.zfill(2) + '-01'
    )

    # ── 2. Flag and drop zero targets ─────────────────────────────────────────
    initial_len = len(df)
    df['co2_kg']      = df['co2_kg'].replace(0, np.nan)
    df['aqi_index']   = df['aqi_index'].replace(0, np.nan)
    df['energie_kwh'] = df['energie_kwh'].replace(0, np.nan)
    df.dropna(subset=['co2_kg', 'aqi_index', 'energie_kwh'], inplace=True)
    logging.info(f"Dropped {initial_len - len(df)} rows with 0/NaN in targets → {len(df)} rows")

    # ── 3. Categorical encoding ───────────────────────────────────────────────
    df['zone_encoded'] = df['zone_sk'].astype('category').cat.codes
    df['mode_encoded'] = df['mode_sk'].astype('category').cat.codes

    # ── 3b. Mode-level mean CO2 (target encoding — computed on full dataset here;
    #        model_regression.py re-computes it on train-only for the final fit,
    #        which eliminates test leakage). This column gives XGBoost a strong
    #        prior for the multi-modal CO2 distribution (modes differ by 100x).
    mode_co2_means     = df.groupby('mode_sk')['co2_kg'].transform('mean')
    df['mode_co2_mean'] = mode_co2_means

    # ── 4. Cyclical month encoding ────────────────────────────────────────────
    df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
    df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)

    # ── 5. Log1p transforms on regression targets ─────────────────────────────
    df['co2_log']     = np.log1p(df['co2_kg'])
    df['energie_log'] = np.log1p(df['energie_kwh'])

    # ── 6. Lag & rolling features per (zone_sk, mode_sk) ─────────────────────
    #    Sort by time first so lags are directionally correct.
    df.sort_values(['zone_sk', 'mode_sk', 'annee_mois_dt'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    group_cols = ['zone_sk', 'mode_sk']

    for lag in [1, 3]:
        df[f'co2_lag{lag}']     = df.groupby(group_cols)['co2_kg'].shift(lag)
        df[f'energie_lag{lag}'] = df.groupby(group_cols)['energie_kwh'].shift(lag)
        df[f'aqi_lag{lag}']     = df.groupby(group_cols)['aqi_index'].shift(lag)
        df[f'pm25_lag{lag}']    = df.groupby(group_cols)['pm25'].shift(lag)

    for col, alias in [('co2_kg', 'co2_roll3'), ('energie_kwh', 'energie_roll3'), ('pm25', 'pm25_roll3')]:
        df[alias] = df.groupby(group_cols)[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

    # Drop rows where ALL lag-1 features are NaN (first observation per group)
    lag1_cols = ['co2_lag1', 'energie_lag1', 'aqi_lag1', 'pm25_lag1']
    before_drop = len(df)
    df.dropna(subset=lag1_cols, inplace=True)
    logging.info(f"Dropped {before_drop - len(df)} rows with missing lag-1 features → {len(df)} rows")

    # ── 7. Carbon intensity — KPI output only ────────────────────────────────
    df['carbon_intensity'] = np.where(
        df['energie_kwh'] > 0,
        df['co2_kg'] / df['energie_kwh'],
        np.nan,
    )

    # ── 8. df_reg for XGBoost regression ─────────────────────────────────────
    df_reg = df.copy()

    # ── 9. df_cluster for K-Means ────────────────────────────────────────────
    #    Light scaling of pm25/no2 only (clustering scaler lives in model_clustering.py)
    df_cluster = df.copy()
    scaler_light = StandardScaler()
    df_cluster[['pm25_scaled', 'no2_scaled']] = scaler_light.fit_transform(
        df_cluster[['pm25', 'no2']]
    )
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(scaler_light, 'outputs/pm25_no2_scaler.pkl')

    # ── 10. df_prophet — monthly aggregated by zone (raw, no lag drops) ──────
    df_prophet = df_clean.copy()
    df_prophet['annee'] = pd.to_numeric(df_prophet['annee'], errors='coerce').astype('Int64')
    df_prophet['mois']  = pd.to_numeric(df_prophet['mois'],  errors='coerce').astype('Int64')
    df_prophet.dropna(subset=['annee', 'mois'], inplace=True)
    df_prophet['annee'] = df_prophet['annee'].astype(int)
    df_prophet['mois']  = df_prophet['mois'].astype(int)
    df_prophet['annee_mois_dt'] = pd.to_datetime(
        df_prophet['annee'].astype(str) + '-' + df_prophet['mois'].astype(str).str.zfill(2) + '-01'
    )
    for col in ['co2_kg', 'aqi_index', 'energie_kwh']:
        df_prophet[col] = df_prophet[col].replace(0, np.nan)

    logging.info(
        f"Feature engineering complete. "
        f"df_reg: {df_reg.shape}  df_cluster: {df_cluster.shape}  df_prophet: {df_prophet.shape}"
    )
    return df_reg, df_cluster, df_prophet


if __name__ == "__main__":
    from data_loader import load_and_clean_data

    df_clean = load_and_clean_data()
    df_reg, df_cluster, df_prophet = engineer_features(df_clean)

    new_cols = [
        'annee_mois_dt', 'zone_encoded', 'mode_encoded',
        'mois_sin', 'mois_cos', 'co2_log', 'energie_log',
        'co2_lag1', 'co2_roll3', 'carbon_intensity',
    ]
    print(df_reg[[c for c in new_cols if c in df_reg.columns]].head(8))
    print(f"\ndf_reg  : {df_reg.shape}")
    print(f"df_cluster: {df_cluster.shape}")
    print(f"df_prophet: {df_prophet.shape}")
    print(f"\nco2_kg stats:\n{df_reg['co2_kg'].describe()}")