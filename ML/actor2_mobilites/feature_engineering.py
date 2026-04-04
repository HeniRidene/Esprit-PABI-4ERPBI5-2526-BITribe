import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def engineer_features(df_raw):
    """
    Transforms clean mobility data into ML-ready feature sets.

    Transformations
    ---------------
    1. Imputation  — vitesse_kmh + temps_trajet_min + stress_1_5 with zone median;
                     global median fallback for zones that are entirely null.
    2. Time feats  — hour (int 0-23), rush_hour (binary), is_weekend (binary).
    3. Encoding    — zone_sk / line_sk / mode_sk as category codes.
    4. retard_flag — alias for annule (only cancellation signal available).
    5. Imbalance   — logs the dynamic scale_pos_weight for the classifier.

    Returns
    -------
    df_ml    : full feature-engineered DataFrame (used by regression + classification)
    df_ts    : minimal DataFrame for Prophet (zone_sk, date, congestion_index)
    """
    logging.info("=== FEATURE ENGINEERING v1 — Actor 2 Mobilités ===")

    df = df_raw.copy()

    # ── 1. Zone-median imputation ─────────────────────────────────────────────
    cols_to_impute = ['vitesse_kmh', 'temps_trajet_min', 'stress_1_5']
    for col in cols_to_impute:
        if col not in df.columns:
            continue
        n_null_before = df[col].isnull().sum()
        if n_null_before == 0:
            continue
        # Per-zone median, then global fallback
        df[col] = df.groupby('zone_sk')[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())
        n_null_after = df[col].isnull().sum()
        logging.info(f"Imputed '{col}': {n_null_before} → {n_null_after} nulls")

    # ── 2. Time features ──────────────────────────────────────────────────────
    if 'heure' in df.columns:
        heure_s = df['heure'].astype(str).str.strip()
        if heure_s.str.contains(':').any():
            # "HH:MM:SS" — extract hour directly from first segment
            df['hour'] = heure_s.str.split(':').str[0].str.extract(r'(\d+)')[0].astype(float).fillna(0).astype(int)
        else:
            # Already an integer hour (0-23)
            df['hour'] = pd.to_numeric(heure_s, errors='coerce').fillna(0).astype(int)
    else:
        logging.warning("'heure' not found after dim_temps join — hour will be 0.")
        df['hour'] = 0

    df['hour'] = df['hour'].fillna(0).astype(int)

    # Rush hour: 7-9 or 17-19
    df['rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)

    # Weekend
    if 'weekend' in df.columns:
        df['is_weekend'] = pd.to_numeric(df['weekend'], errors='coerce').fillna(0).astype(int)
    else:
        logging.warning("'weekend' not found — is_weekend will be 0.")
        df['is_weekend'] = 0

    # ── 3. Categorical encoding ───────────────────────────────────────────────
    for col_sk, col_enc in [('zone_sk', 'zone_encoded'),
                             ('line_sk', 'line_encoded'),
                             ('mode_sk', 'mode_encoded')]:
        if col_sk in df.columns:
            df[col_enc] = df[col_sk].astype('category').cat.codes

    # ── 4. Target aliases ─────────────────────────────────────────────────────
    df['retard_flag'] = df['annule'].astype(int)  # only cancellation signal available

    # ── 5. Class imbalance report ─────────────────────────────────────────────
    pos = int((df['retard_flag'] == 1).sum())
    neg = int((df['retard_flag'] == 0).sum())
    if pos > 0:
        spw = neg / pos
        logging.info(f"Class imbalance: pos={pos} neg={neg} → scale_pos_weight={spw:.2f}")
    else:
        logging.warning("ZERO positive (annule=1) rows — classification will fail.")

    # ── 6. Build df_ts for Prophet ────────────────────────────────────────────
    # Need a real date column: combine annee + mois, then use day from dim_temps
    # or synthesise uniform days-within-month from row order.
    if 'annee' in df.columns and 'mois' in df.columns:
        df['_base_date'] = pd.to_datetime(
            df['annee'].astype(str).str.zfill(4) + '-' +
            df['mois'].astype(str).str.zfill(2) + '-01',
            errors='coerce',
        )
        # Assign sequential day offsets within each (annee, mois, zone_sk) group
        # to spread hourly rows across the calendar month realistically.
        df['_day_offset'] = (
            df.sort_values(['annee', 'mois', 'zone_sk', 'hour'])
              .groupby(['annee', 'mois', 'zone_sk'])
              .cumcount()
        )
        # Cap at 27 so we stay within the month
        df['_day_offset'] = df['_day_offset'] % 28
        df['date'] = df['_base_date'] + pd.to_timedelta(df['_day_offset'], unit='D')
        df.drop(columns=['_base_date', '_day_offset'], inplace=True)
    else:
        logging.warning("'annee'/'mois' not found — date synthesis impossible. Prophet will skip.")
        df['date'] = pd.NaT

    # Aggregate to daily mean congestion per zone (Prophet input)
    df_ts = (
        df.dropna(subset=['date', 'congestion_index'])
          .groupby(['zone_sk', 'date'])['congestion_index']
          .mean()
          .reset_index()
          .rename(columns={'date': 'ds', 'congestion_index': 'y'})
    )
    df_ts['ds'] = pd.to_datetime(df_ts['ds'])

    logging.info(f"Feature engineering complete. df_ml: {df.shape}  df_ts: {df_ts.shape}")
    return df, df_ts


if __name__ == "__main__":
    from data_loader import load_mobility_data
    df_raw = load_mobility_data()
    df_ml, df_ts = engineer_features(df_raw)
    print(f"\ndf_ml shape  : {df_ml.shape}")
    print(f"df_ts shape  : {df_ts.shape}")
    print(f"\nNew columns: {[c for c in df_ml.columns if c not in df_raw.columns]}")
    print(f"\ndf_ts sample:\n{df_ts.head(8)}")