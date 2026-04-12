import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_security_data(
    fact_path=None,
    time_dim_path=None,
    zone_dim_path=None,
):
    """
    Loads fact_impact_territorial.csv and joins dim_temps + dim_zone.
    Applies identical dedup/zone-filter strategy as Actor 1 (same source file,
    independent purpose per architecture decision).

    Dedup key : (time_sk, zone_sk, mode_sk)  — removes weather fan-out duplicates.
    Zone filter: drops zone_sk > 10 (ETL artefact phantom zone).
    Column keep: security + crime columns only; drops environmental columns.

    Returns a clean DataFrame with 1,911 rows and the Actor 3 security columns.
    """
    logging.info("=== DATA LOADER v1 — Actor 3 Sécurité ===")

    if fact_path    is None: fact_path     = os.path.join(BASE_DIR, 'fact_impact_territorial.csv')
    if time_dim_path is None: time_dim_path = os.path.join(BASE_DIR, 'dim_temps.csv')
    if zone_dim_path is None: zone_dim_path = os.path.join(BASE_DIR, 'dim_zone.csv')

    # ── 1. Load facts ─────────────────────────────────────────────────────────
    df = pd.read_csv(fact_path)
    logging.info(f"Loaded facts → shape: {df.shape}")

    # ── 2. Dedup (weather fan-out fix) ────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=['time_sk', 'zone_sk', 'mode_sk'])
    logging.info(f"Dedup removed {n_before - len(df)} rows → {len(df)} remain")

    # ── 3. Drop phantom zone 11 ───────────────────────────────────────────────
    n_before = len(df)
    df = df[df['zone_sk'] <= 10]
    logging.info(f"Dropped {n_before - len(df)} phantom-zone rows → {len(df)} remain")

    # ── 4. Drop ETL artefact columns ─────────────────────────────────────────
    bak_cols  = [c for c in df.columns if c.startswith('bak_')]
    drop_cols = bak_cols + ['weather_sk', 'event_sk',
                             'energie_kwh', 'co2_kg', 'pm25', 'no2', 'aqi_index']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # ── 5. Join time dimension ────────────────────────────────────────────────
    df_temps = pd.read_csv(time_dim_path)
    time_cols = ['time_sk', 'annee', 'mois', 'jour_semaine', 'periode']
    time_cols = [c for c in time_cols if c in df_temps.columns]
    df = pd.merge(df, df_temps[time_cols], on='time_sk', how='left')
    logging.info(f"After ⊕ dim_temps → shape: {df.shape}")

    # ── 6. Join zone dimension (zone_nom, zone_code, ville) ───────────────────
    if os.path.exists(zone_dim_path):
        df_zone = pd.read_csv(zone_dim_path)
        zone_join_cols = [c for c in df_zone.columns if c in ('zone_sk', 'zone_nom', 'zone_code', 'ville')]
        if 'zone_sk' in df_zone.columns:
            df = pd.merge(df, df_zone[zone_join_cols], on='zone_sk', how='left')
            logging.info(f"After ⊕ dim_zone → shape: {df.shape}")

    # ── 7. Keep canonical columns ─────────────────────────────────────────────
    keep = [
        'fact_impact_sk', 'time_sk', 'zone_sk', 'mode_sk',
        'nb_accidents', 'nb_graves', 'nb_mortels',
        'usagers_vulnerables', 'volume_crimes', 'taux_criminalite',
        'annee', 'mois', 'jour_semaine', 'periode',
        'zone_nom', 'zone_code', 'ville',
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # ── 8. Coerce numeric columns ─────────────────────────────────────────────
    num_cols = ['nb_accidents', 'nb_graves', 'nb_mortels', 'usagers_vulnerables',
                'volume_crimes', 'taux_criminalite']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # ── 9. Integrity report ───────────────────────────────────────────────────
    pos_acc  = int((df['nb_accidents'] > 0).sum())
    pos_grav = int((df['nb_graves']    > 0).sum())
    pos_mort = int((df['nb_mortels']   > 0).sum())
    logging.info(f"Rows with accidents>0: {pos_acc} ({pos_acc/len(df)*100:.1f}%)")
    logging.info(f"Rows with graves>0:    {pos_grav}")
    logging.info(f"Rows with mortels>0:   {pos_mort}")
    logging.info(f"Final clean shape: {df.shape}")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        logging.warning(f"Remaining nulls:\n{nulls.to_string()}")
    else:
        logging.info("No nulls in clean DataFrame.")

    return df


if __name__ == "__main__":
    df = load_security_data()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nnb_mortels unique: {sorted(df['nb_mortels'].unique().tolist())}")
