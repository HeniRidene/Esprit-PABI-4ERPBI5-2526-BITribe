import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_clean_data(fact_path="fact_impact_territorial.csv", time_dim_path="dim_temps.csv"):
    """
    Loads fact_impact_territorial.csv, merges time + zone dimensions,
    deduplicates on (time_sk, zone_sk, mode_sk), and enforces the canonical column set.

    Key design decisions
    --------------------
    - Drops zone_sk > 10 and zone_nom == 'UNKNOWN' — ETL artefact with near-zero
      CO2 and energy, which creates a spurious cluster and biases the regression.
    - Does NOT cap co2_kg:  the CO2 distribution is a legitimate multi-modal mix
      of four transport modes (mode_sk 2/3/5/6). Mode_sk=2 (motorised) genuinely
      reaches 600 kg/month. Capping destroys mode_sk=2 training signal.
    - Drops rows where co2_kg <= 0 — these are ETL artefacts (unrecorded trips),
      not real zero-emission events.
    """
    logging.info("=== DATA LOADER v3 ===")

    try:
        # 1. Load facts
        df_facts = pd.read_csv(fact_path)
        logging.info(f"Loaded facts  → shape: {df_facts.shape}")

        # 2. Load & merge time dimension
        df_temps = pd.read_csv(time_dim_path)
        logging.info(f"Loaded dim_temps → shape: {df_temps.shape}")

        if 'time_sk' not in df_facts.columns or 'time_sk' not in df_temps.columns:
            raise KeyError("'time_sk' missing from facts or dim_temps — cannot merge.")

        df = pd.merge(df_facts, df_temps, on='time_sk', how='left')
        logging.info(f"After facts ⊕ dim_temps merge → shape: {df.shape}")

        # 3. Drop columns we don't need
        cols_to_drop = ['weather_sk', 'event_sk', 'temperature', 'condition_text']
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # 4. Dedup on natural business keys
        dedup_keys = ['time_sk', 'zone_sk', 'mode_sk']
        before = len(df)
        df.drop_duplicates(subset=dedup_keys, inplace=True)
        logging.info(f"Dedup removed {before - len(df)} rows → {len(df)} rows remain")

        # 5. Merge zone dimension for names
        df_zone = pd.read_csv('dim_zone.csv')
        df = pd.merge(df, df_zone[['zone_sk', 'zone_nom', 'zone_code', 'ville']], on='zone_sk', how='left')

        # 6. Drop the phantom zone (zone_sk > 10 or zone_nom UNKNOWN / NaN)
        valid_zones = (
            df['zone_sk'].between(1, 10)
            & df['zone_nom'].notna()
            & (df['zone_nom'].str.strip().str.upper() != 'UNKNOWN')
        )
        removed = (~valid_zones).sum()
        df = df[valid_zones].copy()
        logging.info(f"Dropped {removed} phantom-zone rows → {len(df)} rows")

        # 7. Enforce keep-list
        cols_to_keep = [
            'fact_impact_sk', 'time_sk', 'zone_sk', 'mode_sk',
            'energie_kwh', 'co2_kg', 'aqi_index', 'pm25', 'no2',
            'nb_accidents', 'nb_graves', 'nb_mortels',
            'usagers_vulnerables', 'volume_crimes', 'taux_criminalite',
            'annee', 'mois', 'jour_semaine', 'periode',
            'zone_nom', 'zone_code', 'ville', 'mode',
        ]
        actual_keep = [c for c in cols_to_keep if c in df.columns]
        df = df[actual_keep].copy()

        # 8. Coerce numeric targets
        for col in ['co2_kg', 'energie_kwh', 'aqi_index', 'pm25', 'no2']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 9. Drop rows with non-positive CO2 (true ETL artefacts, not zero-emission trips)
        before_co2 = len(df)
        df = df[df['co2_kg'] > 0].copy()
        logging.info(f"Dropped {before_co2 - len(df)} rows with co2_kg <= 0 → {len(df)} rows")

        # 10. Verify temporal columns
        if 'annee' not in df.columns or 'mois' not in df.columns:
            raise ValueError("'annee' / 'mois' missing after merge — check dim_temps columns.")

        # 11. Shape & null report
        logging.info(f"Final clean shape: {df.shape}")
        null_report = df.isnull().sum()
        null_report = null_report[null_report > 0]
        if len(null_report):
            logging.info(f"Null counts:\n{null_report.to_string()}")
        else:
            logging.info("No nulls in the clean DataFrame.")

        zero_counts = {c: int((df[c] == 0).sum())
                       for c in ['co2_kg', 'energie_kwh', 'aqi_index']
                       if c in df.columns}
        logging.info(f"Zero counts in targets: {zero_counts}")
        logging.info(f"CO2 range: [{df['co2_kg'].min():.2f}, {df['co2_kg'].max():.2f}]  "
                     f"median={df['co2_kg'].median():.2f}")

        return df

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"data_loader failed: {e}")
        raise


if __name__ == "__main__":
    df_clean = load_and_clean_data()
    print(f"\nShape: {df_clean.shape}")
    print(f"Zones: {sorted(df_clean['zone_sk'].unique())}")
    print(f"Modes: {sorted(df_clean['mode_sk'].unique())}")
    print(f"co2_kg: min={df_clean['co2_kg'].min():.2f}  max={df_clean['co2_kg'].max():.2f}  "
          f"median={df_clean['co2_kg'].median():.2f}  mean={df_clean['co2_kg'].mean():.2f}")