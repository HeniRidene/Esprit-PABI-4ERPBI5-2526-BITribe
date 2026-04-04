import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mobility_data(
    fact_path=None,
    time_dim_path=None,
):
    """
    Loads fact_service_mobilite.csv, joins dim_temps to recover temporal columns
    (annee, mois, heure, jour_semaine, weekend, periode), drops irrelevant columns,
    and imputes ETL artefact values.

    Key decisions
    -------------
    - NO dedup needed (fan-out issue was specific to fact_impact_territorial).
    - Drops weather_sk (95% NULL), event_sk (99% NULL), bak_* columns.
    - vitesse_kmh == 0 treated as ETL artefact → replaced with NaN for later imputation.
      (bak_vitesse_kmh confirms original values were 0 → NULL in trafic join)
    - stress_1_5 nulls imputed with zone median in feature_engineering.
    - Prints full null / zero report so downstream scripts start from a verified state.
    """
    logging.info("=== DATA LOADER v1 — Actor 2 Mobilités ===")

    if fact_path is None:
        fact_path = os.path.join(BASE_DIR, 'fact_service_mobilite.csv')
    if time_dim_path is None:
        time_dim_path = os.path.join(BASE_DIR, 'dim_temps.csv')

    try:
        # 1. Load facts
        df_facts = pd.read_csv(fact_path)
        logging.info(f"Loaded facts → shape: {df_facts.shape}")

        # 2. Load & merge time dimension (restores annee, mois, heure, etc.)
        df_temps = pd.read_csv(time_dim_path)
        logging.info(f"Loaded dim_temps → shape: {df_temps.shape}")

        if 'time_sk' not in df_facts.columns or 'time_sk' not in df_temps.columns:
            raise KeyError("'time_sk' missing from facts or dim_temps.")

        df = pd.merge(df_facts, df_temps, on='time_sk', how='left')
        logging.info(f"After fact ⊕ dim_temps merge → shape: {df.shape}")

        # 3. Drop columns we don't need
        cols_to_drop = [
            'weather_sk', 'event_sk',
            'bak_vitesse_kmh', 'bak_temps_trajet_min',
            'bak_stress_1_5', 'bak_weather_sk',
        ]
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # 4. Canonical keep list after merge
        cols_to_keep = [
            'fact_mob_sk', 'time_sk', 'zone_sk', 'line_sk', 'mode_sk', 'stop_sk',
            'annule', 'charge_estimee', 'vitesse_kmh', 'temps_trajet_min',
            'congestion_index', 'stress_1_5', 'satisfaction_1_5',
            'annee', 'mois', 'heure', 'jour_semaine', 'periode', 'weekend',
        ]
        actual_keep = [c for c in cols_to_keep if c in df.columns]
        df = df[actual_keep].copy()

        # 5. Coerce key numeric columns
        for col in ['vitesse_kmh', 'temps_trajet_min', 'charge_estimee',
                    'congestion_index', 'stress_1_5']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Mark vitesse_kmh == 0 as NaN (ETL artefact from missing trafic join)
        if 'vitesse_kmh' in df.columns:
            n_zero = (df['vitesse_kmh'] == 0).sum()
            if n_zero > 0:
                df.loc[df['vitesse_kmh'] == 0, 'vitesse_kmh'] = np.nan
                logging.info(f"Replaced {n_zero} zero vitesse_kmh → NaN (ETL artefact)")

        # 7. annule integrity check
        if 'annule' in df.columns:
            pos = int((df['annule'] == 1).sum())
            neg = int((df['annule'] == 0).sum())
            total = len(df)
            logging.info(f"annule: pos={pos} ({pos/total*100:.2f}%)  neg={neg}  total={total}")

        # 8. Null & zero report
        null_report = df.isnull().sum()
        null_report = null_report[null_report > 0]
        if len(null_report):
            logging.info(f"Null counts after load:\n{null_report.to_string()}")
        else:
            logging.info("No nulls in loaded DataFrame.")

        logging.info(f"Final clean shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"data_loader failed: {e}")
        raise


if __name__ == "__main__":
    df = load_mobility_data()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nannule distribution:\n{df['annule'].value_counts().to_string()}")
    print(f"\nvitesse_kmh nulls: {df['vitesse_kmh'].isnull().sum()}")