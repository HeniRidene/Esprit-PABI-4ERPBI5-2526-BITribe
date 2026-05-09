import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Severity merge threshold: if class 2 (mortel) has fewer than this many samples,
# merge class 1 + 2 into a single "serious" class → binary problem.
MIN_MORTEL_SAMPLES = 40


def engineer_features(df_raw):
    """
    Transforms clean security data into ML-ready feature sets.

    Transformations
    ---------------
    1. Severity label  — derives severity_class and severity_label from nb_graves/nb_mortels.
                         Auto-merges class 2 into class 1 if nb_mortels < MIN_MORTEL_SAMPLES in data.
    2. gravite_index   — (nb_graves + nb_mortels) / (nb_accidents + 1); Power BI KPI column.
    3. Encoding        — zone_sk and mode_sk as category codes.
    4. has_accident    — binary flag (nb_accidents > 0).
    5. crime_rate_scaled — z-score normalised taux_criminalite (for clustering).
    6. Outputs         — df_ml (classification + anomaly), df_zone (clustering).

    Returns
    -------
    df_ml   : row-level feature-engineered DataFrame
    df_zone : zone-level aggregated DataFrame for clustering
    """
    logging.info("=== FEATURE ENGINEERING v1 — Actor 3 Sécurité ===")
    df = df_raw.copy()

    # ── 1. Severity class derivation ──────────────────────────────────────────
    n_mortels = int((df['nb_mortels'] > 0).sum())
    logging.info(f"nb_mortels > 0 samples: {n_mortels}")

    if n_mortels < MIN_MORTEL_SAMPLES:
        logging.warning(
            f"nb_mortels > 0 rows ({n_mortels}) < threshold ({MIN_MORTEL_SAMPLES}). "
            f"Merging class 2 (mortel) into class 1 (grave/serious). "
            f"MODEL IS BINARY: 0=none, 1=serious (grave OR mortel). "
            f"This avoids training on near-empty classes — documented for the professor."
        )
        df['severity_class'] = np.where(
            (df['nb_graves'] > 0) | (df['nb_mortels'] > 0), 1, 0
        ).astype(int)
        df['severity_label'] = df['severity_class'].map({0: 'Aucun incident', 1: 'Accident grave'})
        n_classes = 2
    else:
        df['severity_class'] = 0
        df.loc[df['nb_graves']  > 0, 'severity_class'] = 1
        df.loc[df['nb_mortels'] > 0, 'severity_class'] = 2
        df['severity_label'] = df['severity_class'].map(
            {0: 'Aucun incident', 1: 'Accident grave', 2: 'Accident mortel'}
        )
        n_classes = 3

    dist = dict(df['severity_class'].value_counts().sort_index())
    logging.info(f"severity_class distribution ({n_classes} classes): {dist}")

    # ── 2. gravite_index (Power BI KPI) ──────────────────────────────────────
    df['gravite_index'] = (
        (df['nb_graves'] + df['nb_mortels']) / (df['nb_accidents'] + 1)
    ).round(4)

    # ── 3. Categorical encoding ───────────────────────────────────────────────
    df['zone_encoded'] = df['zone_sk'].astype('category').cat.codes
    df['mode_encoded'] = df['mode_sk'].astype('category').cat.codes

    # ── 4. Binary accident flag ───────────────────────────────────────────────
    df['has_accident'] = (df['nb_accidents'] > 0).astype(int)

    # ── 5. Crime rate z-score (for clustering homogenisation) ─────────────────
    mu = df['taux_criminalite'].mean()
    sd = df['taux_criminalite'].std()
    df['crime_rate_scaled'] = ((df['taux_criminalite'] - mu) / sd).fillna(0).round(4)

    # ── 6. Zone-level aggregation for clustering ──────────────────────────────
    zone_meta_cols = [c for c in ['zone_nom', 'zone_code', 'ville'] if c in df.columns]
    agg_dict = {
        'nb_accidents':       'mean',
        'nb_graves':          'mean',
        'nb_mortels':         'mean',
        'volume_crimes':      'mean',
        'taux_criminalite':   'mean',
        'usagers_vulnerables': 'mean',
        'gravite_index':      'mean',
    }
    if zone_meta_cols:
        for c in zone_meta_cols:
            agg_dict[c] = 'first'

    df_zone = df.groupby('zone_sk').agg(agg_dict).reset_index()
    df_zone.columns = ['zone_sk'] + [
        f'mean_{c}' if c not in zone_meta_cols + ['zone_sk'] else c
        for c in df_zone.columns[1:]
    ]
    # Rename mean_ cols cleanly
    df_zone.columns = [
        c.replace('mean_', '') if c.startswith('mean_') else c
        for c in df_zone.columns
    ]

    logging.info(f"Feature engineering complete. df_ml: {df.shape}  df_zone: {df_zone.shape}")
    logging.info(f"New columns: {[c for c in df.columns if c not in df_raw.columns]}")

    return df, df_zone


if __name__ == "__main__":
    from data_loader import load_security_data
    df_raw = load_security_data()
    df_ml, df_zone = engineer_features(df_raw)
    print(f"\ndf_ml: {df_ml.shape}")
    print(f"df_zone:\n{df_zone.to_string()}")
    print(f"\nseverity_class:\n{df_ml['severity_class'].value_counts().to_string()}")
