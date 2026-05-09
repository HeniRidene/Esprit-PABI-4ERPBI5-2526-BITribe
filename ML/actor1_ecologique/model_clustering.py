import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_pollution_clustering(df_cluster):
    """
    Groups the 10 zones into pollution+energy profiles using K-Means.

    Improvements over v1:
    - Aggregates to zone-level means BEFORE scaling (correct order).
    - Tries K=2,3,4; picks by silhouette score.
    - Saves cluster_labels.csv (expected by export_predictions.py).
    - Saves scaler + kmeans model to outputs/.
    - Dynamic cluster naming from centroid values.
    """
    logging.info("Initializing K-Means Clustering Pipeline v2...")
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Aggregate raw values to zone-level means ──────────────────────────
    raw_features = ['pm25', 'no2', 'co2_kg', 'energie_kwh']
    available    = [f for f in raw_features if f in df_cluster.columns]

    if len(available) < 2:
        logging.error(f"Not enough clustering features found: {available}")
        return None

    zone_profiles = (
        df_cluster
        .replace({'co2_kg': {0: np.nan}, 'energie_kwh': {0: np.nan}})
        .dropna(subset=available)
        .groupby('zone_sk')[available]
        .mean()
        .reset_index()
    )

    # Attach zone names if available
    if 'zone_nom' in df_cluster.columns:
        zone_names = df_cluster[['zone_sk', 'zone_nom']].drop_duplicates()
        zone_profiles = pd.merge(zone_profiles, zone_names, on='zone_sk', how='left')
    else:
        zone_profiles['zone_nom'] = 'Zone_' + zone_profiles['zone_sk'].astype(str)

    n_zones = len(zone_profiles)
    logging.info(f"Clustering on {n_zones} unique zones.")

    if n_zones < 3:
        logging.error(f"Only {n_zones} zones — cannot cluster meaningfully.")
        return None

    # ── 2. Scale ─────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(zone_profiles[available])
    joblib.dump(scaler, 'outputs/clustering_scaler.pkl')

    zone_profiles[[f + '_scaled' for f in available]] = X_scaled

    # ── 3. Elbow + Silhouette for K=2,3,4 ───────────────────────────────────
    max_k     = min(4, n_zones - 1)   # can't have more clusters than zones
    k_range   = range(2, max_k + 1)
    inertias  = []
    sil_scores = []

    best_k   = 2
    best_sil = -1

    logging.info(f"Testing K in {list(k_range)}...")
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil    = silhouette_score(X_scaled, labels) if n_zones > k else -1
        inertias.append(km.inertia_)
        sil_scores.append(sil)
        logging.info(f"  K={k}  Silhouette={sil:.4f}  Inertia={km.inertia_:.2f}")

        if sil > best_sil:
            best_sil = sil
            best_k   = k

    logging.info(f"Chosen K={best_k}  (Silhouette={best_sil:.4f})")

    # Elbow plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel('K'); plt.ylabel('Inertia');  plt.title('Elbow Method')
    plt.tight_layout()
    plt.savefig('outputs/clustering_elbow.png', dpi=120)
    plt.close()

    # ── 4. Final model ───────────────────────────────────────────────────────
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    zone_profiles['cluster_id'] = final_km.fit_predict(X_scaled)
    joblib.dump(final_km, 'outputs/kmeans_pollution_zones.pkl')

    # ── 5. Dynamic cluster labelling ─────────────────────────────────────────
    centers = final_km.cluster_centers_
    # Index correspondence to `available` list
    idx = {f: i for i, f in enumerate(available)}

    cluster_labels = {}
    for i in range(best_k):
        c_pm25 = centers[i][idx['pm25']]       if 'pm25'        in idx else 0
        c_co2  = centers[i][idx['co2_kg']]     if 'co2_kg'      in idx else 0
        c_nrj  = centers[i][idx['energie_kwh']] if 'energie_kwh' in idx else 0

        if c_co2 > 0.4 and c_pm25 > 0.4:
            label = 'High Emission & High Pollution'
        elif c_co2 > 0.4 and c_pm25 <= 0.4:
            label = 'High Emission, Clean Air'
        elif c_co2 <= 0.4 and c_pm25 > 0.4:
            label = 'High Pollution, Low Emission'
        else:
            label = 'Clean & Green Zone'

        cluster_labels[i] = label
        logging.info(f"  Cluster {i}: {label}  (CO2={c_co2:.2f}, PM2.5={c_pm25:.2f}, Energy={c_nrj:.2f})")

    zone_profiles['cluster_label'] = zone_profiles['cluster_id'].map(cluster_labels)

    # ── 6. Scatter plot ───────────────────────────────────────────────────────
    co2_scaled_col = 'co2_kg_scaled' if 'co2_kg' in available else available[0] + '_scaled'
    pm25_scaled_col = 'pm25_scaled'  if 'pm25'   in available else available[1] + '_scaled'

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        zone_profiles[co2_scaled_col],
        zone_profiles[pm25_scaled_col],
        c=zone_profiles['cluster_id'],
        cmap='Set1', s=200, alpha=0.85, edgecolors='k', linewidths=0.5,
    )
    plt.colorbar(scatter, label='Cluster ID')
    for _, row in zone_profiles.iterrows():
        plt.annotate(
            row['zone_nom'],
            (row[co2_scaled_col], row[pm25_scaled_col]),
            xytext=(6, 6), textcoords='offset points', fontsize=9,
        )
    plt.xlabel('CO₂ Emissions (Scaled)')
    plt.ylabel('PM2.5 Pollution (Scaled)')
    plt.title(f'Zone Segmentation — K={best_k}  |  Silhouette={best_sil:.3f}')
    plt.tight_layout()
    plt.savefig('outputs/zone_clusters.png', dpi=150)
    plt.close()
    logging.info("Cluster visualization saved → outputs/zone_clusters.png")

    # ── 7. Save cluster_labels.csv ───────────────────────────────────────────
    out_cols = ['zone_sk', 'zone_nom', 'cluster_id', 'cluster_label'] + \
               [f + '_scaled' for f in available]
    export_df = zone_profiles[[c for c in out_cols if c in zone_profiles.columns]].copy()
    export_df.to_csv('outputs/cluster_labels.csv', index=False)
    logging.info(f"Cluster labels saved → outputs/cluster_labels.csv  ({len(export_df)} zones)")

    return export_df


if __name__ == "__main__":
    from data_loader import load_and_clean_data
    from feature_engineering import engineer_features

    try:
        df_clean = load_and_clean_data()
        _, df_cluster, _ = engineer_features(df_clean)

        final_profiles = train_pollution_clustering(df_cluster)
        print("\n" + "="*55)
        print(" FINAL ZONE PROFILES")
        print("="*55)
        if final_profiles is not None:
            print(final_profiles[['zone_sk', 'zone_nom', 'cluster_id', 'cluster_label']])

    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise