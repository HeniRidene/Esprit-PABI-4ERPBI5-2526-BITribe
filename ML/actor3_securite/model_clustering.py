import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLUSTER_FEATURES = [
    'nb_accidents', 'nb_graves',
    'volume_crimes', 'taux_criminalite',
    'usagers_vulnerables',
]
K_FIXED = 3   # Low / Medium / High risk — exactly 3–4 zones per cluster


def _label_clusters(df_cluster, feature_cols):
    """
    Names clusters based on centroid rank: lowest composite score → Low risk.
    Computes a simple composite = mean of z-scored features per cluster.
    """
    centroids = df_cluster.groupby('risk_cluster_id')[feature_cols].mean()
    # Composite score = mean of all normalised features
    centroids['composite'] = centroids.mean(axis=1)
    rank = centroids['composite'].rank(method='first').astype(int)
    label_map = {
        k: {1: 'Low risk', 2: 'Medium risk', 3: 'High risk'}.get(v, f'Cluster {v}')
        for k, v in rank.items()
    }
    logging.info(f"Cluster labels: {label_map}")
    return label_map


def train_zone_clustering(df_zone, k=K_FIXED):
    """
    Groups the 10 zones into risk profiles using K-Means.

    Strategy
    --------
    - Operates on zone-level aggregated means (not individual rows) — 10 data points.
    - Features: nb_accidents, nb_graves, volume_crimes, taux_criminalite, usagers_vulnerables.
    - StandardScaler for homogeneous feature ranges.
    - k=3 fixed (Low/Medium/High risk) for interpretability.
    - Elbow curve saved to validate k=3 choice for the professor.
    - risk_level column powers the colour-coded zone map in Power BI.
    """
    logging.info("=== CLUSTERING — Zone Risk Profiles (K-Means k=3) ===")
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Prepare zone-level features ───────────────────────────────────────
    feat_cols = [f for f in CLUSTER_FEATURES if f in df_zone.columns]
    missing   = set(CLUSTER_FEATURES) - set(feat_cols)
    if missing:
        logging.warning(f"Missing cluster features (skipped): {missing}")

    df_c = df_zone[['zone_sk'] + feat_cols +
                   [c for c in ('zone_nom', 'zone_code', 'ville') if c in df_zone.columns]].copy()

    X_raw = df_c[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    logging.info(f"Clustering {len(df_c)} zones on {len(feat_cols)} features: {feat_cols}")

    # ── 2. Elbow curve (k=2..6) ──────────────────────────────────────────────
    inertias = {}
    sil_scores = {}
    for ki in range(2, min(7, len(df_c))):
        km = KMeans(n_clusters=ki, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias[ki]   = float(km.inertia_)
        sil_scores[ki] = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0

    # Save elbow plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(inertias.keys()), list(inertias.values()), 'b-o', marker='o')
    ax1.axvline(k, color='red', ls='--', label=f'k={k} chosen')
    ax1.set_title('Elbow Curve'); ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(list(sil_scores.keys()), list(sil_scores.values()), 'g-o')
    ax2.axvline(k, color='red', ls='--', label=f'k={k} chosen')
    ax2.set_title('Silhouette Score'); ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/clustering_elbow.png', dpi=150)
    plt.close()

    # ── 3. Final K-Means k=3 ─────────────────────────────────────────────────
    km_final = KMeans(n_clusters=k, random_state=42, n_init=20)
    df_c = df_c.copy()
    df_c['risk_cluster_id'] = km_final.fit_predict(X_scaled)

    sil = float(silhouette_score(X_scaled, df_c['risk_cluster_id'])) if len(np.unique(df_c['risk_cluster_id'])) > 1 else 0.0
    logging.info(f"k={k}  Silhouette={sil:.4f}  Inertia={km_final.inertia_:.4f}")

    # ── 4. Label clusters ─────────────────────────────────────────────────────
    label_map = _label_clusters(df_c, feat_cols)
    df_c['risk_level'] = df_c['risk_cluster_id'].map(label_map)

    logging.info(f"Cluster composition:\n{df_c[['zone_sk','risk_cluster_id','risk_level']].to_string(index=False)}")

    # ── 5. Cluster profile heatmap ────────────────────────────────────────────
    profile = df_c.groupby('risk_level')[feat_cols].mean()
    fig2, ax3 = plt.subplots(figsize=(10, 4))
    sns.heatmap(profile.T, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax3,
                linewidths=0.5, linecolor='white')
    ax3.set_title('Zone Cluster Risk Profiles')
    plt.tight_layout()
    plt.savefig('outputs/clustering_profiles.png', dpi=150)
    plt.close()

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    joblib.dump(km_final, 'outputs/kmeans_risk.pkl')
    joblib.dump(scaler,   'outputs/kmeans_scaler.pkl')
    joblib.dump(feat_cols, 'outputs/kmeans_features.pkl')

    out_path = 'outputs/risk_zone_clusters.csv'
    out_cols = ['zone_sk', 'risk_cluster_id', 'risk_level'] + \
               [c for c in ('zone_nom', 'zone_code', 'ville') if c in df_c.columns] + \
               feat_cols
    df_c[[c for c in out_cols if c in df_c.columns]].to_csv(out_path, index=False)
    logging.info(f"Saved → {out_path}")

    cluster_metrics = {
        'k':              k,
        'silhouette':     round(sil, 4),
        'inertia':        round(float(km_final.inertia_), 4),
        'elbow_inertias': {str(ki): round(v, 4) for ki, v in inertias.items()},
        'cluster_labels': {str(k): v for k, v in label_map.items()},
        'zone_assignments': df_c[['zone_sk', 'risk_cluster_id', 'risk_level']].to_dict(orient='records'),
    }

    return df_c, cluster_metrics


if __name__ == "__main__":
    from data_loader import load_security_data
    from feature_engineering import engineer_features
    df_raw = load_security_data()
    _, df_zone = engineer_features(df_raw)
    df_cluster, metrics = train_zone_clustering(df_zone)
    print(f"\nCluster results:\n{df_cluster[['zone_sk','risk_level']].to_string(index=False)}")
    print(f"\nSilhouette: {metrics['silhouette']}")
