import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

ZONE_NAMES = {1:"Paris",2:"Marseille",3:"Lyon",4:"Toulouse",5:"Nice",
              6:"Nantes",7:"Montpellier",8:"Strasbourg",9:"Bordeaux",10:"Lille"}

st.set_page_config(
    page_title="Actor 1 — Écologique | Transport ML",
    page_icon="🌿",
    layout="wide",
)

# ── Actor access guard ────────────────────────────────────────────────────────
ALLOWED_ACTOR = "actor1"
actor_filter = st.query_params.get("actor", None)
if actor_filter and actor_filter != ALLOWED_ACTOR:
    st.error("⛔ Access restricted. This page is only available to the Ecological Transition role.")
    st.stop()

# ── Role banner (Task 4) ──────────────────────────────────────────────────────
if actor_filter:
    st.success("🌿 Ecological Transition view — Actor 1 only")

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "actor1")

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "co2": "xgboost_co2.pkl",
        "features": "xgboost_features.pkl",
        "nrj": "xgboost_nrj.pkl",
        "kmeans": "kmeans_pollution_zones.pkl",
        "scaler": "clustering_scaler.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(BASE, fname)
        try:
            models[key] = joblib.load(path)
        except Exception as e:
            st.error(f"❌ Could not load **{fname}**: {e}")
            models[key] = None
    return models

models = load_models()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🌿 Directeur Écologique — Environmental Impact Analysis")
st.markdown(
    """
    This module provides **real-time CO₂ emission and energy consumption predictions**
    for urban transport zones, powered by XGBoost models. It also segments transport
    zones into pollution clusters using K-Means to support targeted environmental action.
    """
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — CO2 & ENERGY PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔬 Section A — CO₂ & Energy Prediction")
st.markdown("Configure zone parameters in the sidebar to get instant emission forecasts.")

ZONE_NAMES = {1:"Paris",2:"Marseille",3:"Lyon",4:"Toulouse",5:"Nice",
              6:"Nantes",7:"Montpellier",8:"Strasbourg",9:"Bordeaux",10:"Lille"}

with st.sidebar:
    st.markdown("### 🌿 Écologique — Inputs")
    st.markdown("**Section A: CO₂ & Energy**")
    zone_name  = st.selectbox("Select Zone", list(ZONE_NAMES.values()), key="a1_zone")
    zone_sk    = [k for k, v in ZONE_NAMES.items() if v == zone_name][0]
    mode_sk    = st.selectbox("Transport Mode SK", list(range(1, 6)), key="a1_mode")
    annee      = st.selectbox("Year", [2023, 2024], key="a1_annee")
    mois       = st.selectbox("Month", list(range(1, 13)), key="a1_mois")
    aqi_index  = st.slider("AQI Index", 0, 300, 80, key="a1_aqi")
    pm25       = st.slider("PM2.5 (µg/m³)", 0, 200, 35, key="a1_pm25")

def build_co2_input(features):
    """Build a DataFrame aligned to the model's feature list."""
    row = {
        "zone_sk": zone_sk,
        "mode_sk": mode_sk,
        "annee": annee,
        "mois": mois,
        "aqi_index": aqi_index,
        "pm25": pm25,
    }
    cols = features if features is not None else list(row.keys())
    df = pd.DataFrame([{c: row.get(c, 0) for c in cols}])
    return df

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🏭 CO₂ Emission Forecast")
    if models["co2"] and models["features"] is not None:
        try:
            features = models["features"]
            X = build_co2_input(features)
            co2_pred = float(models["co2"].predict(X)[0])
            st.metric(f"Predicted CO₂ (kg) — {ZONE_NAMES[zone_sk]}", f"{co2_pred:,.2f} kg")
            if co2_pred > 500:
                st.warning("⚠️ **High emission zone — action required**")
            else:
                st.success("✅ Emission level within acceptable range")
            
            try:
                lasso_co2 = joblib.load(os.path.join(BASE, "lasso_co2.pkl"))
                lasso_pred = float(lasso_co2.predict(X)[0])
                comp_df = pd.DataFrame({
                    "Model": ["XGBoost", "Lasso"],
                    "Predicted CO₂": [f"{co2_pred:,.2f} kg", f"{lasso_pred:,.2f} kg"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            except Exception:
                st.caption("Comparison model not available")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Model not loaded — check file paths.")

with col_b:
    st.markdown("#### ⚡ Energy Consumption Forecast")
    if models["nrj"] and models["features"] is not None:
        try:
            features = models["features"]
            X = build_co2_input(features)
            nrj_pred = float(models["nrj"].predict(X)[0])
            st.metric(f"Predicted Energy (kWh) — {ZONE_NAMES[zone_sk]}", f"{nrj_pred:,.2f} kWh")
            if nrj_pred > 1000:
                st.warning("⚠️ High energy consumption detected")
            else:
                st.success("✅ Energy usage within normal range")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Model not loaded — check file paths.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — ZONE POLLUTION CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🗺️ Section B — Zone Pollution Clustering")
st.markdown(
    "Classify a transport zone into a **pollution cluster** based on measured pollutant levels."
)

with st.sidebar:
    st.divider()
    st.markdown("**Section B: Clustering**")
    pm25_clust  = st.slider("PM2.5 (cluster input)", 0, 200, 40, key="b_pm25")
    no2_clust   = st.slider("NO₂ (µg/m³)", 0, 200, 50, key="b_no2")
    co2_clust   = st.slider("CO₂ (kg, cluster input)", 0, 1000, 200, key="b_co2")
    nrj_clust   = st.slider("Energy kWh (cluster input)", 0, 3000, 600, key="b_nrj")

CLUSTER_LABELS = {
    0: ("🟢 Low Pollution Zone", "success"),
    1: ("🟡 Moderate Pollution Zone", "warning"),
    2: ("🔴 High Pollution Zone", "error"),
}

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 🔍 Cluster Prediction")
    if models["kmeans"] and models["scaler"]:
        try:
            X_clust = np.array([[pm25_clust, no2_clust, co2_clust, nrj_clust]])
            # The scaler may have been trained on a subset of features; try with 4
            try:
                X_scaled = models["scaler"].transform(X_clust)
            except Exception:
                # fallback: use only the features the scaler knows
                n = models["scaler"].n_features_in_
                X_scaled = models["scaler"].transform(X_clust[:, :n])

            cluster_id = int(models["kmeans"].predict(X_scaled)[0])
            label, level = CLUSTER_LABELS.get(cluster_id, (f"Cluster {cluster_id}", "info"))

            if level == "success":
                st.success(f"**Assigned Cluster {cluster_id}** — {label}")
            elif level == "warning":
                st.warning(f"**Assigned Cluster {cluster_id}** — {label}")
            else:
                st.error(f"**Assigned Cluster {cluster_id}** — {label}")

            st.metric("Cluster ID", cluster_id)
        except Exception as e:
            st.error(f"Clustering error: {e}")
    else:
        st.info("Clustering model not loaded.")

with col_d:
    st.markdown("#### 📊 Cluster Centroid Profiles")
    if models["kmeans"]:
        try:
            centers = models["kmeans"].cluster_centers_
            n_clusters = centers.shape[0]
            feat_names = ["PM2.5", "NO₂", "CO₂", "Energy kWh"][:centers.shape[1]]
            df_centers = pd.DataFrame(
                centers,
                columns=feat_names,
                index=[f"Cluster {i}" for i in range(n_clusters)],
            )
            st.bar_chart(df_centers.T)
        except Exception as e:
            st.error(f"Chart error: {e}")
    else:
        st.info("Model not loaded.")

    elbow_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "elbow_scores.csv")
    if os.path.exists(elbow_path):
        st.markdown("#### 📉 Elbow Chart Scores")
        st.dataframe(pd.read_csv(elbow_path), use_container_width=True, hide_index=True)
    st.caption("K-Means groups data into a predefined number of spherical clusters, whereas DBSCAN discovers clusters of arbitrary shapes based on density, effectively identifying noise.")


st.caption("🌿 Actor 1 · Directeur Écologique · Transport ML Dashboard")
