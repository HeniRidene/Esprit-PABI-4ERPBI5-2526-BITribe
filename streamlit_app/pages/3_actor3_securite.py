import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

ZONE_NAMES = {1:"Paris",2:"Marseille",3:"Lyon",4:"Toulouse",5:"Nice",
              6:"Nantes",7:"Montpellier",8:"Strasbourg",9:"Bordeaux",10:"Lille"}

st.set_page_config(
    page_title="Actor 3 — Sécurité | Transport ML",
    page_icon="🛡️",
    layout="wide",
)

# ── Actor access guard ────────────────────────────────────────────────────────
ALLOWED_ACTOR = "actor3"
actor_filter = st.query_params.get("actor", None)
if actor_filter and actor_filter != ALLOWED_ACTOR:
    st.error("⛔ Access restricted. This page is only available to the Security role.")
    st.stop()

# ── Role banner (Task 4) ──────────────────────────────────────────────────────
if actor_filter:
    st.warning("🛡️ Security Management view — Actor 3 only")

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "actor3")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "severity":         "rf_severity.pkl",
        "severity_features":"rf_severity_features.pkl",
        "kmeans_risk":      "kmeans_risk.pkl",
        "kmeans_scaler":    "kmeans_scaler.pkl",
        "kmeans_features":  "kmeans_features.pkl",
        "iso_forest":       "isolation_forest.pkl",
        "anomaly_scaler":   "anomaly_scaler.pkl",
        "anomaly_features": "anomaly_features.pkl",
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
st.title("🛡️ Responsable Sécurité — Security Risk Intelligence")
st.markdown(
    """
    This module provides a **three-layer security analysis** for urban transport zones:
    accident severity classification, zone risk clustering, and real-time anomaly detection.
    All models are pre-trained on historical transport safety data.
    """
)
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Sécurité — Inputs")

    st.markdown("**Section A & B Shared Inputs**")
    zone_encoded_s    = st.slider("Zone (encoded)", 0, 9, 2, key="s_zone")
    mode_encoded_s    = st.slider("Mode (encoded)", 0, 4, 1, key="s_mode")
    has_accident      = st.selectbox("Has Accident (0/1)", [0, 1], key="s_hasa")
    nb_accidents      = st.slider("Number of Accidents", 0, 50, 3, key="s_nba")
    nb_graves         = st.slider("Serious Injuries", 0, 20, 1, key="s_nbg")
    congestion_index_s= st.slider("Congestion Index", 0.0, 1.0, 0.3, 0.01, key="s_cong")
    crime_rate_scaled = st.slider("Crime Rate (scaled)", 0.0, 5.0, 1.0, 0.1, key="s_crime")
    gravite_index     = st.slider("Gravity Index", 0.0, 10.0, 2.0, 0.1, key="s_grav")

    st.divider()
    st.markdown("**Section B: Zone Risk**")
    volume_crimes      = st.slider("Volume of Crimes", 0, 500, 80, key="b_volit")
    taux_criminalite   = st.slider("Crime Rate (raw)", 0.0, 10.0, 2.0, 0.1, key="b_taux")
    usagers_vulnerables= st.slider("Vulnerable Users", 0, 200, 40, key="b_usag")

    st.divider()
    st.markdown("**Section C: Anomaly Detection**")
    mois_s = st.selectbox("Month", list(range(1, 13)), key="s_mois")

def build_input(feature_list, row_dict):
    cols = feature_list if feature_list is not None else list(row_dict.keys())
    return pd.DataFrame([{c: row_dict.get(c, 0) for c in cols}])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — ACCIDENT SEVERITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🚨 Section A — Accident Severity Classifier")
st.markdown("Predict whether a transport zone scenario is at risk of a serious incident.")

row_a = {
    "zone_encoded":     zone_encoded_s,
    "mode_encoded":     mode_encoded_s,
    "has_accident":     has_accident,
    "nb_accidents":     nb_accidents,
    "nb_graves":        nb_graves,
    "congestion_index": congestion_index_s,
    "crime_rate_scaled":crime_rate_scaled,
    "gravite_index":    gravite_index,
}

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🔎 Severity Prediction")
    if models["severity"] and models["severity_features"] is not None:
        try:
            feats = models["severity_features"]
            X = build_input(feats, row_a)
            severity_class = int(models["severity"].predict(X)[0])
            severity_proba = float(models["severity"].predict_proba(X)[0][1])

            if severity_class == 0:
                st.success("✅ **No serious incident predicted**")
            else:
                st.error("🚨 **Serious incident risk detected**")

            st.metric("Severity Class", severity_class, help="0 = None · 1 = Serious")
            st.metric("Severity Probability", f"{severity_proba:.3f}")
            st.progress(severity_proba, text=f"Risk: {severity_proba:.1%}")
            
            try:
                svm_model = joblib.load(os.path.join(BASE, "svm_severity.pkl"))
                svm_class = int(svm_model.predict(X)[0])
                svm_proba = float(svm_model.predict_proba(X)[0][1])
                comp_df = pd.DataFrame({
                    "Model": ["Random Forest", "SVM"],
                    "Severity Class": [severity_class, svm_class],
                    "Probability": [f"{severity_proba:.1%}", f"{svm_proba:.1%}"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                st.caption("Random Forest is primary. SVM shown as baseline.")
            except Exception:
                st.caption("Comparison model not available")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Severity model not loaded.")

with col_b:
    st.markdown("#### 📊 Input Summary")
    summary_df = pd.DataFrame(list(row_a.items()), columns=["Feature", "Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — ZONE RISK CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🗺️ Section B — Zone Risk Clustering")
st.markdown("Classify a transport zone into a **risk tier** based on safety indicators.")

RISK_LABELS = {
    0: ("🟢 Low Risk Zone",    "success"),
    1: ("🟡 Medium Risk Zone", "warning"),
    2: ("🔴 High Risk Zone",   "error"),
}

row_b = {
    "nb_accidents":      nb_accidents,
    "nb_graves":         nb_graves,
    "volume_crimes":     volume_crimes,
    "taux_criminalite":  taux_criminalite,
    "usagers_vulnerables": usagers_vulnerables,
}

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 🔍 Risk Cluster Assignment")
    if models["kmeans_risk"] and models["kmeans_scaler"] and models["kmeans_features"] is not None:
        try:
            feats = models["kmeans_features"]
            X_b = build_input(feats, row_b)
            X_scaled = models["kmeans_scaler"].transform(X_b)
            cluster_id = int(models["kmeans_risk"].predict(X_scaled)[0])
            label, level = RISK_LABELS.get(cluster_id, (f"Cluster {cluster_id}", "info"))

            if level == "success":
                st.success(f"**Zone Risk Cluster {cluster_id}** — {label}")
                risk_str = "Low"
            elif level == "warning":
                st.warning(f"**Zone Risk Cluster {cluster_id}** — {label}")
                risk_str = "Med"
            else:
                st.error(f"**Zone Risk Cluster {cluster_id}** — {label}")
                risk_str = "High"

            st.metric("Risk Cluster", cluster_id)
            
            try:
                hier_labels = joblib.load(os.path.join(BASE, "hierarchical_labels.pkl"))
                hier_cluster_id = hier_labels[zone_encoded_s] if len(hier_labels) > zone_encoded_s else hier_labels[0]
                hier_risk_str = "Low" if hier_cluster_id == 0 else ("Med" if hier_cluster_id == 1 else "High")
                comp_df3 = pd.DataFrame({
                    "Model": ["K-Means", "Hierarchical"],
                    "Assigned Cluster": [cluster_id, hier_cluster_id],
                    "Risk Level": [risk_str, hier_risk_str]
                })
                st.dataframe(comp_df3, use_container_width=True, hide_index=True)
            except Exception:
                st.caption("Comparison model not available")

        except Exception as e:
            st.error(f"Clustering error: {e}")
    else:
        st.info("Risk clustering model not loaded.")

with col_d:
    st.markdown("#### 📊 Cluster Centroids")
    if models["kmeans_risk"]:
        try:
            centers = models["kmeans_risk"].cluster_centers_
            feats = models["kmeans_features"] if models["kmeans_features"] is not None else \
                    [f"Feature {i}" for i in range(centers.shape[1])]
            df_c = pd.DataFrame(
                centers,
                columns=feats[:centers.shape[1]],
                index=[f"Cluster {i}" for i in range(centers.shape[0])],
            )
            st.bar_chart(df_c.T)
        except Exception as e:
            st.error(f"Chart error: {e}")
    else:
        st.info("Model not loaded.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔭 Section C — Anomaly Detection")
st.markdown(
    "Detect **unusual security spikes** using an Isolation Forest trained on historical zone data."
)

row_c = {
    "volume_crimes":    volume_crimes,
    "nb_accidents":     nb_accidents,
    "taux_criminalite": taux_criminalite,
    "zone_sk":          zone_encoded_s,
    "mois":             mois_s,
}

col_e, col_f = st.columns(2)

with col_e:
    st.markdown("#### 🚦 Anomaly Status")
    if models["iso_forest"] and models["anomaly_scaler"] and models["anomaly_features"] is not None:
        try:
            feats = models["anomaly_features"]
            X_c = build_input(feats, row_c)
            X_scaled = models["anomaly_scaler"].transform(X_c)
            result = int(models["iso_forest"].predict(X_scaled)[0])
            score  = float(models["iso_forest"].score_samples(X_scaled)[0])

            if result == -1:
                st.error("🚨 **ALERT: Abnormal security spike detected**")
                st.metric("Anomaly Score", f"{score:.4f}", delta="ANOMALY")
            else:
                st.success("✅ **Normal — No anomaly detected**")
                st.metric("Anomaly Score", f"{score:.4f}", delta="NORMAL")

            st.metric("Isolation Forest Output", result, help="-1 = Anomaly · 1 = Normal")
        except Exception as e:
            st.error(f"Anomaly detection error: {e}")
    else:
        st.info("Anomaly detection model not loaded.")

with col_f:
    st.markdown("#### 📋 Anomaly Input Features")
    summary_c = pd.DataFrame(list(row_c.items()), columns=["Feature", "Value"])
    st.dataframe(summary_c, use_container_width=True, hide_index=True)
    st.info(
        "**Isolation Forest** detects data points that deviate significantly from "
        "the historical distribution of security indicators."
    )

st.caption("🛡️ Actor 3 · Responsable Sécurité · Transport ML Dashboard")
