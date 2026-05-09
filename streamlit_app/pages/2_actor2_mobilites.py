import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

ZONE_NAMES = {1:"Paris",2:"Marseille",3:"Lyon",4:"Toulouse",5:"Nice",
              6:"Nantes",7:"Montpellier",8:"Strasbourg",9:"Bordeaux",10:"Lille"}

st.set_page_config(
    page_title="Actor 2 — Mobilités | Transport ML",
    page_icon="🚌",
    layout="wide",
)

# ── Actor access guard ────────────────────────────────────────────────────────
ALLOWED_ACTOR = "actor2"
actor_filter = st.query_params.get("actor", None)
if actor_filter and actor_filter != ALLOWED_ACTOR:
    st.error("⛔ Access restricted. This page is only available to the Mobility role.")
    st.stop()

# ── Role banner (Task 4) ──────────────────────────────────────────────────────
if actor_filter:
    st.info("🚌 Mobility Operations view — Actor 2 only")

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "actor2")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "charge":               "xgboost_charge.pkl",
        "charge_features":      "xgboost_charge_features.pkl",
        "cancellation":         "xgboost_cancellation.pkl",
        "cancellation_features":"xgboost_cancellation_features.pkl",
        "encoding":             "charge_encoding.pkl",
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
st.title("🚌 Directeur Mobilités — Mobility & Scheduling Intelligence")
st.markdown(
    """
    This module predicts **passenger load** and **cancellation risk** for urban transport lines.
    Use it to proactively manage fleet allocation and minimize service disruptions.
    """
)
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚌 Mobilités — Inputs")
    zone_encoded       = st.slider("Zone (encoded)", 0, 9, 3, key="m_zone")
    line_encoded       = st.slider("Line (encoded)", 0, 20, 5, key="m_line")
    mode_encoded       = st.slider("Mode (encoded)", 0, 4, 1, key="m_mode")
    hour               = st.slider("Hour of Day", 0, 23, 8, key="m_hour")
    rush_hour          = st.selectbox("Rush Hour", [0, 1], key="m_rush")
    is_weekend         = st.selectbox("Is Weekend", [0, 1], key="m_weekend")
    congestion_index   = st.slider("Congestion Index", 0.0, 1.0, 0.4, 0.01, key="m_cong")
    vitesse_kmh        = st.slider("Speed (km/h)", 10, 120, 45, key="m_vitesse")
    temps_trajet_min   = st.slider("Journey Time (min)", 5, 120, 30, key="m_temps")

def build_input(feature_list):
    """Build a DataFrame row aligned to a given feature list."""
    row = {
        "zone_encoded":     zone_encoded,
        "line_encoded":     line_encoded,
        "mode_encoded":     mode_encoded,
        "hour":             hour,
        "rush_hour":        rush_hour,
        "is_weekend":       is_weekend,
        "congestion_index": congestion_index,
        "vitesse_kmh":      vitesse_kmh,
        "temps_trajet_min": temps_trajet_min,
    }
    cols = feature_list if feature_list is not None else list(row.keys())
    return pd.DataFrame([{c: row.get(c, 0) for c in cols}])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — PASSENGER LOAD PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🧳 Section A — Passenger Load Prediction")
st.markdown("Forecast the estimated passenger occupancy for a given line and time slot.")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 📈 Predicted Load")
    if models["charge"] and models["charge_features"] is not None:
        try:
            feats = models["charge_features"]
            X = build_input(feats)
            charge = float(models["charge"].predict(X)[0])
            st.metric("Estimated Load (%)", f"{charge:.1f}%")

            # Colour indicator
            if charge < 30:
                st.success(f"🟢 **Low load** ({charge:.1f}%) — Comfortable capacity")
            elif charge < 70:
                st.warning(f"🟡 **Moderate load** ({charge:.1f}%) — Monitor closely")
            else:
                st.error(f"🔴 **High load** ({charge:.1f}%) — Consider adding capacity")
            
            try:
                lasso_charge = joblib.load(os.path.join(BASE, "lasso_charge.pkl"))
                lasso_pred = float(lasso_charge.predict(X)[0])
                comp_df = pd.DataFrame({
                    "Model": ["XGBoost", "Lasso"],
                    "Predicted Load": [f"{charge:.1f}%", f"{lasso_pred:.1f}%"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            except Exception:
                st.caption("Comparison model not available")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Load model not available.")

with col_b:
    st.markdown("#### 📊 Load Level Guide")
    levels_df = pd.DataFrame({
        "Level":      ["Low",    "Moderate", "High"],
        "Range (%)":  ["0 – 29", "30 – 69",  "70+"],
        "Action":     ["No action", "Monitor", "Add capacity"],
    })
    st.dataframe(levels_df, use_container_width=True, hide_index=True)
    st.markdown(
        """
        - **Low** (< 30 %): Fleet utilisation is optimal
        - **Moderate** (30–70 %): Watch for peak-hour spikes
        - **High** (> 70 %): Fleet reinforcement recommended
        """
    )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — CANCELLATION RISK
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🚫 Section B — Cancellation Risk Assessment")
st.markdown(
    "Predict the probability that this trip will be cancelled, based on current conditions."
)

THRESHOLD = 0.047

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 🎲 Risk Probability")
    if models["cancellation"] and models["cancellation_features"] is not None:
        try:
            feats = models["cancellation_features"]
            X = build_input(feats)
            proba = float(models["cancellation"].predict_proba(X)[0][1])

            st.progress(min(proba, 1.0), text=f"Cancellation probability: {proba:.3f}")
            st.metric("Risk Score", f"{proba:.4f}", delta=f"Threshold: {THRESHOLD}")

            if proba < THRESHOLD:
                st.success("🟢 **Low Risk** — Service expected to run normally")
                risk_label = "Low"
            elif proba < 0.15:
                st.warning("🟡 **Medium Risk** — Monitor service conditions")
                risk_label = "Med"
            else:
                st.error("🔴 **High Risk** — Cancellation likely; alert operations team")
                st.warning(
                    "⚠️ **High cancellation risk detected** — Consider deploying backup vehicles "
                    "or issuing passenger notifications."
                )
                risk_label = "High"
                
            try:
                logreg_cancellation = joblib.load(os.path.join(BASE, "logreg_cancellation.pkl"))
                logreg_proba = float(logreg_cancellation.predict_proba(X)[0][1])
                logreg_risk = "Low" if logreg_proba < THRESHOLD else ("Med" if logreg_proba < 0.15 else "High")
                comp_df2 = pd.DataFrame({
                    "Model": ["XGBoost (primary)", "Logistic Reg (base)"],
                    "Probability": [f"{proba:.3f}", f"{logreg_proba:.3f}"],
                    "Risk Label": [risk_label, logreg_risk]
                })
                st.dataframe(comp_df2, use_container_width=True, hide_index=True)
                st.caption("XGBoost is the primary model. Logistic Regression shown as baseline comparison.")
            except Exception:
                st.caption("Comparison model not available")
        except Exception as e:
            st.error(f"Cancellation model error: {e}")
    else:
        st.info("Cancellation model not available.")

with col_d:
    st.markdown("#### 📋 Risk Threshold Reference")
    thresh_df = pd.DataFrame({
        "Risk Level": ["Low", "Medium", "High"],
        "Probability Range": [f"< {THRESHOLD}", f"{THRESHOLD} – 0.15", "> 0.15"],
        "Recommended Action": [
            "No action required",
            "Alert operations team",
            "Deploy backup + notify passengers",
        ],
    })
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)
    st.markdown(
        f"""
        The decision threshold is set at **{THRESHOLD}** based on model calibration
        to balance precision and recall for operational safety.
        """
    )

st.caption("🚌 Actor 2 · Directeur Mobilités · Transport ML Dashboard")
