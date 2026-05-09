import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

ZONE_NAMES = {1:"Paris",2:"Marseille",3:"Lyon",4:"Toulouse",5:"Nice",
              6:"Nantes",7:"Montpellier",8:"Strasbourg",9:"Bordeaux",10:"Lille"}

st.set_page_config(
    page_title="Deep Learning — LSTM Forecast | Transport ML",
    page_icon="🧠",
    layout="wide",
)

# ── Director-only guard ───────────────────────────────────────────────────────
actor_filter = st.query_params.get("actor", None)
if actor_filter:
    st.error("⛔ Access restricted. Deep Learning forecasting is available to Directors only.")
    st.stop()

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(r"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\actor4_advanced_nlp\outputs")
DATA_DIR    = Path(r"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\actor4_advanced_nlp\data")

MODEL_PATH  = OUTPUTS_DIR / "lstm_congestion.keras"
SCALER_PATH = OUTPUTS_DIR / "lstm_scaler.pkl"
CSV_PATH    = DATA_DIR    / "forecast_congestion.csv"

# ── Loaders (cached) ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LSTM model …")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH))

@st.cache_resource(show_spinner="Loading scaler …")
def load_scaler():
    import joblib
    return joblib.load(str(SCALER_PATH))

@st.cache_data(show_spinner="Loading congestion data …")
def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["ds"])
    return df

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='font-size:2.4rem; font-weight:800;
               background:linear-gradient(90deg,#64b5f6,#ce93d8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🧠 Deep Learning — LSTM Congestion Forecast
    </h1>
    <p style='color:#aaa; font-size:1.05rem; max-width:860px; margin-top:-0.4rem;'>
        Autoregressive 7-step-ahead forecast using a stacked LSTM
        (window = 14 days) trained on urban congestion data across 10 zones.
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#64b5f6;'>⚙️ Forecast Settings</h2>",
        unsafe_allow_html=True,
    )
    zone_name = st.selectbox(
        "Select Zone",
        options=list(ZONE_NAMES.values()),
        index=0,
        key="dl_zone_sk",
    )
    zone_sk = [k for k, v in ZONE_NAMES.items() if v == zone_name][0]
    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>Model: <code>lstm_congestion.keras</code><br>"
        "Scaler: <code>lstm_scaler.pkl</code><br>"
        "Window: 14 days · Horizon: 7 days</small>",
        unsafe_allow_html=True,
    )

# ── Load assets ───────────────────────────────────────────────────────────────
model_ok  = MODEL_PATH.exists()
scaler_ok = SCALER_PATH.exists()
data_ok   = CSV_PATH.exists()

if not model_ok:
    st.error(f"❌ Model not found: `{MODEL_PATH}`  — run `lstm_congestion.ipynb` first.")
    st.stop()
if not scaler_ok:
    st.error(f"❌ Scaler not found: `{SCALER_PATH}` — run `lstm_congestion.ipynb` first.")
    st.stop()
if not data_ok:
    st.error(f"❌ Data not found: `{CSV_PATH}`")
    st.stop()

model  = load_model()
scaler = load_scaler()
df_all = load_data()

# ── Filter zone & take last 14 days as seed ───────────────────────────────────
zone_df   = df_all[df_all["zone_sk"] == zone_sk].sort_values("ds").reset_index(drop=True)
last_14   = zone_df.tail(14)
last_30   = zone_df.tail(30)
last_date = zone_df["ds"].max()

# ── Iterative 7-step forecast ─────────────────────────────────────────────────
raw_seed  = last_14["congestion_forecast"].values.reshape(-1, 1)
scaled_seed = scaler.transform(raw_seed).flatten().tolist()

preds_scaled = []
window = scaled_seed.copy()
for _ in range(7):
    x = np.array(window[-14:]).reshape(1, 14, 1)
    p = float(model.predict(x, verbose=0)[0, 0])
    preds_scaled.append(p)
    window.append(p)

preds = scaler.inverse_transform(
    np.array(preds_scaled).reshape(-1, 1)
).flatten()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=7, freq="D"
)

forecast_df = pd.DataFrame({
    "Date":                    [d.strftime("%Y-%m-%d") for d in future_dates],
    "Forecast Congestion Index": [round(float(v), 4) for v in preds],
})

tomorrow_val = float(preds[0])

# ── KPI row ───────────────────────────────────────────────────────────────────
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric(
        label="📅 Tomorrow's Congestion",
        value=f"{tomorrow_val:.3f}",
        delta=f"{tomorrow_val - float(last_30['congestion_forecast'].mean()):.3f} vs 30-day avg",
    )
with col_kpi2:
    st.metric("📊 7-Day Peak",  f"{float(preds.max()):.3f}")
with col_kpi3:
    st.metric("📉 7-Day Min",   f"{float(preds.min()):.3f}")
with col_kpi4:
    st.metric("📈 7-Day Mean",  f"{float(preds.mean()):.3f}")

# ── Warning banner ────────────────────────────────────────────────────────────
if tomorrow_val > 2.5:
    st.warning(
        "⚠️ **High congestion expected — consider alternate routing**  "
        f"(Tomorrow's forecast: **{tomorrow_val:.3f}** > threshold 2.5)"
    )

st.divider()

# ── Matplotlib chart ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#161b22")

# Historical — last 30 days
ax.plot(
    last_30["ds"], last_30["congestion_forecast"],
    color="#64b5f6", linewidth=2.5, label="Actual (last 30 days)", zorder=3,
)
ax.fill_between(
    last_30["ds"], last_30["congestion_forecast"],
    alpha=0.15, color="#64b5f6",
)

# Forecast — 7 days (dashed)
ax.plot(
    future_dates, preds,
    color="#ce93d8", linewidth=2.5, linestyle="--",
    marker="o", markersize=6, markerfacecolor="#fff",
    markeredgecolor="#ce93d8", markeredgewidth=1.5,
    label="LSTM 7-day Forecast", zorder=4,
)

# Vertical delimiter
ax.axvline(x=last_date, color="#888", linestyle=":", linewidth=1.4, label="Now")

# Threshold line
ax.axhline(y=2.5, color="#ef5350", linestyle="--", linewidth=1, alpha=0.7,
           label="High congestion threshold (2.5)")

# Annotations on forecast points
for d, v in zip(future_dates, preds):
    ax.annotate(
        f"{v:.2f}",
        xy=(d, v),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="#ce93d8",
    )

ax.set_title(
    f"{ZONE_NAMES[zone_sk]} — Last 30 Days + 7-Day LSTM Forecast",
    fontsize=14, fontweight="bold", color="#e0e0e0", pad=12,
)
ax.set_xlabel("Date", color="#aaa", fontsize=11)
ax.set_ylabel("Congestion Index", color="#aaa", fontsize=11)
ax.tick_params(colors="#aaa")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="#aaa")
plt.setp(ax.get_yticklabels(), color="#aaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.grid(True, color="#2a2a2a", linewidth=0.8, linestyle="--")
ax.legend(
    facecolor="#1a1a2e", edgecolor="#444",
    labelcolor="#ddd", fontsize=10,
)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# ── Forecast table + download ─────────────────────────────────────────────────
col_l, col_r = st.columns([1.4, 1])

with col_l:
    st.subheader("📋 7-Day Forecast Table")
    # Colour rows where forecast > 2.5
    def highlight_high(row):
        val = row["Forecast Congestion Index"]
        return [
            "background-color:#3b1111; color:#ef9a9a;" if val > 2.5
            else "background-color:#0e1e0e; color:#a5d6a7;"
            for _ in row
        ]
    st.dataframe(
        forecast_df.style.apply(highlight_high, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name=f"lstm_forecast_{ZONE_NAMES[zone_sk].lower()}.csv",
        mime="text/csv",
    )
    
    with st.expander("📊 Model Comparison — LSTM vs Prophet baseline"):
        metrics_path = OUTPUTS_DIR / "lstm_metrics.json"
        try:
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                lstm_mae = metrics.get("mae", "N/A")
                lstm_rmse = metrics.get("rmse", "N/A")
                comp_df = pd.DataFrame({
                    "Model": ["LSTM", "Prophet (baseline)"],
                    "MAE": [lstm_mae, "0.857"],
                    "RMSE": [lstm_rmse, "-"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Comparison model not available")
        except Exception:
            st.caption("Comparison model not available")
        st.caption("Prophet MAE from Actor 2 pipeline (mean 0.857 across 10 zones). LSTM trained on zone-specific sequences with window=14.")

with col_r:
    st.subheader("📐 Model Info")
    st.markdown(
        f"""
        | Property | Value |
        |---|---|
        | Architecture | Stacked LSTM (2 × 64 units) |
        | Dropout | 0.2 per layer |
        | Window | 14 days |
        | Horizon | 7 days (iterative) |
        | Activation | Dense(1) — linear |
        | Selected Zone | **{ZONE_NAMES[zone_sk]}** |
        | Last known date | **{last_date.strftime('%Y-%m-%d')}** |
        | Tomorrow forecast | **{tomorrow_val:.4f}** |
        """
    )

st.divider()
st.caption("🧠 Deep Learning · LSTM Congestion Forecasting · Transport ML Dashboard")
