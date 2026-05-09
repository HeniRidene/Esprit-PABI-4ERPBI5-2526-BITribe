import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Actor 4 — Advanced NLP | Transport ML",
    page_icon="💬",
    layout="wide",
)

# ── Actor access guard ────────────────────────────────────────────────────────
ALLOWED_ACTOR = "actor4"
actor_filter = st.query_params.get("actor", None)
if actor_filter and actor_filter != ALLOWED_ACTOR:
    st.error("⛔ Access restricted. This page is only available to the NLP Analyst role.")
    st.stop()

# ── Role banner (Task 4) ──────────────────────────────────────────────────────
if actor_filter:
    st.info("🔤 Advanced NLP view — Actor 4 only")

# ── Paths ─────────────────────────────────────────────────────────────────────
NLP_DIR  = Path(r"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\actor4_advanced_nlp\outputs")
DATA_DIR = Path(r"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\actor4_advanced_nlp\data")

def nlp_path(filename):
    return NLP_DIR / filename

# ── VADER loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None

# ── LSTM loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    try:
        import tensorflow as tf
        import joblib
        model  = tf.keras.models.load_model(str(nlp_path("lstm_congestion.keras")))
        scaler = joblib.load(str(nlp_path("lstm_scaler.pkl")))
        return model, scaler
    except Exception as e:
        return None, None

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='font-size:2.4rem; font-weight:800; color:#ce93d8;'>
        💬 Actor 4 — Advanced NLP Analyst
    </h1>
    <p style='color:#aaa; font-size:1.05rem; max-width:860px;'>
        Sentiment analysis on transport feedback · Named Entity Recognition ·
        LSTM congestion forecasting
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Load metrics
try:
    metrics = json.loads((nlp_path("lstm_metrics.json")).read_text())
    lstm_mae = metrics.get("lstm_mae", "—")
    prophet_mae = metrics.get("prophet_mae", "—")
except Exception:
    lstm_mae, prophet_mae = "—", "—"

try:
    sent_df = pd.read_csv(nlp_path("sentiment_scores.csv"))
    n_feedbacks = len(sent_df)
    pct_pos = f"{(sent_df['final_sentiment'] == 'positive').mean()*100:.0f}%"
    pct_neg = f"{(sent_df['final_sentiment'] == 'negative').mean()*100:.0f}%"
except Exception:
    n_feedbacks, pct_pos, pct_neg = "—", "—", "—"

CARD = """
<div style='background:{bg}; border-radius:14px; padding:1.2rem;
            text-align:center; border:1px solid {border};'>
    <div style='font-size:1.9rem;'>{icon}</div>
    <div style='color:{label_color}; font-size:0.8rem; margin-top:0.3rem;'>{label}</div>
    <div style='color:#fff; font-size:1.9rem; font-weight:800; margin:0.2rem 0;'>{value}</div>
    <div style='color:#aaa; font-size:0.75rem;'>{sub}</div>
</div>
"""

with kpi1:
    st.markdown(CARD.format(
        bg="linear-gradient(135deg,#2a1f47,#5c35a0)", border="#5c35a0",
        icon="📦", label_color="#ce93d8", label="Feedbacks Analysed",
        value=n_feedbacks, sub="VADER + BERT"
    ), unsafe_allow_html=True)

with kpi2:
    st.markdown(CARD.format(
        bg="linear-gradient(135deg,#1a3a20,#2d6a4f)", border="#2d6a4f",
        icon="😊", label_color="#95d5b2", label="Positive Feedbacks",
        value=pct_pos, sub="Final sentiment label"
    ), unsafe_allow_html=True)

with kpi3:
    st.markdown(CARD.format(
        bg="linear-gradient(135deg,#4a1a1a,#8f2c2c)", border="#8f2c2c",
        icon="😠", label_color="#ef9a9a", label="Negative Feedbacks",
        value=pct_neg, sub="Final sentiment label"
    ), unsafe_allow_html=True)

with kpi4:
    st.markdown(CARD.format(
        bg="linear-gradient(135deg,#1a2a3a,#1565c0)", border="#1565c0",
        icon="🔮", label_color="#90caf9", label="LSTM MAE (Zone 1)",
        value=lstm_mae, sub=f"Prophet baseline: {prophet_mae}"
    ), unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sentiment Analysis",
    "🏷️ Named Entity Recognition",
    "🔮 LSTM Congestion Forecast",
    "✍️ Live VADER Analyser",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Sentiment Analysis — VADER & BERT Results")
    st.markdown(
        "Pre-computed on **772 transport feedback** texts using VADER (full dataset) "
        "and XLM-RoBERTa BERT (200-row sample). Results are production outputs from "
        "`advanced_nlp.ipynb`."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        vader_img = nlp_path("vader_distribution.png")
        if vader_img.exists():
            st.image(str(vader_img), caption="VADER Sentiment Distribution (all 772 feedbacks)", use_container_width=True)
        else:
            st.warning("Run advanced_nlp.ipynb to generate vader_distribution.png")

    with col_b:
        bert_img = nlp_path("vader_vs_bert_distribution.png")
        if bert_img.exists():
            st.image(str(bert_img), caption="VADER vs BERT Distribution (200-sample)", use_container_width=True)
        else:
            st.warning("Run advanced_nlp.ipynb to generate vader_vs_bert_distribution.png")

    st.markdown("---")
    st.subheader("🗺️ Sentiment by City/Zone")
    zone_img = nlp_path("sentiment_by_zone.png")
    if zone_img.exists():
        st.image(str(zone_img), caption="Average VADER Score per City/Zone", use_container_width=True)

    try:
        zone_agg = pd.read_csv(nlp_path("sentiment_by_zone.csv"))
        st.dataframe(
            zone_agg.style.background_gradient(subset=["avg_vader_score"], cmap="RdYlGn"),
            use_container_width=True,
        )
    except Exception:
        pass

    st.markdown("---")
    st.subheader("☁️ Word Clouds by Sentiment Class")
    wc1, wc2, wc3 = st.columns(3)
    for col, label, fname in [
        (wc1, "😊 Positive", "wordcloud_positive.png"),
        (wc2, "😐 Neutral",  "wordcloud_neutral.png"),
        (wc3, "😠 Negative", "wordcloud_negative.png"),
    ]:
        p = nlp_path(fname)
        with col:
            if p.exists():
                st.image(str(p), caption=label, use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Download Sentiment Results")
    try:
        sent_csv = pd.read_csv(nlp_path("sentiment_scores.csv"))
        st.download_button(
            "⬇️ Download sentiment_scores.csv",
            data=sent_csv.to_csv(index=False),
            file_name="sentiment_scores.csv",
            mime="text/csv",
        )
        with st.expander("Preview — sentiment_scores.csv"):
            st.dataframe(sent_csv.head(20), use_container_width=True)
    except Exception:
        st.info("sentiment_scores.csv not found. Run advanced_nlp.ipynb first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Named Entity Recognition
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏷️ Named Entity Recognition — spaCy fr_core_news_sm")
    st.markdown(
        "Extracted **LOC**, **ORG**, and **MISC** entities from all feedback texts using "
        "the French spaCy pipeline. Results identify transport lines, cities, and organisations."
    )

    ent_img = nlp_path("top_entities.png")
    if ent_img.exists():
        st.image(str(ent_img), caption="Top 25 Named Entities in Transport Feedback", use_container_width=True)
    else:
        st.warning("Run advanced_nlp.ipynb to generate top_entities.png")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.subheader("📋 Top Entities Table")
        try:
            ent_df = pd.read_csv(nlp_path("top_entities.csv"))
            st.dataframe(ent_df, use_container_width=True)
            st.download_button("⬇️ Download top_entities.csv",
                               data=ent_df.to_csv(index=False),
                               file_name="top_entities.csv", mime="text/csv")
        except Exception:
            st.info("top_entities.csv not found. Run advanced_nlp.ipynb first.")

    with col_e2:
        st.subheader("📍 Location Entities by Zone")
        try:
            zone_ent = pd.read_csv(nlp_path("zone_entities.csv"))
            st.dataframe(zone_ent, use_container_width=True)
            st.download_button("⬇️ Download zone_entities.csv",
                               data=zone_ent.to_csv(index=False),
                               file_name="zone_entities.csv", mime="text/csv")
        except Exception:
            st.info("zone_entities.csv not found. Run advanced_nlp.ipynb first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — LSTM Congestion Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔮 LSTM Congestion Forecasting")
    st.markdown(
        "A **2-layer stacked LSTM** (64 units each, Dropout 0.2) trained on daily congestion data "
        "across 10 urban zones. Window size: **14 days**. Architecture: `LSTM → Dropout → LSTM → Dropout → Dense(1)`."
    )

    # Training figures
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        loss_img = nlp_path("lstm_training_loss.png")
        if loss_img.exists():
            st.image(str(loss_img), caption="LSTM Training Loss — Zone 1", use_container_width=True)

    with col_t2:
        vs_img = nlp_path("lstm_vs_actual_zone1.png")
        if vs_img.exists():
            st.image(str(vs_img), caption="Actual vs LSTM vs Prophet (Zone 1 test period)", use_container_width=True)

    st.markdown("---")

    # Per-zone metrics
    st.subheader("📊 LSTM vs Prophet MAE — All 10 Zones")
    try:
        zone_met = pd.read_csv(nlp_path("lstm_zone_metrics.csv"))
        zone_met["lstm_wins"] = zone_met["lstm_mae"] < zone_met["prophet_mae"]

        st.dataframe(
            zone_met.style
                .background_gradient(subset=["lstm_mae"], cmap="YlGn_r")
                .background_gradient(subset=["prophet_mae"], cmap="OrRd_r")
                .applymap(lambda v: "color: #2ecc71; font-weight:bold" if v else "color: #e74c3c",
                          subset=["lstm_wins"]),
            use_container_width=True,
        )
        wins = zone_met["lstm_wins"].sum()
        st.info(f"✅ LSTM outperforms Prophet (lower MAE) on **{wins}/10 zones**.")
    except Exception:
        st.info("lstm_zone_metrics.csv not found. Run lstm_congestion.ipynb first.")

    st.markdown("---")

    # 7-day forecast
    st.subheader("📅 7-Day Ahead Congestion Forecast — Zone 1")
    forecast_img = nlp_path("lstm_forecast_7days.png")
    if forecast_img.exists():
        st.image(str(forecast_img), caption="Last 30 Days + 7-Day LSTM Forecast (Zone 1)", use_container_width=True)

    try:
        fc_df = pd.read_csv(nlp_path("lstm_7day_forecast.csv"))
        st.dataframe(fc_df, use_container_width=True)
        st.download_button("⬇️ Download lstm_7day_forecast.csv",
                           data=fc_df.to_csv(index=False),
                           file_name="lstm_7day_forecast.csv", mime="text/csv")
    except Exception:
        st.info("lstm_7day_forecast.csv not found. Run lstm_congestion.ipynb first.")

    st.markdown("---")

    # Live inference section
    st.subheader("⚡ Live LSTM Inference — Predict Next 7 Days")
    ZONE_NAMES = {
        1: "Paris", 2: "Marseille", 3: "Lyon", 4: "Toulouse", 5: "Nice",
        6: "Nantes", 7: "Montpellier", 8: "Strasbourg", 9: "Bordeaux", 10: "Lille",
    }
    zone_name = st.selectbox("Select Zone", list(ZONE_NAMES.values()), index=0, key="lstm_zone_sel")
    zone_sel  = [k for k, v in ZONE_NAMES.items() if v == zone_name][0]

    if st.button("🔮 Generate 7-Day Forecast", type="primary"):
        model_lstm, scaler_lstm = load_lstm()
        if model_lstm is None:
            st.error("❌ LSTM model not found. Run lstm_congestion.ipynb first.")
        else:
            try:
                df_cong = pd.read_csv(DATA_DIR / "forecast_congestion.csv", parse_dates=["ds"])
                zone_data = df_cong[df_cong["zone_sk"] == zone_sel].sort_values("ds")
                raw_vals  = zone_data["congestion_forecast"].values.reshape(-1, 1)

                from sklearn.preprocessing import MinMaxScaler
                sc_live = MinMaxScaler()
                sc_live.fit(raw_vals)
                scaled   = sc_live.transform(raw_vals).flatten()
                seed_win = scaled[-14:].tolist()

                preds_sc = []
                win = seed_win.copy()
                for _ in range(7):
                    x = np.array(win[-14:]).reshape(1, 14, 1)
                    p = model_lstm.predict(x, verbose=0)[0, 0]
                    preds_sc.append(p)
                    win.append(p)

                preds = sc_live.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()
                last_date = zone_data["ds"].max()
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D")

                result_df = pd.DataFrame({
                    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                    "Zone": zone_sel,
                    "Forecast Congestion Index": [round(v, 4) for v in preds],
                })

                st.success(f"✅ 7-day forecast generated for {ZONE_NAMES[zone_sel]}")
                st.dataframe(result_df, use_container_width=True)

                # Inline chart
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                hist_30 = zone_data.tail(30)
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(hist_30["ds"], hist_30["congestion_forecast"],
                        color="royalblue", linewidth=2, label="Historical (last 30 days)")
                ax.plot(future_dates, preds,
                        color="tomato", linewidth=2, linestyle="--", marker="o",
                        markersize=5, label="LSTM 7-day Forecast")
                ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1)
                ax.set_title(f"{ZONE_NAMES[zone_sel]} — LSTM 7-Day Congestion Forecast", fontsize=13)
                ax.set_xlabel("Date"); ax.set_ylabel("Congestion Index")
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.download_button("⬇️ Download forecast CSV",
                                   data=result_df.to_csv(index=False),
                                   file_name=f"lstm_forecast_{zone_name.lower()}.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Forecast error: {e}")

    st.markdown("---")

    # All-zones congestion overview
    st.subheader("🗺️ Congestion Overview — All 10 Zones")
    cong_img = nlp_path("congestion_all_zones.png")
    if cong_img.exists():
        st.image(str(cong_img), caption="Congestion Forecast Across All 10 Zones (2019–2023)", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Live VADER Analyser
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("✍️ Live VADER Sentiment Analyser")
    st.info(
        "📌 Type any transport feedback below for instant sentiment scoring. "
        "VADER works in English and French. For BERT-level accuracy, see the "
        "`advanced_nlp.ipynb` notebook."
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        user_text = st.text_area(
            "Enter transport feedback",
            placeholder=(
                "e.g. 'The metro was packed and delayed by 20 minutes — very frustrating!'\n"
                "or 'Service excellent, les bus étaient ponctuels et propres.'"
            ),
            height=160,
            key="nlp_input_v2",
        )
        analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### 📐 Score Guide")
        st.markdown(
            """
            | Compound | Sentiment |
            |---|---|
            | ≥ 0.05 | 😊 Positive |
            | −0.05 to 0.05 | 😐 Neutral |
            | < −0.05 | 😠 Negative |

            Score runs from **−1** (most negative) to **+1** (most positive).
            """
        )

    if analyse_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text before analysing.")
        else:
            analyser = load_vader()
            if analyser is None:
                st.error("❌ VADER not found. Run: `pip install vaderSentiment`")
            else:
                scores   = analyser.polarity_scores(user_text)
                compound = scores["compound"]

                if compound >= 0.05:
                    label, level, colour = "😊 Positive", "success", "#2d6a4f"
                elif compound <= -0.05:
                    label, level, colour = "😠 Negative", "error",   "#8f2c2c"
                else:
                    label, level, colour = "😐 Neutral",  "warning", "#7d5a00"

                st.divider()
                st.subheader("📊 Analysis Results")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Compound", f"{compound:+.4f}")
                m2.metric("Positive", f"{scores['pos']:.3f}")
                m3.metric("Neutral",  f"{scores['neu']:.3f}")
                m4.metric("Negative", f"{scores['neg']:.3f}")
                m5.metric("Sentiment", label)

                st.divider()
                cl, cr = st.columns([1, 2])
                with cl:
                    if level == "success":
                        st.success(f"**{label}**\n\nCompound: **{compound:+.4f}**")
                    elif level == "error":
                        st.error(f"**{label}**\n\nCompound: **{compound:+.4f}**")
                    else:
                        st.warning(f"**{label}**\n\nCompound: **{compound:+.4f}**")
                with cr:
                    progress_val = (compound + 1) / 2
                    st.markdown("**Sentiment Gauge** (−1 = very negative → +1 = very positive)")
                    st.progress(progress_val, text=f"Compound: {compound:+.4f}")
                    breakdown = pd.DataFrame({
                        "Component": ["Positive", "Neutral", "Negative"],
                        "Score":     [scores["pos"], scores["neu"], scores["neg"]],
                    })
                    st.bar_chart(breakdown.set_index("Component"))

st.divider()
st.caption("💬 Actor 4 · Advanced NLP Analyst · Transport ML Dashboard")
