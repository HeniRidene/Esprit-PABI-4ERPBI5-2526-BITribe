import streamlit as st

# ── Actor filter (injected by urban-mobility-website via ?actor= query param) ──
actor_filter = st.query_params.get("actor", None)

# ── Sidebar page-hiding CSS (Task 1 & 2) ─────────────────────────────────────
# Sidebar nav order: Home(1) | Actor1-Éco(2) | Actor2-Mob(3) | Actor3-Sécu(4) | NLP(5) | DeepLearning(6)
# When an actor filter is active, hide Home + all pages except the allowed one.
if actor_filter == "actor1":
    st.markdown("""
    <style>
    /* actor1: hide Home + pages 2-5 (keep page 1 = Actor1) */
    [data-testid="stSidebarNav"] li:first-child,
    [data-testid="stSidebarNav"] li:nth-child(3),
    [data-testid="stSidebarNav"] li:nth-child(4),
    [data-testid="stSidebarNav"] li:nth-child(5),
    [data-testid="stSidebarNav"] li:nth-child(6) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

elif actor_filter == "actor2":
    st.markdown("""
    <style>
    /* actor2: hide Home + pages 1, 3-5 (keep page 2 = Actor2) */
    [data-testid="stSidebarNav"] li:first-child,
    [data-testid="stSidebarNav"] li:nth-child(2),
    [data-testid="stSidebarNav"] li:nth-child(4),
    [data-testid="stSidebarNav"] li:nth-child(5),
    [data-testid="stSidebarNav"] li:nth-child(6) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

elif actor_filter == "actor3":
    st.markdown("""
    <style>
    /* actor3: hide Home + pages 1-2, 4-5 (keep page 3 = Actor3) */
    [data-testid="stSidebarNav"] li:first-child,
    [data-testid="stSidebarNav"] li:nth-child(2),
    [data-testid="stSidebarNav"] li:nth-child(3),
    [data-testid="stSidebarNav"] li:nth-child(5),
    [data-testid="stSidebarNav"] li:nth-child(6) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

elif actor_filter == "actor4":
    st.markdown("""
    <style>
    /* actor4: hide Home + pages 1-3, 5 (keep page 4 = NLP) */
    [data-testid="stSidebarNav"] li:first-child,
    [data-testid="stSidebarNav"] li:nth-child(2),
    [data-testid="stSidebarNav"] li:nth-child(3),
    [data-testid="stSidebarNav"] li:nth-child(4),
    [data-testid="stSidebarNav"] li:nth-child(6) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# actor_filter is None → Director view: no CSS injection, all pages remain visible

ZONE_NAMES = {1:"Paris",2:"Marseille",3:"Lyon",4:"Toulouse",5:"Nice",
              6:"Nantes",7:"Montpellier",8:"Strasbourg",9:"Bordeaux",10:"Lille"}

st.set_page_config(
    page_title="Transport ML Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0;'>
            <h1 style='font-size:1.8rem; color:#4F8BF9;'>🚆 Transport ML</h1>
            <p style='color:#888; font-size:0.85rem;'>Urban Mobility Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    if actor_filter is None:
        # Full navigation — Director view
        st.markdown("**📌 Navigation**")
        st.info(
            "Use the pages in the sidebar to explore each actor's ML module:\n\n"
            "- 🌿 **Actor 1** — Écologique\n"
            "- 🚌 **Actor 2** — Mobilités\n"
            "- 🛡️ **Actor 3** — Sécurité\n"
            "- 💬 **Actor 4** — NLP"
        )
    else:
        # Restricted view — show only the allowed actor
        ACTOR_NAV = {
            "actor1": "🌿 **Actor 1** — Écologique",
            "actor2": "🚌 **Actor 2** — Mobilités",
            "actor3": "🛡️ **Actor 3** — Sécurité",
            "actor4": "💬 **Actor 4** — NLP",
        }
        st.markdown("**📌 Your Module**")
        st.info(ACTOR_NAV.get(actor_filter, f"Actor: {actor_filter}"))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='font-size:2.6rem; font-weight:800; color:#4F8BF9;'>
        🚆 Transport ML Dashboard
    </h1>
    <p style='font-size:1.1rem; color:#aaa; max-width:820px;'>
        A multi-actor machine learning platform for urban transport intelligence.
        Analyze ecological impact, mobility optimization, security risk assessment,
        and real-time NLP feedback — all in one unified dashboard.
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Role banner (shown when accessed via actor-filtered URL) ─────────────────
if actor_filter == "actor1":
    st.success("🌿 **Ecological Transition View** — Showing Actor 1 modules only")
elif actor_filter == "actor2":
    st.info("🚌 **Mobility Operations View** — Showing Actor 2 modules only")
elif actor_filter == "actor3":
    st.error("🛡️ **Security Management View** — Showing Actor 3 modules only")

# ── Project Description ───────────────────────────────────────────────────────
with st.expander("📖 About this Project", expanded=False):
    st.markdown(
        """
        This dashboard is the operational interface for the **Urban Transport ML Project**.
        It exposes production-ready predictive models trained on real transport data across
        three actor domains:

        | Actor | Role | Key Objective |
        |---|---|---|
        | 🌿 Directeur Écologique | Environmental Director | CO₂ & energy prediction, pollution zone clustering |
        | 🚌 Directeur Mobilités | Mobility Director | Passenger load forecasting, cancellation risk |
        | 🛡️ Responsable Sécurité | Security Director | Accident severity, zone risk, anomaly detection |
        | 💬 NLP Analyst | Feedback Analyst | Sentiment analysis on transport feedback |

        All models are pre-trained XGBoost, Random Forest, K-Means and Isolation Forest pipelines.
        """
    )

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance Summary")

# Each entry: (actor_key, html_content)
# actor_key = "actor1"/"actor2"/"actor3" → shown only for that role (+ Director)
# actor_key = "director_only"            → shown only when actor_filter is None
_KPI_CARDS = [
    ("actor1", """
<div style='background:linear-gradient(135deg,#1a472a,#2d6a4f);
            border-radius:16px; padding:1.4rem; text-align:center;
            border:1px solid #2d6a4f;'>
    <div style='font-size:2rem;'>🌿</div>
    <div style='color:#95d5b2; font-size:0.85rem; margin-top:0.4rem;'>Actor 1 — Écologique</div>
    <div style='color:#fff; font-size:2rem; font-weight:800; margin:0.3rem 0;'>R² = 0.71</div>
    <div style='color:#aaa; font-size:0.8rem;'>CO₂ & Energy XGBoost</div>
</div>"""),
    ("actor2", """
<div style='background:linear-gradient(135deg,#1a2a47,#2c4a8f);
            border-radius:16px; padding:1.4rem; text-align:center;
            border:1px solid #2c4a8f;'>
    <div style='font-size:2rem;'>🚌</div>
    <div style='color:#90caf9; font-size:0.85rem; margin-top:0.4rem;'>Actor 2 — Mobilités</div>
    <div style='color:#fff; font-size:2rem; font-weight:800; margin:0.3rem 0;'>98.22%</div>
    <div style='color:#aaa; font-size:0.8rem;'>On-time Performance</div>
</div>"""),
    ("actor3", """
<div style='background:linear-gradient(135deg,#4a1a1a,#8f2c2c);
            border-radius:16px; padding:1.4rem; text-align:center;
            border:1px solid #8f2c2c;'>
    <div style='font-size:2rem;'>🛡️</div>
    <div style='color:#ef9a9a; font-size:0.85rem; margin-top:0.4rem;'>Actor 3 — Sécurité</div>
    <div style='color:#fff; font-size:2rem; font-weight:800; margin:0.3rem 0;'>F1 = 1.0</div>
    <div style='color:#aaa; font-size:0.8rem;'>Severity Classifier</div>
</div>"""),
    ("director_only", """
<div style='background:linear-gradient(135deg,#2a1f47,#5c35a0);
            border-radius:16px; padding:1.4rem; text-align:center;
            border:1px solid #5c35a0;'>
    <div style='font-size:2rem;'>💬</div>
    <div style='color:#ce93d8; font-size:0.85rem; margin-top:0.4rem;'>Actor 4 — NLP</div>
    <div style='color:#fff; font-size:2rem; font-weight:800; margin:0.3rem 0;'>VADER</div>
    <div style='color:#aaa; font-size:0.8rem;'>Real-time Sentiment</div>
</div>"""),
    ("director_only", """
<div style='background:linear-gradient(135deg,#4F8BF9,#ce93d8);
            border-radius:16px; padding:1.4rem; text-align:center;
            border:1px solid #ce93d8;'>
    <div style='font-size:2rem;'>🧠</div>
    <div style='color:#e3f2fd; font-size:0.85rem; margin-top:0.4rem;'>Deep Learning</div>
    <div style='color:#fff; font-size:1.8rem; font-weight:800; margin:0.3rem 0;'>MAE = 0.169</div>
    <div style='color:#aaa; font-size:0.8rem;'>Congestion Forecasting</div>
</div>"""),
]

# Determine which cards are visible for the current role
if actor_filter is None:
    # Director view: show all 5 cards
    visible_cards = _KPI_CARDS
else:
    # Role-filtered view: show only the matching actor card
    # (actor4/NLP and Deep Learning are director-only)
    visible_cards = [(k, h) for k, h in _KPI_CARDS if k == actor_filter]

# Render with the exact right number of columns
if visible_cards:
    cols = st.columns(len(visible_cards))
    for col, (_, html) in zip(cols, visible_cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


st.divider()

# ── Quick-start guide ─────────────────────────────────────────────────────────
st.subheader("🚀 Quick Start")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        **How to use this dashboard:**
        1. Select a page from the **left sidebar**
        2. Adjust inputs using the sidebar sliders / dropdowns
        3. Predictions update automatically in real-time
        4. Alerts fire when thresholds are exceeded
        """
    )
with c2:
    st.markdown(
        """
        **Technology stack:**
        - 🤖 XGBoost · Random Forest · K-Means · Isolation Forest
        - 🧠 LSTM · TensorFlow/Keras (Deep Learning)
        - 📐 Lasso · Logistic Regression · SVM · Hierarchical Clustering (comparison models)
        - 🐍 Python · scikit-learn · joblib
        - 📊 Streamlit · Pandas · NumPy
        - 💬 VADER Sentiment Analysis
        """
    )

st.caption("Transport ML Dashboard · Urban Mobility Intelligence · 2024")
