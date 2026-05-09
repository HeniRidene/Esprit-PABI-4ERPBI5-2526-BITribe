---

# CLAUDE.md — Master Index
# Urban Mobility Intelligence Platform — BI Tribe ESPRIT PABI 4ERPBI5

## Project State: DONE (ML API, Streamlit, Notebooks) / IN PROGRESS (Next.js Hub integrations)

## Monorepo Map
- `actor4_advanced_nlp`: ✅ Jupyter notebooks for training VADER sentiment, spaCy NER, and LSTM congestion models.
- `ML/actor1_ecologique`: ✅ Legacy training pipeline and XGBoost/K-Means models for the Ecological Director.
- `ML/actor2_mobilites`: ✅ Training scripts for XGBoost passenger load, cancellation risk, and Prophet congestion models.
- `ML/actor3_securite`: ✅ Training scripts for Random Forest severity, K-Means risk clustering, and Isolation Forest anomalies.
- `ml_api_2`: ✅ FastAPI ML backend orchestrating predictions, MLflow tracking, Prometheus monitoring, and n8n workflows.
- `streamlit_app`: ✅ Streamlit multi-page application serving interactive ML dashboards for all actors.
- `streamlit_app/models`: ✅ Backend inference layer documenting the FastAPI dispatch and model registry architecture.
- `urban-mobility-website`: ⚠️ Central Next.js hub integrating BI dashboards, ML predictions, and MLOps control with pending UI updates.

## Global Ports & Auth
- **Next.js Hub**: 3000
- **FastAPI Backend**: 8000
- **Streamlit App**: 8501
- **MLflow**: 5000
- **Prometheus**: 9090
- **Grafana**: 3001 (⚠️ CONFLICT: `ml_api_2/Claude.md` lists 3000)
- **n8n Orchestration**: 5678

**Application Login Credentials:**
- General Director: `admin@urbanmobility.fr` / `admin123`
- Ecological Transition: `eco@urbanmobility.fr` / `eco123`
- Mobility: `mobility@urbanmobility.fr` / `mob123`
- Security: `security@urbanmobility.fr` / `sec123`
- Grafana: `admin` / `admin`

**Environment Variables (.env.local):**
`FASTAPI_URL`, `STREAMLIT_URL`, `MLFLOW_URL`, `GRAFANA_URL`, `PROMETHEUS_URL`, `N8N_URL` (plus `NEXT_PUBLIC_` prefixed equivalents).

**n8n Credentials:**
`YOUR_TELEGRAM_CREDENTIAL_ID`, `YOUR_GMAIL_CREDENTIAL_ID`, `YOUR_SHEETS_CREDENTIAL_ID`, `YOUR_TELEGRAM_CHAT_ID`, `YOUR_GOOGLE_SHEET_ID`.

## ✅ Completed Across All Modules
- [ml_api_2] FastAPI endpoints (/health, /predict, /retrain, /metrics) and MLflow tracking verified.
- [ml_api_2] Docker + Docker Compose deployment configured for API and MLflow.
- [ml_api_2] n8n workflows for prediction, alerting, and scheduled retraining imported and tested.
- [ml_api_2] Prometheus scraping and Grafana 5-panel dashboard setup completed.
- [streamlit_app] Multi-page dashboards implemented with actor-specific access guards.
- [streamlit_app] Streamlit iframe CORS/XSRF protection disabled in config.toml for Next.js embedding.
- [urban-mobility-website] MLOpsDashboard component created with health, prediction, and retrain controls.
- [urban-mobility-website] AuthContext and Sidebar updated for role-based Power BI and ML page access.
- [urban-mobility-website] Production hardening with environment variables, error boundaries, and rate limiting implemented.
- [ML/actor1_ecologique] XGBoost CO2/Energy and K-Means pollution clustering models trained and pickled.
- [ML/actor2_mobilites] XGBoost charge, cancellation classifier, and Prophet congestion models finalized.
- [ML/actor3_securite] Random Forest severity and Isolation Forest anomaly detection models saved.
- [actor4_advanced_nlp] LSTM congestion forecasting model trained and evaluated against Prophet baseline.

## ⚠️ Known Gaps & Open Issues
- [All Modules] ⚠️ CONFLICT: Grafana port is documented as 3000 in ml_api_2/Claude.md but as 3001 in root claude.md.
- [urban-mobility-website] GrafanaPanel component in MLOpsDashboard still hardcodes http://localhost:3001 instead of using NEXT_PUBLIC_GRAFANA_URL.
- [urban-mobility-website] Rate limiting ipMap is in-memory and resets on server restart, unsuitable for multi-instance production.
- [urban-mobility-website] Streamlit onLoad event fires even when iframe is blocked, making load failures silent.
- [urban-mobility-website] NEXT_PUBLIC_ env vars require a dev server restart to take effect after editing .env.local.
- [streamlit_app] Streamlit must be fully restarted after any config.toml changes, hot reload does not apply.
- [ml_api_2] Microsoft Store Python does not add scripts to PATH; must use `python -m mlflow`.
- [ml_api_2] UnicodeEncodeError on Windows console requires PYTHONIOENCODING=utf-8 for emojis.
- [ml_api_2] First /predict call has ~3s latency due to cold model loading from disk.
- [ml_api_2] nvidia_nccl_cu12 pulls 300MB on first Docker build.

## 🔲 Next Steps (All Modules)
- [urban-mobility-website] Apply NEXT_PUBLIC_GRAFANA_URL to GrafanaPanel iframe src and Expand link.
- [urban-mobility-website] Surface drift_detector.py and alerting.py alerts in the dashboard UI.
- [urban-mobility-website] Add UI buttons to trigger n8n webhooks directly from the website.
- [urban-mobility-website] Polish Power BI embeds with loading states, error fallbacks, and status dots.

## Sub-CLAUDE.md Index
| Module | File path | Last known status | Key facts |
|---|---|---|---|
| actor4_advanced_nlp | `actor4_advanced_nlp/CLAUDE.md` | DONE | VADER/BERT sentiment, NER, LSTM congestion models and metrics |
| actor1_ecologique | `ML/actor1_ecologique/CLAUDE.md` | DONE | XGBoost CO2/Energy and K-Means clustering, legacy training pipeline |
| actor2_mobilites | `ML/actor2_mobilites/CLAUDE.md` | DONE | XGBoost charge/cancellation and Prophet models, 98.22% ponctualité |
| actor3_securite | `ML/actor3_securite/CLAUDE.md` | DONE | RF severity and Isolation Forest models, 5% anomaly rate |
| ml_api_2 | `ml_api_2/Claude.md` | DONE | FastAPI serving 18 models, MLflow tracking, n8n orchestration, Prometheus metrics |
| streamlit_app | `streamlit_app/CLAUDE.md` | DONE | Multi-page Streamlit app covering 4 actors and deep learning |
| streamlit_models | `streamlit_app/models/Claude.md` | DONE | FastAPI inference layer documentation and n8n webhook payload mapping |
| urban-mobility-website | `urban-mobility-website/CLAUDE.md` | IN PROGRESS | Next.js Tailwind 4 dashboard system, MD3 design tokens, responsive UI |

---
