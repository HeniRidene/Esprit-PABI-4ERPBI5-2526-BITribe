# 📊 Intelligent Urban Mobility Dashboard & ML Platform
**ESPRIT PABI 4ERPBI5 — BI Tribe**

## 🚀 Project Overview
This project is a comprehensive **Business Intelligence and Machine Learning decision-making suite** designed for urban transport authorities (e.g., Île-de-France Mobilités, RATP). The goal is to transform complex mobility data into actionable insights to improve network performance, attractiveness, safety, and sustainability.

The solution integrates **Power BI** for reporting, **FastAPI & MLflow** for Machine Learning serving and tracking, **n8n** for workflow orchestration, and interactive frontends using **Next.js** and **Streamlit**.

## 👥 Decision-Makers & ML Actors

The platform is structured around specific strategic roles, each powered by dedicated Machine Learning models:

1. **🌳 Ecological Transition Director (Actor 1)**
   - **Focus:** Carbon Intensity (< 0.10 kg/pass.km) and Air Quality.
   - **Models:** XGBoost (CO₂ & Energy prediction), K-Means (Pollution zones clustering), Prophet (AQI & PM2.5 forecasting).

2. **🚆 Mobilities Director (Actor 2)**
   - **Focus:** Punctuality (> 80%), Commercial Speed, Capacity.
   - **Models:** XGBoost (Passenger load / `charge_estimee` & trip cancellation risk), Prophet (Congestion forecasting).

3. **🛡️ Urban Transport Safety Manager (Actor 3)**
   - **Focus:** Accident Density (< 10/km²) and Transit Crime Rates.
   - **Models:** Random Forest (Accident severity), K-Means (Risk zone clustering), Isolation Forest (Anomaly detection for crime/accident spikes).

4. **🧠 Advanced NLP & Deep Learning (Actor 4)**
   - **Focus:** User Feedback Sentiment & Advanced Congestion Tracking.
   - **Models:** VADER / XLM-RoBERTa (Sentiment Analysis), spaCy (Named Entity Recognition), Keras LSTM (Deep learning congestion forecasting).

## 🛠️ Technical Architecture

### 1. Data Modeling & Power BI
- **Schema:** Galaxy Schema (DW-compatible) with zero circular dependencies.
- **Advanced DAX:** Dynamic KPIs with thresholds, Time Intelligence (MoM/YoY), and Top/Bottom ranking.
- **Security:** Row-Level Security (RLS) ensuring stakeholders only see data relevant to their perimeter.

### 2. MLOps & Backend Automation (`ml_api_2`)
- **FastAPI:** Exposes endpoints (`/predict`, `/retrain`, `/health`) to serve 18 different ML models.
- **MLflow Tracking:** Logs parameters, metrics, and `.pkl` artifacts.
- **n8n Orchestration:** Handles automated workflows including live predictions, alerting (Telegram/Gmail/Sheets) on high-risk anomalies, and weekly scheduled retraining.
- **Prometheus & Grafana:** Real-time health tracking, endpoint latency monitoring, and model drift detection.
- **Docker:** Containerized setup for API and MLflow services.

### 3. Frontends
- **Streamlit App (`streamlit_app`):** Multi-page interactive application serving dynamic ML dashboards for all actors.
- **Next.js Hub (`urban-mobility-website`):** Central portal integrating BI dashboards, ML predictions, and MLOps controls.
-  Built with Tailwind CSS and Material Design 3 tokens.

## 📂 Repository Structure

- `ML/actor1_ecologique/` — Training scripts and models for Ecological actor.
- `ML/actor2_mobilites/` — Training scripts and models for Mobility actor.
- `ML/actor3_securite/` — Training scripts and models for Security actor.
- `actor4_advanced_nlp/` — Jupyter notebooks for NLP and LSTM models.
- `ml_api_2/` — FastAPI backend, MLflow storage, Prometheus configs, and n8n workflows (`.json`).
- `streamlit_app/` — Streamlit interactive ML dashboards.
- `urban-mobility-website/` — Next.js central hub application.
- `Mobility_Dashboard.pbix` — Core Power BI report file.
- `TRAIN_MODELS.ps1` — Automated script to train all Actor 1, 2, and 3 ML models.
- `STARTUP.ps1` — Master startup script to launch all microservices.
- `claude.md` (and sub-module `CLAUDE.md` files) — Detailed developer and documentation notes.

## 🚀 How to Run the Platform

1. **Train the Models**  
   Run the training pipeline to generate all required `.pkl` model artifacts:
   ```powershell
   .\TRAIN_MODELS.ps1
   ```

2. **Start the Full Stack**  
   Run the master startup script which spins up MLflow (5000), FastAPI (8000), n8n (5678), Prometheus (9090), Grafana (3001), Streamlit (8501), and Next.js (3000):
   ```powershell
   .\STARTUP.ps1
   ```
   *Note: Ensure Docker is running if relying on the containerized API setup, or that the Python virtual environment is activated.*

## ✍️ Authors
Developed by a group of 6 students at **ESPRIT (2025-2026)**:
- **Heni Ridene**
- **Mohamed Sbissi**
- **Sirine Ben Chouikha**
- **Mohamed Amjed Chemchik**
- **Emna Baya Ben Romdhane**
- **Hammami Eya**
