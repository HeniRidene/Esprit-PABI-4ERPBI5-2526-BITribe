# Esprit-PABI-4ERPBI5-2526-BITribe
# 📊 Intelligent Urban Mobility Dashboard

## 🚀 Project Overview
This project is a high-level **Business Intelligence decision-making suite** designed for urban transport authorities (e.g., Île-de-France Mobilités, RATP). The goal is to transform complex mobility data into actionable insights to improve network performance, attractiveness, and sustainability.

This report follows professional **data visualization principles**, ensuring that the main insight of any visual can be understood in **less than 5 seconds**.

## 👥 Decision-Makers (Personas)
The dashboard is structured to meet the needs of three distinct strategic roles:
1. **Mobilities Director:** Focuses on Punctuality (Target > 80%), Commercial Speed, and Capacity
2. **Ecological Transition Director:** Focuses on Carbon Intensity (Target < 0.10 kg/pass.km) and Air Quality.
3. **Urban Transport Safety Manager:** Monitors Accident Density (Target < 10/km²) and Transit Crime Rates.

## 🛠️ Technical Architecture
### Data Modeling
- **Schema:** Galaxy Schema (DW-compatible) with zero circular dependencies to ensure optimal performance.
- **Tables:** Includes fact tables for `Service Mobility` and `Territorial Impact` connected to dimensions such as `Time`, `Zone`, `Lines`, and `Vehicles`.

### Advanced DAX Implementation
- **KPIs with Thresholds:** Dynamic business logic using `CALCULATE`, `VAR`, and `DIVIDE` to trigger visual alerts.
- **Time Intelligence:** Implementation of MoM and YoY growth comparisons (e.g., `SAMEPERIODLASTYEAR`).
- **Ranking:** Use of `TOPN` and `RANKX` for identifying the bottom-performing transport lines.

### Security & Deployment
- **Row-Level Security (RLS):** Configured and tested static/dynamic roles ensuring stakeholders only see data relevant to their perimeter.
- **Interactivity:** Full UX integration with page navigation, bookmarks for storytelling, and drill-through capabilities.
## 📈 Dashboard Structure
- **Page 1: Strategic Executive Summary** – High-level KPIs and network health status.
- **Page 2: Operational Reliability** – Deep-dive into delays and geographical hotspot analysis.
- **Page 3: Capacity & Competitiveness** – Analysis of passenger load vs. user satisfaction and commercial speed by mode.

## 📂 Repository Structure
- `Mobility_Dashboard.pbix`: The core Power BI report file.
- `Documentation/`: Detailed DAX measure definitions and data dictionary.
- `Assets/`: Project logo and custom theme JSON file.
- `README.md`: Project overview and technical documentation.

## ✍️ Authors
Developed by a group of 6 students at ESPRIT (2025-2026) : 
- **Heni Ridene**
- **Mohamed Sbissi**
- **Sirine Ben Chouikha**
- **Mohamed Amjed Chemchik**
- **Emna Baya Ben Romdhane**
- **Hammami Eya**

