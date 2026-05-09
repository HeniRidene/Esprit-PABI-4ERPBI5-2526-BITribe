# Streamlit App — Transport ML Dashboard

## Pages
| Page | Actor | Models Used |
|---|---|---|
| app.py | Home | Summary dashboard |
| 1_actor1_ecologique.py | Actor 1 | XGBoost CO2, XGBoost Energy, K-Means, Lasso |
| 2_actor2_mobilites.py | Actor 2 | XGBoost charge, XGBoost cancellation, Logistic Regression |
| 3_actor3_securite.py | Actor 3 | Random Forest, SVM, K-Means, Isolation Forest |
| 4_advanced_nlp.py | Actor 4 | VADER sentiment, spaCy NER |
| 5_deep_learning.py | Deep Learning | LSTM congestion forecasting |

## How to Run
cd streamlit_app
streamlit run app.py

## Models Directory
models/
├── actor1/   ← XGBoost CO2, Energy, K-Means, Lasso pkl files
├── actor2/   ← XGBoost charge, cancellation, encoding pkl files
└── actor3/   ← RF severity, K-Means risk, Isolation Forest pkl files

## Requirements
pip install streamlit joblib pandas numpy xgboost scikit-learn vaderSentiment tensorflow matplotlib

## Bonus
This app covers the professor's bonus requirement:
Model deployment (web application) — Streamlit multi-page app
Git versioning — see root README.md
