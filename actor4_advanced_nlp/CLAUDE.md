# Actor 4 — Advanced NLP & Deep Learning

## Models
- VADER: rule-based sentiment analysis on transport feedbacks (full dataset)
- XLM-RoBERTa (BERT): transformer sentiment model (200-row sample)
- spaCy fr_core_news_sm: Named Entity Recognition on feedback text
- LSTM (Keras): deep learning congestion forecasting (window=14, horizon=7 days)

## Notebooks
| Notebook | Purpose |
|---|---|
| advanced_nlp.ipynb | Sentiment Analysis + NER + LSTM |
| lstm_congestion.ipynb | Dedicated LSTM training and evaluation |

## Output Files
| File | Description |
|---|---|
| outputs/sentiment_scores.csv | VADER + BERT scores per feedback |
| outputs/lstm_congestion.keras | Trained LSTM model |
| outputs/lstm_scaler.pkl | MinMaxScaler for congestion data |
| outputs/lstm_7day_forecast.csv | 7-day ahead forecast per zone |
| outputs/lstm_metrics.json | LSTM vs Prophet MAE comparison |

## How to Run
jupyter notebook advanced_nlp.ipynb
jupyter notebook lstm_congestion.ipynb

## Key Metrics
| Metric | Value |
|---|---|
| LSTM MAE (Zone 1) | 0.169 |
| Prophet baseline MAE | 0.288 |
| Feedbacks analysed | 772 |
| NER entities extracted | top 10 transport entities |
