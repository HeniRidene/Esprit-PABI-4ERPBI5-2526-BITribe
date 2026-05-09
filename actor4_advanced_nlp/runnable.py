import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#!pip install vaderSentiment transformers spacy torch tensorflow wordcloud scikit-learn nltk
#!python -m spacy download fr_core_news_sm

#%matplotlib inline
import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)
print('Setup complete.')
os.chdir(r'C:\Users\sbiss\OneDrive - ESPRIT\Desktop\actor4_advanced_nlp')

# --- 1.1 Load & Clean ---
df = pd.read_csv('data/feedbacks.csv')

# Drop duplicate header row if present
df = df[df['City'] != 'City'].reset_index(drop=True)
df['feedback_id'] = df.index + 1
df['Feedback'] = df['Feedback'].fillna('')

# French + English stopwords
stop_words = set(stopwords.words('french') + stopwords.words('english'))

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

df['clean_text'] = df['Feedback'].apply(clean_text)
print(f'Loaded {len(df)} feedbacks. Sample:')
df[['City', 'Feedback', 'clean_text']].head(3)

# --- 1.2 MODEL 1: VADER ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def vader_score(text):
    return analyzer.polarity_scores(text)['compound'] if text.strip() else 0.0

def vader_label(score):
    if score >= 0.05: return 'positive'
    elif score <= -0.05: return 'negative'
    return 'neutral'

df['vader_score'] = df['clean_text'].apply(vader_score)
df['vader_label'] = df['vader_score'].apply(vader_label)

# Plot distribution
colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
counts = df['vader_label'].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
counts.plot(kind='bar', color=[colors.get(c, 'gray') for c in counts.index], ax=ax)
ax.set_title('VADER Sentiment Distribution', fontsize=14)
ax.set_xlabel('Sentiment'); ax.set_ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/vader_distribution.png', dpi=150)
#plt.show()
print(counts)

# --- 1.3 MODEL 2: Transformers (sample=200) ---
# NOTE: Running on 200-row sample due to transformer inference time.
from transformers import pipeline

SAMPLE_SIZE = min(200, len(df))
print(f'[WARNING]  Running BERT on {SAMPLE_SIZE} rows (subsample) — full inference would be slow.')

df_sample = df.sample(SAMPLE_SIZE, random_state=42).copy()

# cardiffnlp/twitter-xlm-roberta-base-sentiment supports French
bert_pipe = pipeline(
    'sentiment-analysis',
    model='cardiffnlp/twitter-xlm-roberta-base-sentiment',
    truncation=True, max_length=128
)

def get_bert_label(text):
    if not text.strip():
        return 'neutral'
    try:
        result = bert_pipe(text[:256])[0]['label'].lower()
        # model returns: positive / negative / neutral
        return result if result in ['positive', 'negative', 'neutral'] else 'neutral'
    except:
        return 'neutral'

df_sample['bert_label'] = df_sample['clean_text'].apply(get_bert_label)

# Merge bert_label back to full df (NaN for non-sampled rows)
df = df.merge(df_sample[['feedback_id', 'bert_label']], on='feedback_id', how='left')
print('BERT scoring done.')
df_sample[['Feedback', 'vader_label', 'bert_label']].head(5)

# --- 1.4 Comparison Table: VADER vs BERT ---
# Only on the 200-row sample where both labels exist
cmp = df_sample[['vader_label', 'bert_label']].copy()
crosstab = pd.crosstab(cmp['vader_label'], cmp['bert_label'], margins=True)
print('=== VADER vs BERT Crosstab ===')
print(crosstab)

agreement = (cmp['vader_label'] == cmp['bert_label']).mean()
print(f'\nAgreement Rate: {agreement:.1%}')

# Distribution comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cmp['vader_label'].value_counts().plot(kind='bar', ax=axes[0], title='VADER (sample)', color='steelblue')
cmp['bert_label'].value_counts().plot(kind='bar', ax=axes[1], title='BERT (sample)', color='coral')
for ax in axes: ax.set_xlabel(''); ax.tick_params(rotation=0)
plt.tight_layout()
plt.savefig('outputs/vader_vs_bert_distribution.png', dpi=150)
#plt.show()

# --- 1.5 Output: sentiment_scores.csv ---
# final_sentiment = bert_label if available else vader_label
df['final_sentiment'] = df['bert_label'].fillna(df['vader_label'])

# zone_sk: categorical code from City
df['zone_sk'] = df['City'].astype('category').cat.codes

out_cols = ['feedback_id', 'vader_score', 'bert_label', 'final_sentiment', 'zone_sk', 'City', 'Date']
sentiment_out = df[out_cols].copy()
sentiment_out.to_csv('outputs/sentiment_scores.csv', index=False)
print('Saved → outputs/sentiment_scores.csv')
sentiment_out.head()

# --- 1.6 Wordcloud per sentiment class ---
from wordcloud import WordCloud

def make_wordcloud(label, color):
    texts = ' '.join(df[df['vader_label'] == label]['clean_text'].dropna())
    if not texts.strip():
        print(f'No text for {label}'); return
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap=color, max_words=80).generate(texts)
    plt.figure(figsize=(10, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Wordcloud — {label.capitalize()} Feedbacks', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'outputs/wordcloud_{label}.png', dpi=150)
    #plt.show()

make_wordcloud('positive', 'Greens')
make_wordcloud('negative', 'Reds')
make_wordcloud('neutral',  'Blues')

# --- 1.7 Sentiment score by zone_sk (Power BI ready) ---
zone_agg = df.groupby(['zone_sk', 'City']).agg(
    avg_vader_score=('vader_score', 'mean'),
    count=('feedback_id', 'count'),
    pct_negative=('vader_label', lambda x: (x == 'negative').mean() * 100)
).reset_index()

zone_agg.to_csv('outputs/sentiment_by_zone.csv', index=False)
print('Saved → outputs/sentiment_by_zone.csv')

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(zone_agg['City'], zone_agg['avg_vader_score'],
              color=['#e74c3c' if s < 0 else '#2ecc71' for s in zone_agg['avg_vader_score']])
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Average VADER Score by City/Zone', fontsize=14)
ax.set_ylabel('Avg Compound Score')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/sentiment_by_zone.png', dpi=150)
#plt.show()
zone_agg

import spacy
from collections import Counter

nlp = spacy.load('fr_core_news_sm')

# Extract entities from original (uncleaned) Feedback for better NER precision
records = []
for _, row in df.iterrows():
    text = str(row['Feedback'])
    if not text.strip() or text == 'nan': continue
    doc = nlp(text[:512])  # cap length for speed
    for ent in doc.ents:
        if ent.label_ in ('LOC', 'ORG', 'MISC'):
            records.append({
                'entity': ent.text.strip(),
                'label': ent.label_,
                'city': row['City'],
                'zone_sk': row['zone_sk']
            })

ent_df = pd.DataFrame(records)
print(f'Total entity mentions: {len(ent_df)}')
ent_df.head()

# --- 2.1 Top entities frequency table ---
top_ents = (
    ent_df.groupby(['entity', 'label'])
    .size().reset_index(name='frequency')
    .sort_values('frequency', ascending=False)
    .head(25)
)
top_ents.to_csv('outputs/top_entities.csv', index=False)
print('Saved → outputs/top_entities.csv')

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
color_map = {'LOC': '#3498db', 'ORG': '#e67e22', 'MISC': '#9b59b6'}
colors_list = [color_map.get(l, 'gray') for l in top_ents['label']]
ax.barh(top_ents['entity'], top_ents['frequency'], color=colors_list)
ax.invert_yaxis()
ax.set_title('Top 25 Named Entities in Feedbacks', fontsize=13)
ax.set_xlabel('Frequency')
# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('outputs/top_entities.png', dpi=150)
#plt.show()
top_ents

# --- 2.2 Link extracted zones to zone_sk ---
zone_entities = (
    ent_df[ent_df['label'] == 'LOC']
    .groupby(['entity', 'zone_sk', 'city'])
    .size().reset_index(name='frequency')
    .sort_values('frequency', ascending=False)
    .head(20)
)
zone_entities.to_csv('outputs/zone_entities.csv', index=False)
print('Saved → outputs/zone_entities.csv')
print('\nTop LOC entities linked to zone_sk:')
zone_entities

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Load congestion data from actor2 outputs — fallback to simulation
actor2_paths = [
    '../../actor2_mobilites/outputs/forecast_congestion.csv',
    '../outputs/forecast_congestion.csv',
    '../../actor2_mobilites/data/congestion.csv',
]
ts_df = None
for path in actor2_paths:
    if os.path.exists(path):
        ts_df = pd.read_csv(path)
        print(f'Loaded congestion data from: {path}')
        break

if ts_df is None:
    print('[WARNING]  Actor 2 output not found — using simulated congestion data (500 days).')
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    trend = np.linspace(0.3, 0.7, 500)
    seasonal = 0.2 * np.sin(2 * np.pi * np.arange(500) / 7)
    noise = np.random.normal(0, 0.05, 500)
    ts_df = pd.DataFrame({'date': dates.astype(str),
                           'congestion_index': trend + seasonal + noise})

# Ensure column names
if 'congestion_index' not in ts_df.columns:
    numeric_cols = ts_df.select_dtypes(include=np.number).columns.tolist()
    ts_df.rename(columns={numeric_cols[0]: 'congestion_index'}, inplace=True)

ts_values = ts_df['congestion_index'].values.reshape(-1, 1)
print(f'Time-series length: {len(ts_values)} points')

# --- 3.1 Sliding window + train/test split ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(ts_values)

WINDOW = 7
X, y = [], []
for i in range(len(scaled) - WINDOW):
    X.append(scaled[i:i+WINDOW, 0])
    y.append(scaled[i+WINDOW, 0])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape(*X_train.shape, 1)
X_test  = X_test.reshape(*X_test.shape, 1)
print(f'Train: {X_train.shape} | Test: {X_test.shape}')

# --- 3.2 Build & Train LSTM ---
tf.random.set_seed(42)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20, batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Training loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history['loss'], label='Train Loss', color='steelblue')
ax.plot(history.history['val_loss'], label='Val Loss', color='tomato')
ax.set_title('LSTM Training Loss (MSE)', fontsize=13)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/lstm_loss.png', dpi=150)
#plt.show()

# --- 3.3 Predict + Inverse Transform ---
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

lstm_mae = mean_absolute_error(y_actual, y_pred)
print(f'LSTM  MAE: {lstm_mae:.4f}')

# Simulate Prophet predictions (naive persistence = last known value)
# Replace with actual prophet output if available in actor2 exports
prophet_pred = np.roll(y_actual, 1); prophet_pred[0] = y_actual[0]
prophet_mae  = mean_absolute_error(y_actual, prophet_pred)
print(f'Prophet MAE (naive baseline): {prophet_mae:.4f}')

# Comparison table
mae_df = pd.DataFrame({
    'Model':  ['LSTM', 'Prophet (naive)'],
    'MAE':    [round(lstm_mae, 4), round(prophet_mae, 4)]
})
mae_df.to_csv('outputs/model_mae_comparison.csv', index=False)
print('\n'); print(mae_df.to_string(index=False))

# --- 3.4 Plot: Actual vs LSTM vs Prophet ---
n = len(y_actual)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(n), y_actual,      label='Actual',           color='black',       linewidth=1.5)
ax.plot(range(n), y_pred,        label=f'LSTM  (MAE={lstm_mae:.3f})',    color='royalblue',   linewidth=1.5)
ax.plot(range(n), prophet_pred,  label=f'Prophet (MAE={prophet_mae:.3f})', color='darkorange', linewidth=1.2, linestyle='--')
ax.set_title('Congestion Index — Test Period Forecast vs Actual', fontsize=14)
ax.set_xlabel('Time step'); ax.set_ylabel('Congestion Index')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/forecast_comparison.png', dpi=150)
#plt.show()

# Save predictions
pred_df = pd.DataFrame({'actual': y_actual.flatten(), 'lstm_pred': y_pred.flatten(), 'prophet_pred': prophet_pred.flatten()})
pred_df.to_csv('outputs/lstm_predictions.csv', index=False)
print('Saved → outputs/lstm_predictions.csv')
# Save model and scaler
model.save('outputs/lstm_model.keras')
import joblib
joblib.dump(scaler, 'outputs/scaler.pkl')
print('Saved → outputs/lstm_model.keras and outputs/scaler.pkl')



# Save model and scaler
try:
    model.save('outputs/lstm_model.keras')
    import joblib
    joblib.dump(scaler, 'outputs/scaler.pkl')
    print('Saved → outputs/lstm_model.keras and outputs/scaler.pkl')
except Exception as e:
    print('Failed to save model:', e)
