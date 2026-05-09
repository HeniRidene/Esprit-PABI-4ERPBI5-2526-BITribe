import matplotlib
matplotlib.use('Agg')

#!pip install tensorflow scikit-learn pandas numpy matplotlib joblib --quiet

import os
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\sbiss\OneDrive - ESPRIT\Desktop\actor4_advanced_nlp')
os.makedirs('outputs', exist_ok=True)

print('Working directory:', os.getcwd())
print('Setup complete.')

# Load dataset
df = pd.read_csv('data/forecast_congestion.csv')
df['ds'] = pd.to_datetime(df['ds'])

print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Zones (zone_sk):', sorted(df['zone_sk'].unique()))
print('Date range:', df['ds'].min(), '->', df['ds'].max())
print('Rows per zone:')
print(df.groupby('zone_sk').size())
df.head()

# Plot congestion_forecast for all 10 zones
fig, ax = plt.subplots(figsize=(14, 6))

colors = plt.cm.tab10.colors
for i, zone in enumerate(sorted(df['zone_sk'].unique())):
    zone_df = df[df['zone_sk'] == zone].sort_values('ds')
    ax.plot(zone_df['ds'], zone_df['congestion_forecast'],
            label=f'Zone {zone}', color=colors[i % 10], linewidth=1.2, alpha=0.85)

ax.set_title('Congestion Forecast — All 10 Zones', fontsize=15, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Congestion Index')
ax.legend(loc='upper left', ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/congestion_all_zones.png', dpi=150)
plt.show()
print('Saved -> outputs/congestion_all_zones.png')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

WINDOW_SIZE = 14

def create_sequences(data, window):
    """Convert a 1-D scaled array into (X, y) supervised learning pairs."""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i: i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# Isolate zone 1, sort chronologically
zone1_df = df[df['zone_sk'] == 1].sort_values('ds').reset_index(drop=True)
values_z1 = zone1_df['congestion_forecast'].values.reshape(-1, 1)

# Scale
scaler_z1 = MinMaxScaler()
scaled_z1 = scaler_z1.fit_transform(values_z1).flatten()

# Sequences
X_z1, y_z1 = create_sequences(scaled_z1, WINDOW_SIZE)

# Temporal train/test split (80/20 — NO shuffle)
split = int(len(X_z1) * 0.8)
X_train, X_test = X_z1[:split], X_z1[split:]
y_train, y_test = y_z1[:split], y_z1[split:]

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
X_test  = X_test.reshape(-1, WINDOW_SIZE, 1)

print(f'Zone 1 — Total samples: {len(X_z1)}')
print(f'  Train: X={X_train.shape}, y={y_train.shape}')
print(f'  Test:  X={X_test.shape},  y={y_test.shape}')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)

def build_lstm_model(window_size):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model_z1 = build_lstm_model(WINDOW_SIZE)
model_z1.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model_z1.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Plot training loss vs validation loss
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history.history['loss'],     label='Train Loss', color='royalblue', linewidth=2)
ax.plot(history.history['val_loss'], label='Val Loss',   color='tomato',    linewidth=2)
ax.set_title('LSTM Training Loss (MSE) — Zone 1', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/lstm_training_loss.png', dpi=150)
plt.show()
print('Saved -> outputs/lstm_training_loss.png')

# Predictions on test set
y_pred_scaled = model_z1.predict(X_test)
y_pred   = scaler_z1.inverse_transform(y_pred_scaled).flatten()
y_actual = scaler_z1.inverse_transform(y_test.reshape(-1, 1)).flatten()

# LSTM metrics
lstm_mae  = mean_absolute_error(y_actual, y_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
print(f'LSTM   MAE:  {lstm_mae:.4f}')
print(f'LSTM   RMSE: {lstm_rmse:.4f}')

# Prophet baseline: 14-day rolling mean on zone 1 test period
all_values_z1 = zone1_df['congestion_forecast'].values
rolling_mean   = pd.Series(all_values_z1).rolling(window=14, min_periods=1).mean().values
# Align with test set indices
test_start_idx = split + WINDOW_SIZE  # first actual target index in original array
prophet_baseline = rolling_mean[test_start_idx: test_start_idx + len(y_actual)]

# Ensure same length
min_len = min(len(y_actual), len(prophet_baseline))
y_actual_cmp = y_actual[:min_len]
y_pred_cmp   = y_pred[:min_len]
prophet_cmp  = prophet_baseline[:min_len]

prophet_mae  = mean_absolute_error(y_actual_cmp, prophet_cmp)
prophet_rmse = np.sqrt(mean_squared_error(y_actual_cmp, prophet_cmp))
print(f'Prophet MAE:  {prophet_mae:.4f}')
print(f'Prophet RMSE: {prophet_rmse:.4f}')

# Comparison table
metrics_df = pd.DataFrame({
    'Model': ['LSTM', 'Prophet (baseline)'],
    'MAE':   [round(lstm_mae, 4),  round(prophet_mae, 4)],
    'RMSE':  [round(lstm_rmse, 4), round(prophet_rmse, 4)]
})
print(metrics_df.style.highlight_min(subset=['MAE', 'RMSE'], color='#d4edda'))

# Save metrics
metrics_dict = {
    'zone': 1,
    'lstm_mae':     round(lstm_mae, 4),
    'lstm_rmse':    round(lstm_rmse, 4),
    'prophet_mae':  round(prophet_mae, 4),
    'prophet_rmse': round(prophet_rmse, 4)
}
with open('outputs/lstm_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print('Saved -> outputs/lstm_metrics.json')

# Plot: Actual vs LSTM predicted on test period
n = len(y_actual_cmp)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(range(n), y_actual_cmp, label='Actual',                             color='black',      linewidth=1.5)
ax.plot(range(n), y_pred_cmp,   label=f'LSTM  (MAE={lstm_mae:.3f})',        color='royalblue',  linewidth=1.5)
ax.plot(range(n), prophet_cmp,  label=f'Prophet baseline (MAE={prophet_mae:.3f})',
        color='darkorange', linewidth=1.2, linestyle='--')
ax.set_title('Congestion Index — Test Period: Actual vs LSTM vs Prophet (Zone 1)', fontsize=13, fontweight='bold')
ax.set_xlabel('Time step (test)')
ax.set_ylabel('Congestion Index')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/lstm_vs_actual_zone1.png', dpi=150)
plt.show()
print('Saved -> outputs/lstm_vs_actual_zone1.png')

zones = sorted(df['zone_sk'].unique())
zone_results = []

zone1_model_saved = False

for zone in zones:
    print(f'\n--- Zone {zone} ---')
    z_df = df[df['zone_sk'] == zone].sort_values('ds').reset_index(drop=True)
    vals = z_df['congestion_forecast'].values.reshape(-1, 1)

    # Scale
    scaler_z = MinMaxScaler()
    scaled_z  = scaler_z.fit_transform(vals).flatten()

    # Sequences
    X_z, y_z = create_sequences(scaled_z, WINDOW_SIZE)
    if len(X_z) < 30:
        print(f'  Skipping zone {zone}: insufficient data ({len(X_z)} samples)')
        continue

    # Split
    sp = int(len(X_z) * 0.8)
    Xtr, Xte  = X_z[:sp].reshape(-1, WINDOW_SIZE, 1), X_z[sp:].reshape(-1, WINDOW_SIZE, 1)
    ytr, yte  = y_z[:sp], y_z[sp:]

    # Build & Train
    tf.random.set_seed(42)
    m = build_lstm_model(WINDOW_SIZE)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    m.fit(Xtr, ytr, epochs=50, batch_size=16,
          validation_data=(Xte, yte), callbacks=[es], verbose=0)

    # Evaluate
    ypred_sc = m.predict(Xte, verbose=0)
    ypred    = scaler_z.inverse_transform(ypred_sc).flatten()
    yact     = scaler_z.inverse_transform(yte.reshape(-1, 1)).flatten()
    z_mae    = mean_absolute_error(yact, ypred)

    # Prophet baseline
    all_vals_z = z_df['congestion_forecast'].values
    roll_z     = pd.Series(all_vals_z).rolling(14, min_periods=1).mean().values
    t_start    = sp + WINDOW_SIZE
    prop_z     = roll_z[t_start: t_start + len(yact)]
    mn = min(len(yact), len(prop_z))
    z_prophet_mae = mean_absolute_error(yact[:mn], prop_z[:mn])

    zone_results.append({
        'zone_sk':     zone,
        'lstm_mae':    round(z_mae, 4),
        'prophet_mae': round(z_prophet_mae, 4)
    })
    print(f'  LSTM MAE={z_mae:.4f} | Prophet MAE={z_prophet_mae:.4f}')

    # Save zone 1 model and scaler
    if zone == 1 and not zone1_model_saved:
        m.save('outputs/lstm_congestion.keras')
        joblib.dump(scaler_z, 'outputs/lstm_scaler.pkl')
        zone1_model_saved = True
        print('  Model saved to outputs/lstm_congestion.keras')
        print('  Scaler saved to outputs/lstm_scaler.pkl')

# Save per-zone metrics
zone_metrics_df = pd.DataFrame(zone_results)
zone_metrics_df.to_csv('outputs/lstm_zone_metrics.csv', index=False)
print('\nSaved -> outputs/lstm_zone_metrics.csv')
print('\nSummary:')
print(zone_metrics_df.to_string(index=False))

# Reload zone1 model (robust — handles first-run and re-run scenarios)
try:
    lstm_z1 = tf.keras.models.load_model('outputs/lstm_congestion.keras')
    scaler_reload = joblib.load('outputs/lstm_scaler.pkl')
    print('Zone 1 model loaded from outputs/lstm_congestion.keras')
except Exception as e:
    print(f'Could not load saved model ({e}). Using in-memory model_z1.')
    lstm_z1       = model_z1
    scaler_reload = scaler_z1

# Seed: last 14 days of zone 1 scaled data
zone1_df_full = df[df['zone_sk'] == 1].sort_values('ds').reset_index(drop=True)
full_scaled   = scaler_reload.transform(zone1_df_full['congestion_forecast'].values.reshape(-1, 1)).flatten()
seed_window   = full_scaled[-WINDOW_SIZE:].tolist()

# Iterative 7-step forecast
HORIZON = 7
forecast_scaled = []
current_window  = seed_window.copy()

for _ in range(HORIZON):
    x_input = np.array(current_window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
    pred    = lstm_z1.predict(x_input, verbose=0)[0, 0]
    forecast_scaled.append(pred)
    current_window.append(pred)

# Inverse transform
forecast_values = scaler_reload.inverse_transform(
    np.array(forecast_scaled).reshape(-1, 1)
).flatten()

# Future dates
last_date      = zone1_df_full['ds'].max()
future_dates   = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON, freq='D')

print('7-day forecast (Zone 1):')
for d, v in zip(future_dates, forecast_values):
    print(f'  {d.date()}  ->  {v:.4f}')

# Plot: last 30 historical days + 7-day forecast
hist_30   = zone1_df_full.tail(30)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(hist_30['ds'], hist_30['congestion_forecast'],
        label='Historical (last 30 days)', color='royalblue', linewidth=2)
ax.plot(future_dates, forecast_values,
        label='LSTM 7-day Forecast', color='tomato', linewidth=2,
        linestyle='--', marker='o', markersize=5)
ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.2, label='Forecast start')
ax.set_title('Zone 1 — Congestion: Last 30 Days + 7-Day LSTM Forecast', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Congestion Index')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/lstm_forecast_7days.png', dpi=150)
plt.show()
print('Saved -> outputs/lstm_forecast_7days.png')

# Save 7-day forecast CSV
forecast_df = pd.DataFrame({
    'day':            [d.strftime('%Y-%m-%d') for d in future_dates],
    'zone_sk':        [1] * HORIZON,
    'forecast_value': [round(v, 4) for v in forecast_values]
})
forecast_df.to_csv('outputs/lstm_7day_forecast.csv', index=False)
print('Saved -> outputs/lstm_7day_forecast.csv')
forecast_df

