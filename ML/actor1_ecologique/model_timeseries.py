import os
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings
warnings.filterwarnings('ignore')   # suppress Prophet's verbose Stan output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# How many future months to forecast beyond the training data
FORECAST_HORIZON = 24   # 2 years — gives future-year predictions for Power BI


def _tune_prophet(df_zone_monthly, target_col, n_holdout=3):
    """
    Tries two changepoint_prior_scale values and picks the one with lower
    MAE on the last n_holdout months. Returns the best fitted model.
    """
    if len(df_zone_monthly) <= n_holdout + 6:
        # Not enough data to do a meaningful holdout — just use default
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        m.fit(df_zone_monthly.rename(columns={'annee_mois_dt': 'ds', target_col: 'y'}))
        return m

    train_df = df_zone_monthly.iloc[:-n_holdout]
    val_df   = df_zone_monthly.iloc[-n_holdout:]

    best_mae   = float('inf')
    best_scale = 0.05

    for scale in [0.05, 0.3]:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=scale,
        )
        m.fit(train_df.rename(columns={'annee_mois_dt': 'ds', target_col: 'y'}))
        fut = m.make_future_dataframe(periods=n_holdout, freq='MS')
        fc  = m.predict(fut).tail(n_holdout)
        mae = np.mean(np.abs(fc['yhat'].values - val_df[target_col].values))

        if mae < best_mae:
            best_mae   = mae
            best_scale = scale

    # Refit on the full series with the best scale
    m_final = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=best_scale,
    )
    m_final.fit(df_zone_monthly.rename(columns={'annee_mois_dt': 'ds', target_col: 'y'}))
    logging.info(f"    Best changepoint_prior_scale={best_scale} (holdout MAE={best_mae:.3f})")
    return m_final


def train_prophet_forecasts(df_prophet):
    """
    Trains Prophet models for AQI, PM2.5, CO2, and Energy per zone.

    Improvements over v1:
    - Clamp negative forecast values to 0 (pollution can't be negative).
    - Lightweight hyperparameter tuning (changepoint_prior_scale).
    - Extended forecast horizon (FORECAST_HORIZON months = 2 years).
    - Saves per-zone forecast plots for visual validation.
    - Returns both historical fitted values AND future forecasts in one flat CSV.
    """
    logging.info(f"Initializing Prophet Time Series Pipeline v2 (horizon={FORECAST_HORIZON} months)...")
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/prophet_plots', exist_ok=True)

    df = df_prophet.copy()
    df['annee_mois_dt'] = pd.to_datetime(df['annee_mois_dt'])

    all_forecasts = []
    # (zone_sk, target) → holdout MAE for evaluate.py
    holdout_metrics = {}

    unique_zones = sorted(df['zone_sk'].unique())
    logging.info(f"Found {len(unique_zones)} unique zones.")

    targets = {
        'aqi_index':   ('aqi_forecast',   'aqi_lower',   'aqi_upper'),
        'pm25':        ('pm25_forecast',   'pm25_lower',  'pm25_upper'),
        'co2_kg':      ('co2_forecast',    'co2_lower',   'co2_upper'),
        'energie_kwh': ('nrj_forecast',    'nrj_lower',   'nrj_upper'),
    }

    for zone in unique_zones:
        logging.info(f"  → Zone {zone}")
        zone_df = df[df['zone_sk'] == zone].copy()

        # Aggregate to monthly means (one row per month)
        zone_monthly = (
            zone_df
            .groupby('annee_mois_dt')[list(targets.keys())]
            .mean()
            .reset_index()
            .sort_values('annee_mois_dt')
        )

        if len(zone_monthly) < 18:
            logging.warning(f"    Zone {zone}: only {len(zone_monthly)} months — skipping (need ≥18).")
            continue

        zone_forecast_parts = []

        for target_col, (fc_col, lo_col, hi_col) in targets.items():
            sub = zone_monthly[['annee_mois_dt', target_col]].dropna()
            if len(sub) < 18:
                continue

            try:
                m = _tune_prophet(sub, target_col, n_holdout=3)

                future = m.make_future_dataframe(periods=FORECAST_HORIZON, freq='MS')
                fc_df  = m.predict(future)

                # Clamp to 0 — pollution metrics can't go negative
                fc_df['yhat']       = fc_df['yhat'].clip(lower=0)
                fc_df['yhat_lower'] = fc_df['yhat_lower'].clip(lower=0)
                fc_df['yhat_upper'] = fc_df['yhat_upper'].clip(lower=0)

                # Mark rows as historical or future
                last_hist_date = sub['annee_mois_dt'].max()
                fc_df['row_type'] = np.where(fc_df['ds'] <= last_hist_date, 'historical', 'forecast')

                fc_out = fc_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'row_type']].rename(columns={
                    'yhat':       fc_col,
                    'yhat_lower': lo_col,
                    'yhat_upper': hi_col,
                })

                # Compute holdout MAE on last 3 historical months
                hist_fc   = fc_out[fc_out['row_type'] == 'historical'].tail(3)
                hist_act  = sub[sub['annee_mois_dt'].isin(hist_fc['ds'])][target_col].values
                if len(hist_act) == 3:
                    mae = np.mean(np.abs(hist_fc[fc_col].values - hist_act))
                    holdout_metrics[(zone, target_col)] = round(float(mae), 4)

                zone_forecast_parts.append(fc_out)

                # Save per-zone per-target plot
                fig = m.plot(m.predict(future))
                plt.title(f"Zone {zone} — {target_col} forecast")
                plt.savefig(f'outputs/prophet_plots/zone{zone}_{target_col}.png', dpi=100, bbox_inches='tight')
                plt.close()

            except Exception as e:
                logging.warning(f"    Zone {zone} / {target_col} failed: {e}")

        if zone_forecast_parts:
            # Merge all targets on 'ds' and 'row_type'
            merged = zone_forecast_parts[0]
            for part in zone_forecast_parts[1:]:
                merge_on = [c for c in ['ds', 'row_type'] if c in part.columns]
                merged = pd.merge(merged, part, on=merge_on, how='outer')
            merged.insert(0, 'zone_sk', zone)
            all_forecasts.append(merged)

    if not all_forecasts:
        logging.error("No forecasts generated. Check data density.")
        return None, {}

    final_df = pd.concat(all_forecasts, ignore_index=True)

    # Round float columns
    float_cols = [c for c in final_df.columns if final_df[c].dtype == float]
    final_df[float_cols] = final_df[float_cols].round(4)

    export_path = 'outputs/prophet_forecasts.csv'
    final_df.to_csv(export_path, index=False)
    logging.info(f"Forecasts saved → {export_path}  shape={final_df.shape}")

    # Save holdout metrics for evaluate.py
    import json
    with open('outputs/prophet_holdout_metrics.json', 'w') as f:
        json.dump({str(k): v for k, v in holdout_metrics.items()}, f, indent=2)

    return final_df, holdout_metrics


if __name__ == "__main__":
    from data_loader import load_and_clean_data
    from feature_engineering import engineer_features

    try:
        df_clean = load_and_clean_data()
        _, _, df_prophet = engineer_features(df_clean)

        forecast_df, metrics = train_prophet_forecasts(df_prophet)
        if forecast_df is not None:
            print("\nPreview of Prophet Forecast Output:")
            print(forecast_df[forecast_df['row_type'] == 'forecast'].head(12))
            print("\nHoldout MAE by zone/target:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.3f}")
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise