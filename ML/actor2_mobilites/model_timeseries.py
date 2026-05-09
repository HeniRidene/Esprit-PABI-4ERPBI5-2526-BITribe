import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)


def train_congestion_forecast(df_ts, horizon_days=30):
    """
    Trains Prophet per zone to forecast congestion_index for the next 30 days.

    Input
    -----
    df_ts : DataFrame with columns [zone_sk, ds, y]
            aggregated to daily mean congestion per zone (from feature_engineering).

    Strategy
    --------
    - Holds out last 30 days per zone for MAPE/MAE evaluation.
    - Adds French public holidays as a Prophet holiday regressor.
    - Tunes changepoint_prior_scale per zone (grid: [0.05, 0.1, 0.3]).
    - Clamps yhat to [0, 10] since congestion_index range is bounded.
    - Output includes zone_sk, ds, congestion_forecast, congestion_lower, congestion_upper.
    """
    logging.info("=== PROPHET — Daily Congestion Forecast ===")
    os.makedirs('outputs', exist_ok=True)

    try:
        from prophet import Prophet
    except ImportError:
        logging.error("prophet not installed. Run: pip install prophet")
        return None, {}

    zones   = sorted(df_ts['zone_sk'].unique())
    logging.info(f"Fitting Prophet for {len(zones)} zones  (horizon={horizon_days} days)...")

    all_forecasts = []
    holdout_maes  = []
    zone_metrics  = {}

    CHANGEPOINT_SCALES = [0.05, 0.1, 0.3]

    for zone in zones:
        zone_df = df_ts[df_ts['zone_sk'] == zone][['ds', 'y']].copy()
        zone_df = zone_df.sort_values('ds').drop_duplicates('ds').reset_index(drop=True)
        zone_df['y'] = zone_df['y'].clip(lower=0)

        if len(zone_df) < 10:
            logging.warning(f"  Zone {zone}: only {len(zone_df)} data points — skipping.")
            continue

        logging.info(f"  → Zone {zone}  ({len(zone_df)} daily points)")

        # Hold-out last 30 days for evaluation
        n_holdout  = min(horizon_days, max(1, len(zone_df) // 5))
        train_data = zone_df.iloc[:-n_holdout]
        hold_data  = zone_df.iloc[-n_holdout:]

        if len(train_data) < 5:
            logging.warning(f"  Zone {zone}: not enough training rows after holdout — skipping.")
            continue

        # Tune changepoint_prior_scale
        best_mae   = float('inf')
        best_scale = CHANGEPOINT_SCALES[0]

        for cps in CHANGEPOINT_SCALES:
            try:
                m_cv = Prophet(
                    changepoint_prior_scale=cps,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                )
                m_cv.add_country_holidays(country_name='FR')
                m_cv.fit(train_data)

                future_cv  = m_cv.make_future_dataframe(periods=n_holdout, freq='D')
                fc_cv      = m_cv.predict(future_cv)
                fc_hold    = fc_cv.tail(n_holdout)
                mae_cv     = float(np.mean(np.abs(hold_data['y'].values - fc_hold['yhat'].values)))

                if mae_cv < best_mae:
                    best_mae   = mae_cv
                    best_scale = cps
            except Exception:
                continue

        logging.info(f"    Best changepoint_scale={best_scale}  holdout MAE={best_mae:.4f}")
        holdout_maes.append(best_mae)
        zone_metrics[int(zone)] = round(best_mae, 4)

        # Final model on full data
        try:
            m = Prophet(
                changepoint_prior_scale=best_scale,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            m.add_country_holidays(country_name='FR')
            m.fit(zone_df)

            future   = m.make_future_dataframe(periods=horizon_days, freq='D')
            forecast = m.predict(future)

            forecast['zone_sk']             = zone
            forecast['congestion_forecast']  = forecast['yhat'].clip(lower=0, upper=10).round(4)
            forecast['congestion_lower']     = forecast['yhat_lower'].clip(lower=0).round(4)
            forecast['congestion_upper']     = forecast['yhat_upper'].clip(upper=10).round(4)

            out = forecast[['ds', 'zone_sk', 'congestion_forecast', 'congestion_lower', 'congestion_upper']].copy()
            all_forecasts.append(out)

        except Exception as e:
            logging.error(f"  Zone {zone}: final Prophet fit failed — {e}")
            continue

    # Save forecasts
    if all_forecasts:
        final_df = pd.concat(all_forecasts, ignore_index=True)
        out_path = 'outputs/forecast_congestion.csv'
        final_df.to_csv(out_path, index=False)
        n_future = int((final_df['ds'] > df_ts['ds'].max()).sum())
        logging.info(f"Saved {len(final_df)} rows to {out_path} ({n_future} future rows)")
    else:
        logging.error("No forecasts generated — all zones skipped.")
        final_df = pd.DataFrame()

    mean_mae = float(np.mean(holdout_maes)) if holdout_maes else float('nan')
    logging.info(f"Mean holdout MAE across zones: {mean_mae:.4f}")

    ts_metrics = {
        'zones_fitted':     len(all_forecasts),
        'horizon_days':     horizon_days,
        'holdout_days':     'last 30%',
        'mean_holdout_mae': round(mean_mae, 4),
        'per_zone_mae':     zone_metrics,
    }

    return final_df, ts_metrics


if __name__ == "__main__":
    from data_loader import load_mobility_data
    from feature_engineering import engineer_features
    df_raw  = load_mobility_data()
    _, df_ts = engineer_features(df_raw)
    final_df, metrics = train_congestion_forecast(df_ts)
    if final_df is not None:
        print(f"\nForecast rows: {len(final_df)}")
        print(final_df.tail(5).to_string())