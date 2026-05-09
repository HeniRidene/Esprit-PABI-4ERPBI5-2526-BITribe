import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_evaluation():
    """
    Central evaluation script for Actor 1.
    - Rebuilds the same temporal split used by model_regression.py (no data leakage).
    - Loads clustering silhouette score dynamically from the saved model.
    - Loads Prophet holdout MAE from the JSON written by model_timeseries.py.
    - Writes a single clean metrics_report.json for the professor.
    """
    logging.info("Starting central evaluation for Actor 1...")

    metrics   = {}
    report_ok = True

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Regression Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from data_loader import load_and_clean_data
        from feature_engineering import engineer_features

        df_clean = load_and_clean_data()
        df_reg, _, _ = engineer_features(df_clean)

        # Reload the exact feature list that was used during training
        feature_path = 'outputs/xgboost_features.pkl'
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"{feature_path} missing — run model_regression.py first.")
        features = joblib.load(feature_path)

        xgb_co2 = joblib.load('outputs/xgboost_co2.pkl')
        xgb_nrj = joblib.load('outputs/xgboost_nrj.pkl')

        # Reconstruct the temporal test set (last 8 months — matches model_regression v4)
        cutoff    = df_reg['annee_mois_dt'].max() - pd.DateOffset(months=8)
        train_df  = df_reg[df_reg['annee_mois_dt'] <= cutoff].copy()
        test_df   = df_reg[df_reg['annee_mois_dt']  > cutoff].copy()

        if len(test_df) == 0:
            raise ValueError("Test set is empty — not enough temporal range.")

        # Add mode_co2_mean (target encoding, train-only) if used
        mode_enc_path = 'outputs/mode_co2_encoding.pkl'
        if os.path.exists(mode_enc_path):
            mode_enc = joblib.load(mode_enc_path)
            test_df['mode_co2_mean'] = test_df['mode_sk'].map(mode_enc['mode_means']).fillna(mode_enc['global_mean'])

        X_test        = test_df[[f for f in features if f in test_df.columns]]
        y_co2_test    = test_df['co2_log']
        y_nrj_test    = test_df['energie_log']

        # ── CO2: use per-mode models if available ────────────────────────────
        per_mode_path     = 'outputs/xgboost_co2_per_mode.pkl'
        per_mode_feat_path = 'outputs/xgboost_co2_per_mode_features.pkl'

        if os.path.exists(per_mode_path) and os.path.exists(per_mode_feat_path):
            mode_models_co2 = joblib.load(per_mode_path)
            mode_feat_cols  = joblib.load(per_mode_feat_path)
            all_actual_co2 = []
            all_pred_co2   = []
            for mode_sk, m_model in mode_models_co2.items():
                m_test = test_df[test_df['mode_sk'] == mode_sk]
                if len(m_test) == 0:
                    continue
                m_feat = mode_feat_cols.get(mode_sk, [f for f in features if f in m_test.columns])
                m_feat = [f for f in m_feat if f in m_test.columns]
                preds  = np.expm1(m_model.predict(m_test[m_feat]))
                actual = np.expm1(m_test['co2_log'])
                all_actual_co2.extend(actual.tolist())
                all_pred_co2.extend(preds.tolist())
            actual_co2 = np.array(all_actual_co2)
            pred_co2   = np.array(all_pred_co2)
        else:
            preds_co2_log = xgb_co2.predict(X_test)
            actual_co2 = np.expm1(y_co2_test)
            pred_co2   = np.expm1(preds_co2_log)

        # ── Energy: combined model ────────────────────────────────────────────
        preds_nrj_log = xgb_nrj.predict(X_test)
        actual_nrj = np.expm1(y_nrj_test)
        pred_nrj   = np.expm1(preds_nrj_log)

        def safe_mape(actual, pred):
            mask = actual > 0
            if mask.sum() == 0:
                return None
            return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

        metrics['regression'] = {
            'test_rows':       int(len(test_df)),
            'train_rows':      int(len(df_reg) - len(test_df)),
            'test_period_start': df_reg[df_reg['annee_mois_dt'] > cutoff]['annee_mois_dt'].min().strftime('%Y-%m'),
            'test_period_end':   df_reg['annee_mois_dt'].max().strftime('%Y-%m'),
            'co2': {
                'rmse_kg':  round(float(np.sqrt(mean_squared_error(actual_co2, pred_co2))), 4),
                'mae_kg':   round(float(mean_absolute_error(actual_co2, pred_co2)), 4),
                'r2':       round(float(r2_score(actual_co2, pred_co2)), 4),
                'mape_pct': round(safe_mape(actual_co2, pred_co2) or -1, 2),
            },
            'energie': {
                'rmse_kwh': round(float(np.sqrt(mean_squared_error(actual_nrj, pred_nrj))), 4),
                'mae_kwh':  round(float(mean_absolute_error(actual_nrj, pred_nrj)), 4),
                'r2':       round(float(r2_score(actual_nrj, pred_nrj)), 4),
                'mape_pct': round(safe_mape(actual_nrj.values, pred_nrj) or -1, 2),
            },
        }
        logging.info(f"[Regression] CO2   R²={metrics['regression']['co2']['r2']}  RMSE={metrics['regression']['co2']['rmse_kg']} kg")
        logging.info(f"[Regression] Energy R²={metrics['regression']['energie']['r2']}  RMSE={metrics['regression']['energie']['rmse_kwh']} kWh")

    except Exception as e:
        logging.error(f"Regression evaluation failed: {e}")
        metrics['regression'] = {'status': 'FAILED', 'error': str(e)}
        report_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Clustering Evaluation (dynamically from saved model)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        kmeans_path = 'outputs/kmeans_pollution_zones.pkl'
        scaler_path = 'outputs/clustering_scaler.pkl'

        if not os.path.exists(kmeans_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Clustering model or scaler not found — run model_clustering.py first.")

        kmeans = joblib.load(kmeans_path)
        scaler = joblib.load(scaler_path)

        df_clean_c = load_and_clean_data()   # already has UNKNOWN zone removed
        features_c = ['pm25', 'no2', 'co2_kg', 'energie_kwh']
        zone_profiles = (
            df_clean_c
            .replace({'co2_kg': {0: np.nan}, 'energie_kwh': {0: np.nan}})
            .dropna(subset=features_c)
            .groupby('zone_sk')[features_c]
            .mean()
            .reset_index()
        )

        X_cluster = scaler.transform(zone_profiles[features_c])
        labels     = kmeans.predict(X_cluster)
        sil        = silhouette_score(X_cluster, labels)

        metrics['clustering'] = {
            'n_zones':          int(len(zone_profiles)),
            'optimal_k':        int(kmeans.n_clusters),
            'silhouette_score': round(float(sil), 4),
            'inertia':          round(float(kmeans.inertia_), 4),
        }
        logging.info(f"[Clustering] K={kmeans.n_clusters}  Silhouette={sil:.4f}  Inertia={kmeans.inertia_:.2f}")

    except Exception as e:
        logging.error(f"Clustering evaluation failed: {e}")
        metrics['clustering'] = {'status': 'FAILED', 'error': str(e)}
        report_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Time-Series Evaluation (holdout MAE from Prophet run)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        holdout_path = 'outputs/prophet_holdout_metrics.json'
        if not os.path.exists(holdout_path):
            raise FileNotFoundError("Prophet holdout metrics not found — run model_timeseries.py first.")

        with open(holdout_path) as f:
            raw = json.load(f)

        # The keys look like "('1', 'aqi_index')" — parse them
        parsed = {}
        for k, v in raw.items():
            parsed[k] = v

        # Summary stats
        aqi_maes   = [v for k, v in parsed.items() if 'aqi_index'   in k]
        pm25_maes  = [v for k, v in parsed.items() if 'pm25'        in k]
        co2_maes   = [v for k, v in parsed.items() if 'co2_kg'      in k]
        nrj_maes   = [v for k, v in parsed.items() if 'energie_kwh' in k]

        metrics['timeseries_prophet'] = {
            'holdout_months':   3,
            'aqi_mean_mae':    round(np.mean(aqi_maes),  4) if aqi_maes  else None,
            'pm25_mean_mae':   round(np.mean(pm25_maes), 4) if pm25_maes else None,
            'co2_mean_mae':    round(np.mean(co2_maes),  4) if co2_maes  else None,
            'energie_mean_mae':round(np.mean(nrj_maes),  4) if nrj_maes  else None,
            'detail':          parsed,
        }
        logging.info(f"[Prophet] AQI holdout MAE={metrics['timeseries_prophet']['aqi_mean_mae']}")

    except Exception as e:
        logging.error(f"Prophet evaluation failed: {e}")
        metrics['timeseries_prophet'] = {'status': 'FAILED', 'error': str(e)}
        report_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # 4. KPI Summary
    # ─────────────────────────────────────────────────────────────────────────
    try:
        df_clean_kpi = load_and_clean_data()
        df_clean_kpi['co2_kg']      = df_clean_kpi['co2_kg'].replace(0, np.nan)
        df_clean_kpi['energie_kwh'] = df_clean_kpi['energie_kwh'].replace(0, np.nan)
        valid_mask = df_clean_kpi['energie_kwh'] > 0
        ci = df_clean_kpi.loc[valid_mask, 'co2_kg'] / df_clean_kpi.loc[valid_mask, 'energie_kwh']

        metrics['kpi'] = {
            'carbon_intensity_mean':   round(float(ci.mean()),   4),
            'carbon_intensity_median': round(float(ci.median()), 4),
            'carbon_intensity_p90':    round(float(ci.quantile(0.9)), 4),
            'co2_total_kg':            round(float(df_clean_kpi['co2_kg'].sum()), 2),
            'energie_total_kwh':       round(float(df_clean_kpi['energie_kwh'].sum()), 2),
        }

    except Exception as e:
        logging.error(f"KPI summary failed: {e}")
        metrics['kpi'] = {'status': 'FAILED', 'error': str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Write report
    # ─────────────────────────────────────────────────────────────────────────
    report = {
        'metadata': {
            'actor':            'Actor 1 — Transition Écologique',
            'execution_date':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status':   '✅ ALL PASSED' if report_ok else '⚠️ SOME STEPS FAILED',
        },
        'metrics': metrics,
    }

    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/metrics_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logging.info(f"Evaluation complete. Report saved → {out_path}")
    return report


if __name__ == "__main__":
    run_evaluation()