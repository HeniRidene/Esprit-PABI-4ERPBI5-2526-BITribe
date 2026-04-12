import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GENERATED_AT = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')


def evaluate_actor2(df_ml=None):
    """
    Central evaluation hub for Actor 2.

    - Reloads saved models and reproduces the exact same test split.
    - Reports Regression (RMSE, MAE, R²), Classification (F1, AUC, Precision, Recall,
      confusion matrix at two thresholds), and KPI (Taux de Ponctualité).
    - Reads Prophet holdout metrics from the cached classification_report.json if available.
    - Writes outputs/metrics_report.json.
    """
    logging.info("Starting central evaluation for Actor 2...")
    metrics = {
        'metadata': {
            'actor': 'Actor 2 — Directeur Mobilités',
            'execution_date': GENERATED_AT,
        }
    }

    # ── 1. Load data if not provided ─────────────────────────────────────────
    if df_ml is None:
        from data_loader import load_mobility_data
        from feature_engineering import engineer_features
        df_raw = load_mobility_data()
        df_ml, _ = engineer_features(df_raw)

    # ── 2. Regression — Passenger Load ───────────────────────────────────────
    try:
        from sklearn.model_selection import train_test_split
        reg_model    = joblib.load('outputs/xgboost_charge.pkl')
        reg_features = joblib.load('outputs/xgboost_charge_features.pkl')

        # Load target encoding maps (needed for line_charge_mean, zone_hour_charge)
        charge_enc = joblib.load('outputs/charge_encoding.pkl') if os.path.exists('outputs/charge_encoding.pkl') else None

        # Reproduce the exact base columns and split
        base_cols = ['zone_encoded', 'line_encoded', 'mode_encoded', 'hour', 'rush_hour',
                     'is_weekend', 'congestion_index', 'vitesse_kmh', 'temps_trajet_min',
                     'charge_estimee']
        df_reg = df_ml[[c for c in base_cols if c in df_ml.columns]].dropna()

        _, df_test_r = train_test_split(df_reg, test_size=0.2, shuffle=True, random_state=42)

        # Apply saved target encoding to test set
        if charge_enc is not None:
            gm = charge_enc['global_mean']
            df_test_r = df_test_r.copy()
            df_test_r['line_charge_mean'] = df_test_r['line_encoded'].map(charge_enc['line_charge_means']).fillna(gm)
            zh = charge_enc['zone_hour_means']
            df_test_r['zone_hour_charge'] = [zh.get((z, h), gm) for z, h in zip(df_test_r['zone_encoded'], df_test_r['hour'])]
            df_test_r['hour_x_congestion'] = df_test_r['hour'] * df_test_r['congestion_index']
            df_test_r['weekend_x_zone']    = df_test_r['is_weekend'] * df_test_r['zone_encoded']

        X_test_r = df_test_r[[f for f in reg_features if f in df_test_r.columns]]
        y_test_r = df_test_r['charge_estimee']
        preds_r  = reg_model.predict(X_test_r)
        rmse = float(np.sqrt(mean_squared_error(y_test_r, preds_r)))
        mae  = float(mean_absolute_error(y_test_r, preds_r))
        r2   = float(r2_score(y_test_r, preds_r))

        metrics['regression_charge'] = {
            'target': 'charge_estimee (passenger load factor)',
            'test_rows': int(len(X_test_r)),
            'rmse': round(rmse, 4),
            'mae':  round(mae, 4),
            'r2':   round(r2, 4),
        }
        logging.info(f"[Regression] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

    except Exception as e:
        logging.error(f"Regression evaluation failed: {e}")
        metrics['regression_charge'] = {'error': str(e)}

    # ── 3. Classification — Cancellation Risk ─────────────────────────────────
    try:
        clf_model    = joblib.load('outputs/xgboost_cancellation.pkl')
        clf_features = joblib.load('outputs/xgboost_cancellation_features.pkl')

        df_clf = df_ml[clf_features + ['retard_flag']].dropna()
        X_clf  = df_clf[clf_features]
        y_clf  = df_clf['retard_flag'].astype(int)

        try:
            _, X_test_c, _, y_test_c = train_test_split(
                X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
            )
        except ValueError:
            _, X_test_c, _, y_test_c = train_test_split(
                X_clf, y_clf, test_size=0.2, random_state=42
            )

        proba_c  = clf_model.predict_proba(X_test_c)[:, 1]
        preds_c  = (proba_c >= 0.50).astype(int)

        auc  = float(roc_auc_score(y_test_c, proba_c)) if y_test_c.sum() > 0 else 0.0
        f1   = float(f1_score(y_test_c, preds_c, zero_division=0))
        prec = float(precision_score(y_test_c, preds_c, zero_division=0))
        rec  = float(recall_score(y_test_c, preds_c, zero_division=0))
        cm   = confusion_matrix(y_test_c, preds_c).tolist()

        # Load best-threshold info from saved report
        best_thresh_info = {}
        if os.path.exists('outputs/classification_report.json'):
            with open('outputs/classification_report.json', 'r', encoding='utf-8') as f:
                clf_rpt = json.load(f)
            best_thresh_info = clf_rpt.get('threshold_best_f1', {})

        metrics['classification_annule'] = {
            'target': 'annule (trip cancellation, 0/1)',
            'test_rows':          int(len(X_test_c)),
            'pos_in_test':        int(y_test_c.sum()),
            'default_threshold': {
                'threshold':  0.50,
                'auc_roc':    round(auc, 4),
                'f1':         round(f1, 4),
                'precision':  round(prec, 4),
                'recall':     round(rec, 4),
                'confusion_matrix': cm,
            },
            'best_f1_threshold': best_thresh_info,
            'threshold_note': (
                'The threshold is a business decision. '
                'Lower → higher recall (fewer missed cancellations). '
                'Higher → higher precision (fewer false alarms to operators).'
            ),
        }
        logging.info(f"[Classification] AUC={auc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")

    except Exception as e:
        logging.error(f"Classification evaluation failed: {e}")
        metrics['classification_annule'] = {'error': str(e)}

    # ── 4. Time Series — Prophet Congestion ───────────────────────────────────
    try:
        ts_path = 'outputs/forecast_congestion.csv'
        if os.path.exists(ts_path):
            ts_df   = pd.read_csv(ts_path)
            n_zones = ts_df['zone_sk'].nunique() if 'zone_sk' in ts_df.columns else 0
            metrics['timeseries_congestion'] = {
                'model':     'Prophet (daily congestion per zone)',
                'zones':     int(n_zones),
                'rows':      int(len(ts_df)),
                'note':      'Holdout MAE per zone stored in classification_report.json',
            }
            logging.info(f"[Prophet] {n_zones} zones, {len(ts_df)} forecast rows")
        else:
            metrics['timeseries_congestion'] = {'error': 'forecast_congestion.csv not found'}
    except Exception as e:
        metrics['timeseries_congestion'] = {'error': str(e)}

    # ── 5. KPI — Taux de Ponctualité ─────────────────────────────────────────
    try:
        total  = len(df_ml)
        annule = int(df_ml['annule'].sum()) if 'annule' in df_ml.columns else 0
        ponctu = round((1 - annule / total) * 100, 4) if total > 0 else None
        metrics['kpi'] = {
            'taux_ponctualite_pct':  ponctu,
            'total_trips':           total,
            'cancelled_trips':       annule,
        }
        logging.info(f"[KPI] Taux de Ponctualité = {ponctu}%  ({annule}/{total} cancelled)")
    except Exception as e:
        metrics['kpi'] = {'error': str(e)}

    # ── 6. Overall status ─────────────────────────────────────────────────────
    has_errors = any('error' in v for v in metrics.values() if isinstance(v, dict))
    metrics['metadata']['overall_status'] = '❌ SOME FAILED' if has_errors else '✅ ALL PASSED'

    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/metrics_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logging.info(f"Evaluation complete. Report saved → {out_path}")

    return metrics


if __name__ == "__main__":
    m = evaluate_actor2()
    import json
    print(json.dumps(m, indent=2, ensure_ascii=False))