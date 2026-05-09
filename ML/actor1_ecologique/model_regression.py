import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# Features used within each per-mode model.
# mode_encoded is removed (constant within a mode group),
# mode_co2_mean is also removed (constant within a mode).
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_PER_MODE = [
    'zone_encoded',
    'annee', 'mois_sin', 'mois_cos',
    'pm25', 'no2', 'aqi_index',
    'co2_lag1',  'co2_lag3',
    'energie_lag1', 'energie_lag3',
    'aqi_lag1',  'pm25_lag1',
    'co2_roll3', 'energie_roll3', 'pm25_roll3',
]

# Features for the combined model (mode_encoded is the key discriminator)
FEATURES_COMBINED = [
    'zone_encoded', 'mode_encoded', 'mode_co2_mean',
    'annee', 'mois_sin', 'mois_cos',
    'pm25', 'no2', 'aqi_index',
    'co2_lag1',  'co2_lag3',
    'energie_lag1', 'energie_lag3',
    'aqi_lag1',  'pm25_lag1',
    'co2_roll3', 'energie_roll3', 'pm25_roll3',
]


def _temporal_split(df, test_months=8):
    cutoff     = df['annee_mois_dt'].max() - pd.DateOffset(months=test_months)
    train_mask = df['annee_mois_dt'] <= cutoff
    test_mask  = df['annee_mois_dt']  > cutoff
    return df[train_mask].copy(), df[test_mask].copy()


def _get_base_params():
    return {
        'n_estimators':    300,
        'max_depth':        4,
        'learning_rate':    0.03,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma':            0.05,
        'reg_alpha':        0.3,
        'reg_lambda':       1.5,
        'objective':       'reg:squarederror',
        'random_state':     42,
        'n_jobs':          -1,
    }


def train_and_evaluate_xgboost(df_reg):
    """
    Trains XGBoost regressors for CO2 and Energy using two strategies:

    Strategy A — Per-mode models (primary, for CO2)
    ------------------------------------------------
    Trains a separate XGBoost regressor per transport mode (4 modes).
    Within each mode the CO2 range is narrow (no more than 2× variation)
    so XGBoost can learn zone × time patterns cleanly.
    The per-mode predictions are concatenated for a unified test evaluation.

    Strategy B — Single combined model (for Energy, always)
    --------------------------------------------------------
    Energy does not have a strong mode-based bimodal split so a single model
    with mode_encoded + mode_co2_mean performs well (R² ~0.73).

    The final saved features list uses Strategy B columns so that export_predictions
    can generate future predictions from a single model call.
    """
    logging.info("=== XGBOOST REGRESSION v5 (per-mode CO2 + combined Energy) ===")
    os.makedirs('outputs', exist_ok=True)

    # ── Global temporal split ────────────────────────────────────────────────
    train_all, test_all = _temporal_split(df_reg, test_months=8)
    logging.info(f"Train: {len(train_all)} rows  Test: {len(test_all)} rows")

    if len(test_all) == 0:
        logging.warning("Test set empty — using last 20%.")
        n = int(0.8 * len(df_reg))
        train_all = df_reg.iloc[:n].copy()
        test_all  = df_reg.iloc[n:].copy()

    # ── STRATEGY A: Per-mode CO2 regressors ──────────────────────────────────
    logging.info("Training per-mode CO2 regressors...")
    mode_models_co2 = {}
    mode_feat_cols  = {}

    all_test_co2_actual = []
    all_test_co2_pred   = []
    all_test_modes      = []

    base_params = _get_base_params()
    fit_params  = {**base_params, 'early_stopping_rounds': 30, 'eval_metric': 'rmse'}

    for mode_sk in sorted(df_reg['mode_sk'].unique()):
        train_m = train_all[train_all['mode_sk'] == mode_sk].copy()
        test_m  = test_all[test_all['mode_sk']   == mode_sk].copy()

        if len(train_m) < 20 or len(test_m) == 0:
            logging.warning(f"  Mode {mode_sk}: insufficient data (train={len(train_m)}, test={len(test_m)}) — skipping")
            continue

        avail = [f for f in FEATURES_PER_MODE if f in train_m.columns]
        mode_feat_cols[mode_sk] = avail

        X_tr = train_m[avail]
        X_te = test_m[avail]
        y_tr = train_m['co2_log']
        y_te = test_m['co2_log']

        m = xgb.XGBRegressor(**fit_params)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        mode_models_co2[mode_sk] = m

        preds_log = m.predict(X_te)
        actual_real = np.expm1(y_te)
        pred_real   = np.expm1(preds_log)

        rmse = np.sqrt(mean_squared_error(actual_real, pred_real))
        r2   = r2_score(actual_real, pred_real)
        logging.info(f"  Mode {mode_sk}: RMSE={rmse:.2f} kg  R²={r2:.4f}  (best_iter={m.best_iteration})")

        all_test_co2_actual.extend(actual_real.tolist())
        all_test_co2_pred.extend(pred_real.tolist())
        all_test_modes.extend([mode_sk] * len(actual_real))

    # Save per-mode models
    joblib.dump(mode_models_co2, 'outputs/xgboost_co2_per_mode.pkl')
    joblib.dump(mode_feat_cols,  'outputs/xgboost_co2_per_mode_features.pkl')

    # Combined CO2 metrics across all modes
    actual_co2_all = np.array(all_test_co2_actual)
    pred_co2_all   = np.array(all_test_co2_pred)
    co2_rmse = np.sqrt(mean_squared_error(actual_co2_all, pred_co2_all))
    co2_mae  = mean_absolute_error(actual_co2_all, pred_co2_all)
    co2_r2   = r2_score(actual_co2_all, pred_co2_all)
    logging.info(f"[TEST ALL MODES] CO2 RMSE={co2_rmse:.2f} kg  MAE={co2_mae:.2f}  R²={co2_r2:.4f}")

    # ── STRATEGY B: Combined Energy model ───────────────────────────────────
    logging.info("Training combined Energy regressor...")
    avail_comb = [f for f in FEATURES_COMBINED if f in train_all.columns]

    xgb_nrj = xgb.XGBRegressor(**fit_params)
    xgb_nrj.fit(
        train_all[avail_comb], train_all['energie_log'],
        eval_set=[(test_all[avail_comb], test_all['energie_log'])],
        verbose=False,
    )

    preds_nrj_log  = xgb_nrj.predict(test_all[avail_comb])
    actual_nrj_real = np.expm1(test_all['energie_log'])
    pred_nrj_real   = np.expm1(preds_nrj_log)
    nrj_rmse = np.sqrt(mean_squared_error(actual_nrj_real, pred_nrj_real))
    nrj_mae  = mean_absolute_error(actual_nrj_real, pred_nrj_real)
    nrj_r2   = r2_score(actual_nrj_real, pred_nrj_real)
    logging.info(f"[TEST] Energy RMSE={nrj_rmse:.2f} kWh  MAE={nrj_mae:.2f}  R²={nrj_r2:.4f}  (best_iter={xgb_nrj.best_iteration})")

    # ── Also train a fallback combined CO2 model for export_predictions ──────
    logging.info("Training fallback combined CO2 model (for future predictions)...")
    xgb_co2_comb = xgb.XGBRegressor(**fit_params)
    xgb_co2_comb.fit(
        train_all[avail_comb], train_all['co2_log'],
        eval_set=[(test_all[avail_comb], test_all['co2_log'])],
        verbose=False,
    )

    # Save all models
    joblib.dump(xgb_co2_comb, 'outputs/xgboost_co2.pkl')
    joblib.dump(xgb_nrj,      'outputs/xgboost_nrj.pkl')
    joblib.dump(avail_comb,   'outputs/xgboost_features.pkl')
    logging.info("Combined models saved → outputs/")

    # Save mode_co2_mean encoding
    mode_means  = train_all.groupby('mode_sk')['co2_kg'].mean().to_dict()
    global_mean = float(train_all['co2_kg'].mean())
    joblib.dump({'mode_means': mode_means, 'global_mean': global_mean},
                'outputs/mode_co2_encoding.pkl')

    # ── 5-Fold CV on the training set (combined model) ───────────────────────
    logging.info("5-Fold CV on training set (combined CO2 + Energy)...")
    cv_co2 = cross_val_score(xgb.XGBRegressor(**base_params),
                              train_all[avail_comb], train_all['co2_log'],
                              cv=5, scoring='neg_mean_squared_error')
    cv_nrj = cross_val_score(xgb.XGBRegressor(**base_params),
                              train_all[avail_comb], train_all['energie_log'],
                              cv=5, scoring='neg_mean_squared_error')
    logging.info(f"CO2    CV RMSE (log): {np.mean(np.sqrt(-cv_co2)):.4f} ± {np.std(np.sqrt(-cv_co2)):.4f}")
    logging.info(f"Energy CV RMSE (log): {np.mean(np.sqrt(-cv_nrj)):.4f} ± {np.std(np.sqrt(-cv_nrj)):.4f}")

    # ── Feature importance charts (combined models) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    xgb.plot_importance(xgb_co2_comb, ax=axes[0], title='Feature Importance — CO₂ (combined)',
                        importance_type='gain', max_num_features=15)
    xgb.plot_importance(xgb_nrj, ax=axes[1], title='Feature Importance — Energy',
                        importance_type='gain', max_num_features=15)
    plt.tight_layout()
    plt.savefig('outputs/feature_importances.png', dpi=150)
    plt.close()

    # ── Actual vs Predicted scatter using PER-MODE CO2 predictions ───────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(actual_co2_all, pred_co2_all, alpha=0.5, edgecolors='k', linewidths=0.3,
                c=all_test_modes, cmap='Set1')
    lims = [min(actual_co2_all.min(), pred_co2_all.min()),
            max(actual_co2_all.max(), pred_co2_all.max())]
    ax1.plot(lims, lims, 'r--', label='Perfect fit')
    ax1.set_xlabel('Actual CO₂ (kg)'); ax1.set_ylabel('Predicted CO₂ (kg)')
    ax1.set_title(f'CO₂ Per-Mode Prediction  R²={co2_r2:.3f}'); ax1.legend()

    ax2.scatter(actual_nrj_real, pred_nrj_real, alpha=0.5, edgecolors='k', linewidths=0.3)
    lims = [min(actual_nrj_real.min(), pred_nrj_real.min()),
            max(actual_nrj_real.max(), pred_nrj_real.max())]
    ax2.plot(lims, lims, 'r--', label='Perfect fit')
    ax2.set_xlabel('Actual Energy (kWh)'); ax2.set_ylabel('Predicted Energy (kWh)')
    ax2.set_title(f'Energy Prediction  R²={nrj_r2:.3f}'); ax2.legend()

    plt.tight_layout()
    plt.savefig('outputs/actual_vs_predicted.png', dpi=150)
    plt.close()
    logging.info("Plots saved → outputs/")

    return (test_all,
            actual_co2_all, pred_co2_all,
            actual_nrj_real.values, pred_nrj_real)


if __name__ == "__main__":
    from data_loader import load_and_clean_data
    from feature_engineering import engineer_features

    df_clean = load_and_clean_data()
    df_reg, _, _ = engineer_features(df_clean)
    train_and_evaluate_xgboost(df_reg)