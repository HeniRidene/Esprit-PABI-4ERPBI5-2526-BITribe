import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES = [
    'zone_encoded', 'line_encoded', 'mode_encoded',
    'hour', 'rush_hour', 'is_weekend',
    'congestion_index', 'vitesse_kmh', 'temps_trajet_min',
    # Target-encoded features (train-only, no leakage)
    'line_charge_mean',   # mean charge_estimee per line (strong: line identity drives load)
    'zone_hour_charge',   # mean charge per zone×hour stratum
    # Interaction features
    'hour_x_congestion',  # congestion during peak hours
    'weekend_x_zone',     # weekend load differs by zone
]
TARGET = 'charge_estimee'


def _add_target_features(train_df, test_df):
    """
    Adds target-encoded and interaction features (computed on train only = no leakage).
      line_charge_mean  — mean charge per line_encoded
      zone_hour_charge  — mean charge per zone_encoded x hour stratum
      hour_x_congestion — multiplicative interaction
      weekend_x_zone    — multiplicative interaction
    """
    line_means  = train_df.groupby('line_encoded')['charge_estimee'].mean()
    global_mean = float(train_df['charge_estimee'].mean())

    zh_means = train_df.groupby(['zone_encoded', 'hour'])['charge_estimee'].mean().to_dict()

    for df in [train_df, test_df]:
        df = df.copy()
        df['line_charge_mean']  = df['line_encoded'].map(line_means).fillna(global_mean)
        df['zone_hour_charge']  = [
            zh_means.get((z, h), global_mean)
            for z, h in zip(df['zone_encoded'], df['hour'])
        ]
        df['hour_x_congestion'] = df['hour'] * df['congestion_index']
        df['weekend_x_zone']    = df['is_weekend'] * df['zone_encoded']

        if df is train_df:
            train_df = df
        else:
            test_df = df

    return train_df, test_df


def train_load_regression(df_ml):
    """
    Trains an XGBoost regressor to predict charge_estimee (passenger load factor).

    Strategy
    --------
    - 80/20 random split, then target-encoded features added (train-only, no leakage).
    - n_estimators=300, max_depth=6 — can afford deeper model with 4k+ rows.
    - 5-fold CV on training set for stable metric estimation.
    - Returns X_test, y_test, preds for downstream evaluate.py use.
    - Saves feature importance plot and actual-vs-predicted scatter.
    """
    logging.info("=== REGRESSION — Passenger Load (charge_estimee) ===")
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Feature selection ──────────────────────────────────────────────────
    base_cols = ['zone_encoded', 'line_encoded', 'mode_encoded', 'hour', 'rush_hour',
                 'is_weekend', 'congestion_index', 'vitesse_kmh', 'temps_trajet_min',
                 TARGET]
    df_clean = df_ml[[c for c in base_cols if c in df_ml.columns]].dropna()
    logging.info(f"Regression set: {len(df_clean)} rows ({len(df_ml)-len(df_clean)} dropped for NaN)")

    if TARGET not in df_clean.columns:
        raise ValueError(f"Target '{TARGET}' not found in df_ml.")

    # ── 2. Train/Test split ───────────────────────────────────────────────────
    df_train, df_test = train_test_split(
        df_clean, test_size=0.2, shuffle=True, random_state=42
    )
    logging.info(f"Split: train={len(df_train)}  test={len(df_test)}")

    # ── 2b. Add target-encoded + interaction features (train-only, no leakage) ─
    df_train, df_test = _add_target_features(df_train, df_test)

    available = [f for f in FEATURES if f in df_train.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        logging.warning(f"Missing features (skipped): {missing}")
    logging.info(f"Training on {len(available)} features: {available}")

    X_train = df_train[available]
    X_test  = df_test[available]
    y_train = df_train[TARGET]
    y_test  = df_test[TARGET]

    # ── 3. Model ──────────────────────────────────────────────────────────────
    params = {
        'n_estimators':    300,
        'max_depth':         6,
        'learning_rate':   0.05,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_weight':  3,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'objective':       'reg:squarederror',
        'random_state':     42,
        'n_jobs':           -1,
    }

    # ── 4. CV on training set ─────────────────────────────────────────────────
    logging.info("5-Fold CV on training set...")
    cv_scores = cross_val_score(
        xgb.XGBRegressor(**params), X_train, y_train,
        cv=5, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores)
    logging.info(f"CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

    # ── 5. Final fit ──────────────────────────────────────────────────────────
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))
    logging.info(f"[TEST] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

    # ── 7. Save model, feature list & encoding maps ───────────────────────────
    joblib.dump(model,     'outputs/xgboost_charge.pkl')
    joblib.dump(available, 'outputs/xgboost_charge_features.pkl')

    # Save target encoding maps for reproducible inference in export/evaluate
    line_means_map  = df_train.groupby('line_encoded')['charge_estimee'].mean().to_dict()
    zh_means_map    = df_train.groupby(['zone_encoded', 'hour'])['charge_estimee'].mean().to_dict()
    global_charge_mean = float(df_train['charge_estimee'].mean())
    joblib.dump({
        'line_charge_means':  line_means_map,
        'zone_hour_means':    zh_means_map,
        'global_mean':        global_charge_mean,
    }, 'outputs/charge_encoding.pkl')
    logging.info("Model & encoding saved → outputs/xgboost_charge.pkl")

    # ── 8. Feature importance ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax, title='Feature Importance — Passenger Load',
                        importance_type='gain', max_num_features=12)
    plt.tight_layout()
    plt.savefig('outputs/charge_feature_importance.png', dpi=150)
    plt.close()

    # ── 9. Actual vs Predicted ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    ax2.scatter(y_test, preds, alpha=0.3, edgecolors='k', linewidths=0.2)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax2.plot(lims, lims, 'r--', label='Perfect fit')
    ax2.set_xlabel('Actual charge_estimee'); ax2.set_ylabel('Predicted')
    ax2.set_title(f'Passenger Load  R²={r2:.3f}'); ax2.legend()
    plt.tight_layout()
    plt.savefig('outputs/charge_actual_vs_predicted.png', dpi=150)
    plt.close()
    logging.info("Plots saved → outputs/")

    return X_test, y_test, preds, {'rmse': rmse, 'mae': mae, 'r2': r2}


if __name__ == "__main__":
    from data_loader import load_mobility_data
    from feature_engineering import engineer_features
    df_raw = load_mobility_data()
    df_ml, _ = engineer_features(df_raw)
    train_load_regression(df_ml)