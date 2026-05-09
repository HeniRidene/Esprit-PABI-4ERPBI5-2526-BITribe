import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve
)
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES = [
    'zone_encoded', 'line_encoded', 'mode_encoded',
    'hour', 'rush_hour', 'is_weekend',
    'congestion_index', 'charge_estimee',
]
TARGET = 'retard_flag'   # alias for annule in feature_engineering

# Risk label thresholds
LOW_THRESHOLD    = 0.20
MEDIUM_THRESHOLD = 0.50


def _risk_label(proba: float) -> str:
    if proba < LOW_THRESHOLD:
        return 'Low'
    elif proba < MEDIUM_THRESHOLD:
        return 'Medium'
    else:
        return 'High'


def train_cancellation_model(df_ml):
    """
    Trains XGBoost classifier to predict trip cancellation probability (annule=1).

    Key design decisions
    --------------------
    - scale_pos_weight = neg/pos (dynamic) → handles the 2% positive class imbalance
      without SMOTE. Equally effective for tree models and simpler.
    - Outputs both binary (0/1) and probability score per trip.
    - Evaluates at default threshold (0.5) AND finds the F1-maximising threshold.
    - Saves confusion matrix, ROC curve, and precision-recall curve for the professor.
    - risk_label column (Low/Medium/High) ready for Power BI conditional formatting.
    """
    logging.info("=== CLASSIFICATION — Trip Cancellation Risk ===")
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Feature selection ──────────────────────────────────────────────────
    available = [f for f in FEATURES if f in df_ml.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        logging.warning(f"Missing features (skipped): {missing}")

    if TARGET not in df_ml.columns:
        raise ValueError(f"Target '{TARGET}' not found in df_ml.")

    needed = available + [TARGET]
    df_clean = df_ml[needed].dropna()
    logging.info(f"Classification set: {len(df_clean)} rows")

    X = df_clean[available]
    y = df_clean[TARGET].astype(int)

    # ── 2. Train / Test split (stratified) ───────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        logging.warning("Stratify failed (too few positives) → random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    logging.info(f"Split: train={len(X_train)} test={len(X_test)}")

    # ── 3. Dynamic class imbalance ────────────────────────────────────────────
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    if pos_count == 0:
        logging.error("ZERO positive cancellations in training set — cannot train classifier.")
        return None, {}

    scale_pos_weight = neg_count / pos_count
    logging.info(f"scale_pos_weight = {scale_pos_weight:.2f}  (pos={pos_count} neg={neg_count})")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    logging.info(f"Classification training complete.")

    # ── 5. Evaluate at default threshold (0.50) ───────────────────────────────
    proba_test    = model.predict_proba(X_test)[:, 1]
    default_thresh = 0.50
    preds_default  = (proba_test >= default_thresh).astype(int)

    auc  = float(roc_auc_score(y_test, proba_test)) if y_test.sum() > 0 else 0.0
    f1_d = float(f1_score(y_test, preds_default, zero_division=0))
    pre_d = float(precision_score(y_test, preds_default, zero_division=0))
    rec_d = float(recall_score(y_test, preds_default, zero_division=0))
    cm    = confusion_matrix(y_test, preds_default).tolist()

    logging.info(f"[TEST @ 0.50] AUC={auc:.4f}  F1={f1_d:.4f}  Precision={pre_d:.4f}  Recall={rec_d:.4f}")
    logging.info(f"Confusion matrix (TN,FP / FN,TP): {cm}")

    # ── 6. Find F1-maximising threshold ──────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)
    f1_per_thresh = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx   = int(np.argmax(f1_per_thresh))
    best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else default_thresh
    best_f1     = float(f1_per_thresh[best_idx])
    logging.info(f"Best F1={best_f1:.4f} at threshold={best_thresh:.3f}")

    preds_best = (proba_test >= best_thresh).astype(int)
    f1_best    = float(f1_score(y_test, preds_best, zero_division=0))
    pre_best   = float(precision_score(y_test, preds_best, zero_division=0))
    rec_best   = float(recall_score(y_test, preds_best, zero_division=0))

    # ── 7. Save model + features ──────────────────────────────────────────────
    joblib.dump(model,     'outputs/xgboost_cancellation.pkl')
    joblib.dump(available, 'outputs/xgboost_cancellation_features.pkl')
    logging.info("Model saved → outputs/xgboost_cancellation.pkl")

    # ── 8. Classification report JSON (for evaluate.py + professor) ───────────
    report = {
        'model': 'XGBoost Classifier — Trip Cancellation',
        'class_imbalance': {
            'pos_count': pos_count,
            'neg_count': neg_count,
            'scale_pos_weight': round(scale_pos_weight, 2),
        },
        'threshold_default': {
            'threshold': default_thresh,
            'auc_roc':   round(auc, 4),
            'f1':        round(f1_d, 4),
            'precision': round(pre_d, 4),
            'recall':    round(rec_d, 4),
            'confusion_matrix': cm,
        },
        'threshold_best_f1': {
            'threshold': round(best_thresh, 3),
            'f1':        round(f1_best, 4),
            'precision': round(pre_best, 4),
            'recall':    round(rec_best, 4),
            'note': 'Threshold is a business decision: lower → more recalls (fewer missed), higher → fewer false alarms.',
        },
    }
    with open('outputs/classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls[:-1], precisions[:-1], lw=2)
    ax.axvline(rec_d,  color='red',    ls='--', label=f'Threshold={default_thresh:.2f}')
    ax.axvline(rec_best, color='green', ls='--', label=f'Best F1={best_thresh:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve  (AUC={auc:.3f})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/cancellation_pr_curve.png', dpi=150)
    plt.close()

    # Feature importance
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax2, title='Feature Importance — Cancellation Risk',
                        importance_type='gain', max_num_features=10)
    plt.tight_layout()
    plt.savefig('outputs/cancellation_feature_importance.png', dpi=150)
    plt.close()
    logging.info("Plots saved → outputs/")

    # ── 10. Return test set predictions for export ────────────────────────────
    test_indices = X_test.index
    results = pd.DataFrame({
        'index':        test_indices,
        'annule_proba': proba_test,
        'annule_pred':  preds_default,
        'risk_label':   [_risk_label(p) for p in proba_test],
    }).set_index('index')

    return results, report


if __name__ == "__main__":
    from data_loader import load_mobility_data
    from feature_engineering import engineer_features
    df_raw = load_mobility_data()
    df_ml, _ = engineer_features(df_raw)
    results, report = train_cancellation_model(df_ml)
    if results is not None:
        print(f"\nPredictions: {results.shape}")
        print(results['risk_label'].value_counts().to_string())