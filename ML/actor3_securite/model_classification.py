import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES = [
    'zone_encoded', 'mode_encoded',
    'annee', 'mois',
    'volume_crimes', 'taux_criminalite',
    'usagers_vulnerables',
    # Derived features
    'has_accident', 'gravite_index',
    'crime_rate_scaled',
]
TARGET = 'severity_class'


def train_severity_classifier(df_ml):
    """
    Trains a Random Forest classifier to predict accident severity per zone/month.

    Design decisions
    ----------------
    - Random Forest over XGBoost for small imbalanced multiclass problems:
      ensemble variance is lower and class_weight='balanced' is natively supported.
    - class_weight='balanced': automatically weights minority class inversely to its frequency.
    - 5-fold STRATIFIED CV: ensures each fold sees all severity classes proportionally.
    - Reports F1-macro (equal weight) + per-class precision/recall for the professor.
    - Saves confusion matrix heatmap + feature importance plot.
    - Both binary (0/1) and multiclass (0/1/2) are handled transparently.
    """
    logging.info("=== CLASSIFICATION — Accident Severity (Random Forest) ===")
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Prepare features ───────────────────────────────────────────────────
    available = [f for f in FEATURES if f in df_ml.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        logging.warning(f"Missing features (skipped): {missing}")

    df_clean = df_ml[available + [TARGET]].dropna()
    logging.info(f"Classification set: {len(df_clean)} rows  ({len(df_ml)-len(df_clean)} dropped for NaN)")

    X = df_clean[available].values
    y = df_clean[TARGET].values
    classes = sorted(np.unique(y).tolist())
    n_classes = len(classes)
    logging.info(f"Classes: {classes}  |  distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 2. Stratified 5-fold CV ───────────────────────────────────────────────
    logging.info("Running 5-fold Stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_cv = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,          # let trees grow — generalises better on small data
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    cv_results = cross_validate(
        rf_cv, X, y, cv=cv,
        scoring=['f1_macro', 'f1_weighted', 'accuracy'],
        return_train_score=False,
    )
    cv_f1_macro = float(cv_results['test_f1_macro'].mean())
    cv_f1_std   = float(cv_results['test_f1_macro'].std())
    logging.info(f"CV F1-macro: {cv_f1_macro:.4f} ± {cv_f1_std:.4f}")

    # ── 3. Final model on full data (for predict_proba + feature importance) ──
    rf_final = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X, y)

    # ── 4. In-sample evaluation for confusion matrix ──────────────────────────
    # Use last CV fold as held-out test for confusion matrix
    train_idx, test_idx = list(cv.split(X, y))[-1]
    X_test_cv, y_test_cv = X[test_idx], y[test_idx]

    rf_fold = RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_fold.fit(X[train_idx], y[train_idx])
    preds_cv = rf_fold.predict(X_test_cv)
    proba_cv = rf_fold.predict_proba(X_test_cv)

    f1_mac  = float(f1_score(y_test_cv, preds_cv, average='macro',    zero_division=0))
    f1_wei  = float(f1_score(y_test_cv, preds_cv, average='weighted', zero_division=0))
    clf_rpt = classification_report(y_test_cv, preds_cv, zero_division=0, output_dict=True)
    cm      = confusion_matrix(y_test_cv, preds_cv, labels=classes).tolist()

    # AUC one-vs-rest
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_test_cv, proba_cv[:, 1]))
        else:
            y_bin = label_binarize(y_test_cv, classes=classes)
            auc   = float(roc_auc_score(y_bin, proba_cv, multi_class='ovr', average='macro'))
    except Exception:
        auc = 0.0

    logging.info(f"[Test fold] F1-macro={f1_mac:.4f}  F1-weighted={f1_wei:.4f}  AUC={auc:.4f}")
    logging.info(f"Confusion matrix: {cm}")

    # ── 5. Save model ─────────────────────────────────────────────────────────
    joblib.dump(rf_final,  'outputs/rf_severity.pkl')
    joblib.dump(available, 'outputs/rf_severity_features.pkl')

    # ── 6. Confusion matrix heatmap ───────────────────────────────────────────
    label_names = {0: 'Aucun', 1: 'Grave', 2: 'Mortel'}
    tick_labels = [label_names.get(c, str(c)) for c in classes]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — Accident Severity\nF1-macro={f1_mac:.3f}')
    plt.tight_layout()
    plt.savefig('outputs/severity_confusion_matrix.png', dpi=150)
    plt.close()

    # ── 7. Feature importance ─────────────────────────────────────────────────
    importances = pd.Series(rf_final.feature_importances_, index=available).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    importances.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Feature Importance — Accident Severity (RF Gini)')
    ax2.set_ylabel('Importance'); ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/severity_feature_importance.png', dpi=150)
    plt.close()

    logging.info("Model & plots saved → outputs/")

    # ── 8. Build report dict ──────────────────────────────────────────────────
    class_keys, class_counts = np.unique(y, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(class_keys, class_counts)}

    report = {
        'model':              'RandomForestClassifier — Accident Severity',
        'n_classes':          n_classes,
        'class_merge_note':   (
            'Classes 1+2 merged (mortel+grave → serious) because nb_mortels=0 in all rows. '
            'Binary classification: 0=Aucun incident, 1=Accident grave/mortel.'
            if n_classes == 2 else '3-class problem: 0=Aucun, 1=Grave, 2=Mortel.'
        ),
        'cv_f1_macro':        round(cv_f1_macro, 4),
        'cv_f1_std':          round(cv_f1_std, 4),
        'test_fold': {
            'f1_macro':       round(f1_mac, 4),
            'f1_weighted':    round(f1_wei, 4),
            'auc_roc':        round(auc, 4),
            'confusion_matrix': cm,
        },
        'feature_importances': importances.round(4).to_dict(),
        'class_distribution':  counts,
    }

    with open('outputs/classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logging.info("Report saved → outputs/classification_report.json")

    return rf_final, report


if __name__ == "__main__":
    from data_loader import load_security_data
    from feature_engineering import engineer_features
    df_raw = load_security_data()
    df_ml, _ = engineer_features(df_raw)
    rf, report = train_severity_classifier(df_ml)
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))
