"""
Actor 3 — Responsable Sécurité des Transports Urbains
======================================================
ML Pipeline Orchestrator

Steps
-----
  1. Data Loader       — fact_impact_territorial.csv (security columns only)
  2. Feature Eng.      — severity_class, gravite_index, zone/mode encoding
  3. Classification    — Random Forest → accident severity per zone/month
  4. Clustering        — K-Means k=3 → zone risk profiles (Low/Medium/High)
  5. Anomaly Detection — Isolation Forest → flag abnormal crime/accident spikes
  6. Evaluation        — Central metrics report (JSON)
  7. Export            — severity_predictions.csv + risk_zone_clusters.csv + anomaly_flags.csv
"""

import os
import sys
import time
import logging
from datetime import datetime

# Always run relative to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/pipeline.log', mode='w', encoding='utf-8'),
    ],
)
os.makedirs('outputs', exist_ok=True)

BANNER = """
================================================================
  ACTOR 3 — RESPONSABLE SÉCURITÉ ML PIPELINE
================================================================
"""


def _step(name, fn, *args, **kwargs):
    """Runs a pipeline step with timing and error isolation."""
    start = time.time()
    logging.info(f">>> STEP: {name}")
    try:
        result  = fn(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"    ✅ {name} — {elapsed:.1f}s")
        return result, True
    except Exception as e:
        elapsed = time.time() - start
        logging.error(f"    ❌ {name} FAILED ({elapsed:.1f}s): {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None, False


def run_pipeline():
    print(BANNER)
    t0       = time.time()
    statuses = {}

    # ── 1. Data Loading ───────────────────────────────────────────────────────
    from data_loader import load_security_data
    df_raw, ok = _step("Data Loader", load_security_data)
    statuses["1. Data Loader"] = "✅" if ok else "❌"
    if not ok:
        logging.critical("Cannot continue without data. Exiting.")
        return

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    from feature_engineering import engineer_features
    result_fe, ok_fe = _step("Feature Engineering", engineer_features, df_raw)
    statuses["2. Feature Engineering"] = "✅" if ok_fe else "❌"
    if not ok_fe:
        logging.critical("Feature engineering failed. Exiting.")
        return
    df_ml, df_zone = result_fe

    # ── 3. Classification — Accident Severity ─────────────────────────────────
    from model_classification import train_severity_classifier
    clf_result, ok_clf = _step("Classification (Severity)", train_severity_classifier, df_ml)
    statuses["3. Classification"] = "✅" if ok_clf else "❌"

    # ── 4. Clustering — Zone Risk Profiles ────────────────────────────────────
    from model_clustering import train_zone_clustering
    cluster_result, ok_cl = _step("Clustering (Zone Risk)", train_zone_clustering, df_zone)
    statuses["4. Clustering"] = "✅" if ok_cl else "❌"

    # ── 5. Anomaly Detection ──────────────────────────────────────────────────
    from model_anomaly import detect_anomalies
    anom_result, ok_anom = _step("Anomaly Detection", detect_anomalies, df_ml)
    statuses["5. Anomaly Detection"] = "✅" if ok_anom else "❌"

    # ── 6. Central Evaluation ─────────────────────────────────────────────────
    from evaluate import evaluate_actor3
    df_cluster = cluster_result[0] if cluster_result else None
    eval_result, ok_eval = _step("Evaluation", evaluate_actor3, df_ml, df_zone)
    statuses["6. Evaluation"] = "✅" if ok_eval else "❌"

    # ── 7. Export ─────────────────────────────────────────────────────────────
    from export_predictions import export_predictions
    _, ok_exp = _step("Export Predictions", export_predictions, df_ml, df_zone, df_cluster)
    statuses["7. Export"] = "✅" if ok_exp else "❌"

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    n_ok   = sum(1 for v in statuses.values() if v.startswith("✅"))
    n_fail = sum(1 for v in statuses.values() if v.startswith("❌"))

    print(f"\n{'='*64}")
    print(f"  PIPELINE SUMMARY  ({elapsed_total:.1f}s)  |  ✅ {n_ok} OK  ❌ {n_fail} FAILED")
    print(f"{'='*64}")
    for step, status in statuses.items():
        padding = ' ' * max(0, 35 - len(step))
        print(f"  {status} {step}")
    print(f"{'='*64}\n")

    if n_fail == 0:
        print("  All outputs are in the outputs/ folder. Ready for Power BI.\n")
    else:
        print(f"  {n_fail} step(s) failed. Check outputs/pipeline.log for details.\n")


if __name__ == "__main__":
    run_pipeline()
