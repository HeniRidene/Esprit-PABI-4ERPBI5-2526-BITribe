"""
Actor 2 — Directeur Mobilités (RATP / Île-de-France Mobilités)
===============================================================
ML Pipeline Orchestrator

Steps
-----
  1. Data Loader          — fact_service_mobilite.csv + dim_temps merge
  2. Feature Engineering  — imputation, encoding, time features, Prophet aggregation
  3. Regression           — XGBoost → predict charge_estimee (passenger load)
  4. Classification       — XGBoost → predict cancellation risk (annule)
  5. Time Series          — Prophet → 30-day daily congestion forecast per zone
  6. Evaluation           — Central metrics report (JSON)
  7. Export               — load_predictions.csv, cancellation_risk.csv, congestion_forecast.csv
"""

import os
import sys
import time
import logging
from datetime import datetime

# Always run relative to this file's directory (important when invoked from actor1 venv)
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
  ACTOR 2 — DIRECTEUR MOBILITÉS ML PIPELINE
================================================================
"""


def _step(name, fn, *args, **kwargs):
    """Runs a pipeline step with timing and error isolation."""
    start = time.time()
    logging.info(f">>> STEP: {name}")
    try:
        result = fn(*args, **kwargs)
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
    from data_loader import load_mobility_data
    df_raw, ok = _step("Data Loader", load_mobility_data)
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
    df_ml, df_ts = result_fe

    # ── 3. Regression — Passenger Load ───────────────────────────────────────
    from model_regression import train_load_regression
    reg_result, ok_reg = _step("Regression (Passenger Load)", train_load_regression, df_ml)
    statuses["3. Regression"] = "✅" if ok_reg else "❌"

    # ── 4. Classification — Cancellation Risk ─────────────────────────────────
    from model_classification import train_cancellation_model
    clf_result, ok_clf = _step("Classification (Cancellation)", train_cancellation_model, df_ml)
    statuses["4. Classification"] = "✅" if ok_clf else "❌"

    # ── 5. Time Series — Congestion Forecast ──────────────────────────────────
    from model_timeseries import train_congestion_forecast
    ts_result, ok_ts = _step("Prophet Congestion Forecast", train_congestion_forecast, df_ts)
    statuses["5. Time Series (Prophet)"] = "✅" if ok_ts else "❌"

    # ── 6. Central Evaluation ─────────────────────────────────────────────────
    from evaluate import evaluate_actor2
    eval_result, ok_eval = _step("Evaluation", evaluate_actor2, df_ml)
    statuses["6. Evaluation"] = "✅" if ok_eval else "❌"

    # ── 7. Export ─────────────────────────────────────────────────────────────
    from export_predictions import export_predictions
    _, ok_exp = _step("Export Predictions", export_predictions, df_ml, df_ts)
    statuses["7. Export"] = "✅" if ok_exp else "❌"

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    n_ok   = sum(1 for v in statuses.values() if v.startswith("✅"))
    n_fail = sum(1 for v in statuses.values() if v.startswith("❌"))

    print(f"\n{'='*64}")
    print(f"  PIPELINE SUMMARY  ({elapsed_total:.1f}s)  |  ✅ {n_ok} OK  ❌ {n_fail} FAILED")
    print(f"{'='*64}")
    for step, status in statuses.items():
        print(f"  {status} {step}")
    print(f"{'='*64}\n")

    if n_fail == 0:
        print("  All outputs are in the outputs/ folder. Ready for Power BI.\n")
    else:
        print(f"  {n_fail} step(s) failed. Check outputs/pipeline.log for details.\n")


if __name__ == "__main__":
    run_pipeline()