import logging
import time
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Make sure the project root is on the path regardless of where you call this
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('outputs/pipeline.log', mode='w', encoding='utf-8'),
    ]
)


def run_pipeline():
    """
    Master orchestrator for the Actor 1: Transition Écologique pipeline.

    Steps
    -----
    1. Data Loader          — load + dedup fact_impact_territorial.csv
    2. Feature Engineering  — lags, cyclical encoding, log transforms
    3. Regression Models    — XGBoost (CO2 + Energy), temporal split, 5-fold CV
    4. Time Series          — Prophet per zone, 2-year horizon
    5. Clustering           — K-Means on zone profiles
    6. Evaluation           — unified metrics → metrics_report.json
    7. Export               — CSVs for Power BI + optional DB push
    """
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "="*60)
    print("  ACTOR 1 — TRANSITION ÉCOLOGIQUE ML PIPELINE  v2")
    print("="*60 + "\n")

    start_time      = time.time()
    pipeline_status = {}

    df_clean  = None
    df_reg    = None
    df_cluster = None
    df_prophet = None

    # ── STEP 1: Data Loader ──────────────────────────────────────────────────
    try:
        logging.info(">>> STEP 1: Loading and Cleaning Data")
        from data_loader import load_and_clean_data
        df_clean = load_and_clean_data()
        logging.info(f"    Clean data shape: {df_clean.shape}")
        pipeline_status['1. Data Loader'] = f"✅ SUCCESS — {df_clean.shape[0]} rows"
    except Exception as e:
        logging.error(f"Data Loader FAILED: {e}")
        pipeline_status['1. Data Loader'] = f"❌ FAILED: {e}"
        print("\nCRITICAL: Cannot proceed without data. Aborting.\n")
        _print_summary(pipeline_status, time.time() - start_time)
        return pipeline_status

    # ── STEP 2: Feature Engineering ──────────────────────────────────────────
    try:
        logging.info(">>> STEP 2: Feature Engineering")
        from feature_engineering import engineer_features
        df_reg, df_cluster, df_prophet = engineer_features(df_clean)
        logging.info(f"    df_reg={df_reg.shape}  df_cluster={df_cluster.shape}  df_prophet={df_prophet.shape}")
        pipeline_status['2. Feature Engineering'] = f"✅ SUCCESS — {df_reg.shape[0]} ML rows"
    except Exception as e:
        logging.error(f"Feature Engineering FAILED: {e}")
        pipeline_status['2. Feature Engineering'] = f"❌ FAILED: {e}"
        print("\nCRITICAL: Cannot proceed without features. Aborting.\n")
        _print_summary(pipeline_status, time.time() - start_time)
        return pipeline_status

    # ── STEP 3: Regression Models ─────────────────────────────────────────────
    try:
        logging.info(">>> STEP 3: XGBoost Regression (CO2 + Energy)")
        from model_regression import train_and_evaluate_xgboost
        train_and_evaluate_xgboost(df_reg)
        pipeline_status['3. Regression Models'] = "✅ SUCCESS"
    except Exception as e:
        logging.error(f"Regression FAILED: {e}")
        pipeline_status['3. Regression Models'] = f"❌ FAILED: {e}"

    # ── STEP 4: Time Series Forecasting ──────────────────────────────────────
    try:
        logging.info(">>> STEP 4: Prophet Time Series (per zone, 24-month horizon)")
        from model_timeseries import train_prophet_forecasts
        forecast_df, _ = train_prophet_forecasts(df_prophet)
        n = len(forecast_df) if forecast_df is not None else 0
        pipeline_status['4. Time Series Forecasts'] = f"✅ SUCCESS — {n} forecast rows"
    except Exception as e:
        logging.error(f"Time Series FAILED: {e}")
        pipeline_status['4. Time Series Forecasts'] = f"❌ FAILED: {e}"

    # ── STEP 5: Clustering ───────────────────────────────────────────────────
    try:
        logging.info(">>> STEP 5: K-Means Zone Clustering")
        from model_clustering import train_pollution_clustering
        profiles = train_pollution_clustering(df_cluster)
        n = len(profiles) if profiles is not None else 0
        pipeline_status['5. Clustering'] = f"✅ SUCCESS — {n} zones clustered"
    except Exception as e:
        logging.error(f"Clustering FAILED: {e}")
        pipeline_status['5. Clustering'] = f"❌ FAILED: {e}"

    # ── STEP 6: Evaluation ───────────────────────────────────────────────────
    try:
        logging.info(">>> STEP 6: Central Evaluation & Metrics")
        from evaluate import run_evaluation
        report = run_evaluation()
        status = report.get('metadata', {}).get('overall_status', 'DONE')
        pipeline_status['6. Evaluation'] = f"✅ SUCCESS — {status}"
    except Exception as e:
        logging.error(f"Evaluation FAILED: {e}")
        pipeline_status['6. Evaluation'] = f"❌ FAILED: {e}"

    # ── STEP 7: Export ───────────────────────────────────────────────────────
    try:
        logging.info(">>> STEP 7: Exporting Predictions (CSV + optional DB)")
        from export_predictions import generate_power_bi_export
        generate_power_bi_export()
        pipeline_status['7. Export'] = "✅ SUCCESS"
    except Exception as e:
        logging.error(f"Export FAILED: {e}")
        pipeline_status['7. Export'] = f"❌ FAILED: {e}"

    _print_summary(pipeline_status, time.time() - start_time)
    return pipeline_status


def _print_summary(status_dict, elapsed):
    n_ok   = sum(1 for v in status_dict.values() if v.startswith("✅"))
    n_fail = sum(1 for v in status_dict.values() if v.startswith("❌"))
    print("\n" + "="*60)
    print(f"  PIPELINE SUMMARY  ({elapsed:.1f}s)  |  ✅ {n_ok} OK  ❌ {n_fail} FAILED")
    print("="*60)
    for step, st in status_dict.items():
        print(f"  {step:<32}  {st}")
    print("="*60)
    if n_fail == 0:
        print("\n  All outputs are in the outputs/ folder. Ready for Power BI.\n")
    else:
        print(f"\n  {n_fail} step(s) failed. Check outputs/pipeline.log for details.\n")


if __name__ == "__main__":
    run_pipeline()