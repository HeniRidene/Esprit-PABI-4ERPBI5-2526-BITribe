# ESPRIT ML Production Gateway — Architecture Notes

> Last updated: 2026-04-16  
> Maintained by: Senior ML Engineer / Platform Architect

---

## 1. Project Overview

This folder serves as the **backend inference layer** for the ESPRIT Smart-City ML platform.  
It exposes 18 scikit-learn / XGBoost models as a single REST API and is orchestrated by an n8n workflow.

```
models/
├── main.py                  ← FastAPI backend (v2.0.0)
├── n8n_workflow.json        ← n8n orchestration workflow
├── requirements.txt         ← Python dependencies
├── Claude.md                ← This file
├── actor1/                  ← Ecological actor models
│   ├── xgboost_features.pkl         (feature order schema)
│   ├── mode_co2_encoding.pkl        (categorical encoder)
│   ├── clustering_scaler.pkl        (StandardScaler)
│   ├── xgboost_co2.pkl              (CO₂ emission regressor)
│   ├── xgboost_nrj.pkl              (energy consumption regressor)
│   └── kmeans_pollution_zones.pkl   (pollution zone clusterer)
├── actor2/                  ← Mobility actor models
│   ├── xgboost_cancellation_features.pkl
│   ├── xgboost_charge_features.pkl
│   ├── charge_encoding.pkl
│   ├── xgboost_cancellation.pkl     (trip cancellation classifier)
│   └── xgboost_charge.pkl           (fare-charge regressor)
└── actor3/                  ← Security actor models
    ├── anomaly_features.pkl
    ├── anomaly_scaler.pkl
    ├── isolation_forest.pkl         (anomaly detector, returns ±1)
    ├── kmeans_features.pkl
    ├── kmeans_scaler.pkl
    ├── kmeans_risk.pkl              (risk zone clusterer)
    ├── rf_severity_features.pkl
    └── rf_severity.pkl              (incident severity classifier)
```

---

## 2. FastAPI Backend (`main.py`)

### Design Decisions

| Decision | Rationale |
|---|---|
| **Lazy loading** | Models are loaded on first request and cached in `_registry`. Keeps RAM low at startup; rarely-used models don't consume memory. |
| **Thread-safe locking** | Double-checked locking (`threading.Lock`) prevents duplicate disk reads under concurrent requests. |
| **Dispatch table** | `PIPELINE_ROUTER` maps `(actor, task)` tuples to pipeline functions. Adding a new model = one line. |
| **Pydantic v2 validators** | `@field_validator` on `actor` and `task` ensures 400-level errors hit before any model I/O. |
| **Consistent response schema** | Every response (success or error) follows `{status, actor, task, prediction, latency_ms}`. |

### Supported Pipelines

| Actor | Task | Pipeline Steps |
|---|---|---|
| `actor1` | `co2` | xgboost_features → mode_co2_encoding → clustering_scaler → xgboost_co2 |
| `actor1` | `nrj` | xgboost_features → mode_co2_encoding → clustering_scaler → xgboost_nrj |
| `actor1` | `cluster` | clustering_scaler → kmeans_pollution_zones |
| `actor2` | `cancellation` | xgboost_cancellation_features → xgboost_cancellation |
| `actor2` | `charge` | xgboost_charge_features → charge_encoding → xgboost_charge |
| `actor3` | `severity` | rf_severity_features → rf_severity |
| `actor3` | `risk` | kmeans_features → kmeans_scaler → kmeans_risk |
| `actor3` | `anomaly` | anomaly_features → anomaly_scaler → isolation_forest |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe; lists cached models |
| `GET` | `/models` | Lists all registered actor/task combos |
| `POST` | `/predict` | Main inference endpoint |

### Request / Response Shape

```jsonc
// POST /predict
{
  "actor": "actor1",
  "task": "co2",
  "features": {
    "km_parcourus": 12000,
    "type_carburant": "diesel",
    "age_vehicule": 5
    // ... all feature columns required by xgboost_features.pkl
  }
}

// 200 OK
{
  "status": "success",
  "actor": "actor1",
  "task": "co2",
  "prediction": 2.41,
  "latency_ms": 14.3,
  "metadata": { "pipeline_key": "actor1/co2" }
}

// 400 Bad Request (missing feature / wrong actor)
{ "detail": "Missing required feature columns: ['type_carburant']..." }

// 422 Unprocessable Entity (Pydantic validation failure)
{ "detail": [{ "loc": ["body","actor"], "msg": "actor must be one of ..." }] }
```

### How to Run

```powershell
# 1. Install dependencies (once)
pip install -r requirements.txt

# 2. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Open interactive docs
# http://localhost:8000/docs
```

---

## 3. n8n Workflow (`n8n_workflow.json`)

### Node Architecture

```
Webhook (POST /ml-predict)
    │
    ▼
Feature Mapper [Code Node]
    • Reads actor + task from body
    • Applies alias map (rename client keys → model keys)
    • Builds { actor, task, features } payload
    │
    ▼
POST /predict [HTTP Request Node]
    • Method: POST
    • URL: http://localhost:8000/predict
    • Body: JSON payload from Feature Mapper
    • continueOnFail: true  ← lets error branch trigger
    │
    ▼
Success? [IF Node]
    • Condition: response.body.status == "success"
    │                    │
   TRUE                FALSE
    │                    │
    ▼                    ▼
Format Success      Handle Error [Code Node]
[Code Node]            • Extracts error.detail
    │                  • Logs actor/task/status
    ▼                    │
Return 200             Return 422
[Respond Webhook]    [Respond Webhook]
```

### How to Import

1. Open n8n → **Workflows** → **Import from File**
2. Select `n8n_workflow.json`
3. Activate the workflow
4. Webhook URL will be: `https://<your-n8n-host>/webhook/ml-predict`

### Adding Feature Aliases

Edit the `ALIAS_MAPS` object inside the **Feature Mapper** Code Node:

```js
const ALIAS_MAPS = {
  actor1: {
    "fuel": "type_carburant",   // client sends "fuel", model expects "type_carburant"
  },
  actor2: {},
  actor3: {}
};
```

---

## 4. Extending the System

### Adding a New Model

1. Drop the `.pkl` file into the correct `actor*/` folder.
2. Add a pipeline function in `main.py`:
   ```python
   def _pipeline_actor1_new_task(raw_df: pd.DataFrame) -> Any:
       feature_list = load_pkl("actor1", "new_task_features.pkl")
       df = _reorder_columns(raw_df, feature_list)
       model = load_pkl("actor1", "new_task_model.pkl")
       return model.predict(df)
   ```
3. Register it in the dispatch table:
   ```python
   PIPELINE_ROUTER[("actor1", "new_task")] = _pipeline_actor1_new_task
   ```
4. Add `"new_task"` to `VALID_TASKS["actor1"]`.
5. No n8n changes needed — the payload routing is dynamic.

### Scaling to Production

- Run behind **nginx** or **Caddy** as a reverse proxy.
- Use **Gunicorn** with uvicorn workers: `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker`
- Add **Redis** caching for high-frequency identical inputs.
- Monitor with **Prometheus** metrics via `prometheus-fastapi-instrumentator`.

---

## 5. Logging

All log lines are emitted in JSON format:

```json
{"time": "2026-04-16T21:00:00", "level": "INFO", "module": "main", "message": "Loading model from disk: actor1/xgboost_co2.pkl"}
{"time": "2026-04-16T21:00:01", "level": "INFO", "module": "main", "message": "Prediction success — actor=actor1 task=co2 prediction=2.41 latency_ms=14.32"}
{"time": "2026-04-16T21:00:02", "level": "ERROR", "module": "main", "message": "Pipeline error [ValueError] — actor=actor2 task=charge error=\"Missing required feature columns: ['nb_trajets']\" latency_ms=3.10"}
```

Pipe stdout to **Loki**, **CloudWatch**, or **Datadog** for centralised observability.
