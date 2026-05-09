# ============================================================
# URBAN MOBILITY — FULL STACK STARTUP
# ============================================================

# T1 — MLflow (port 5000)
Start-Process powershell -ArgumentList "-NoExit -Command `"cd \`"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ml_api_2\`"; python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlflow/mlruns --default-artifact-root ./mlflow/mlruns --workers 1`""
Start-Sleep -Seconds 8

# T2 — FastAPI (port 8000)
Start-Process powershell -ArgumentList "-NoExit -Command `"cd \`"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ml_api_2\`"; python -m uvicorn main:app --host 0.0.0.0 --port 8000`""
Start-Sleep -Seconds 6

# T3 — n8n (port 5678)
$n8nProcess = Get-NetTCPConnection -LocalPort 5678 -ErrorAction SilentlyContinue
if ($n8nProcess) {
    Stop-Process -Id (Get-Process -Id $n8nProcess.OwningProcess).Id -Force
    Write-Host "Killed existing process on port 5678" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
Start-Process powershell -ArgumentList "-NoExit -Command `"n8n`""
Start-Sleep -Seconds 5

# T4 — Prometheus (port 9090)
if (Test-Path "C:\prometheus-3.11.3.windows-amd64\prometheus-3.11.3.windows-amd64\prometheus.exe") {
    Start-Process powershell -ArgumentList "-NoExit -Command `"& \`"C:\prometheus-3.11.3.windows-amd64\prometheus-3.11.3.windows-amd64\prometheus.exe\`" --config.file=\`"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ml_api_2\prometheus.yml\`"`""
    Start-Sleep -Seconds 3
} else {
    Write-Host "⚠️ MISSING: Prometheus binary not found at C:\prometheus-3.11.3.windows-amd64\prometheus-3.11.3.windows-amd64\prometheus.exe. Skipping..." -ForegroundColor Red
}

# T5 — Grafana (port 3001)
$grafanaLog = "C:\Program Files\GrafanaLabs\grafana\data\log"
if (-not (Test-Path $grafanaLog)) {
    New-Item -ItemType Directory -Path $grafanaLog -Force
}

if (Test-Path "C:\Program Files\GrafanaLabs\grafana\bin\grafana-server.exe") {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoExit -Command `"& \`"C:\Program Files\GrafanaLabs\grafana\bin\grafana-server.exe\`" --homepath \`"C:\Program Files\GrafanaLabs\grafana\`"`""
    Start-Sleep -Seconds 3
} else {
    Write-Host "⚠️ MISSING: Grafana binary not found at C:\Program Files\GrafanaLabs\grafana\bin\grafana-server.exe. Skipping..." -ForegroundColor Red
}

# T6 — Streamlit (port 8501)
Start-Process powershell -ArgumentList "-NoExit -Command `"cd \`"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\streamlit_app\`"; python -m streamlit run app.py`""
Start-Sleep -Seconds 6

# T7 — Next.js (port 3000)
Start-Process powershell -ArgumentList "-NoExit -Command `"cd \`"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\urban-mobility-website\`"; npm run dev`""
Start-Sleep -Seconds 8

# T8 — Simulation (LAST)
$modelsExist = (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor1_ecologique\outputs\xgboost_co2.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor1_ecologique\outputs\xgboost_nrj.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor1_ecologique\outputs\kmeans_pollution_zones.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor2_mobilites\outputs\xgboost_charge.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor2_mobilites\outputs\xgboost_cancellation.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor3_securite\outputs\rf_severity.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor3_securite\outputs\kmeans_risk.pkl") -and
               (Test-Path "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ML\actor3_securite\outputs\isolation_forest.pkl")

if ($modelsExist) {
    Start-Process powershell -ArgumentList "-NoExit -Command `"cd \`"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ml_api_2\`"; python simulate_scenarios.py`""
    Start-Sleep -Seconds 10
} else {
    Write-Host "⚠️ Models not trained. Run TRAIN_MODELS.ps1 first!" -ForegroundColor Red
}

# ============================================================
# HEALTH CHECK URLS
# ============================================================
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " ALL SERVICES STARTING..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Next.js      -> http://localhost:3000" -ForegroundColor Green
Write-Host " FastAPI      -> http://localhost:8000/health" -ForegroundColor Green
Write-Host " FastAPI Docs -> http://localhost:8000/docs" -ForegroundColor Green
Write-Host " Streamlit    -> http://localhost:8501" -ForegroundColor Green
Write-Host " MLflow       -> http://localhost:5000" -ForegroundColor Green
Write-Host " Prometheus   -> http://localhost:9090" -ForegroundColor Green
Write-Host " Grafana      -> http://localhost:3001" -ForegroundColor Green
Write-Host " n8n          -> http://localhost:5678" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " LOG TAIL: Get-Content `"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe\ml_api_2\ml_api.log`" -Wait -Tail 20" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Cyan
