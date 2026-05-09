$base = "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\Esprit-PABI-4ERPBI5-2526-BITribe"
$env:PYTHONIOENCODING="utf-8"

# Create target directories
New-Item -ItemType Directory -Force -Path "$base\actor1_ecologique\outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\actor2_mobilites\outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\actor3_securite\outputs" | Out-Null

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " URBAN MOBILITY - TRAIN ALL MODELS" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Actor 1 - Ecological
Write-Host "`n[1/3] Training Actor 1 (Ecological)..." -ForegroundColor Yellow
Set-Location "$base\ML\actor1_ecologique"
python main.py
if ($LASTEXITCODE -eq 0) {
    Copy-Item "outputs\*" -Destination "$base\actor1_ecologique\outputs\" -Force -Recurse
    Write-Host "[OK] Actor 1 complete" -ForegroundColor Green
} else {
    Write-Host "[X] Actor 1 FAILED - check errors above" -ForegroundColor Red
    exit 1
}

# Actor 2 - Mobility
Write-Host "`n[2/3] Training Actor 2 (Mobility)..." -ForegroundColor Yellow
Set-Location "$base\ML\actor2_mobilites"
python main.py
if ($LASTEXITCODE -eq 0) {
    Copy-Item "outputs\*" -Destination "$base\actor2_mobilites\outputs\" -Force -Recurse
    Write-Host "[OK] Actor 2 complete" -ForegroundColor Green
} else {
    Write-Host "[X] Actor 2 FAILED - check errors above" -ForegroundColor Red
    exit 1
}

# Actor 3 - Security
Write-Host "`n[3/3] Training Actor 3 (Security)..." -ForegroundColor Yellow
Set-Location "$base\ML\actor3_securite"
python main.py
if ($LASTEXITCODE -eq 0) {
    Copy-Item "outputs\*" -Destination "$base\actor3_securite\outputs\" -Force -Recurse
    Write-Host "[OK] Actor 3 complete" -ForegroundColor Green
} else {
    Write-Host "[X] Actor 3 FAILED - check errors above" -ForegroundColor Red
    exit 1
}

# Verify all 8 .pkl files exist
Write-Host "`n[CHECK] Verifying model files..." -ForegroundColor Cyan
$models = @(
    "$base\actor1_ecologique\outputs\xgboost_co2.pkl",
    "$base\actor1_ecologique\outputs\xgboost_nrj.pkl",
    "$base\actor1_ecologique\outputs\kmeans_pollution_zones.pkl",
    "$base\actor2_mobilites\outputs\xgboost_charge.pkl",
    "$base\actor2_mobilites\outputs\xgboost_cancellation.pkl",
    "$base\actor3_securite\outputs\rf_severity.pkl",
    "$base\actor3_securite\outputs\kmeans_risk.pkl",
    "$base\actor3_securite\outputs\isolation_forest.pkl"
)
$allOk = $true
foreach ($m in $models) {
    if (Test-Path $m) {
        Write-Host "  [OK] $(Split-Path $m -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  [X] MISSING: $m" -ForegroundColor Red
        $allOk = $false
    }
}
if ($allOk) {
    Write-Host "`n[OK] ALL MODELS READY - Run STARTUP.ps1 now" -ForegroundColor Green
} else {
    Write-Host "`n[X] Some models missing - check training errors above" -ForegroundColor Red
}
