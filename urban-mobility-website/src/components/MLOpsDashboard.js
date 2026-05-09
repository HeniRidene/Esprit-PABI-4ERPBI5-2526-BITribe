"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Activity,
  BarChart3,
  FlaskConical,
  Workflow,
  RefreshCw,
  Play,
  RotateCcw,
  ExternalLink,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  Database,
  AlertTriangle,
  Zap,
} from "lucide-react";

// ── Demo payloads ────────────────────────────────────────────────────────────
const DEMO_PAYLOADS = {
  actor1: {
    co2: {
      actor: "actor1", task: "co2",
      features: {
        zone_encoded: 2, mode_encoded: 1, mode_co2_mean: 45.3,
        annee: 2023, mois_sin: 0.5, mois_cos: 0.866,
        pm25: 18.4, no2: 32.1, aqi_index: 75.0,
        co2_lag1: 42.0, co2_lag3: 39.5,
        energie_lag1: 120.0, energie_lag3: 115.0,
        aqi_lag1: 72.0, pm25_lag1: 17.8,
        co2_roll3: 40.8, energie_roll3: 117.5, pm25_roll3: 18.0,
      },
    },
    energy: {
      actor: "actor1", task: "energy",
      features: {
        zone_encoded: 2, mode_encoded: 1, mode_co2_mean: 45.3,
        annee: 2023, mois_sin: 0.5, mois_cos: 0.866,
        pm25: 18.4, no2: 32.1, aqi_index: 75.0,
        co2_lag1: 42.0, co2_lag3: 39.5,
        energie_lag1: 120.0, energie_lag3: 115.0,
        aqi_lag1: 72.0, pm25_lag1: 17.8,
        co2_roll3: 40.8, energie_roll3: 117.5, pm25_roll3: 18.0,
      },
    },
    cluster: {
      actor: "actor1", task: "cluster",
      features: {
        zone_encoded: 2, mode_encoded: 1, mode_co2_mean: 45.3,
        annee: 2023, mois_sin: 0.5, mois_cos: 0.866,
        pm25: 18.4, no2: 32.1, aqi_index: 75.0,
        co2_lag1: 42.0, co2_lag3: 39.5,
        energie_lag1: 120.0, energie_lag3: 115.0,
        aqi_lag1: 72.0, pm25_lag1: 17.8,
        co2_roll3: 40.8, energie_roll3: 117.5, pm25_roll3: 18.0,
      },
    },
  },
  actor2: {
    charge: {
      actor: "actor2", task: "charge",
      features: {
        heure: 14, jour_semaine: 2, zone_encoded: 1,
        temperature: 22.5, precipitation: 0, evenement: 0,
        capacite_ligne: 350, retard_moyen: 3.2,
        charge_lag1: 0.72, charge_lag3: 0.68, charge_roll3: 0.70,
      },
    },
    cancellation: {
      actor: "actor2", task: "cancellation",
      features: {
        heure: 14, jour_semaine: 2, zone_encoded: 1,
        temperature: 22.5, precipitation: 0, evenement: 0,
        capacite_ligne: 350, retard_moyen: 3.2,
        charge_lag1: 0.72, charge_lag3: 0.68, charge_roll3: 0.70,
      },
    },
  },
  actor3: {
    severity: {
      actor: "actor3", task: "severity",
      features: {
        heure: 10, type_incident_encoded: 2, zone_encoded: 3,
        meteo_encoded: 1, visibilite: 8.5, densite_trafic: 0.65,
        infrastructure_age: 12, maintenance_score: 7.2,
        incidents_lag1: 2, incidents_lag3: 5,
      },
    },
    risk_cluster: {
      actor: "actor3", task: "risk_cluster",
      features: {
        heure: 10, type_incident_encoded: 2, zone_encoded: 3,
        meteo_encoded: 1, visibilite: 8.5, densite_trafic: 0.65,
        infrastructure_age: 12, maintenance_score: 7.2,
        incidents_lag1: 2, incidents_lag3: 5,
      },
    },
    anomaly: {
      actor: "actor3", task: "anomaly",
      features: {
        heure: 10, type_incident_encoded: 2, zone_encoded: 3,
        meteo_encoded: 1, visibilite: 8.5, densite_trafic: 0.65,
        infrastructure_age: 12, maintenance_score: 7.2,
        incidents_lag1: 2, incidents_lag3: 5,
      },
    },
  },
};

const ACTOR_TASKS = {
  actor1: ["co2", "energy", "cluster"],
  actor2: ["charge", "cancellation"],
  actor3: ["severity", "risk_cluster", "anomaly"],
};

const ACTOR_LABELS = {
  actor1: "🌿 Ecological Director",
  actor2: "🚌 Mobility Director",
  actor3: "🛡️ Security Manager",
};

const EXTERNAL_SERVICES = [
  { name: "MLflow",     port: "5000", url: process.env.NEXT_PUBLIC_MLFLOW_URL     ?? "http://localhost:5000", Icon: FlaskConical, color: "#006b5a", bg: "rgba(0,107,90,0.08)" },
  { name: "Grafana",    port: "3001", url: process.env.NEXT_PUBLIC_GRAFANA_URL    ?? "http://localhost:3001", Icon: BarChart3,    color: "#000018", bg: "rgba(0,0,24,0.06)" },
  { name: "Prometheus", port: "9090", url: process.env.NEXT_PUBLIC_PROMETHEUS_URL ?? "http://localhost:9090", Icon: Activity,     color: "#e6522c", bg: "rgba(230,82,44,0.08)" },
  { name: "n8n",        port: "5678", url: process.env.NEXT_PUBLIC_N8N_URL        ?? "http://localhost:5678", Icon: Workflow,     color: "#7c3aed", bg: "rgba(124,58,237,0.08)" },
];

const GRAFANA_PANELS = [
  { id: "traffic",    title: "API Traffic",         panelId: 1, description: "ml_api_requests_total — requests per second by actor" },
  { id: "latency",    title: "Request Latency",      panelId: 2, description: "ml_api_request_duration_seconds — p50/p95/p99" },
  { id: "errors",     title: "Error Rate",            panelId: 3, description: "ml_api_error_rate — errors per actor" },
  { id: "confidence", title: "Model Confidence",      panelId: 4, description: "ml_api_model_confidence — per actor" },
  { id: "health",     title: "Data Health",           panelId: 5, description: "Feature drift & data quality scores" },
];

// ── Skeleton loader ──────────────────────────────────────────────────────────
function Skeleton({ className = "" }) {
  return (
    <div className={`bg-gray-200 rounded animate-pulse ${className}`} />
  );
}

// ── JSON pretty renderer ─────────────────────────────────────────────────────
function JsonDisplay({ data, isError }) {
  if (!data) return null;
  return (
    <pre
      className={`mt-3 p-3 rounded-xl text-xs font-mono overflow-auto max-h-48 leading-relaxed
        ${isError
          ? "bg-red-50 border border-red-200 text-red-800"
          : "bg-[#f0f7f5] border border-[#006b5a]/20 text-[#003d2e]"
        }`}
    >
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

// ── Section title ────────────────────────────────────────────────────────────
function SectionTitle({ children }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <h2 className="text-[13px] font-bold text-[#777682] uppercase tracking-[0.12em]">
        {children}
      </h2>
      <div className="flex-1 h-px bg-gray-200" />
    </div>
  );
}

// ── SECTION 1: System Health ─────────────────────────────────────────────────
function HealthCard() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/mlops/health");
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth({ status: "error", detail: "Cannot reach API" });
    } finally {
      setLoading(false);
      setLastUpdated(new Date());
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    const id = setInterval(fetchHealth, 30000);
    return () => clearInterval(id);
  }, [fetchHealth]);

  const isOk = health?.status === "ok";

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-6 animate-fade-in-up">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          {loading ? (
            <div className="w-3 h-3 rounded-full bg-gray-300 animate-pulse" />
          ) : (
            <div className={`w-3 h-3 rounded-full ${isOk ? "bg-green-500" : "bg-red-500"}`}
              style={isOk ? { boxShadow: "0 0 0 3px rgba(34,197,94,0.25)" } : {}} />
          )}
          <span className="text-[15px] font-bold text-[#0b1c30]">API Health</span>
        </div>
        <button
          onClick={fetchHealth}
          className="flex items-center gap-1.5 text-[12px] text-[#777682] hover:text-[#000018] transition-colors cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          {lastUpdated && <span>{lastUpdated.toLocaleTimeString()}</span>}
        </button>
      </div>

      {loading ? (
        <div className="space-y-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-56" />
          <div className="grid grid-cols-3 gap-3 mt-4">
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </div>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-4 mb-4">
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wide
              ${isOk ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
              {isOk ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
              {health?.status}
            </span>
            {health?.version && (
              <span className="text-[12px] text-[#777682] font-mono">v{health.version}</span>
            )}
            {health?.predictions_stored !== undefined && (
              <span className="text-[12px] text-[#777682]">
                <span className="font-bold text-[#000018]">{health.predictions_stored}</span> predictions stored
              </span>
            )}
          </div>

          {health?.actors && (
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(health.actors).map(([actorId, meta]) => (
                <div key={actorId}
                  className="bg-[#f8f9ff] rounded-xl p-3.5 border border-[#e5eeff]">
                  <div className="text-[12px] font-bold text-[#000018] mb-1">{actorId}</div>
                  <div className="text-[11px] text-[#777682] mb-2 leading-tight">{meta.description?.split("(")[0].trim()}</div>
                  <div className="flex flex-wrap gap-1">
                    {meta.tasks?.map((t) => (
                      <span key={t}
                        className="px-1.5 py-0.5 bg-[#006b5a]/10 text-[#006b5a] rounded-md text-[10px] font-semibold">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {!isOk && health?.detail && (
            <p className="mt-3 text-[12px] text-red-600 bg-red-50 rounded-lg px-3 py-2">{health.detail}</p>
          )}
        </>
      )}
    </div>
  );
}

// ── SECTION 1B: Drift Monitor ─────────────────────────────────────────────────────
const ACTOR_DRIFT_LABELS = {
  actor1: { label: "🌿 Ecological", color: "#006b5a" },
  actor2: { label: "🚌 Mobility",   color: "#1a56db" },
  actor3: { label: "🛡️ Security",   color: "#ba1a1a" },
};

function KsBar({ value }) {
  // value 0.0–1.0; threshold 0.3 = drift
  const pct    = Math.min(value * 100, 100);
  const isDrift = value > 0.3;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
        <div
          className="h-1.5 rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            backgroundColor: isDrift ? "#ba1a1a" : "#006b5a",
          }}
        />
      </div>
      <span className={`text-[10px] font-mono font-bold shrink-0 ${isDrift ? "text-red-600" : "text-[#777682]"}`}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function DriftCard() {
  const [drift, setDrift]           = useState(null);
  const [loading, setLoading]       = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchDrift = useCallback(async () => {
    try {
      const res  = await fetch("/api/mlops/drift");
      const data = await res.json();
      setDrift(data);
    } catch {
      setDrift({ overall_status: "error", actors: {} });
    } finally {
      setLoading(false);
      setLastUpdated(new Date());
    }
  }, []);

  useEffect(() => {
    fetchDrift();
    const id = setInterval(fetchDrift, 60000);
    return () => clearInterval(id);
  }, [fetchDrift]);

  const hasDrift = drift?.overall_status === "drift";
  const actors   = drift?.actors ?? {};

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-6 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-[#000018]" />
          <span className="text-[15px] font-bold text-[#0b1c30]">Drift Monitor</span>
        </div>
        <button
          onClick={fetchDrift}
          className="flex items-center gap-1.5 text-[12px] text-[#777682] hover:text-[#000018] transition-colors cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          {lastUpdated && <span>{lastUpdated.toLocaleTimeString()}</span>}
        </button>
      </div>

      {/* Drift warning banner */}
      {!loading && hasDrift && (
        <div className="flex items-center gap-2 px-3 py-2 mb-4 rounded-xl bg-amber-50 border border-amber-200">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
          <span className="text-[12px] font-semibold text-amber-700">
            Distribution drift or confidence drop detected — review models
          </span>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="space-y-3">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      ) : (
        <div className="space-y-3">
          {Object.entries(actors).map(([actorId, info]) => {
            const meta     = ACTOR_DRIFT_LABELS[actorId] ?? { label: actorId, color: "#777682" };
            const isDrift  = info.status === "drift";
            return (
              <div key={actorId} className="bg-[#f8f9ff] rounded-xl px-4 py-3 border border-[#e5eeff]">
                {/* Row 1: actor + status badge */}
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[12px] font-bold" style={{ color: meta.color }}>
                    {meta.label}
                  </span>
                  <div className="flex items-center gap-2">
                    {info.confidence_drop && (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-red-50 border border-red-200 text-red-600">
                        CONF ↓
                      </span>
                    )}
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide
                      ${isDrift
                        ? "bg-red-100 text-red-700 border border-red-200"
                        : "bg-green-100 text-green-700 border border-green-200"}`}
                    >
                      {isDrift
                        ? <><XCircle className="w-2.5 h-2.5" /> DRIFT</>
                        : <><CheckCircle2 className="w-2.5 h-2.5" /> OK</>
                      }
                    </span>
                  </div>
                </div>

                {/* Row 2: KS bar + record count */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-[#777682] font-semibold">KS stat</span>
                    <span className="text-[10px] text-[#777682]">{info.record_count ?? 0} records</span>
                  </div>
                  <KsBar value={info.ks_stat ?? 0} />
                </div>
              </div>
            );
          })}

          {Object.keys(actors).length === 0 && (
            <div className="text-center py-6 text-[12px] text-[#777682]">
              No prediction records yet — run predictions first
            </div>
          )}
        </div>
      )}

      {/* Footer: last checked */}
      {drift?.last_checked && (
        <p className="mt-3 text-[10px] text-[#777682] text-right font-mono">
          Checked: {new Date(drift.last_checked).toLocaleTimeString()} · auto-refresh 60s
        </p>
      )}
    </div>
  );
}

// ── SECTION 2A: Run Prediction ─────────────────────────────────────────────────
function PredictionCard() {
  const [actor, setActor] = useState("actor1");
  const [task, setTask] = useState("co2");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isError, setIsError] = useState(false);

  // Sync task when actor changes
  useEffect(() => {
    setTask(ACTOR_TASKS[actor][0]);
    setResult(null);
  }, [actor]);

  const handleExecute = async () => {
    setLoading(true);
    setResult(null);
    setIsError(false);
    try {
      const payload = DEMO_PAYLOADS[actor]?.[task] || DEMO_PAYLOADS[actor][ACTOR_TASKS[actor][0]];
      const res = await fetch("/api/mlops/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResult(data);
      setIsError(!res.ok);
    } catch (e) {
      setResult({ status: "error", detail: e.message });
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-6 flex flex-col animate-fade-in-up">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-[#000018]/5 flex items-center justify-center">
          <Play className="w-4 h-4 text-[#000018]" />
        </div>
        <div>
          <div className="text-[14px] font-bold text-[#0b1c30]">Run Prediction</div>
          <div className="text-[11px] text-[#777682]">Live inference via FastAPI</div>
        </div>
      </div>

      <div className="space-y-3 mb-4">
        {/* Actor dropdown */}
        <div className="relative">
          <select
            value={actor}
            onChange={(e) => setActor(e.target.value)}
            className="w-full appearance-none bg-[#f8f9ff] border border-[#e5eeff] rounded-xl px-3 py-2.5 text-[13px] text-[#0b1c30] font-medium cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#000018]/20"
          >
            {Object.keys(ACTOR_TASKS).map((a) => (
              <option key={a} value={a}>{ACTOR_LABELS[a]}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#777682] pointer-events-none" />
        </div>

        {/* Task dropdown */}
        <div className="relative">
          <select
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="w-full appearance-none bg-[#f8f9ff] border border-[#e5eeff] rounded-xl px-3 py-2.5 text-[13px] text-[#0b1c30] font-medium cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#000018]/20"
          >
            {ACTOR_TASKS[actor].map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#777682] pointer-events-none" />
        </div>
      </div>

      <button
        onClick={handleExecute}
        disabled={loading}
        className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl bg-[#000018] text-white text-[13px] font-semibold
          hover:bg-[#000018]/85 active:scale-[0.98] transition-all duration-150 disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? "Running…" : "Execute"}
      </button>

      {result && (
        <>
          <div className="flex items-center gap-1.5 mt-3">
            {isError
              ? <XCircle className="w-3.5 h-3.5 text-red-500" />
              : <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />}
            <span className={`text-[11px] font-semibold ${isError ? "text-red-500" : "text-green-600"}`}>
              {isError ? "Error" : `${result.latency_ms?.toFixed(0) ?? "—"} ms`}
            </span>
          </div>
          <JsonDisplay data={result} isError={isError} />
        </>
      )}
    </div>
  );
}

// ── SECTION 2B: Retrain ──────────────────────────────────────────────────────
function RetrainCard() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isError, setIsError] = useState(false);

  const handleRetrain = async () => {
    if (!window.confirm("⚠️ This will retrain all 3 actor models sequentially.\nThis may take several minutes. Continue?")) return;
    setLoading(true);
    setResult(null);
    setIsError(false);
    try {
      const res = await fetch("/api/mlops/retrain", { method: "POST" });
      const data = await res.json();
      setResult(data);
      setIsError(!res.ok || !data.all_success);
    } catch (e) {
      setResult({ status: "error", detail: e.message });
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-6 flex flex-col animate-fade-in-up">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-amber-50 flex items-center justify-center">
          <RotateCcw className="w-4 h-4 text-amber-600" />
        </div>
        <div>
          <div className="text-[14px] font-bold text-[#0b1c30]">Retrain All Models</div>
          <div className="text-[11px] text-[#777682]">Triggers all 3 actor pipelines</div>
        </div>
      </div>

      <div className="flex-1 flex flex-col justify-between">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 mb-4">
          <p className="text-[12px] text-amber-700 leading-relaxed">
            Runs <span className="font-bold">actor1 → actor2 → actor3</span> training scripts sequentially via <code className="font-mono bg-amber-100 px-1 rounded">POST /retrain</code>. Up to 10 min timeout.
          </p>
        </div>

        <button
          onClick={handleRetrain}
          disabled={loading}
          className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl
            bg-amber-500 text-white text-[13px] font-semibold
            hover:bg-amber-600 active:scale-[0.98] transition-all duration-150
            disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
        >
          {loading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Retraining… (may take minutes)</>
            : <><RotateCcw className="w-4 h-4" /> Retrain All Models</>}
        </button>

        {result && (
          <>
            <div className="flex items-center gap-1.5 mt-3">
              {isError
                ? <XCircle className="w-3.5 h-3.5 text-red-500" />
                : <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />}
              <span className={`text-[11px] font-semibold ${isError ? "text-red-500" : "text-green-600"}`}>
                {isError ? "Partial failure" : "All models retrained"}
              </span>
            </div>
            <JsonDisplay data={result} isError={isError} />
          </>
        )}
      </div>
    </div>
  );
}

// ── SECTION 2C: Prediction History ──────────────────────────────────────────
function HistoryCard() {
  const [count, setCount] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/mlops/health")
      .then((r) => r.json())
      .then((d) => setCount(d.predictions_stored ?? 0))
      .catch(() => setCount("—"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-6 flex flex-col animate-fade-in-up">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-[#006b5a]/10 flex items-center justify-center">
          <Database className="w-4 h-4 text-[#006b5a]" />
        </div>
        <div>
          <div className="text-[14px] font-bold text-[#0b1c30]">Prediction History</div>
          <div className="text-[11px] text-[#777682]">Stored in results/predictions.json</div>
        </div>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center py-4">
        {loading ? (
          <Skeleton className="h-16 w-24" />
        ) : (
          <>
            <div className="text-5xl font-black text-[#000018] leading-none mb-1">{count}</div>
            <div className="text-[12px] text-[#777682] uppercase tracking-wide font-semibold">predictions stored</div>
          </>
        )}
      </div>

      <div className="space-y-2 mt-4">
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl
            bg-[#006b5a] text-white text-[13px] font-semibold
            hover:bg-[#005a4b] active:scale-[0.98] transition-all duration-150"
        >
          <ExternalLink className="w-4 h-4" />
          View in API Docs
        </a>
        <a
          href="http://localhost:8000/predictions"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl
            border border-[#e5eeff] text-[#0b1c30] text-[13px] font-semibold
            hover:bg-[#f8f9ff] active:scale-[0.98] transition-all duration-150"
        >
          <Database className="w-4 h-4 text-[#777682]" />
          Browse Raw JSON
        </a>
      </div>
    </div>
  );
}

// ── SECTION 2D: n8n Workflow Triggers ────────────────────────────────────────────
function N8nWorkflowCard() {
  const [n8nStatus, setN8nStatus]           = useState(null);   // null=checking, { online, workflows, ... }
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResult, setPredictResult]   = useState(null);
  const [predictError, setPredictError]     = useState(false);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [retrainResult, setRetrainResult]   = useState(null);
  const [retrainError, setRetrainError]     = useState(false);

  // Check n8n status through authenticated server-side route
  const checkN8nStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/mlops/n8n-status", { signal: AbortSignal.timeout(8000) });
      const data = await res.json();
      setN8nStatus(data);
    } catch {
      setN8nStatus({ online: false, detail: "Cannot reach status endpoint" });
    }
  }, []);

  useEffect(() => {
    checkN8nStatus();
    const id = setInterval(checkN8nStatus, 30000);
    return () => clearInterval(id);
  }, [checkN8nStatus]);

  const n8nOnline = n8nStatus?.online ?? null;

  const handleTriggerPredict = async () => {
    setPredictLoading(true);
    setPredictResult(null);
    setPredictError(false);
    try {
      const res = await fetch("/api/mlops/n8n-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor: "actor1", task: "co2", features: {} }),
        signal: AbortSignal.timeout(40000),
      });
      const data = await res.json();
      setPredictResult(data);
      setPredictError(!res.ok || data.status === "error");
    } catch (e) {
      setPredictResult({ status: "error", detail: e.message });
      setPredictError(true);
    } finally {
      setPredictLoading(false);
    }
  };

  const handleTriggerRetrain = async () => {
    if (!window.confirm(
      "This will trigger the n8n Retraining Workflow (or /retrain directly as fallback).\nThis may take several minutes. Continue?"
    )) return;
    setRetrainLoading(true);
    setRetrainResult(null);
    setRetrainError(false);
    try {
      const res = await fetch("/api/mlops/n8n-retrain", {
        method: "POST",
        signal: AbortSignal.timeout(640000),
      });
      const data = await res.json();
      setRetrainResult(data);
      setRetrainError(!res.ok || data.status === "error");
    } catch (e) {
      setRetrainResult({ status: "error", detail: e.message });
      setRetrainError(true);
    } finally {
      setRetrainLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-6 flex flex-col animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-[#7c3aed]/10 flex items-center justify-center">
            <Workflow className="w-4 h-4 text-[#7c3aed]" />
          </div>
          <div>
            <div className="text-[14px] font-bold text-[#0b1c30]">n8n Workflows</div>
            <div className="text-[11px] text-[#777682]">Orchestration triggers</div>
          </div>
        </div>
        {/* n8n status badge */}
        <div className="flex items-center gap-2">
          <button
            onClick={checkN8nStatus}
            className="text-[#777682] hover:text-[#000018] transition-colors cursor-pointer"
            title="Refresh n8n status"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
          <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border ${
            n8nOnline === null
              ? "bg-gray-50 border-gray-200 text-gray-400"
              : n8nOnline
                ? "bg-green-50 border-green-200 text-green-700"
                : "bg-amber-50 border-amber-200 text-amber-700"
          }`}>
            {n8nOnline === null && <Loader2 className="w-2.5 h-2.5 animate-spin" />}
            {n8nOnline === true  && <CheckCircle2 className="w-2.5 h-2.5" />}
            {n8nOnline === false && <AlertTriangle className="w-2.5 h-2.5" />}
            {n8nOnline === null ? "Checking..." : n8nOnline ? "n8n online" : "n8n offline"}
          </span>
        </div>
      </div>

      {/* Workflow list when online */}
      {n8nStatus?.online && n8nStatus.workflows?.length > 0 && (
        <div className="mb-4 space-y-1.5">
          <div className="text-[11px] text-[#777682] font-semibold uppercase tracking-wide">
            {n8nStatus.workflow_count} Workflow{n8nStatus.workflow_count !== 1 ? "s" : ""}
          </div>
          {n8nStatus.workflows.map((wf) => (
            <div key={wf.id ?? wf.name}
              className="flex items-center justify-between bg-[#f8f9ff] rounded-lg px-3 py-1.5 border border-[#e5eeff]">
              <span className="text-[12px] text-[#0b1c30] font-medium truncate">{wf.name}</span>
              <span className={`inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full ${
                wf.active
                  ? "bg-green-100 text-green-700"
                  : "bg-gray-100 text-gray-500"
              }`}>
                {wf.active ? <CheckCircle2 className="w-2 h-2" /> : <XCircle className="w-2 h-2" />}
                {wf.active ? "Active" : "Inactive"}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 space-y-4">
        {/* Button 1: Trigger Prediction Pipeline */}
        <div className="space-y-2">
          <button
            onClick={handleTriggerPredict}
            disabled={predictLoading || retrainLoading}
            className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl
              bg-[#7c3aed] text-white text-[13px] font-semibold
              hover:bg-[#6d28d9] active:scale-[0.98] transition-all duration-150
              disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
          >
            {predictLoading
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Running pipeline...</>
              : <><Zap className="w-4 h-4" /> Trigger Prediction Pipeline</>}
          </button>
          {predictResult && (
            <>
              <div className="flex items-center gap-1.5">
                {predictError
                  ? <XCircle className="w-3.5 h-3.5 text-red-500" />
                  : <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />}
                <span className={`text-[11px] font-semibold ${
                  predictError ? "text-red-500" : "text-green-600"
                }`}>
                  {predictError ? "Pipeline error" : `Triggered via ${predictResult.webhook ? "webhook" : "API"}`}
                </span>
              </div>
              <JsonDisplay data={predictResult} isError={predictError} />
            </>
          )}
        </div>

        {/* Divider */}
        <div className="h-px bg-[#e5eeff]" />

        {/* Button 2: Trigger Retraining Pipeline */}
        <div className="space-y-2">
          <button
            onClick={handleTriggerRetrain}
            disabled={retrainLoading || predictLoading}
            className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl
              border-2 border-[#7c3aed] text-[#7c3aed] text-[13px] font-semibold
              hover:bg-[#7c3aed]/5 active:scale-[0.98] transition-all duration-150
              disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
          >
            {retrainLoading
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Triggering... (may take minutes)</>
              : <><RotateCcw className="w-4 h-4" /> Trigger Retraining Pipeline</>}
          </button>
          {retrainResult && (
            <>
              <div className="flex items-center gap-1.5">
                {retrainError
                  ? <XCircle className="w-3.5 h-3.5 text-red-500" />
                  : <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />}
                <span className={`text-[11px] font-semibold ${
                  retrainError ? "text-red-500" : "text-green-600"
                }`}>
                  {retrainError ? "Error" : `Triggered via ${retrainResult.method ?? "n8n"}`}
                </span>
              </div>
              <JsonDisplay data={retrainResult} isError={retrainError} />
            </>
          )}
        </div>
      </div>

      {/* Footer note when n8n offline */}
      {n8nOnline === false && (
        <p className="mt-3 text-[10px] text-amber-600 bg-amber-50 rounded-lg px-2 py-1.5 leading-relaxed">
          n8n unreachable — retraining will fall back to direct <code className="font-mono">/retrain</code> call.
        </p>
      )}
    </div>
  );
}

// ── SECTION 3: Monitoring Panels ─────────────────────────────────────────────
function GrafanaPanel({ title, description, panelId, iframeKey, onError, hasError }) {
  const src = `http://localhost:3001/d-solo/ml-api-dashboard/ml-api?orgId=1&panelId=${panelId}&refresh=10s&theme=light`;

  if (hasError) {
    return (
      <div className="bg-white rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] p-4 flex flex-col items-center justify-center gap-2 animate-fade-in-up"
        style={{ minHeight: "200px" }}>
        <AlertTriangle className="w-6 h-6 text-amber-500" />
        <div className="text-[13px] font-bold text-[#0b1c30]">{title}</div>
        <div className="text-[11px] text-[#777682] text-center">Grafana offline — start on :3001</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] overflow-hidden animate-fade-in-up">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#e5eeff]">
        <div>
          <div className="text-[13px] font-bold text-[#0b1c30]">{title}</div>
          <div className="text-[10px] text-[#777682] font-mono">{description}</div>
        </div>
        <a
          href={`http://localhost:3001/d/ml-api-dashboard/ml-api?orgId=1&viewPanel=${panelId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[10px] text-[#777682] hover:text-[#000018] flex items-center gap-1 transition-colors"
        >
          Expand <ExternalLink className="w-3 h-3" />
        </a>
      </div>
      <iframe
        key={iframeKey}
        src={src}
        title={title}
        width="100%"
        height="200"
        style={{ border: "none", display: "block" }}
        onError={() => onError(true)}
      />
    </div>
  );
}

function MonitoringSection() {
  const [iframeKey, setIframeKey] = useState(0);
  const [grafanaError, setGrafanaError] = useState(false);

  const handleRefresh = () => {
    setIframeKey((k) => k + 1);
    setGrafanaError(false);
  };

  return (
    <div className="space-y-4">
      {/* Header row with refresh button */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-[#000018]" />
          <span className="text-[13px] font-bold text-[#0b1c30]">Live Grafana Panels</span>
          <span className="text-[10px] font-mono text-[#777682] bg-[#f8f9ff] px-1.5 py-0.5 rounded">:3001</span>
        </div>
        <button
          onClick={handleRefresh}
          className="flex items-center gap-1.5 text-[12px] text-[#777682] hover:text-[#000018] transition-colors cursor-pointer px-2 py-1 rounded-lg hover:bg-[#f8f9ff]"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh All
        </button>
      </div>

      {/* Panels grid: 3 on top row, 2 on bottom */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {GRAFANA_PANELS.slice(0, 3).map((panel) => (
          <GrafanaPanel
            key={panel.id}
            {...panel}
            iframeKey={iframeKey}
            hasError={grafanaError}
            onError={setGrafanaError}
          />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {GRAFANA_PANELS.slice(3).map((panel) => (
          <GrafanaPanel
            key={panel.id}
            {...panel}
            iframeKey={iframeKey}
            hasError={grafanaError}
            onError={setGrafanaError}
          />
        ))}
      </div>
    </div>
  );
}

// ── Compact service link ─────────────────────────────────────────────────────
function CompactServiceLink({ name, port, url, Icon, color, bg }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="bg-white rounded-xl shadow-[0_2px_10px_rgba(0,0,0,0.04)] px-4 py-3 flex items-center gap-3
        hover:scale-[1.02] hover:shadow-md transition-all duration-200 cursor-pointer animate-fade-in-up group"
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
        style={{ backgroundColor: bg }}
      >
        <Icon className="w-4 h-4" style={{ color }} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[13px] font-bold text-[#0b1c30] group-hover:text-[#000018] truncate">{name}</div>
      </div>
      <span
        className="text-[10px] font-bold px-1.5 py-0.5 rounded-full border font-mono shrink-0"
        style={{ color, borderColor: `${color}30`, backgroundColor: bg }}
      >
        :{port}
      </span>
      <ExternalLink className="w-3.5 h-3.5 text-[#777682] group-hover:text-[#000018] shrink-0 transition-colors" />
    </a>
  );
}

// ── Root Component ───────────────────────────────────────────────────────────
export default function MLOpsDashboard() {
  return (
    <div className="space-y-8">
      {/* Page header */}
      <div className="animate-fade-in-up">
        <h1 className="text-[22px] font-black text-[#000018] tracking-tight">MLOps Control Panel</h1>
        <p className="text-[13px] text-[#777682] mt-0.5">
          Monitor, predict, and retrain — integrated with FastAPI · MLflow · Prometheus · Grafana · n8n
        </p>
      </div>

      {/* Section 1: Health + Drift */}
      <div>
        <SectionTitle>System Health</SectionTitle>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          <HealthCard />
          <DriftCard />
        </div>
      </div>

      {/* Section 2: ML Operations */}
      <div>
        <SectionTitle>ML Operations</SectionTitle>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <PredictionCard />
          <N8nWorkflowCard />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-5">
          <RetrainCard />
          <HistoryCard />
        </div>
      </div>

      {/* Section 3: Monitoring */}
      <div>
        <SectionTitle>Monitoring</SectionTitle>
        <MonitoringSection />
      </div>

      {/* Section 4: External Services (compact) */}
      <div>
        <SectionTitle>External Services</SectionTitle>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {EXTERNAL_SERVICES.map((svc) => (
            <CompactServiceLink key={svc.name} {...svc} />
          ))}
        </div>
      </div>

      {/* Keyframe styles */}
      <style>{`
        @keyframes fade-in-up {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.35s ease-out both;
        }
      `}</style>
    </div>
  );
}
