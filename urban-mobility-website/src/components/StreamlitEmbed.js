"use client";

import { useState, useEffect, useRef } from "react";
import { Brain, AlertTriangle, RefreshCw, ExternalLink, Loader2 } from "lucide-react";

const ACTOR_BADGES = {
  actor1: { label: "🌿 Ecological — Actor 1", color: "#006b5a", bg: "rgba(0,107,90,0.25)" },
  actor2: { label: "🚌 Mobility — Actor 2",   color: "#93c5fd", bg: "rgba(26,86,219,0.30)" },
  actor3: { label: "🛡️ Security — Actor 3",   color: "#fca5a5", bg: "rgba(186,26,26,0.30)" },
};

const STREAMLIT_BASE = process.env.NEXT_PUBLIC_STREAMLIT_URL ?? "http://localhost:8501";

export default function StreamlitEmbed({ actorFilter = null }) {
  // Connectivity check state
  const [checking,  setChecking]  = useState(true);  // pinging health on mount
  const [reachable, setReachable] = useState(null);   // null | true | false

  // Iframe state
  const [loaded,  setLoaded]  = useState(false);
  const [error,   setError]   = useState(false);
  const iframeKey = useRef(0);
  const [keyVal,  setKeyVal]  = useState(0);

  const src   = `${STREAMLIT_BASE}${actorFilter ? `?actor=${actorFilter}` : ""}`;
  const badge = actorFilter ? ACTOR_BADGES[actorFilter] : null;

  // ── On mount: server-side connectivity check via /api/mlops/health ──────────
  useEffect(() => {
    let cancelled = false;
    setChecking(true);

    fetch("/api/mlops/health", { signal: AbortSignal.timeout(5000) })
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) {
          setReachable(!!data.streamlit_reachable);
          setChecking(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          // Health endpoint itself unreachable — assume Streamlit also down
          setReachable(false);
          setChecking(false);
        }
      });

    return () => { cancelled = true; };
  }, [keyVal]); // re-run on retry

  const handleRetry = () => {
    iframeKey.current += 1;
    setKeyVal(iframeKey.current);
    setLoaded(false);
    setError(false);
    setChecking(true);
    setReachable(null);
  };

  // ── Shared error card ────────────────────────────────────────────────────────
  const ErrorCard = ({ title, message }) => (
    <div className="absolute inset-0 flex items-center justify-center bg-[#f8f9ff] z-10">
      <div className="text-center max-w-sm px-6">
        <div className="w-14 h-14 rounded-2xl bg-red-50 border border-red-100 flex items-center justify-center mx-auto mb-4">
          <AlertTriangle className="w-7 h-7 text-[#ba1a1a]" />
        </div>
        <h4 className="text-[15px] font-bold text-[#0b1c30] mb-1">{title}</h4>
        <p className="text-[13px] text-[#777682] mb-2 leading-relaxed">{message}</p>
        <p className="text-[12px] text-[#777682] mb-5">
          Start Streamlit with:{" "}
          <code className="font-mono text-[#000018] bg-gray-100 px-1.5 py-0.5 rounded text-[11px]">
            streamlit run app.py
          </code>
        </p>
        <div className="flex flex-col gap-2">
          <button
            onClick={handleRetry}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl
              bg-[#000018] text-white text-[13px] font-semibold
              hover:bg-[#000018]/85 active:scale-[0.98] transition-all duration-150 cursor-pointer"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
          <a
            href={STREAMLIT_BASE}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl
              border border-[#e5eeff] text-[#0b1c30] text-[13px] font-semibold
              hover:bg-[#f8f9ff] active:scale-[0.98] transition-all duration-150"
          >
            <ExternalLink className="w-4 h-4" />
            Open directly in browser
          </a>
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] overflow-hidden">
      {/* Header bar */}
      <div className="bg-[#000018] text-white px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center shrink-0">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-[15px] font-bold tracking-tight leading-tight">
              ML Predictions Dashboard
            </h3>
            <p className="text-[11px] text-white/50 uppercase tracking-[0.1em] font-semibold">
              Streamlit · {STREAMLIT_BASE.replace("http://", "")}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Streamlit reachability status pill */}
          <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1 rounded-full border ${
            checking
              ? "bg-gray-900/20 border-white/10 text-white/50"
              : reachable
                ? "bg-green-500/20 border-green-400/30 text-green-300"
                : "bg-red-500/20 border-red-400/30 text-red-300"
          }`}>
            {checking && <Loader2 className="w-2.5 h-2.5 animate-spin" />}
            {!checking && reachable  && <span className="w-1.5 h-1.5 rounded-full bg-green-400" />}
            {!checking && !reachable && <span className="w-1.5 h-1.5 rounded-full bg-red-400" />}
            {checking ? "Checking…" : reachable ? "Online" : "Offline"}
          </span>

          {badge && (
            <span
              className="text-[11px] font-bold px-3 py-1 rounded-full border"
              style={{
                color: badge.color,
                backgroundColor: badge.bg,
                borderColor: `${badge.color}50`,
              }}
            >
              {badge.label}
            </span>
          )}
        </div>
      </div>

      {/* Iframe / Status area */}
      <div className="relative w-full" style={{ minHeight: "850px" }}>

        {/* 1. Checking connectivity */}
        {checking && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[#f8f9ff] z-10">
            <div className="w-12 h-12 rounded-2xl bg-[#000018]/5 flex items-center justify-center animate-pulse">
              <Brain className="w-6 h-6 text-[#000018]/30" />
            </div>
            <div className="space-y-2 w-48">
              <div className="h-2.5 bg-gray-200 rounded animate-pulse" />
              <div className="h-2.5 bg-gray-200 rounded animate-pulse w-3/4 mx-auto" />
            </div>
            <p className="text-[12px] text-[#777682]">Checking Streamlit connectivity…</p>
          </div>
        )}

        {/* 2. Streamlit not reachable — show error immediately */}
        {!checking && !reachable && (
          <ErrorCard
            title="Streamlit is not running"
            message={`Could not reach ${STREAMLIT_BASE}. The server may be stopped or still starting up.`}
          />
        )}

        {/* 3. Reachable but iframe errored */}
        {!checking && reachable && error && (
          <ErrorCard
            title="Iframe failed to load"
            message="Streamlit is reachable but the iframe could not render. Check browser console for details."
          />
        )}

        {/* 4. Reachable — show iframe (with loading shimmer until onLoad) */}
        {!checking && reachable && !error && (
          <>
            {/* Skeleton while iframe loads */}
            {!loaded && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[#f8f9ff] z-10">
                <div className="w-12 h-12 rounded-2xl bg-[#000018]/5 flex items-center justify-center animate-pulse">
                  <Brain className="w-6 h-6 text-[#000018]/30" />
                </div>
                <div className="space-y-2 w-48">
                  <div className="h-2.5 bg-gray-200 rounded animate-pulse" />
                  <div className="h-2.5 bg-gray-200 rounded animate-pulse w-3/4 mx-auto" />
                </div>
                <p className="text-[12px] text-[#777682]">Loading Streamlit…</p>
              </div>
            )}

            <iframe
              key={keyVal}
              src={src}
              title="ML Predictions Dashboard"
              width="100%"
              style={{ minHeight: "850px", border: "none", display: "block" }}
              sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
              referrerPolicy="no-referrer-when-downgrade"
              onLoad={() => setLoaded(true)}
              onError={() => setError(true)}
            />
          </>
        )}
      </div>
    </div>
  );
}
