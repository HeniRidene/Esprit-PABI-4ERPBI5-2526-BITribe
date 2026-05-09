"use client";

import { useAuth, ROLES, getPowerBiEmbedUrl } from "@/context/AuthContext";
import { useState, useEffect, useRef, useCallback } from "react";
import { BarChart3, ExternalLink, AlertTriangle } from "lucide-react";
import ErrorBoundary from "./ErrorBoundary";
import IntroductionPage from "./IntroductionPage";
import LoginPage from "./LoginPage";
import AboutUsPage from "./AboutUsPage";
import MLOpsDashboard from "./MLOpsDashboard";
import StreamlitEmbed from "./StreamlitEmbed";

// ── localStorage key helpers ─────────────────────────────────────────────────
const PBI_LOAD_KEY = (pageId) => `pbi_loaded_${pageId}`;

export function markPbiLoaded(pageId) {
  try {
    localStorage.setItem(PBI_LOAD_KEY(pageId), "1");
    // Dispatch custom event so Sidebar can react without a full re-render
    window.dispatchEvent(new CustomEvent("pbi-loaded", { detail: { pageId } }));
  } catch (_) {}
}

export function isPbiLoaded(pageId) {
  try {
    return localStorage.getItem(PBI_LOAD_KEY(pageId)) === "1";
  } catch (_) {
    return false;
  }
}

// ── Top loading bar ───────────────────────────────────────────────────────────
function TopLoadingBar({ visible }) {
  return (
    <>
      <div
        className="top-loading-bar"
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: "3px",
          zIndex: 9999,
          backgroundColor: "#006b5a",
          transformOrigin: "left center",
          animation: visible ? "loading-bar-in 0.8s ease-out forwards" : "none",
          opacity: visible ? 1 : 0,
          transition: "opacity 0.2s",
        }}
      />
      <style>{`
        @keyframes loading-bar-in {
          0%   { transform: scaleX(0); opacity: 1; }
          70%  { transform: scaleX(0.85); opacity: 1; }
          100% { transform: scaleX(1); opacity: 0; }
        }
      `}</style>
    </>
  );
}

// ── Power BI iframe with skeleton + error boundary ───────────────────────────
const PBI_PAGE_LABELS = {
  "eco-page-1": "Emissions & Energy",
  "eco-page-2": "Green Projects",
  "mob-page-1": "Network Performance",
  "sec-page-1": "Security & Maintenance",
};

function PowerBiEmbed({ embedUrl, pageId, pageLabel }) {
  const [loaded, setLoaded] = useState(false);
  const [timedOut, setTimedOut] = useState(false);
  const timeoutRef = useRef(null);

  // Reset state when page changes
  useEffect(() => {
    setLoaded(false);
    setTimedOut(false);
    // 10-second error boundary
    timeoutRef.current = setTimeout(() => {
      setTimedOut(true);
    }, 10000);
    return () => clearTimeout(timeoutRef.current);
  }, [pageId]);

  const handleLoad = useCallback(() => {
    clearTimeout(timeoutRef.current);
    setLoaded(true);
    setTimedOut(false);
    markPbiLoaded(pageId);
  }, [pageId]);

  const reportLabel = PBI_PAGE_LABELS[pageId] || pageLabel;

  // Error boundary: iframe timed out
  if (timedOut && !loaded) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#f8f9ff] min-h-[700px]">
        <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.06)] p-10 max-w-md w-full mx-4 text-center">
          {/* Power BI logo placeholder */}
          <div className="w-16 h-16 rounded-2xl bg-[#f2c811]/10 border border-[#f2c811]/30 flex items-center justify-center mx-auto mb-5">
            <BarChart3 className="w-8 h-8 text-[#f2c811]" />
          </div>
          <div
            className="text-[11px] font-bold uppercase tracking-[0.15em] mb-2"
            style={{ color: "#f2c811" }}
          >
            Power BI
          </div>
          <h3 className="text-[17px] font-bold text-[#0b1c30] mb-2">{reportLabel}</h3>
          <p className="text-[13px] text-[#777682] mb-6 leading-relaxed">
            The report did not load within 10 seconds. This usually means you need
            to authenticate with your Microsoft account.
          </p>
          <div className="space-y-3">
            <a
              href={embedUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 w-full py-3 px-5 rounded-xl
                bg-[#000018] text-white text-[13px] font-semibold
                hover:bg-[#000018]/85 active:scale-[0.98] transition-all duration-150"
            >
              <ExternalLink className="w-4 h-4" />
              Open in Power BI
            </a>
            <button
              onClick={() => { setTimedOut(false); setLoaded(false); }}
              className="flex items-center justify-center gap-2 w-full py-2.5 px-5 rounded-xl
                border border-[#e5eeff] text-[#0b1c30] text-[13px] font-semibold
                hover:bg-[#f8f9ff] active:scale-[0.98] transition-all duration-150 cursor-pointer"
            >
              Retry
            </button>
          </div>
          <div className="flex items-center gap-1.5 justify-center mt-5 text-[11px] text-[#777682]">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
            Embed requires Power BI Pro + active Microsoft session
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex-1 w-full min-h-[700px] md:min-h-[850px]">
      {/* Animated skeleton — shown while loading */}
      {!loaded && (
        <div className="absolute inset-0 z-10 bg-[#f8f9ff] flex flex-col gap-4 p-6">
          {/* Skeleton header bar */}
          <div className="h-10 w-56 rounded-xl bg-gray-200 animate-pulse" />
          {/* Skeleton chart rows */}
          <div className="flex gap-4 mt-2">
            <div className="h-32 flex-1 rounded-2xl bg-gray-200 animate-pulse" style={{ animationDelay: "0.05s" }} />
            <div className="h-32 flex-1 rounded-2xl bg-gray-200 animate-pulse" style={{ animationDelay: "0.1s" }} />
            <div className="h-32 flex-1 rounded-2xl bg-gray-200 animate-pulse" style={{ animationDelay: "0.15s" }} />
          </div>
          <div className="h-64 w-full rounded-2xl bg-gray-200 animate-pulse mt-1" style={{ animationDelay: "0.2s" }} />
          <div className="flex gap-4">
            <div className="h-48 flex-1 rounded-2xl bg-gray-200 animate-pulse" style={{ animationDelay: "0.25s" }} />
            <div className="h-48 w-1/3 rounded-2xl bg-gray-200 animate-pulse" style={{ animationDelay: "0.3s" }} />
          </div>
          {/* Loading label */}
          <div className="absolute bottom-6 left-0 right-0 flex items-center justify-center">
            <span className="text-[12px] text-[#777682] bg-white px-3 py-1.5 rounded-full shadow-sm border border-[#e5eeff]">
              Loading {reportLabel}…
            </span>
          </div>
        </div>
      )}

      {/* The actual iframe */}
      <iframe
        id="powerbi-embed"
        key={pageId}
        title={`Power BI — ${reportLabel}`}
        src={embedUrl}
        className="w-full h-full border-0 absolute inset-0"
        style={{ opacity: loaded ? 1 : 0, transition: "opacity 0.3s ease" }}
        allowFullScreen={true}
        onLoad={handleLoad}
      />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function DashboardMain() {
  const { user, activePage, showLoginPrompt } = useAuth();
  const role = user ? ROLES[user.role] : null;

  // ── Loading bar state ──────────────────────────────────────────────────────
  const [showBar, setShowBar] = useState(false);
  const prevPageId = useRef(activePage?.id);

  useEffect(() => {
    if (activePage?.id !== prevPageId.current) {
      prevPageId.current = activePage?.id;
      setShowBar(true);
      const t = setTimeout(() => setShowBar(false), 850);
      return () => clearTimeout(t);
    }
  }, [activePage?.id]);

  if (showLoginPrompt && !user) {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff] flex items-center justify-center">
        <TopLoadingBar visible={showBar} />
        <LoginPage />
      </main>
    );
  }

  // If the active page is the introduction/accueil page
  if (activePage?.id === "accueil") {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
        <TopLoadingBar visible={showBar} />
        <IntroductionPage />
      </main>
    );
  }

  // If the active page is about us
  if (activePage?.id === "about") {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
        <TopLoadingBar visible={showBar} />
        <AboutUsPage />
      </main>
    );
  }

  // MLOps Control Panel
  if (activePage?.id === "mlops") {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
        <TopLoadingBar visible={showBar} />
        <ErrorBoundary label="MLOps Dashboard">
          <MLOpsDashboard />
        </ErrorBoundary>
      </main>
    );
  }

  // Streamlit ML Predictions (all actors or actor-filtered)
  if (activePage?.id === "streamlit" || activePage?.id?.startsWith("streamlit-")) {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
        <TopLoadingBar visible={showBar} />
        <ErrorBoundary label="ML Predictions">
          <StreamlitEmbed actorFilter={activePage?.actorFilter || null} />
        </ErrorBoundary>
      </main>
    );
  }

  // Build the Power BI embed URL based on active page
  const embedUrl = activePage
    ? getPowerBiEmbedUrl(user.role, activePage.pageName)
    : getPowerBiEmbedUrl(user.role);

  const getSectionTitle = () => activePage?.label || "Analytics Dashboard";

  return (
    <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
      <TopLoadingBar visible={showBar} />

      {/* Power BI Dashboard Container */}
      <div className="bg-white rounded-2xl overflow-hidden animate-fade-in-up border border-outline-variant/20 shadow-[0px_4px_20px_rgba(0,0,0,0.04)] flex flex-col min-h-full">
        {/* Container Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-outline-variant/20 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/5 flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="text-[16px] font-bold text-primary tracking-tight">
                {getSectionTitle()}
              </h3>
              <p className="text-[10px] font-bold text-outline uppercase tracking-wider">Fabric Integrated Intelligence</p>
            </div>
          </div>
        </div>

        {/* Power BI iframe with skeleton + error boundary */}
        <PowerBiEmbed
          embedUrl={embedUrl}
          pageId={activePage?.id || "default"}
          pageLabel={getSectionTitle()}
        />
      </div>
    </main>
  );
}

