"use client";

import { useAuth, ROLES } from "@/context/AuthContext";
import { isPbiLoaded } from "./DashboardMain";
import { useState, useEffect } from "react";
import {
  Home, BarChart3, FolderTree, ShieldCheck, Leaf, Bus,
  HelpCircle, Headphones, LogOut, ChevronRight, LayoutDashboard,
  Settings, Brain, FlaskConical,
} from "lucide-react";

/* Map page IDs to icons */
const pageIcons = {
  accueil: Home,
  "pbi-overview": LayoutDashboard,
  "eco-page-1": Leaf,
  "mob-page-1": Bus,
  "sec-page-1": ShieldCheck,
  "mlops": Settings,
  "streamlit": Brain,
  "streamlit-actor1": FlaskConical,
  "streamlit-actor2": Bus,
  "streamlit-actor3": ShieldCheck,
};

/* Pages that have Power BI embeds and show load-status dots */
const PBI_PAGE_IDS = new Set(["pbi-overview", "eco-page-1", "mob-page-1", "sec-page-1"]);

export default function Sidebar() {
  const { user, logout, activePage, handlePageChange } = useAuth();
  const pagesToDisplay = user ? ROLES[user.role].pages : ROLES.DIRECTOR.pages;

  // Track which PBI pages have loaded successfully (via localStorage)
  const [loadedPages, setLoadedPages] = useState(() => {
    const set = new Set();
    if (typeof window !== "undefined") {
      PBI_PAGE_IDS.forEach((id) => { if (isPbiLoaded(id)) set.add(id); });
    }
    return set;
  });

  useEffect(() => {
    const handler = (e) => {
      setLoadedPages((prev) => {
        const next = new Set(prev);
        next.add(e.detail.pageId);
        return next;
      });
    };
    window.addEventListener("pbi-loaded", handler);
    return () => window.removeEventListener("pbi-loaded", handler);
  }, []);

  return (
    <aside
      id="sidebar"
      className="fixed left-0 top-0 h-screen w-[280px] border-r border-outline-variant/30 bg-[#fcfdff] flex flex-col p-6 z-50"
    >
      {/* Brand Section */}
      <div className="mb-10 mt-2 px-2">
        <div className="flex items-center gap-3">
          <div className="w-14 h-14 rounded-xl bg-white flex items-center justify-center shadow-sm overflow-hidden border border-outline-variant/30">
            <img src="/logo.png" alt="Logo" width={48} height={48} className="object-contain" />
          </div>
          <div>
            <h2 className="text-[16px] font-bold text-primary tracking-tight leading-tight">Urban mobility</h2>
            <p className="text-[10px] font-bold text-outline uppercase tracking-[0.1em]">BI Tribe</p>
          </div>
        </div>
      </div>

      {/* Main Nav — pages from role config */}
      <nav className="flex-1 space-y-1.5">
        <p className="px-3 text-[11px] font-bold text-outline uppercase tracking-[0.1em] mb-4">Analytics Dashboard</p>
        {pagesToDisplay.map((page, idx) => {
          const Icon = pageIcons[page.id] || BarChart3;
          const isActive = activePage?.id === page.id;
          const isMLPage = page.id === "mlops" || page.id.startsWith("streamlit");
          const prevPage = pagesToDisplay[idx - 1];
          const isFirstMLPage = isMLPage && (!prevPage || (prevPage.id !== "mlops" && !prevPage.id.startsWith("streamlit")));
          return (
            <div key={page.id}>
              {isFirstMLPage && (
                <div className="pt-4 pb-2">
                  <hr className="border-outline-variant/30 mb-3" />
                  <p className="px-3 text-[11px] font-bold text-outline uppercase tracking-[0.1em]">
                    ML &amp; Operations
                  </p>
                </div>
              )}
              <button
                id={`nav-${page.id}`}
                onClick={() => handlePageChange(page)}
                className={`w-full flex items-center justify-between px-3 py-3 rounded-xl transition-all duration-200 group cursor-pointer
                  ${isActive
                    ? "bg-primary text-white"
                    : "text-outline hover:bg-surface hover:text-primary"
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-5 h-5 ${isActive ? "text-white" : "text-outline group-hover:text-primary"}`} />
                  <span className={`text-[14px] font-medium tracking-tight`}>
                    {page.label}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  {/* PBI load-status dot — only for Power BI pages */}
                  {PBI_PAGE_IDS.has(page.id) && !isActive && (
                    <span
                      title={loadedPages.has(page.id) ? "Report loaded" : "Not yet loaded"}
                      className="w-1.5 h-1.5 rounded-full shrink-0 transition-colors duration-300"
                      style={{
                        backgroundColor: loadedPages.has(page.id) ? "#22c55e" : "#c8c8d0",
                        boxShadow: loadedPages.has(page.id) ? "0 0 0 2px rgba(34,197,94,0.2)" : "none",
                      }}
                    />
                  )}
                  {!isActive && (
                    <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                  )}
                </div>
              </button>
            </div>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="mt-auto space-y-2 pt-6 border-t border-outline-variant/30">
        <div className="flex flex-col gap-1">
          <button className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-surface text-outline hover:text-primary transition-all duration-200 cursor-pointer">
            <HelpCircle className="w-4 h-4" />
            <span className="text-[13px] font-medium">Help Center</span>
          </button>
          <button className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-surface text-outline hover:text-primary transition-all duration-200 cursor-pointer">
            <Headphones className="w-4 h-4" />
            <span className="text-[13px] font-medium">Support</span>
          </button>
          {user && (
            <button
              id="logout-button"
              onClick={logout}
              className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-error/80 hover:text-error hover:bg-error-container/20 transition-all duration-200 cursor-pointer mt-2"
            >
              <LogOut className="w-4 h-4" />
              <span className="text-[13px] font-medium">Logout</span>
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
