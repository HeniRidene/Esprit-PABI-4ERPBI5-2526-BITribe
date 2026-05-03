"use client";

import { useAuth, ROLES, getPowerBiEmbedUrl } from "@/context/AuthContext";
import { BarChart3 } from "lucide-react";

export default function DashboardMain() {
  const { user, activePage } = useAuth();
  const role = user ? ROLES[user.role] : null;

  if (!role) return null;

  // Build the Power BI embed URL based on active page
  const embedUrl = activePage
    ? getPowerBiEmbedUrl(user.role, activePage.pageName)
    : getPowerBiEmbedUrl(user.role);

  // Determine the dashboard section title based on active page
  const getSectionTitle = () => {
    if (activePage?.id === "accueil") return "Executive Summary Overview";
    return activePage?.label || "Analytics Dashboard";
  };

  return (
    <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
      {/* Power BI Dashboard Container — full focus */}
      <div className="bg-white rounded-2xl overflow-hidden animate-fade-in-up border border-outline-variant/20 shadow-[0px_4px_20px_rgba(0,0,0,0.04)] h-full flex flex-col">
        {/* Container Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-outline-variant/20">
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

        {/* Power BI iframe — with filters & page nav hidden */}
        <div className="relative min-h-[900px]">
          <iframe
            id="powerbi-embed"
            key={activePage?.id || "default"}
            title={`Power BI - ${getSectionTitle()}`}
            src={embedUrl}
            className="w-full h-full border-0 absolute inset-0"
            allowFullScreen={true}
          />
        </div>
      </div>
    </main>
  );
}
