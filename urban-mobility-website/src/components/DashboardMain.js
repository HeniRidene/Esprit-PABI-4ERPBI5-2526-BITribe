"use client";

import { useAuth, ROLES, getPowerBiEmbedUrl } from "@/context/AuthContext";
import { BarChart3 } from "lucide-react";
import IntroductionPage from "./IntroductionPage";
import LoginPage from "./LoginPage";
import AboutUsPage from "./AboutUsPage";

export default function DashboardMain() {
  const { user, activePage, showLoginPrompt } = useAuth();
  const role = user ? ROLES[user.role] : null;

  if (showLoginPrompt && !user) {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff] flex items-center justify-center">
        <LoginPage />
      </main>
    );
  }

  // If the active page is the introduction/accueil page
  if (activePage?.id === "accueil") {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
        <IntroductionPage />
      </main>
    );
  }

  // If the active page is about us
  if (activePage?.id === "about") {
    return (
      <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
        <AboutUsPage />
      </main>
    );
  }

  // Build the Power BI embed URL based on active page
  const embedUrl = activePage
    ? getPowerBiEmbedUrl(user.role, activePage.pageName)
    : getPowerBiEmbedUrl(user.role);

  // Determine the dashboard section title based on active page
  const getSectionTitle = () => {
    return activePage?.label || "Analytics Dashboard";
  };

  return (
    <main className="flex-1 overflow-y-auto p-6 bg-[#f8f9ff]">
      {/* Power BI Dashboard Container — full focus */}
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

        {/* Power BI iframe — with filters & page nav hidden */}
        <div className="relative flex-1 w-full min-h-[700px] md:min-h-[850px]">
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
