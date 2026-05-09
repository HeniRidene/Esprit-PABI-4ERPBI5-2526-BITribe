"use client";

import { useAuth, ROLES } from "@/context/AuthContext";
import { Info, Map, Leaf, ShieldCheck, Bus, LayoutDashboard, ArrowRight } from "lucide-react";

export default function IntroductionPage() {
  const { user, handlePageChange } = useAuth();
  const role = user ? ROLES[user.role] : null;

  const showEcological = !user || user.role === "DIRECTOR" || user.role === "TRANSITION_ECOLOGIQUE";
  const showMobility = !user || user.role === "DIRECTOR" || user.role === "MOBILITE1";
  const showSecurity = !user || user.role === "DIRECTOR" || user.role === "SECURITE1";

  // Helper to find the correct page object from the user's role pages
  const getPage = (pageId) => {
    if (!role) return null;
    return role.pages.find((p) => p.id === pageId) || null;
  };

  const navigateTo = (pageId) => {
    const page = getPage(pageId);
    if (page) handlePageChange(page);
  };

  return (
    <div className="h-full overflow-y-auto pr-2 pb-6">
      <div className="bg-white rounded-2xl p-8 mb-6 animate-fade-in-up border border-outline-variant/20 shadow-premium relative overflow-hidden">
        {/* Decorative background element */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-bl-full -z-10" />
        
        <div className="max-w-3xl">
          <h1 className="text-display-kpi mb-4">
            Welcome to Urban Mobility
          </h1>
          <p className="text-lg text-outline-variant mb-6 leading-relaxed">
            Your business intelligence platform for urban mobility management. 
            Discover an overview of your strategic indicators, track your performance 
            in real time, and make informed decisions using our integrated dashboards.
          </p>
          <div className="flex items-center gap-4">
            <div className="px-4 py-2 bg-primary/10 text-primary font-semibold rounded-lg border border-primary/20">
              Active Role: {role?.label || 'Guest'}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
        
        {showEcological && (
          <button
            onClick={() => navigateTo("eco-page-1")}
            className="bg-white rounded-2xl p-6 border border-outline-variant/20 shadow-premium hover-scale cursor-pointer transition-all duration-300 group text-left hover:border-secondary/40 hover:shadow-lg"
          >
            <div className="w-12 h-12 rounded-xl bg-secondary/10 flex items-center justify-center mb-4 group-hover:bg-secondary/20 transition-colors">
              <Leaf className="w-6 h-6 text-secondary" />
            </div>
            <h3 className="text-[18px] font-bold text-primary tracking-tight mb-2">Ecological Transition</h3>
            <p className="text-sm text-outline-variant leading-relaxed mb-4">
              Tracking of carbon emissions, energy consumption, and the impact of green initiatives.
            </p>
            <div className="flex items-center gap-2 text-secondary text-sm font-semibold opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <span>View Dashboard</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </button>
        )}

        {showMobility && (
          <button
            onClick={() => navigateTo("mob-page-1")}
            className="bg-white rounded-2xl p-6 border border-outline-variant/20 shadow-premium hover-scale cursor-pointer transition-all duration-300 group text-left hover:border-[#4355b9]/40 hover:shadow-lg"
          >
            <div className="w-12 h-12 rounded-xl bg-[#4355b9]/10 flex items-center justify-center mb-4 group-hover:bg-[#4355b9]/20 transition-colors">
              <Bus className="w-6 h-6 text-[#4355b9]" />
            </div>
            <h3 className="text-[18px] font-bold text-primary tracking-tight mb-2">Mobility & Network</h3>
            <p className="text-sm text-outline-variant leading-relaxed mb-4">
              Analysis of passenger flows, punctuality, and optimization of transport routes.
            </p>
            <div className="flex items-center gap-2 text-[#4355b9] text-sm font-semibold opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <span>View Dashboard</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </button>
        )}

        {showSecurity && (
          <button
            onClick={() => navigateTo("sec-page-1")}
            className="bg-white rounded-2xl p-6 border border-outline-variant/20 shadow-premium hover-scale cursor-pointer transition-all duration-300 group text-left hover:border-[#6b5c00]/40 hover:shadow-lg"
          >
            <div className="w-12 h-12 rounded-xl bg-[#6b5c00]/10 flex items-center justify-center mb-4 group-hover:bg-[#6b5c00]/20 transition-colors">
              <ShieldCheck className="w-6 h-6 text-[#6b5c00]" />
            </div>
            <h3 className="text-[18px] font-bold text-primary tracking-tight mb-2">Security & Maintenance</h3>
            <p className="text-sm text-outline-variant leading-relaxed mb-4">
              Predicting breakdowns, tracking incidents, and proactively managing infrastructure health.
            </p>
            <div className="flex items-center gap-2 text-[#6b5c00] text-sm font-semibold opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <span>View Dashboard</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </button>
        )}

      </div>

      <div className="mt-6 bg-primary rounded-2xl p-8 text-white relative overflow-hidden animate-fade-in-up shadow-glow-primary" style={{ animationDelay: "0.2s" }}>
        <div className="absolute -right-10 -bottom-10 w-48 h-48 bg-white/10 rounded-full blur-2xl" />
        <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-6">
          <div>
            <h3 className="text-xl font-bold mb-2">Ready to explore your data?</h3>
            <p className="text-primary-fixed-dim text-sm max-w-xl">
              Click on any dashboard card above or use the sidebar menu to start analyzing KPIs specific to your domain.
            </p>
          </div>
          <div className="p-3 bg-white/10 rounded-xl backdrop-blur-md border border-white/20 flex-shrink-0">
            <LayoutDashboard className="w-8 h-8 text-primary-fixed" />
          </div>
        </div>
      </div>
    </div>
  );
}
