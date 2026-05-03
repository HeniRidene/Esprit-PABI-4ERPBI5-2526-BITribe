"use client";

import { useAuth, ROLES } from "@/context/AuthContext";
import {
  Home, BarChart3, FolderTree, ShieldCheck, Leaf, Bus,
  HelpCircle, Headphones, LogOut, ChevronRight, LayoutDashboard
} from "lucide-react";
import Image from "next/image";

/* Map page IDs to icons */
const pageIcons = {
  accueil: Home,
  "eco-page-1": Leaf,
  "eco-page-2": FolderTree,
  "mob-page-1": Bus,
  "sec-page-1": ShieldCheck,
};

export default function Sidebar() {
  const { user, logout, activePage, setActivePage } = useAuth();
  const role = user ? ROLES[user.role] : null;

  return (
    <aside
      id="sidebar"
      className="fixed left-0 top-0 h-screen w-[280px] border-r border-outline-variant/30 bg-[#fcfdff] flex flex-col p-6 z-50"
    >
      {/* Brand Section */}
      <div className="mb-10 mt-2 px-2">
        <div className="flex items-center gap-3">
          <div className="w-14 h-14 rounded-xl bg-white flex items-center justify-center shadow-sm overflow-hidden border border-outline-variant/30">
            <Image src="/logo.png" alt="Logo" width={48} height={48} className="object-contain" />
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
        {(role?.pages || []).map((page) => {
          const Icon = pageIcons[page.id] || BarChart3;
          const isActive = activePage?.id === page.id;
          return (
            <button
              key={page.id}
              id={`nav-${page.id}`}
              onClick={() => setActivePage(page)}
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
              {!isActive && (
                <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
              )}
            </button>
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
          <button
            id="logout-button"
            onClick={logout}
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-error/80 hover:text-error hover:bg-error-container/20 transition-all duration-200 cursor-pointer mt-2"
          >
            <LogOut className="w-4 h-4" />
            <span className="text-[13px] font-medium">Déconnexion</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
