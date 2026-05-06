"use client";

import { useAuth, ROLES } from "@/context/AuthContext";
import { Search, Bell, Settings, Command, BarChart3, Download, Share2 } from "lucide-react";

export default function TopBar() {
  const { user, setActivePage } = useAuth();
  const role = user ? ROLES[user.role] : null;

  return (
    <header
      id="topbar"
      className="sticky top-0 z-40 bg-white border-b border-outline-variant/30 flex justify-between items-center w-full px-8 py-4 h-[80px]"
    >
      {/* Context / Search */}
      <div className="flex items-center gap-8">
        {/* Quick Search */}
        <div className="relative hidden lg:flex items-center group">
          <Search className="absolute left-4 w-4 h-4 text-outline group-focus-within:text-primary transition-colors" />
          <input
            type="text"
            placeholder="Search analytics, reports..."
            className="pl-12 pr-12 py-2.5 bg-surface-low rounded-full text-[14px] border border-outline-variant/30 focus:border-primary/40 focus:bg-white focus:outline-none w-[360px] text-on-surface transition-all placeholder-outline"
          />
          <div className="absolute right-4 flex items-center gap-1">
            <Command className="w-3.5 h-3.5 text-outline" />
            <span className="text-[11px] font-bold text-outline">K</span>
          </div>
        </div>
      </div>

      {/* Center Actions */}
      <div className="hidden lg:flex items-center gap-3">
        {role?.pages?.length > 1 && (
          <button
            onClick={() => setActivePage(role.pages[1])}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-xl hover:bg-[#002d6d] hover:shadow-md transition-all text-sm font-medium"
          >
            <BarChart3 className="w-4 h-4" />
            Dashboards Power BI
          </button>
        )}
        <button className="flex items-center gap-2 px-4 py-2 bg-white text-primary border border-outline-variant/30 rounded-xl hover:bg-surface-low transition-all text-sm font-medium">
          <Download className="w-4 h-4" />
          Export PDF
        </button>
        <button className="flex items-center gap-2 px-4 py-2 bg-white text-primary border border-outline-variant/30 rounded-xl hover:bg-surface-low transition-all text-sm font-medium">
          <Share2 className="w-4 h-4" />
          Partager
        </button>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2">
          <button
            id="topbar-notifications"
            className="p-2.5 text-outline hover:text-primary hover:bg-surface-low transition-all duration-200 rounded-full cursor-pointer relative"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-2.5 right-2.5 w-2 h-2 bg-secondary rounded-full border-2 border-white" />
          </button>

          <button
            id="topbar-settings"
            className="p-2.5 text-outline hover:text-primary hover:bg-surface-low transition-all duration-200 rounded-full cursor-pointer"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>

        <div className="h-8 w-[1px] bg-outline-variant/30" />

        {/* User Profile */}
        {user && (
          <button className="flex items-center gap-3 py-1.5 px-2 hover:bg-surface-low rounded-full transition-all duration-200 cursor-pointer group">
            <div className="w-9 h-9 rounded-full bg-primary flex items-center justify-center">
              <span className="text-[12px] font-bold text-white">
                {user.avatarInitials}
              </span>
            </div>
            <div className="hidden md:flex flex-col items-start pr-2">
              <span className="text-[14px] font-semibold text-primary leading-tight">{user.name}</span>
              <span className="text-[11px] font-bold text-outline uppercase tracking-wider">{user.role.replace('_', ' ')}</span>
            </div>
          </button>
        )}
      </div>
    </header>
  );
}
