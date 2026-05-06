"use client";

import { AuthProvider, useAuth } from "@/context/AuthContext";
import LoginPage from "@/components/LoginPage";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import DashboardMain from "@/components/DashboardMain";

function AppContent() {
  const { user, isInitialized } = useAuth();

  if (!isInitialized) {
    return null; // Or a subtle loading state to prevent flash
  }

  if (!user) {
    return <LoginPage />;
  }

  return (
    <div className="flex h-screen overflow-hidden mesh-bg">
      <Sidebar />
      {/* Main content — offset by sidebar width */}
      <div className="flex-1 flex flex-col ml-[280px] relative">
        <TopBar />
        <DashboardMain />
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
