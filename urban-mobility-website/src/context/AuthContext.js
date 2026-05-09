"use client";

import { createContext, useContext, useState, useCallback, useEffect } from "react";

/**
 * Base Power BI embed URL for the report.
 * Includes filterPaneEnabled=false and navContentPaneEnabled=false
 * to hide the native Power BI filter pane and bottom page navigation tabs.
 */
const POWERBI_BASE_URL =
  "https://app.powerbi.com/reportEmbed?reportId=9e02c322-1fbe-4b2c-89fd-2c77955a6b78&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&filterPaneEnabled=false&navContentPaneEnabled=false";

/**
 * Role definitions — each role determines which Power BI pages a user can see.
 * The user's role is determined after login via the /api/auth endpoint.
 *
 * DIRECTOR sees ALL pages across every domain.
 * Other roles only see their specific domain pages.
 */
export const ROLES = {
  DIRECTOR: {
    id: "DIRECTOR",
    label: "General Director",
    title: "Global Executive Dashboard",
    subtitle: "Comprehensive view of all strategic indicators.",
    pages: [
      { id: "accueil", label: "Home", pageName: "" },
      { id: "eco-page-1", label: "Emissions & Energy", pageName: "ReportSection1" },
      { id: "eco-page-2", label: "Green Projects", pageName: "ReportSection2" },
      { id: "mob-page-1", label: "Network Performance", pageName: "ReportSection3" },
      { id: "sec-page-1", label: "Security & Maintenance", pageName: "ReportSection4" },
      { id: "mlops", label: "MLOps Control", pageName: "" },
      { id: "streamlit", label: "ML Predictions", pageName: "" },
    ],
  },
  TRANSITION_ECOLOGIQUE: {
    id: "TRANSITION_ECOLOGIQUE",
    label: "Ecological Transition",
    title: "Sustainability & Emissions Analytics",
    subtitle: "Real-time tracking of the environmental impact of transport networks.",
    pages: [
      { id: "accueil", label: "Home", pageName: "" },
      { id: "eco-page-1", label: "Emissions & Energy", pageName: "ReportSection1" },
      { id: "eco-page-2", label: "Green Projects", pageName: "ReportSection2" },
      { id: "streamlit-actor1", label: "Eco Predictions", pageName: "", actorFilter: "actor1" },
    ],
  },
  MOBILITE1: {
    id: "MOBILITE1",
    label: "Mobility",
    title: "Strategic Territory Overview",
    subtitle: "Real-time performance metrics for urban hubs.",
    pages: [
      { id: "accueil", label: "Home", pageName: "" },
      { id: "mob-page-1", label: "Network Performance", pageName: "ReportSection3" },
      { id: "streamlit-actor2", label: "Mobility Predictions", pageName: "", actorFilter: "actor2" },
    ],
  },
  SECURITE1: {
    id: "SECURITE1",
    label: "Security",
    title: "Operational Safety & Maintenance",
    subtitle: "Infrastructure health and real-time incident reports.",
    pages: [
      { id: "accueil", label: "Home", pageName: "" },
      { id: "sec-page-1", label: "Security & Maintenance", pageName: "ReportSection4" },
      { id: "streamlit-actor3", label: "Security Predictions", pageName: "", actorFilter: "actor3" },
    ],
  },
};

/**
 * Returns the Power BI embed URL for a given page.
 */
export function getPowerBiEmbedUrl(roleId, pageName = "") {
  const role = ROLES[roleId];
  if (!role) return "";

  if (pageName) {
    return `${POWERBI_BASE_URL}&pageName=${pageName}`;
  }
  return POWERBI_BASE_URL;
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loginError, setLoginError] = useState(null);
  const [activePage, setActivePage] = useState({ id: "accueil", label: "Home", pageName: "" });
  const [isInitialized, setIsInitialized] = useState(false);
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);

  const handlePageChange = useCallback((page) => {
    if (page.id !== "accueil" && page.id !== "about" && !user) {
      setShowLoginPrompt(true);
    } else {
      setShowLoginPrompt(false);
      setActivePage(page);
    }
  }, [user]);

  // Restore user from localStorage on mount
  useEffect(() => {
    const storedUser = localStorage.getItem("urban_mobility_user");
    if (storedUser) {
      try {
        const parsed = JSON.parse(storedUser);
        setUser(parsed);
        const role = ROLES[parsed.role];
        if (role) {
          setActivePage(role.pages[0]);
        }
      } catch (err) {
        console.error("Failed to parse user from localStorage", err);
      }
    }
    setIsInitialized(true);
  }, []);

  /**
   * Login with email + password via the /api/auth endpoint.
   * On success, sets the user and navigates to the dashboard.
   */
  const login = useCallback(async (email, password) => {
    setIsLoading(true);
    setLoginError(null);

    try {
      const res = await fetch("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setLoginError(data.error || "Login error");
        setIsLoading(false);
        return false;
      }

      const role = ROLES[data.user.role];
      if (!role) {
        setLoginError("User role not configured");
        setIsLoading(false);
        return false;
      }

      const userData = {
        name: data.user.name,
        email: data.user.email,
        role: data.user.role,
        accessLevel: data.user.accessLevel,
        avatarInitials: data.user.name
          .split(" ")
          .map((n) => n[0])
          .join("")
          .toUpperCase()
          .slice(0, 2),
      };

      setUser(userData);
      localStorage.setItem("urban_mobility_user", JSON.stringify(userData));

      setActivePage(role.pages[0]);
      setShowLoginPrompt(false);
      setIsLoading(false);
      return true;
    } catch (err) {
      setLoginError("Unable to contact server");
      setIsLoading(false);
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setActivePage({ id: "accueil", label: "Home", pageName: "" });
    setShowLoginPrompt(false);
    setLoginError(null);
    localStorage.removeItem("urban_mobility_user");
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        login,
        logout,
        isLoading,
        loginError,
        setLoginError,
        activePage,
        setActivePage,
        handlePageChange,
        showLoginPrompt,
        setShowLoginPrompt,
        isInitialized,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
