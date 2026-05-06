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
    label: "Directeur Général",
    title: "Global Executive Dashboard",
    subtitle: "Vue complète de tous les indicateurs stratégiques.",
    pages: [
      { id: "accueil", label: "Accueil", pageName: "" },
      { id: "eco-page-1", label: "Émissions & Énergie", pageName: "ReportSection1" },
      { id: "eco-page-2", label: "Projets Verts", pageName: "ReportSection2" },
      { id: "mob-page-1", label: "Performance Réseau", pageName: "ReportSection3" },
      { id: "sec-page-1", label: "Sécurité & Maintenance", pageName: "ReportSection4" },
    ],
  },
  TRANSITION_ECOLOGIQUE: {
    id: "TRANSITION_ECOLOGIQUE",
    label: "Transition Écologique",
    title: "Sustainability & Emissions Analytics",
    subtitle: "Suivi en temps réel de l'impact environnemental des réseaux de transport.",
    pages: [
      { id: "accueil", label: "Accueil", pageName: "" },
      { id: "eco-page-1", label: "Émissions & Énergie", pageName: "ReportSection1" },
      { id: "eco-page-2", label: "Projets Verts", pageName: "ReportSection2" },
    ],
  },
  MOBILITE1: {
    id: "MOBILITE1",
    label: "Mobilité",
    title: "Strategic Territory Overview",
    subtitle: "Métriques de performance en temps réel pour les hubs urbains.",
    pages: [
      { id: "accueil", label: "Accueil", pageName: "" },
      { id: "mob-page-1", label: "Performance Réseau", pageName: "ReportSection3" },
    ],
  },
  SECURITE1: {
    id: "SECURITE1",
    label: "Sécurité",
    title: "Operational Safety & Maintenance",
    subtitle: "Santé de l'infrastructure et rapports d'incidents en temps réel.",
    pages: [
      { id: "accueil", label: "Accueil", pageName: "" },
      { id: "sec-page-1", label: "Sécurité & Maintenance", pageName: "ReportSection4" },
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
  const [activePage, setActivePage] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);

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
        setLoginError(data.error || "Erreur de connexion");
        setIsLoading(false);
        return false;
      }

      const role = ROLES[data.user.role];
      if (!role) {
        setLoginError("Rôle utilisateur non configuré");
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
      setIsLoading(false);
      return true;
    } catch (err) {
      setLoginError("Impossible de contacter le serveur");
      setIsLoading(false);
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setActivePage(null);
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
