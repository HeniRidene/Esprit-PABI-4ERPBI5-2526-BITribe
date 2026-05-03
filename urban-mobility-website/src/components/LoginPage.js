"use client";

import { useState } from "react";
import { useAuth, ROLES } from "@/context/AuthContext";
import { Globe, KeyRound, Loader2, ChevronRight, Leaf, Bus, ShieldCheck } from "lucide-react";

const roleIcons = { TRANSITION_ECOLOGIQUE: Leaf, MOBILITE1: Bus, SECURITE1: ShieldCheck };

export default function LoginPage() {
  const { login, isLoading, loginError } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="min-h-screen flex items-center justify-center mesh-bg relative overflow-hidden">
      {/* Background Decorative Elements */}
      <div className="absolute top-[-10%] left-[-5%] w-[40%] h-[40%] bg-primary/5 rounded-full blur-[100px]" />
      <div className="absolute bottom-[-10%] right-[-5%] w-[40%] h-[40%] bg-secondary/5 rounded-full blur-[100px]" />

      <div className="relative w-full max-w-lg mx-4 animate-fade-in-up">
        <div className="glass shadow-premium rounded-3xl overflow-hidden border border-white/50">
          {/* Branding Header */}
          <div className="bg-primary/95 px-10 py-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent" />
            <div className="relative z-10">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 mb-6 shadow-glow-primary">
                <Globe className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-3xl font-extrabold text-white tracking-tight">
                UrbanMobility BI
              </h1>
              <p className="text-blue-200/80 text-sm mt-2 font-medium uppercase tracking-[0.2em]">
                Intelligent Urban Mobility Suite
              </p>
            </div>
          </div>

          {/* Login Body */}
          <div className="p-10 bg-white">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-primary">Connexion sécurisée</h2>
              <p className="text-outline mt-2 text-[14px]">
                Veuillez entrer vos identifiants pour accéder au tableau de bord.
              </p>
            </div>

            <form 
              onSubmit={async (e) => {
                e.preventDefault();
                await login(email, password);
              }}
              className="space-y-5"
            >
              {loginError && (
                <div className="p-3 rounded-xl bg-error-container/30 text-error text-[13px] font-medium border border-error/20 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 shrink-0" />
                  {loginError}
                </div>
              )}

              <div className="space-y-1.5">
                <label className="text-[12px] font-bold text-primary uppercase tracking-wider ml-1">
                  Adresse Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="nom.prenom@esprit.tn"
                  className="w-full px-4 py-3 rounded-xl border border-outline-variant/40 bg-surface-low focus:bg-white focus:border-primary focus:outline-none transition-all text-[14px]"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-[12px] font-bold text-primary uppercase tracking-wider ml-1">
                  Mot de passe
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-4 py-3 rounded-xl border border-outline-variant/40 bg-surface-low focus:bg-white focus:border-primary focus:outline-none transition-all text-[14px]"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-2 py-3.5 px-6 rounded-xl bg-primary text-white font-bold text-[14px] hover:bg-primary-container transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed mt-2"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <KeyRound className="w-4 h-4" />
                    Se connecter
                  </>
                )}
              </button>
            </form>

            {/* SSO Button (Mock) */}
            <div className="mt-6 text-center">
              <button
                type="button"
                className="text-[13px] font-semibold text-outline hover:text-primary transition-colors"
              >
                Connexion via Microsoft SSO
              </button>
            </div>
          </div>

          <div className="px-10 py-5 bg-surface-low/50 border-t border-outline-variant/10 text-center">
            <p className="text-[10px] text-outline font-bold uppercase tracking-[0.15em]">Plateforme sécurisée · Gouvernance des données Urbaines</p>
          </div>
        </div>
      </div>
    </div>
  );
}
