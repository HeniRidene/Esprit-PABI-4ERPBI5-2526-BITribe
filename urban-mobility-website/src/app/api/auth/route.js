/**
 * POST /api/auth
 * Authenticates a user by email and password.
 * Returns user profile + role config on success.
 */
import { NextResponse } from "next/server";

/* ── User Database ─────────────────────────────────────
 * In production, replace this with a real database (PostgreSQL, MongoDB, etc.)
 * and hash passwords with bcrypt. This is a simplified version for the project.
 * ──────────────────────────────────────────────────────── */
const USERS = [
  {
    email: "heni.ridene@esprit.tn",
    password: "director2024",
    name: "Heni Ridene",
    role: "DIRECTOR",
    accessLevel: "director",
  },
  {
    email: "sbissi.mohamed@esprit.tn",
    password: "service2024",
    name: "Mohamed Sbissi",
    role: "TRANSITION_ECOLOGIQUE",
    accessLevel: "service",
  },
  {
    email: "eya.hammami@esprit.tn",
    password: "safety2024",
    name: "Eya Hammami",
    role: "SECURITE1",
    accessLevel: "safety",
  },
  {
    email: "emnabaya.benromdhane@esprit.tn",
    password: "mobility2024",
    name: "Emnabaya Ben Romdhane",
    role: "MOBILITE1",
    accessLevel: "mobility",
  },
];

export async function POST(request) {
  try {
    const { email, password } = await request.json();

    if (!email || !password) {
      return NextResponse.json(
        { error: "Email et mot de passe requis" },
        { status: 400 }
      );
    }

    /* Find user by email (case-insensitive) */
    const user = USERS.find(
      (u) => u.email.toLowerCase() === email.toLowerCase()
    );

    if (!user || user.password !== password) {
      return NextResponse.json(
        { error: "Email ou mot de passe incorrect" },
        { status: 401 }
      );
    }

    /* Return user profile (never send the password back) */
    return NextResponse.json({
      success: true,
      user: {
        email: user.email,
        name: user.name,
        role: user.role,
        accessLevel: user.accessLevel,
      },
    });
  } catch (err) {
    return NextResponse.json(
      { error: "Erreur serveur interne" },
      { status: 500 }
    );
  }
}
