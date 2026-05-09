import { NextResponse } from "next/server";

// ── In-memory rate limiter ────────────────────────────────────────────────────
const RATE_LIMIT = 30;
const WINDOW_MS  = 60_000;
const ipMap      = new Map();

function checkRateLimit(request) {
  const ip  = request.headers.get("x-forwarded-for")?.split(",")[0].trim()
               ?? request.headers.get("x-real-ip")
               ?? "unknown";
  const now = Date.now();
  const rec = ipMap.get(ip);

  if (!rec || now > rec.resetAt) {
    ipMap.set(ip, { count: 1, resetAt: now + WINDOW_MS });
    return null;
  }
  if (rec.count >= RATE_LIMIT) {
    return NextResponse.json(
      { error: "rate limit", retryAfter: Math.ceil((rec.resetAt - now) / 1000) },
      { status: 429, headers: { "Retry-After": String(Math.ceil((rec.resetAt - now) / 1000)) } }
    );
  }
  rec.count++;
  return null;
}

// ── Route ─────────────────────────────────────────────────────────────────────
const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function POST(request) {
  const limited = checkRateLimit(request);
  if (limited) return limited;

  try {
    const body = await request.json();
    const res  = await fetch(`${FASTAPI_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ status: "error", detail: e.message }, { status: 502 });
  }
}
