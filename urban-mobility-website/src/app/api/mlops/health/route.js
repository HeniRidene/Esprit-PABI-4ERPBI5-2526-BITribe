import { NextResponse } from "next/server";

// ── In-memory rate limiter ────────────────────────────────────────────────────
// Max 30 requests per minute per IP
const RATE_LIMIT = 30;
const WINDOW_MS  = 60_000;
const ipMap      = new Map(); // ip → { count, resetAt }

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

// ── Service URLs ──────────────────────────────────────────────────────────────
const FASTAPI_URL   = process.env.FASTAPI_URL    ?? "http://localhost:8000";
const STREAMLIT_URL = process.env.STREAMLIT_URL  ?? "http://localhost:8501";

// ── Ping a URL server-side — returns true if reachable, false if not ──────────
async function pingService(url, timeoutMs = 3000) {
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { method: "GET", signal: controller.signal });
    clearTimeout(t);
    return res.ok || res.status < 500; // 200–4xx = server is up (even 403/404 = reachable)
  } catch {
    return false; // ECONNREFUSED, timeout, etc.
  }
}

// ── Route ─────────────────────────────────────────────────────────────────────
export async function GET(request) {
  const limited = checkRateLimit(request);
  if (limited) return limited;

  // Fetch FastAPI health + ping Streamlit in parallel
  const [fastapiResult, streamlitReachable] = await Promise.allSettled([
    fetch(`${FASTAPI_URL}/health`).then((r) => r.json()),
    pingService(STREAMLIT_URL),
  ]);

  const fastapi = fastapiResult.status === "fulfilled"
    ? fastapiResult.value
    : { status: "error", detail: fastapiResult.reason?.message ?? "unreachable" };

  return NextResponse.json({
    ...fastapi,
    streamlit_reachable: streamlitReachable.status === "fulfilled"
      ? streamlitReachable.value
      : false,
    streamlit_url: STREAMLIT_URL,
  });
}
