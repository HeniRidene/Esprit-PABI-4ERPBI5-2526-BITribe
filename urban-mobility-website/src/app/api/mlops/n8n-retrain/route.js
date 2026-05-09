import { NextResponse } from "next/server";
import { getN8nAuthHeaders } from "@/lib/n8nAuth";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";
const N8N_URL = process.env.N8N_URL ?? "http://localhost:5678";

export async function POST() {
  try {
    // Attempt to authenticate with n8n and trigger the webhook
    let n8nHeaders;
    try {
      n8nHeaders = await getN8nAuthHeaders();
    } catch {
      // n8n offline — return structured error so frontend shows fallback
      return NextResponse.json(
        { status: "error", detail: "n8n offline", offline: true },
        { status: 503 }
      );
    }

    // Try n8n webhook first
    try {
      const n8nRes = await fetch(`${N8N_URL}/webhook/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...n8nHeaders },
        signal: AbortSignal.timeout(600000), // 10 min
      });
      if (n8nRes.ok) {
        const data = await n8nRes.json();
        return NextResponse.json({ ...data, method: "n8n" }, { status: 200 });
      }
    } catch {
      // n8n webhook failed, fall through to direct FastAPI call
    }

    // Fallback: call FastAPI /retrain directly
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 630000);
    try {
      const res = await fetch(`${FASTAPI_URL}/trigger-n8n-retrain`, {
        method: "POST",
        signal: controller.signal,
      });
      const data = await res.json();
      return NextResponse.json({ ...data, method: "direct" }, { status: res.status });
    } finally {
      clearTimeout(timeout);
    }
  } catch (e) {
    return NextResponse.json({ status: "error", detail: e.message }, { status: 502 });
  }
}
