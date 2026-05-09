import { NextResponse } from "next/server";
import { getN8nAuthHeaders } from "@/lib/n8nAuth";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";
const N8N_URL = process.env.N8N_URL ?? "http://localhost:5678";

export async function POST(request) {
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

    const body = await request.json().catch(() => ({}));

    // Try n8n webhook first
    try {
      const n8nRes = await fetch(`${N8N_URL}/webhook/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...n8nHeaders },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30000),
      });
      if (n8nRes.ok) {
        const data = await n8nRes.json();
        return NextResponse.json({ ...data, webhook: true }, { status: 200 });
      }
    } catch {
      // n8n webhook failed, fall through to direct FastAPI call
    }

    // Fallback: call FastAPI directly
    const res = await fetch(`${FASTAPI_URL}/trigger-n8n-predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(40000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ status: "error", detail: e.message }, { status: 502 });
  }
}
