import { NextResponse } from "next/server";
import { getN8nAuthHeaders } from "@/lib/n8nAuth";

const N8N_URL = process.env.N8N_URL ?? "http://localhost:5678";

export async function GET() {
  try {
    const headers = await getN8nAuthHeaders();

    const res = await fetch(`${N8N_URL}/rest/workflows`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });

    if (!res.ok) {
      return NextResponse.json(
        { online: false, detail: `n8n returned ${res.status}` },
        { status: 200 }
      );
    }

    const body = await res.json();
    const workflows = (body.data ?? body ?? []).map((wf) => ({
      name: wf.name,
      active: wf.active,
      id: wf.id,
    }));

    return NextResponse.json({
      online: true,
      workflow_count: workflows.length,
      workflows,
    });
  } catch (e) {
    return NextResponse.json(
      { online: false, detail: e.message },
      { status: 200 }
    );
  }
}
