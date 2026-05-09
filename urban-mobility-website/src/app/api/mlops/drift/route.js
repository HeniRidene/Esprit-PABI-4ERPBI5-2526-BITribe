import { NextResponse } from "next/server";

export async function GET() {
  try {
    const res = await fetch("http://localhost:8000/drift");
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ status: "error", detail: e.message }, { status: 502 });
  }
}
