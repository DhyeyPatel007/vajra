import { NextRequest, NextResponse } from "next/server";
import { WorkloadId } from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";
import { tavilyExtractKernel } from "@/lib/tavily";

export const maxDuration = 15;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { url, workloadId } = body;

    if (!url || typeof url !== "string") {
      return NextResponse.json(
        { ok: false, error: "Missing or invalid 'url' parameter in request body." },
        { status: 400 }
      );
    }

    if (!workloadId || !WORKLOADS[workloadId as WorkloadId]) {
      return NextResponse.json(
        { ok: false, error: `Invalid or missing 'workloadId': ${workloadId}` },
        { status: 400 }
      );
    }

    const result = await tavilyExtractKernel(url, workloadId as WorkloadId);

    if (!result.ok) {
      return NextResponse.json(
        { ok: false, error: result.error, requiredBindings: result.requiredBindings },
        { status: 422 }
      );
    }

    return NextResponse.json(
      {
        ok: true,
        wgsl: result.wgsl,
        originalUrl: result.originalUrl,
        requiredBindings: result.requiredBindings,
      },
      { status: 200 }
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { ok: false, error: `Internal server error during Tavily kernel import: ${msg}` },
      { status: 500 }
    );
  }
}

