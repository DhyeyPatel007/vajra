import { NextRequest, NextResponse } from "next/server";
import {
  GenerateErrorResponse,
  GenerateRequest,
  GenerateResponse,
  WorkloadId,
} from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";
import { getMockVariants, getGen2MockVariant } from "@/lib/mockVariants";
import {
  buildNemotronSystemPrompt,
  buildNemotronUserPrompt,
  parseNemotronResponse,
  buildNemotronGen2SystemPrompt,
  buildNemotronGen2UserPrompt,
  parseNemotronGen2Response,
} from "@/lib/nemotronPrompt";

export const maxDuration = 60;

export async function POST(req: NextRequest) {
  const startTime = Date.now();

  try {
    const body = (await req.json()) as GenerateRequest;
    const {
      workloadId,
      mockMode,
      simulateCompileError,
      generation = 1,
      previousWinner,
      failedVariants,
    } = body;

    if (!workloadId || !WORKLOADS[workloadId as WorkloadId]) {
      const errRes: GenerateErrorResponse = {
        error: `Invalid or missing workloadId: "${workloadId}". Expected one of: matmul, softmax, conv1d, sobel.`,
        code: "INVALID_WORKLOAD",
        retryable: false,
      };
      return NextResponse.json(errRes, { status: 400 });
    }

    const workload = WORKLOADS[workloadId as WorkloadId];
    const serverMockDefault = process.env.VAJRA_MOCK_MODE !== "false";
    const isMock = mockMode === true || (mockMode === undefined && serverMockDefault);

    // 1. MOCK MODE PATH (Zero external API calls)
    if (isMock) {
      if (generation === 2) {
        const parentWinnerId = previousWinner?.id || (simulateCompileError ? "variant-b" : "variant-c");
        const gen2Mock = getGen2MockVariant(workloadId as WorkloadId, parentWinnerId);
        const response: GenerateResponse = {
          workloadId: workloadId as WorkloadId,
          source: "mock",
          modelId: "mock-nemotron-local",
          usedFallbackModel: false,
          latencyMs: Math.max(1, Date.now() - startTime),
          variants: [gen2Mock],
        };
        return NextResponse.json(response, { status: 200 });
      }

      const mockVariants = getMockVariants(
        workloadId as WorkloadId,
        Boolean(simulateCompileError)
      );

      const response: GenerateResponse = {
        workloadId: workloadId as WorkloadId,
        source: "mock",
        modelId: "mock-nemotron-local",
        usedFallbackModel: false,
        latencyMs: Math.max(1, Date.now() - startTime),
        variants: mockVariants,
      };

      return NextResponse.json(response, { status: 200 });
    }

    // 2. LIVE NEBIUS TOKEN FACTORY PATH
    const apiKey = process.env.NEBIUS_API_KEY;
    if (!apiKey || apiKey === "your_nebius_token_factory_api_key_here") {
      const errRes: GenerateErrorResponse = {
        error: "NEBIUS_API_KEY is not configured in server environment (.env.local or Vercel environment variables).",
        code: "MISSING_API_KEY",
        retryable: true,
        details: "Please paste your Nebius Token Factory API key into .env.local as NEBIUS_API_KEY.",
      };
      return NextResponse.json(errRes, { status: 502 });
    }

    const baseUrl = (process.env.NEBIUS_BASE_URL || "https://api.tokenfactory.nebius.com/v1").replace(/\/+$/, "");
    const superModel = process.env.NEBIUS_MODEL_SUPER || "nvidia/nemotron-3-super-120b-a12b";
    const ultraModel = process.env.NEBIUS_MODEL_ULTRA || "nvidia/Nemotron-3-Ultra-550b-a55b";

    const isGen2 = generation === 2;
    const systemPrompt = isGen2 ? buildNemotronGen2SystemPrompt() : buildNemotronSystemPrompt();
    const userPrompt = isGen2
      ? buildNemotronGen2UserPrompt(
          workload,
          previousWinner
            ? {
                id: previousWinner.id,
                name: previousWinner.name,
                gflops: previousWinner.gflops,
                medianTimeMs: previousWinner.medianTimeMs,
                wgslBody: previousWinner.fullWgslCode,
              }
            : undefined,
          failedVariants
        )
      : buildNemotronUserPrompt(workload);

    const callNebiusModel = async (model: string, timeoutMs: number) => {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          temperature: 0.15,
          top_p: 0.9,
          max_tokens: 3200,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
        }),
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (!res.ok) {
        const errorText = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} from ${model}: ${errorText.slice(0, 300)}`);
      }

      const data = await res.json();
      const content = data.choices?.[0]?.message?.content;
      if (!content) {
        throw new Error(`Empty content returned by ${model}`);
      }

      return isGen2
        ? parseNemotronGen2Response(content, previousWinner?.id || "variant-c")
        : parseNemotronResponse(content);
    };

    // Attempt 1: Primary Super Model (25s timeout)
    try {
      const variants = await callNebiusModel(superModel, 25000);
      const response: GenerateResponse = {
        workloadId: workloadId as WorkloadId,
        source: "nebius-super",
        modelId: superModel,
        usedFallbackModel: false,
        latencyMs: Date.now() - startTime,
        variants,
      };
      return NextResponse.json(response, { status: 200 });
    } catch (superErr: unknown) {
      console.warn(`[Vajra] Super model (${superModel}) failed, attempting Ultra fallback:`, superErr);

      // Attempt 2: Fallback Ultra Model (30s timeout)
      try {
        const variants = await callNebiusModel(ultraModel, 30000);
        const response: GenerateResponse = {
          workloadId: workloadId as WorkloadId,
          source: "nebius-ultra-fallback",
          modelId: ultraModel,
          usedFallbackModel: true,
          latencyMs: Date.now() - startTime,
          variants,
        };
        return NextResponse.json(response, { status: 200 });
      } catch (ultraErr: unknown) {
        console.error(`[Vajra] Both Super and Ultra models failed:`, {
          superErr: superErr instanceof Error ? superErr.message : superErr,
          ultraErr: ultraErr instanceof Error ? ultraErr.message : ultraErr,
        });

        const errRes: GenerateErrorResponse = {
          error: "Upstream generation failed: both Nemotron Super and Nemotron Ultra returned errors or timed out.",
          code: "UPSTREAM_SUPER_AND_ULTRA_FAILED",
          retryable: true,
          details: `Super: ${superErr instanceof Error ? superErr.message : String(superErr)} | Ultra: ${ultraErr instanceof Error ? ultraErr.message : String(ultraErr)}`,
        };
        return NextResponse.json(errRes, { status: 502 });
      }
    }
  } catch (err: unknown) {
    const errRes: GenerateErrorResponse = {
      error: "Unexpected server error while processing kernel generation request.",
      code: "PARSE_ERROR",
      retryable: true,
      details: err instanceof Error ? err.message : String(err),
    };
    return NextResponse.json(errRes, { status: 500 });
  }
}
