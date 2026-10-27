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

/**
 * Strips sensitive authorization token patterns to ensure credentials are never leaked.
 */
function sanitizeErrorMessage(msg: string): string {
  if (!msg) return "";
  return msg
    .replace(/Bearer\s+[A-Za-z0-9\-_\.]+/gi, "Bearer [REDACTED]")
    .replace(/(?:api[_-]?key|token)=["']?[A-Za-z0-9\-_\.]+["']?/gi, "$1=[REDACTED]");
}

// Known placeholder string values indicating an unconfigured key
const PLACEHOLDER_KEYS = new Set([
  "your_nebius_api_key_here",
  "your_nebius_token_factory_api_key_here",
  "your_actual_nebius_key",
  "your_nvidia_api_key_here",
  "your_actual_nvidia_key",
]);

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

    // 1. MOCK MODE PATH (Zero external API calls, NO key required)
    if (isMock) {
      if (generation === 2) {
        const parentWinnerId =
          previousWinner?.id || (simulateCompileError ? "variant-b" : "variant-c");
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

    // 2. LIVE LLM PROVIDER SELECTION
    // NOTE:
    // - Token Factory (api.tokenfactory.nebius.com) is the final submission path (default: VAJRA_PROVIDER=nebius).
    // - NVIDIA path (integrate.api.nvidia.com) is for temporary checking only (VAJRA_PROVIDER=nvidia).
    const provider = (process.env.VAJRA_PROVIDER || "nebius").toLowerCase().trim();
    const isNvidia = provider === "nvidia";

    const apiKeyEnvVar = isNvidia ? "NVIDIA_API_KEY" : "NEBIUS_API_KEY";
    const rawApiKey = isNvidia ? process.env.NVIDIA_API_KEY : process.env.NEBIUS_API_KEY;
    const apiKey = rawApiKey ? rawApiKey.trim() : "";

    if (!apiKey || PLACEHOLDER_KEYS.has(apiKey)) {
      const errRes: GenerateErrorResponse = {
        error: `${apiKeyEnvVar} is not configured in server environment (.env.local or Vercel environment variables).`,
        code: "MISSING_API_KEY",
        retryable: true,
        details: `Please set ${apiKeyEnvVar} in .env.local to enable live kernel generation via ${
          isNvidia ? "NVIDIA API (check path)" : "Nebius Token Factory (submission path)"
        }.`,
      };
      return NextResponse.json(errRes, { status: 502 });
    }

    const baseUrl = isNvidia
      ? (process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1").replace(/\/+$/, "")
      : (process.env.NEBIUS_BASE_URL || "https://api.tokenfactory.nebius.com/v1").replace(/\/+$/, "");

    const primaryModel = isNvidia
      ? process.env.NVIDIA_MODEL_SUPER || "nvidia/nemotron-3-super-120b-a12b"
      : process.env.NEBIUS_MODEL_SUPER || "nvidia/nemotron-3-super-120b-a12b";

    const fallbackModel = isNvidia
      ? process.env.NVIDIA_MODEL_NANO || "nvidia/llama-3.1-nemotron-70b-instruct"
      : process.env.NEBIUS_MODEL_ULTRA || "nvidia/Nemotron-3-Ultra-550b-a55b";

    const primaryTimeoutMs = 25000;
    const fallbackTimeoutMs = 30000;

    const primarySource = isNvidia ? "nvidia-super" : "nebius-super";
    const fallbackSource = isNvidia ? "nvidia-nano-fallback" : "nebius-ultra-fallback";

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

    const callProviderModel = async (model: string, timeoutMs: number) => {
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
        const rawErrorText = await res.text().catch(() => "");
        const safeErrorText = sanitizeErrorMessage(rawErrorText.slice(0, 300));
        throw new Error(`HTTP ${res.status} from ${model}: ${safeErrorText}`);
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

    // Attempt 1: Primary Model (Super, 25s timeout)
    try {
      const variants = await callProviderModel(primaryModel, primaryTimeoutMs);
      const response: GenerateResponse = {
        workloadId: workloadId as WorkloadId,
        source: primarySource,
        modelId: primaryModel,
        usedFallbackModel: false,
        latencyMs: Date.now() - startTime,
        variants,
      };
      return NextResponse.json(response, { status: 200 });
    } catch (primaryErr: unknown) {
      const safePrimaryErr = sanitizeErrorMessage(
        primaryErr instanceof Error ? primaryErr.message : String(primaryErr)
      );
      console.warn(
        `[Vajra] Primary model (${primaryModel}) failed on ${provider}, attempting fallback:`,
        safePrimaryErr
      );

      // Attempt 2: Fallback Model (Ultra / Nano, 30s timeout)
      try {
        const variants = await callProviderModel(fallbackModel, fallbackTimeoutMs);
        const response: GenerateResponse = {
          workloadId: workloadId as WorkloadId,
          source: fallbackSource,
          modelId: fallbackModel,
          usedFallbackModel: true,
          latencyMs: Date.now() - startTime,
          variants,
        };
        return NextResponse.json(response, { status: 200 });
      } catch (fallbackErr: unknown) {
        const safeFallbackErr = sanitizeErrorMessage(
          fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr)
        );
        console.error(
          `[Vajra] Both primary and fallback models failed on ${provider}:`,
          {
            primary: safePrimaryErr,
            fallback: safeFallbackErr,
          }
        );

        const errRes: GenerateErrorResponse = {
          error: `Upstream generation failed: both ${primaryModel} and ${fallbackModel} returned errors or timed out.`,
          code: "UPSTREAM_SUPER_AND_ULTRA_FAILED",
          retryable: true,
          details: `Primary (${primaryModel}): ${safePrimaryErr} | Fallback (${fallbackModel}): ${safeFallbackErr}`,
        };
        return NextResponse.json(errRes, { status: 502 });
      }
    }
  } catch (err: unknown) {
    const safeErr = sanitizeErrorMessage(err instanceof Error ? err.message : String(err));
    const errRes: GenerateErrorResponse = {
      error: "Unexpected server error while processing kernel generation request.",
      code: "PARSE_ERROR",
      retryable: true,
      details: safeErr,
    };
    return NextResponse.json(errRes, { status: 500 });
  }
}
