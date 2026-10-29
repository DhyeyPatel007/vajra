import { NextRequest, NextResponse } from "next/server";
import {
  GenerateErrorResponse,
  GenerateRequest,
  GenerateResponse,
  GeneratedVariant,
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

export const maxDuration = 120;

// One-shot telemetry logging for the next live call only
let hasLoggedNextLiveCall = false;

interface UpstreamCallError extends Error {
  status?: number;
  isEmptyContent?: boolean;
  hasReasoning?: boolean;
  retryable?: boolean;
  rejectedThinkingParam?: boolean;
}

/**
 * Strips sensitive authorization token patterns to ensure credentials are never leaked.
 */
function sanitizeErrorMessage(msg: string): string {
  if (!msg) return "";
  return msg
    .replace(/Bearer\s+[A-Za-z0-9\-_\.]+/gi, "Bearer [REDACTED]")
    .replace(/(?:api[_-]?key|token)=["']?[A-Za-z0-9\-_\.]+["']?/gi, "$1=[REDACTED]");
}

/**
 * Robustly detects whether an upstream fetch failed due to a timeout/abort.
 */
function isTimeoutError(err: unknown, elapsedMs: number, timeoutMs: number): boolean {
  if (elapsedMs >= timeoutMs - 1000) return true;
  if (err instanceof Error) {
    if (err.name === "TimeoutError" || err.name === "AbortError") return true;
    const msg = err.message.toLowerCase();
    if (msg.includes("timeout") || msg.includes("timed out") || msg.includes("aborted")) return true;
    if ((err.cause as Error)?.name === "TimeoutError") return true;
  }
  return false;
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
      provider: requestedProvider,
      simulateCompileError,
      generation = 1,
      previousWinner,
      failedVariants,
      maxWorkgroupSize = 256,
      forceStandardGen2 = false,
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
    const provider = (requestedProvider || process.env.VAJRA_PROVIDER || "nebius").toLowerCase().trim();
    const isNvidia = provider === "nvidia";

    const apiKeyEnvVar = isNvidia ? "NVIDIA_API_KEY" : "NEBIUS_API_KEY";
    const rawApiKey = isNvidia ? process.env.NVIDIA_API_KEY : process.env.NEBIUS_API_KEY;
    const apiKey = rawApiKey ? rawApiKey.trim() : "";

    if (!apiKey || PLACEHOLDER_KEYS.has(apiKey)) {
      const errRes: GenerateErrorResponse = {
        error: isNvidia
          ? "NVIDIA_API_KEY is not configured in server environment (.env.local or Vercel environment variables). Please set NVIDIA_API_KEY in .env.local to enable live kernel generation via NVIDIA NIM (check path only)."
          : "NEBIUS_API_KEY is not configured in server environment (.env.local or Vercel environment variables). Please set NEBIUS_API_KEY in .env.local to enable live kernel generation via Nebius Token Factory (submission path).",
        code: "MISSING_API_KEY",
        retryable: true,
        details: isNvidia
          ? "Please set NVIDIA_API_KEY in .env.local to enable live kernel generation via NVIDIA NIM (check path only)."
          : "Please set NEBIUS_API_KEY in .env.local to enable live kernel generation via Nebius Token Factory (submission path).",
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
      ? process.env.NVIDIA_MODEL_NANO || "nvidia/nemotron-3-nano-30b-a3b"
      : process.env.NEBIUS_MODEL_ULTRA || "nvidia/Nemotron-3-Ultra-550b-a55b";

    const primaryTimeoutMs = 90000;
    const fallbackTimeoutMs = 90000;

    const primarySource = isNvidia ? "nvidia-super" : "nebius-super";
    const fallbackSource = isNvidia ? "nvidia-nano-fallback" : "nebius-ultra-fallback";

    const isGen2 = generation === 2;
    const systemPrompt = isGen2
      ? buildNemotronGen2SystemPrompt(maxWorkgroupSize)
      : buildNemotronSystemPrompt(maxWorkgroupSize);
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
          failedVariants,
          maxWorkgroupSize,
          forceStandardGen2
        )
      : buildNemotronUserPrompt(workload, maxWorkgroupSize);

    const callSingleAttempt = async (
      model: string,
      timeoutMs: number,
      customUserPrompt?: string,
      disableThinkingOverride?: boolean
    ) => {
      const activeUserPrompt = customUserPrompt || userPrompt;
      const attemptStart = Date.now();

      const requestBody: Record<string, unknown> = {
        model,
        temperature: 0.05,
        top_p: 0.9,
        max_tokens: 3200,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: activeUserPrompt },
        ],
      };

      if (isNvidia) {
        requestBody.chat_template_kwargs = { enable_thinking: false };
      } else if (disableThinkingOverride) {
        requestBody.chat_template_kwargs = { enable_thinking: false };
      }

      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(requestBody),
        signal: AbortSignal.timeout(timeoutMs),
      });

      const attemptLatencyMs = Date.now() - attemptStart;

      if (!res.ok) {
        const rawErrorText = await res.text().catch(() => "");
        const safeErrorText = sanitizeErrorMessage(rawErrorText.slice(0, 300));

        if (!hasLoggedNextLiveCall && !isMock) {
          hasLoggedNextLiveCall = true;
          console.log(
            `[Vajra Live Call Telemetry] Provider: ${provider} | Model: ${model} | Status: ${res.status} | Latency: ${attemptLatencyMs}ms | Content Length: 0 | reasoning_content present: false`
          );
        }

        const callErr = new Error(`HTTP ${res.status} from ${model}: ${safeErrorText}`) as UpstreamCallError;
        callErr.status = res.status;
        if (disableThinkingOverride && (res.status === 400 || res.status === 422)) {
          const lower = safeErrorText.toLowerCase();
          if (
            lower.includes("chat_template_kwargs") ||
            lower.includes("enable_thinking") ||
            lower.includes("extra fields") ||
            lower.includes("unrecognized")
          ) {
            callErr.rejectedThinkingParam = true;
          }
        }
        throw callErr;
      }

      const data = await res.json();
      const message = data.choices?.[0]?.message;
      const rawContent = message?.content;
      const reasoning = message?.reasoning_content;
      const hasReasoning = Boolean(reasoning && String(reasoning).trim().length > 0);
      const contentLength = typeof rawContent === "string" ? rawContent.length : 0;
      const isContentEmpty = !rawContent || (typeof rawContent === "string" && rawContent.trim().length === 0);

      if (!hasLoggedNextLiveCall && !isMock) {
        hasLoggedNextLiveCall = true;
        console.log(
          `[Vajra Live Call Telemetry] Provider: ${provider} | Model: ${model} | Status: ${res.status} | Latency: ${attemptLatencyMs}ms | Content Length: ${contentLength} | reasoning_content present: ${hasReasoning}`
        );
      }

      if (isContentEmpty) {
        const emptyErr = new Error(
          hasReasoning
            ? `Empty content returned by ${model} (reasoning_content present)`
            : `Empty content returned by ${model}`
        ) as UpstreamCallError;
        emptyErr.isEmptyContent = true;
        emptyErr.hasReasoning = hasReasoning;
        emptyErr.status = res.status;
        throw emptyErr;
      }

      return isGen2
        ? parseNemotronGen2Response(rawContent, previousWinner?.id || "variant-c", maxWorkgroupSize)
        : parseNemotronResponse(rawContent, maxWorkgroupSize);
    };

    const callProviderModelWithRetry = async (model: string, timeoutMs: number) => {
      const startMs = Date.now();
      let variants: GeneratedVariant[];

      try {
        variants = await callSingleAttempt(model, timeoutMs, undefined, false);
      } catch (err: unknown) {
        const elapsedMs = Date.now() - startMs;
        const typedErr = err as UpstreamCallError;
        const timedOut = isTimeoutError(typedErr, elapsedMs, timeoutMs);
        const status = typedErr.status;
        const isEmpty =
          typedErr.isEmptyContent ||
          (typedErr.message && typedErr.message.includes("Empty content"));
        const is502or503 =
          status === 502 ||
          status === 503 ||
          (typedErr.message &&
            (typedErr.message.includes("HTTP 502") || typedErr.message.includes("HTTP 503")));
        const is401_403_404 =
          status === 401 ||
          status === 403 ||
          status === 404 ||
          (typedErr.message &&
            (typedErr.message.includes("HTTP 401") ||
              typedErr.message.includes("HTTP 403") ||
              typedErr.message.includes("HTTP 404")));

        // Never retry on 401/403/404
        if (is401_403_404) {
          console.error(
            `[Vajra] Model ${model} failed with non-retryable HTTP ${status}. Rethrowing immediately.`
          );
          throw typedErr;
        }

        // ONE retry after 5-second backoff, ONLY on timeout, empty content, 502, or 503
        const canRetry = timedOut || isEmpty || is502or503;
        if (!canRetry) {
          console.error(
            `[Vajra] Model ${model} failed with non-retryable error: ${typedErr.message}. Rethrowing immediately.`
          );
          throw typedErr;
        }

        console.warn(
          `[Vajra] Model ${model} attempt 1 failed (${
            timedOut ? "timeout" : isEmpty ? "empty content" : `HTTP ${status}`
          }) after ${elapsedMs}ms on ${provider}. Retrying once after 5s backoff...`
        );

        // 5-second backoff
        await new Promise((resolve) => setTimeout(resolve, 5000));

        const retryStartMs = Date.now();
        try {
          if (isEmpty && typedErr.hasReasoning) {
            console.log(
              `[Vajra] Attempt 1 had empty content but reasoning_content present. Retrying once with enable_thinking disabled...`
            );
            try {
              variants = await callSingleAttempt(model, timeoutMs, undefined, true);
              console.log(
                `[Vajra] Token Factory accepted chat_template_kwargs. Retry succeeded with ${variants.length} variants.`
              );
            } catch (thinkingErr: unknown) {
              const typedThinkingErr = thinkingErr as UpstreamCallError;
              const msg = typedThinkingErr.message?.toLowerCase() || "";
              const rejected =
                typedThinkingErr.rejectedThinkingParam ||
                msg.includes("chat_template_kwargs") ||
                msg.includes("enable_thinking") ||
                msg.includes("extra fields") ||
                msg.includes("unrecognized");

              if (rejected) {
                console.warn(
                  `[Vajra] Token Factory rejected chat_template_kwargs (${typedThinkingErr.message}). Retrying without parameter with 90s timeout...`
                );
                variants = await callSingleAttempt(model, timeoutMs, undefined, false);
              } else {
                throw thinkingErr;
              }
            }
          } else {
            variants = await callSingleAttempt(model, timeoutMs, undefined, false);
          }
        } catch (retryErr: unknown) {
          const retryElapsedMs = Date.now() - retryStartMs;
          const retryTimedOut = isTimeoutError(retryErr, retryElapsedMs, timeoutMs);
          if (retryTimedOut) {
            console.error(
              `[Vajra] Model ${model} timed out on retry after ${retryElapsedMs}ms on ${provider}. Total elapsed: ${
                elapsedMs + 5000 + retryElapsedMs
              }ms.`
            );
            throw new Error(
              `Model ${model} timed out after initial ${elapsedMs}ms and retry ${retryElapsedMs}ms (timeout: ${timeoutMs}ms)`
            );
          }
          throw retryErr;
        }
      }

      // Generation 1 validation: ensure 3 variants returned
      if (!isGen2 && variants.length < 3) {
        console.warn(
          `[Vajra] Generation on ${model} returned fewer than 3 variants (${variants.length}/3). Retrying once with 'return exactly 3 variants'...`
        );
        try {
          const retryPrompt = `${userPrompt}\n\nCRITICAL INSTRUCTION: You returned only ${variants.length} variants previously. You MUST return exactly 3 complete variants: Variant A (workgroup_tuning), Variant B (shared_memory_tiling), and Variant C (vec4_vectorized). Return valid JSON matching the schema.`;
          const retryVariants = await callSingleAttempt(model, timeoutMs, retryPrompt, true);
          if (retryVariants.length >= 3) {
            console.log(`[Vajra] Retry succeeded: received ${retryVariants.length} variants.`);
            variants = retryVariants;
          } else {
            console.warn(
              `[Vajra] Retry on ${model} still returned fewer than 3 variants (${retryVariants.length}/3). Marking Variant C DROPPED with reason 'generator returned 2/3'.`
            );
            variants = retryVariants;
          }
        } catch (retryGenErr: unknown) {
          console.warn(`[Vajra] Retry attempt failed:`, retryGenErr);
        }

        if (variants.length < 3) {
          // If still fewer than 3 (e.g. 2), mark C DROPPED with reason "generator returned 2/3"
          variants.push({
            id: "variant-c",
            name: "Variant C: Vectorized",
            strategy: "vec4_vectorized",
            rationale: "generator returned 2/3",
            workgroupSize: [1, 1, 1],
            elementsPerThread: [1, 1, 1],
            wgslBody: "",
          });
        }
      }

      return variants;
    };

    // Attempt 1: Primary Model (90s timeout + 1 retry on timeout)
    try {
      const variants = await callProviderModelWithRetry(primaryModel, primaryTimeoutMs);
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

      // Attempt 2: Fallback Model (90s timeout + 1 retry on timeout)
      try {
        const variants = await callProviderModelWithRetry(fallbackModel, fallbackTimeoutMs);
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
