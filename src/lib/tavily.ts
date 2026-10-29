import { WorkloadId } from "@/types/vajra";
import { sanitizeWgslBody } from "./nemotronPrompt";

export interface TavilySource {
  title: string;
  url: string;
}

export interface TavilyGroundingResult {
  query: string;
  sources: TavilySource[];
  contentSnippets?: string[];
}

export interface TavilyExtractResult {
  ok: boolean;
  wgsl?: string;
  error?: string;
  originalUrl?: string;
  requiredBindings?: string[];
}

/**
 * Expected storage buffer bindings from src/lib/workloads.ts
 */
export const WORKLOAD_BINDINGS: Record<WorkloadId, string[]> = {
  matmul: ["matrixA", "matrixB", "matrixC"],
  softmax: ["inputBuf", "outputBuf"],
  conv1d: ["signalIn", "filterWeight", "signalOut"],
  sobel: ["imgIn", "imgOut"],
};

/**
 * Perform a single Tavily Search call (3.5s timeout, 1 credit) during Gen 1 Live Token Factory
 * generation to ground Nemotron in W3C WGSL uniform control-flow and memory-tiling specifications.
 * Gracefully returns null on any error or missing key so generation is never blocked.
 */
export async function tavilySearchGrounding(
  workloadId: WorkloadId
): Promise<TavilyGroundingResult | null> {
  const apiKey = (process.env.TAVILY_API_KEY || "").trim();
  if (!apiKey || apiKey.startsWith("your_") || apiKey.includes("placeholder")) {
    return null;
  }

  const query = `W3C WebGPU WGSL compute shader ${workloadId} uniform workgroupBarrier memory tiling`;

  try {
    const res = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        search_depth: "basic",
        max_results: 3,
        include_answer: false,
      }),
      signal: AbortSignal.timeout(3500),
    });

    if (!res.ok) {
      console.warn(`[Vajra] Tavily search returned HTTP ${res.status}`);
      return null;
    }

    const data = await res.json();
    const results = Array.isArray(data.results) ? data.results : [];

    const sources: TavilySource[] = results.slice(0, 3).map((r: { title?: string; url?: string }) => ({
      title: r.title || "W3C WGSL Specification Reference",
      url: r.url || "https://www.w3.org/TR/WGSL/",
    }));

    const contentSnippets: string[] = results
      .slice(0, 3)
      .map((r: { content?: string }) => r.content || "")
      .filter((c: string) => c.length > 0);

    return {
      query,
      sources,
      contentSnippets,
    };
  } catch (err: unknown) {
    console.warn("[Vajra] Tavily search grounding non-fatal failure:", err);
    return null;
  }
}

/**
 * Perform honest Tavily Extract on a user-provided GitHub URL (4.5s timeout, 1 credit).
 * Enforces the strict compatibility gate matching exact bindings from src/lib/workloads.ts.
 * Does NOT perform any fake direct raw.githubusercontent.com fallback.
 */
export async function tavilyExtractKernel(
  url: string,
  workloadId: WorkloadId
): Promise<TavilyExtractResult> {
  const trimmedUrl = (url || "").trim();
  if (!trimmedUrl) {
    return { ok: false, error: "Please enter a valid GitHub repository or raw file URL." };
  }

  // Validate GitHub domain
  try {
    const parsed = new URL(trimmedUrl);
    const host = parsed.hostname.toLowerCase();
    if (host !== "github.com" && host !== "raw.githubusercontent.com" && host !== "gist.github.com") {
      return {
        ok: false,
        error: "URL must point to a public GitHub file (github.com, raw.githubusercontent.com, or gist.github.com).",
      };
    }
  } catch {
    return { ok: false, error: "Invalid URL format." };
  }

  const apiKey = (process.env.TAVILY_API_KEY || "").trim();
  if (!apiKey || apiKey.startsWith("your_") || apiKey.includes("placeholder")) {
    return {
      ok: false,
      error: "TAVILY_API_KEY is not configured in the server environment (.env.local). Please set TAVILY_API_KEY to use Tavily Extract.",
    };
  }

  try {
    const res = await fetch("https://api.tavily.com/extract", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_key: apiKey,
        urls: [trimmedUrl],
      }),
      signal: AbortSignal.timeout(4500),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return {
        ok: false,
        error: `Tavily Extract API returned HTTP ${res.status}: ${errText.slice(0, 150)}`,
      };
    }

    const data = await res.json();
    const resultItem = data.results?.[0];
    const rawContent = resultItem?.raw_content || resultItem?.content || "";

    if (!rawContent || rawContent.trim().length === 0) {
      return {
        ok: false,
        error: "Tavily Extract succeeded but returned empty content from the specified URL.",
      };
    }

    // Extract WGSL code from content (either inside markdown fence or raw text)
    let extractedWgsl = rawContent;
    const fenceMatch = rawContent.match(/```(?:wgsl|rust|c|cpp)?\s*([\s\S]*?)\s*```/);
    if (fenceMatch && fenceMatch[1]) {
      extractedWgsl = fenceMatch[1].trim();
    }

    // Strict Compatibility Gate: check exact storage buffer bindings for the workload
    const requiredBindings = WORKLOAD_BINDINGS[workloadId] || [];
    const missingBindings = requiredBindings.filter((bindingName) => {
      // Must contain the identifier as a whole word
      const reg = new RegExp(`\\b${bindingName}\\b`);
      return !reg.test(extractedWgsl);
    });

    if (missingBindings.length > 0) {
      return {
        ok: false,
        error: `Compatibility Gate Rejected: The extracted kernel does not match the Vajra ${workloadId.toUpperCase()} binding contract. Missing required buffer bindings: [${missingBindings.join(", ")}]. Expected: [${requiredBindings.join(", ")}].`,
        requiredBindings,
      };
    }

    // Check for compute entry point
    if (!extractedWgsl.includes("@compute") || !extractedWgsl.includes("fn main")) {
      return {
        ok: false,
        error: "Compatibility Gate Rejected: The extracted code must contain a WebGPU compute entry point (@compute fn main).",
      };
    }

    const sanitized = sanitizeWgslBody(extractedWgsl);

    return {
      ok: true,
      wgsl: sanitized,
      originalUrl: trimmedUrl,
      requiredBindings,
    };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("timeout") || msg.includes("aborted")) {
      return {
        ok: false,
        error: "Tavily Extract timed out after 4.5 seconds. Please try again or check the URL.",
      };
    }
    return {
      ok: false,
      error: `Tavily Extract failed: ${msg}`,
    };
  }
}

