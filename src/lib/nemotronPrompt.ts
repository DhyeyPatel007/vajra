import { GeneratedVariant, WorkloadId, WorkloadSpec } from "@/types/vajra";
import { parseWorkgroupSize } from "./webgpuHarness";

export function buildNemotronSystemPrompt(): string {
  return `You are an elite GPU kernel engineer specializing in WebGPU WGSL compute shader performance optimization on physical silicon.
Your task is to take a compute workload specification, review the naive baseline WGSL shader, and produce exactly 3 mathematically equivalent, hardware-accelerated WGSL compute shader variants.

CRITICAL HARDWARE & LANGUAGE CONSTRAINTS:
1. PREAMBLE IS FIXED: The WebGPU harness has already declared the uniform buffer (struct Params at @group(0) @binding(0)) and storage buffers (@group(0) @binding(1..N)).
2. DO NOT REDECLARE PREAMBLE: Never output struct Params or any @group(0) @binding(...) declarations. Output ONLY helper functions, optional var<workgroup> shared memory arrays, and the @compute @workgroup_size(X, Y, Z) fn main(...) entry point.
3. THREE STRICT STRATEGIES:
   - Variant A (workgroup_tuning): Tune workgroup geometry (e.g., 16x16, 128x1, 256x1) and unroll loops for coalesced global DRAM memory accesses.
   - Variant B (shared_memory_tiling): Allocate var<workgroup> shared tile memory with workgroupBarrier() to cache operands and minimize DRAM bus traffic.
   - Variant C (vec4_vectorized): Leverage SIMD vector types (vec4<f32>, vec3<f32>), vector arithmetic, or dot products to maximize ALU execution density.
4. NUMERICAL EQUIVALENCE: Every kernel must produce element-wise results identical to the baseline within 1e-4 relative error.
5. STRICT JSON ONLY: Return ONLY a raw JSON object with no explanatory text before or after it. Do not include markdown commentary.

Output JSON structure:
{
  "variants": [
    {
      "id": "variant-a",
      "name": "Variant A: Title",
      "strategy": "workgroup_tuning",
      "rationale": "One-sentence hardware performance rationale.",
      "elementsPerThread": [1, 1, 1],
      "wgslBody": "..."
    },
    {
      "id": "variant-b",
      "name": "Variant B: Title",
      "strategy": "shared_memory_tiling",
      "rationale": "One-sentence hardware performance rationale.",
      "elementsPerThread": [1, 1, 1],
      "wgslBody": "..."
    },
    {
      "id": "variant-c",
      "name": "Variant C: Title",
      "strategy": "vec4_vectorized",
      "rationale": "One-sentence hardware performance rationale.",
      "elementsPerThread": [1, 1, 1],
      "wgslBody": "..."
    }
  ]
}`;
}

export function buildNemotronUserPrompt(workload: WorkloadSpec): string {
  return `Target Workload: ${workload.name} (${workload.dimensionsLabel})
Description: ${workload.description}
Problem Dimensions: [${workload.problemDims.join(", ")}]
Total FLOPs: ${workload.totalFlops}
Total Bytes Moved: ${workload.totalBytesMoved}

Fixed Harness WGSL Preamble (ALREADY PROVIDED - DO NOT REPEAT IN OUTPUT):
\`\`\`wgsl
${workload.wgslPreamble}
\`\`\`

Naive Baseline WGSL Kernel:
\`\`\`wgsl
${workload.baselineWgslBody}
\`\`\`

Generate 3 hardware-optimized variants (Variant A: workgroup_tuning, Variant B: shared_memory_tiling, Variant C: vec4_vectorized). Return valid JSON matching the schema.`;
}

export function sanitizeWgslBody(rawBody: string): string {
  let cleaned = rawBody;

  // Strip markdown code fences
  cleaned = cleaned.replace(/```(?:wgsl|rust|c|cpp)?/gi, "").replace(/```/g, "");

  // Strip accidental harness preamble bindings if echoed by LLM
  cleaned = cleaned.replace(/@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*[^;]+;/g, "");

  // Strip duplicate struct Params definitions
  cleaned = cleaned.replace(/struct\s+Params\s*\{[^}]*\};?/g, "");

  return cleaned.trim();
}

export function parseNemotronResponse(rawText: string): GeneratedVariant[] {
  let jsonString = rawText.trim();

  // Strip outer markdown json fences if present
  const fenceMatch = jsonString.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) {
    jsonString = fenceMatch[1].trim();
  } else {
    // Find first '{' and last '}'
    const start = jsonString.indexOf("{");
    const end = jsonString.lastIndexOf("}");
    if (start !== -1 && end !== -1 && end > start) {
      jsonString = jsonString.substring(start, end + 1);
    }
  }

  const parsed = JSON.parse(jsonString);
  const rawVariants = Array.isArray(parsed) ? parsed : parsed.variants;

  if (!Array.isArray(rawVariants) || rawVariants.length < 3) {
    throw new Error(`Expected at least 3 variants in response, received ${rawVariants?.length || 0}`);
  }

  const variantIds: Array<"variant-a" | "variant-b" | "variant-c"> = [
    "variant-a",
    "variant-b",
    "variant-c",
  ];

  return rawVariants.slice(0, 3).map((v: Partial<GeneratedVariant>, index: number) => {
    const id = variantIds[index];
    const sanitizedBody = sanitizeWgslBody(v.wgslBody || "");
    const parsedWgSize = parseWorkgroupSize(sanitizedBody);

    return {
      id,
      name: v.name || `Variant ${id.toUpperCase().replace("VARIANT-", "")}`,
      strategy: v.strategy || (index === 0 ? "workgroup_tuning" : index === 1 ? "shared_memory_tiling" : "vec4_vectorized"),
      rationale: v.rationale || "Hardware-tuned WGSL kernel generated by Nemotron via Nebius Token Factory.",
      workgroupSize: parsedWgSize,
      elementsPerThread: v.elementsPerThread || [1, 1, 1],
      wgslBody: sanitizedBody,
    };
  });
}

export function buildNemotronGen2SystemPrompt(): string {
  return `You are an elite GPU kernel engineer specializing in WebGPU WGSL evolutionary autotuning.
You are executing a Generation 2 refinement pass. You will receive the Generation 1 winning kernel (with measured hardware telemetry) and any compiler errors from dropped variants.
Your task is to synthesize exactly ONE evolved variant: "variant-d".

CRITICAL GEN 2 CONSTRAINTS:
1. PREAMBLE IS FIXED: Do NOT output struct Params or any @group(0) @binding(...) declarations.
2. FUSED EVOLUTION: Fuse the winning Gen 1 strategy with register coarsening (WPT), unrolled loop iterations, or vectorized arithmetic.
3. GUARDRAILS: Do NOT repeat any syntax or intrinsic errors that caused compiler failures in dropped variants.
4. NUMERICAL EQUIVALENCE: Must maintain exact IEEE-754 correctness within 1e-4 relative error.
5. STRICT JSON ONLY: Return ONLY a raw JSON object with no markdown commentary:
{
  "variants": [
    {
      "id": "variant-d",
      "name": "Variant D: Gen 2 Fused Tile + Register Coarsened",
      "strategy": "gen2_evolved",
      "rationale": "One-sentence explanation of the evolutionary synthesis and coarsening.",
      "elementsPerThread": [1, 1, 1],
      "wgslBody": "..."
    }
  ]
}`;
}

export function buildNemotronGen2UserPrompt(
  workload: WorkloadSpec,
  previousWinner?: { id: string; name: string; gflops?: number | null; medianTimeMs?: number | null; wgslBody: string },
  failedVariants?: Array<{ name: string; compileErrors: string[] }>
): string {
  const winnerInfo = previousWinner
    ? `GEN 1 WINNER KERNEL:
Name: ${previousWinner.name} (${previousWinner.id})
Measured Throughput: ${previousWinner.gflops ? previousWinner.gflops.toFixed(1) : "N/A"} GFLOPS
Measured Median Latency: ${previousWinner.medianTimeMs ? previousWinner.medianTimeMs.toFixed(3) : "N/A"} ms
Winning WGSL Body:
\`\`\`wgsl
${previousWinner.wgslBody}
\`\`\``
    : "No prior winner recorded.";

  const failureInfo =
    failedVariants && failedVariants.length > 0
      ? `GEN 1 DROPPED / COMPILER FAILED VARIANTS (AVOID THESE ERRORS):
${failedVariants
  .map(
    (f) =>
      `- ${f.name}:\n  Compiler Errors: ${f.compileErrors.join("; ") || "Syntax / Linker Fault"}`
  )
  .join("\n")}`
      : "No compiler failures logged in Gen 1.";

  return `Target Workload: ${workload.name} (${workload.dimensionsLabel})
Description: ${workload.description}
Problem Dimensions: [${workload.problemDims.join(", ")}]

Fixed Harness WGSL Preamble (DO NOT REPEAT):
\`\`\`wgsl
${workload.wgslPreamble}
\`\`\`

${winnerInfo}

${failureInfo}

Produce ONE evolved Generation 2 kernel ("variant-d") that improves upon the Gen 1 winner while strictly avoiding the compiler faults above. Return valid JSON matching the schema.`;
}

export function parseNemotronGen2Response(
  rawText: string,
  parentWinnerId: string = "variant-c"
): GeneratedVariant[] {
  let jsonString = rawText.trim();

  const fenceMatch = jsonString.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) {
    jsonString = fenceMatch[1].trim();
  } else {
    const start = jsonString.indexOf("{");
    const end = jsonString.lastIndexOf("}");
    if (start !== -1 && end !== -1 && end > start) {
      jsonString = jsonString.substring(start, end + 1);
    }
  }

  const parsed = JSON.parse(jsonString);
  const rawVariants = Array.isArray(parsed) ? parsed : parsed.variants;
  const raw = Array.isArray(rawVariants) ? rawVariants[0] : parsed;

  const sanitizedBody = sanitizeWgslBody(raw?.wgslBody || "");
  const parsedWgSize = parseWorkgroupSize(sanitizedBody);

  return [
    {
      id: "variant-d",
      name: raw?.name || "Variant D: Gen 2 Fused Tile + Register Coarsened",
      strategy: "gen2_evolved",
      rationale:
        raw?.rationale ||
        "Evolutionary synthesis: double-buffered shared tile memory with coarsened registers and unrolled inner loops.",
      workgroupSize: parsedWgSize,
      elementsPerThread: raw?.elementsPerThread || [1, 1, 1],
      wgslBody: sanitizedBody,
      generation: 2,
      parentWinnerId,
    },
  ];
}
