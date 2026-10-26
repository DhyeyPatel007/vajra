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
