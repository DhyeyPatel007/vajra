import { GeneratedVariant, WorkloadSpec } from "@/types/vajra";
import { parseWorkgroupSize, repairEarlyReturnBeforeBarrier } from "./webgpuHarness";

/**
 * Repairs module-scope 'let' declarations into 'const'.
 * In WGSL, 'let' is strictly forbidden at module scope (outside function bodies).
 */
export function repairModuleScopeLet(wgsl: string): string {
  const lines = wgsl.split("\n");
  let depth = 0;
  const repairedLines = lines.map((line) => {
    let modified = line;
    // Check if line is at module scope (depth === 0)
    if (depth === 0) {
      // Replace module-scope "let foo" with "const foo"
      modified = modified.replace(/^(\s*)let(\s+[a-zA-Z_])/, "$1const$2");
    }

    // Update brace depth
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === "{") depth++;
      else if (ch === "}") depth = Math.max(0, depth - 1);
    }

    return modified;
  });

  return repairedLines.join("\n");
}

/**
 * Detects whether any module-scope 'let' remains in the WGSL string.
 */
export function hasModuleScopeLet(wgsl: string): boolean {
  const lines = wgsl.split("\n");
  let depth = 0;
  for (const line of lines) {
    if (depth === 0 && /^\s*let\s+[a-zA-Z_]/.test(line)) {
      return true;
    }
    for (let i = 0; i < line.length; i++) {
      if (line[i] === "{") depth++;
      else if (line[i] === "}") depth = Math.max(0, depth - 1);
    }
  }
  return false;
}

/**
 * Clamps @workgroup_size(x, y, z) so total threads x*y*z never exceeds maxLimit.
 */
export function clampWorkgroupSizeInWgsl(wgsl: string, maxLimit: number = 256): string {
  return wgsl.replace(
    /@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?(?:,\s*(\d+)\s*)?\)/g,
    (match, p1, p2, p3) => {
      let x = parseInt(p1, 10) || 1;
      let y = p2 ? parseInt(p2, 10) || 1 : 1;
      let z = p3 ? parseInt(p3, 10) || 1 : 1;

      if (x * y * z <= maxLimit) {
        return match;
      }

      if (y === 1 && z === 1) {
        x = maxLimit;
      } else if (z === 1) {
        if (maxLimit <= 256) {
          x = Math.min(x, 16);
          y = Math.min(y, 16);
        }
        while (x * y > maxLimit && y > 1) y = Math.floor(y / 2);
        while (x * y > maxLimit && x > 1) x = Math.floor(x / 2);
      } else {
        while (x * y * z > maxLimit && z > 1) z = Math.floor(z / 2);
        while (x * y * z > maxLimit && y > 1) y = Math.floor(y / 2);
        while (x * y * z > maxLimit && x > 1) x = Math.floor(x / 2);
      }

      return p2
        ? p3
          ? `@workgroup_size(${x}, ${y}, ${z})`
          : `@workgroup_size(${x}, ${y}, 1)`
        : `@workgroup_size(${x}, 1, 1)`;
    }
  );
}

/**
 * Robustly parses JSON from LLM text and recovers complete variants even if the output
 * was truncated by token limit.
 */
export function extractValidVariantsJson(rawText: string): any {
  let jsonString = rawText.trim();
  const fenceMatch = jsonString.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch && fenceMatch[1]) {
    jsonString = fenceMatch[1].trim();
  }

  const start = jsonString.indexOf("{");
  if (start === -1) return null;

  try {
    return JSON.parse(jsonString.substring(start));
  } catch {
    // Truncated recovery: look backwards for the last valid variant object '}'
    let pos = jsonString.lastIndexOf("}");
    while (pos > start) {
      const candidate = jsonString.substring(start, pos + 1);
      const attempts = [candidate, candidate + "]}", candidate + "}"];
      for (const att of attempts) {
        try {
          const parsed = JSON.parse(att);
          if (parsed && (Array.isArray(parsed) || Array.isArray(parsed.variants))) {
            return parsed;
          }
        } catch {}
      }
      pos = jsonString.lastIndexOf("}", pos - 1);
    }
  }
  return null;
}

export function buildNemotronSystemPrompt(maxWorkgroupSize: number = 256): string {
  return `You are an elite GPU kernel engineer specializing in WebGPU WGSL compute shader performance optimization on physical silicon.
Your task is to take a compute workload specification, review the naive baseline WGSL shader, and produce exactly 3 mathematically equivalent, hardware-accelerated WGSL compute shader variants.

CRITICAL HARDWARE & LANGUAGE CONSTRAINTS:
1. PREAMBLE IS FIXED: The WebGPU harness has already declared the uniform buffer (struct Params at @group(0) @binding(0)) and storage buffers (@group(0) @binding(1..N)).
2. DO NOT REDECLARE PREAMBLE: Never output struct Params or any @group(0) @binding(...) declarations. Output ONLY helper functions, optional var<workgroup> shared memory arrays, and the @compute @workgroup_size(X, Y, Z) fn main(...) entry point.
3. THREE STRICT STRATEGIES:
   - Variant A (workgroup_tuning): Tune workgroup geometry (e.g., 16x16, 128x1, 256x1) and unroll loops for coalesced global DRAM memory accesses.
   - Variant B (shared_memory_tiling): Allocate var<workgroup> shared tile memory with workgroupBarrier() to cache operands and minimize DRAM bus traffic.
   - Variant C (vec4_vectorized): Leverage SIMD vector types (vec4<f32>), vector arithmetic, or dot products to maximize ALU execution density.
4. **WORKGROUP BARRIER RULES & UNIFORM FLOW:**
   All threads in the workgroup MUST execute workgroupBarrier() in uniform control flow. NEVER put workgroupBarrier() after an early return or inside a divergent if!
   - For Softmax: each thread independently processes one row (row = gid.x, problemDims = [1024, 1, 1]); Variant B MUST NOT use workgroupBarrier(). Instead, use @workgroup_size(256, 1, 1) with hoisted reciprocal math (let invSum = 1.0 / sumExp;) and 4-way unrolled scalar/register accumulators without workgroupBarrier().
   - For Matmul: 512 is an exact multiple of 16 (32 tiles). Do NOT write any early return before workgroupBarrier(). Inside the tile loop 'for (var t = 0u; t < 32u; t = t + 1u)', you MUST include TWO workgroupBarrier() calls: (1) load tileA and tileB, then workgroupBarrier(); (2) accumulate sum over the 16 elements; (3) call a SECOND workgroupBarrier() before the end of the 't' loop so the next tile load never overwrites tiles while threads are still reading!
   - For Conv1D: preload the 64-element filter into var<workgroup> sharedFilter: array<f32, 64>; via 'if (lid.x < 64u) { sharedFilter[lid.x] = filterWeight[lid.x]; } workgroupBarrier();' (with NO return before workgroupBarrier()), then accumulate k = 0u..64u.
   - For Sobel: Border pixels (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) write 0.0 to imgOut[y * params.width + x]. Interior pixels compute exact Sobel gradients: gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br; gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br; imgOut[y * params.width + x] = sqrt(gx * gx + gy * gy);. Keep uniform barriers.
5. **VEC4 VECTORIZATION & TAIL BOUNDS RULES:**
   Set elementsPerThread to [1, 1, 1]. 1 thread computes 1 output element (or 1 row for Softmax).
   You MUST use explicit vec4<f32> types and dot() intrinsics in Variant C, constructed from scalar storage arrays:
   - For matmul: Step k from 0u to params.K step 4u:
     let aVec = vec4<f32>(matrixA[row * params.K + k], matrixA[row * params.K + k + 1u], matrixA[row * params.K + k + 2u], matrixA[row * params.K + k + 3u]);
     let bVec = vec4<f32>(matrixB[(k + 0u) * params.N + col], matrixB[(k + 1u) * params.N + col], matrixB[(k + 2u) * params.N + col], matrixB[(k + 3u) * params.N + col]);
     sum += dot(aVec, bVec);
   - For conv1d: Whenever reading signalIn[idx + k], you MUST guard out-of-bounds indices at the end of the signal using select(0.0, signalIn[idx + k], (idx + k) < params.signalLen). Step k by 4u and accumulate with dot(vSig, vW).
   - For softmax: Step c by 4u, loading let v = vec4<f32>(inputBuf[base+c], inputBuf[base+c+1u], inputBuf[base+c+2u], inputBuf[base+c+3u]); in all 3 passes.
6. MODULE-SCOPE VARIABLES: In WGSL, 'let' is strictly forbidden at module scope (outside functions). Module scope may ONLY use 'const' (for compile-time constants like filters/masks) or 'var<workgroup>' (for shared memory). 'let' is ONLY valid inside function bodies (e.g. inside fn main). NEVER declare 'let' at module scope.
7. WORKGROUP SIZE HARDWARE LIMIT: The target GPU hardware limit is STRICTLY ${maxWorkgroupSize} total threads per workgroup (X * Y * Z <= ${maxWorkgroupSize}).
   NEVER generate @workgroup_size(1024, 1, 1), @workgroup_size(32, 32, 1), or any workgroup where X * Y * Z > ${maxWorkgroupSize}. For 2D workloads use (16, 16, 1) or (8, 8, 1). For 1D workloads use (256, 1, 1), (128, 1, 1), or (64, 1, 1).
8. ZERO COMMENTS & CONCISE KERNEL BODIES: Write ZERO comments (no // or /* */) inside wgslBody so JSON output stays compact and never truncates on Sobel. Keep rationale to 1 sentence.
9. NUMERICAL EQUIVALENCE: Every kernel must produce element-wise results identical to the baseline within 1e-4 relative error.
10. STRICT JSON ONLY: Return ONLY a raw JSON object with no explanatory text before or after it. Do not include markdown commentary.

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

export function buildNemotronUserPrompt(workload: WorkloadSpec, maxWorkgroupSize: number = 256): string {
  let workloadSpecificRules = "";
  if (workload.id === "matmul") {
    workloadSpecificRules = `
CRITICAL MATMUL (512x512) RULES:
- Variant B (shared_memory_tiling): Use @workgroup_size(16, 16, 1) and var<workgroup> tileA: array<f32, 256>; var<workgroup> tileB: array<f32, 256>;. Because 512 is an exact multiple of 16 (32 tiles), do NOT write any early return before workgroupBarrier(). Inside the tile loop 'for (var t = 0u; t < 32u; t = t + 1u)', you MUST include TWO workgroupBarrier() calls: (1) load tileA[lid.y * 16u + lid.x] = matrixA[row * params.K + t * 16u + lid.x] and tileB[lid.y * 16u + lid.x] = matrixB[(t * 16u + lid.y) * params.N + col], then call workgroupBarrier(); (2) loop k = 0u..16u accumulating sum += tileA[lid.y * 16u + k] * tileB[k * 16u + lid.x]; (3) call a SECOND workgroupBarrier() before the end of the 't' loop so the next tile load never overwrites tileA/tileB while threads are still reading!
- Variant C (vec4_vectorized): Use @workgroup_size(16, 16, 1), elementsPerThread: [1, 1, 1]. Loop k from 0u to params.K step 4u: let aVec = vec4<f32>(matrixA[row * params.K + k], matrixA[row * params.K + k + 1u], matrixA[row * params.K + k + 2u], matrixA[row * params.K + k + 3u]); let bVec = vec4<f32>(matrixB[(k + 0u) * params.N + col], matrixB[(k + 1u) * params.N + col], matrixB[(k + 2u) * params.N + col], matrixB[(k + 3u) * params.N + col]); sum += dot(aVec, bVec);`;
  } else if (workload.id === "conv1d") {
    workloadSpecificRules = `
CRITICAL CONV1D RULES:
- All Variants: Whenever reading signalIn[idx + k], you MUST guard out-of-bounds indices at the end of the signal using select(0.0, signalIn[idx + k], (idx + k) < params.signalLen). Never read signalIn[idx + k] unguarded or clamp to signalLen - 1u, or the last 63 elements will fail < 1e-4 verification!
- Variant B (shared_memory_tiling): Use @workgroup_size(256, 1, 1) and preload ONLY the 64-element filter into var<workgroup> sharedFilter: array<f32, 64>; via 'if (lid.x < 64u) { sharedFilter[lid.x] = filterWeight[lid.x]; } workgroupBarrier();' (with NO return before workgroupBarrier()), then accumulate k = 0u..64u reading sharedFilter[k] and select(0.0, signalIn[idx + k], (idx + k) < params.signalLen).
- Variant C (vec4_vectorized): Step k by 4u, elementsPerThread: [1, 1, 1], construct vec4<f32> filter values and vec4<f32> signal values (each element using select) and accumulate using dot().`;
  } else if (workload.id === "sobel") {
    workloadSpecificRules = `
CRITICAL SOBEL RULES:
- Border pixels (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) MUST write 0.0 to imgOut[y * params.width + x].
- Interior pixels MUST compute exact Sobel gradients:
  gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
  gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
  imgOut[y * params.width + x] = sqrt(gx * gx + gy * gy);
- For Variant B: avoid complex apron indexing bugs by caching the current row neighborhood or keeping uniform barriers before the border check.`;
  } else if (workload.id === "softmax") {
    workloadSpecificRules = `
CRITICAL SOFTMAX RULES:
- Each thread independently processes one row (row = gid.x, problemDims = [1024, 1, 1]).
- Variant B MUST NOT use workgroupBarrier(); instead use @workgroup_size(256, 1, 1) with hoisted reciprocal (let invSum = 1.0 / sumExp;) and 4-way unrolled scalar/register accumulators.
- Variant C: set elementsPerThread to [1, 1, 1]. Step c by 4u loading vec4<f32>(inputBuf[base+c], inputBuf[base+c+1u], inputBuf[base+c+2u], inputBuf[base+c+3u]) in all 3 passes.`;
  }

  return `Target Workload: ${workload.name} (${workload.dimensionsLabel})
Description: ${workload.description}
Problem Dimensions: [${workload.problemDims.join(", ")}]
Total FLOPs: ${workload.totalFlops}
Total Bytes Moved: ${workload.totalBytesMoved}
Hardware Workgroup Limit: Maximum ${maxWorkgroupSize} invocations per workgroup (X*Y*Z <= ${maxWorkgroupSize}). DO NOT exceed ${maxWorkgroupSize}.
WGSL Rules:
- In WGSL, 'let' is forbidden at module scope; use 'const' for module constants, or declare 'let' inside fn main.
- Write ZERO comments (no // or /* */) inside wgslBody.
- Never put workgroupBarrier() after an early return or inside a divergent if.
- For vec4_vectorized: set elementsPerThread to [1, 1, 1]. 1 thread computes 1 output (or 1 row for softmax).
${workloadSpecificRules}

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

export function sanitizeWgslBody(rawBody: string, maxWorkgroupSize: number = 256): string {
  let cleaned = rawBody;

  // Strip markdown code fences
  cleaned = cleaned.replace(/```(?:wgsl|rust|c|cpp)?/gi, "").replace(/```/g, "");

  // Strip accidental harness preamble bindings if echoed by LLM
  cleaned = cleaned.replace(/@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*[^;]+;/g, "");

  // Strip duplicate struct Params definitions
  cleaned = cleaned.replace(/struct\s+Params\s*\{[^}]*\};?/g, "");

  // Repair module-scope 'let' to 'const'
  cleaned = repairModuleScopeLet(cleaned);

  // Repair early return before workgroupBarrier
  cleaned = repairEarlyReturnBeforeBarrier(cleaned);

  // Clamp workgroup size so X*Y*Z <= maxWorkgroupSize
  cleaned = clampWorkgroupSizeInWgsl(cleaned, maxWorkgroupSize);

  return cleaned.trim();
}

export function parseNemotronResponse(rawText: string, maxWorkgroupSize: number = 256): GeneratedVariant[] {
  const parsed = extractValidVariantsJson(rawText);
  if (!parsed) {
    throw new Error("Failed to extract valid JSON variants from model output");
  }

  const rawVariants = Array.isArray(parsed) ? parsed : parsed.variants;

  if (!Array.isArray(rawVariants) || rawVariants.length === 0) {
    throw new Error(`Expected at least 1 variant in response, received ${rawVariants?.length || 0}`);
  }

  const variantIds: Array<"variant-a" | "variant-b" | "variant-c"> = [
    "variant-a",
    "variant-b",
    "variant-c",
  ];

  return rawVariants.slice(0, 3).map((v: Partial<GeneratedVariant>, index: number) => {
    const id = variantIds[index] || `variant-${index}`;
    const sanitizedBody = sanitizeWgslBody(v.wgslBody || "", maxWorkgroupSize);
    const parsedWgSize = parseWorkgroupSize(sanitizedBody);

    let strategy =
      v.strategy ||
      (index === 0
        ? "workgroup_tuning"
        : index === 1
        ? "shared_memory_tiling"
        : "vec4_vectorized");
    let name = v.name || `Variant ${id.toUpperCase().replace("VARIANT-", "")}`;

    // Vec4 honesty validator: if strategy claims vec4 but the WGSL has no vec4 type, rename to 4-way unroll
    const hasVec4Type = /\b(?:vec4<f32>|vec4f|vec4<[a-zA-Z0-9_]+>|vec4)\b/.test(sanitizedBody);
    if ((strategy === "vec4_vectorized" || strategy.toLowerCase().includes("vec4")) && !hasVec4Type) {
      strategy = "4-way unroll";
      name = name.replace(/vec4\s*vectorized/gi, "4-Way Unroll").replace(/vec4/gi, "4-Way Unroll");
    }

    return {
      id,
      name,
      strategy,
      rationale:
        v.rationale ||
        "Hardware-tuned WGSL kernel generated by Nemotron.",
      workgroupSize: parsedWgSize,
      elementsPerThread: [1, 1, 1],
      wgslBody: sanitizedBody,
    };
  });
}

export function buildNemotronGen2SystemPrompt(maxWorkgroupSize: number = 256): string {
  return `You are an elite GPU kernel engineer specializing in WebGPU WGSL evolutionary autotuning.
You are executing a Generation 2 refinement pass. You will receive the Generation 1 winning kernel (with measured hardware telemetry) and any compiler errors from dropped variants.
Your task is to synthesize exactly ONE evolved variant: "variant-d".

CRITICAL GEN 2 CONSTRAINTS:
1. PREAMBLE IS FIXED: Do NOT output struct Params or any @group(0) @binding(...) declarations.
2. FUSED EVOLUTION: Fuse the winning Gen 1 strategy with register caching, 4-way unrolled fma() accumulation, and uniform shared memory tiling.
3. **BARRIER UNIFORMITY: NEVER put workgroupBarrier() after an early return or inside a divergent if.** All barriers must execute uniformly.
4. MODULE-SCOPE 'let' FORBIDDEN: Module scope may use ONLY 'const' or 'var<workgroup>'. 'let' is only allowed inside functions.
5. WORKGROUP SIZE HARDWARE LIMIT: Maximum ${maxWorkgroupSize} invocations per workgroup (workgroup_size X*Y*Z <= ${maxWorkgroupSize}).
6. GUARDRAILS: Do NOT repeat any syntax or intrinsic errors that caused compiler failures in dropped variants.
7. NUMERICAL EQUIVALENCE: Must maintain exact IEEE-754 correctness within 1e-4 relative error.
8. STRICT JSON ONLY: Return ONLY a raw JSON object with no markdown commentary:
{
  "variants": [
    {
      "id": "variant-d",
      "name": "Variant D: Gen 2 Fused Tile + Unrolled FMA Pipeline",
      "strategy": "gen2_evolved",
      "rationale": "One-sentence explanation of the evolutionary synthesis.",
      "elementsPerThread": [1, 1, 1],
      "wgslBody": "..."
    }
  ]
}`;
}

export function buildNemotronGen2UserPrompt(
  workload: WorkloadSpec,
  previousWinner?: { id: string; name: string; gflops?: number | null; medianTimeMs?: number | null; wgslBody: string },
  failedVariants?: Array<{ name: string; compileErrors: string[] }>,
  maxWorkgroupSize: number = 256,
  _forceStandardGen2: boolean = false
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

  let gen2Instruction = "";
  if (workload.id === "matmul") {
    gen2Instruction = `
SGEMM GEN 2 OPTIMIZATION DIRECTIVE:
Use @workgroup_size(16, 16, 1) and var<workgroup> tileA: array<f32, 256>; var<workgroup> tileB: array<f32, 256>;. Because 512 is an exact multiple of 16 (32 tiles), do NOT write any early return before workgroupBarrier(). Inside the tile loop 'for (var t = 0u; t < 32u; t = t + 1u)', you MUST include TWO workgroupBarrier() calls: (1) load tileA[lid.y * 16u + lid.x] = matrixA[row * params.K + t * 16u + lid.x] and tileB[lid.y * 16u + lid.x] = matrixB[(t * 16u + lid.y) * params.N + col], then call workgroupBarrier(); (2) 4-way unrolled fma() accumulation sum += tileA[lid.y * 16u + k] * tileB[k * 16u + lid.x]; (3) call a SECOND workgroupBarrier() before the end of the 't' loop so the next tile load never overwrites tileA/tileB while threads are still reading! Set elementsPerThread: [1, 1, 1].`;
  } else if (workload.id === "conv1d") {
    gen2Instruction = `
CONV1D GEN 2 OPTIMIZATION DIRECTIVE:
Preload the 64-element filter into var<workgroup> sharedFilter: array<f32, 64>; with uniform workgroupBarrier() and no return before the barrier. Whenever reading signalIn[idx + k], you MUST guard out-of-bounds indices using select(0.0, signalIn[idx + k], (idx + k) < params.signalLen). Accumulate with 4-way unrolled fma(). Set elementsPerThread: [1, 1, 1].`;
  } else if (workload.id === "sobel") {
    gen2Instruction = `
SOBEL GEN 2 OPTIMIZATION DIRECTIVE:
Border pixels (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) write 0.0 to imgOut. Interior pixels compute exact Sobel gradients:
gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
imgOut[y * params.width + x] = sqrt(gx * gx + gy * gy);
Keep uniform control flow and elementsPerThread: [1, 1, 1].`;
  } else if (workload.id === "softmax") {
    gen2Instruction = `
SOFTMAX GEN 2 OPTIMIZATION DIRECTIVE:
Each thread row = gid.x processes 1 row. Do NOT use workgroupBarrier(). Use @workgroup_size(256, 1, 1), elementsPerThread: [1, 1, 1], hoisted reciprocal (let invSum = 1.0 / sumExp;), and vec4<f32> vectorization across columns.`;
  }

  return `Target Workload: ${workload.name} (${workload.dimensionsLabel})
Description: ${workload.description}
Problem Dimensions: [${workload.problemDims.join(", ")}]
Hardware Workgroup Limit: Maximum ${maxWorkgroupSize} invocations per workgroup (X*Y*Z <= ${maxWorkgroupSize}).
WGSL Rule: In WGSL, 'let' is forbidden at module scope; use 'const' for module constants, or declare 'let' inside fn main. NEVER put workgroupBarrier() after an early return.
${gen2Instruction}

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
  parentWinnerId: string = "variant-c",
  maxWorkgroupSize: number = 256
): GeneratedVariant[] {
  const parsed = extractValidVariantsJson(rawText);
  if (!parsed) {
    throw new Error("Failed to extract valid JSON variant from Gen 2 model output");
  }

  const rawVariants = Array.isArray(parsed) ? parsed : parsed.variants;
  const raw = Array.isArray(rawVariants) ? rawVariants[0] : parsed;

  const sanitizedBody = sanitizeWgslBody(raw?.wgslBody || "", maxWorkgroupSize);
  const parsedWgSize = parseWorkgroupSize(sanitizedBody);

  return [
    {
      id: "variant-d",
      name: raw?.name || "Variant D: Gen 2 Fused Tile + Unrolled FMA Pipeline",
      strategy: "gen2_evolved",
      rationale:
        raw?.rationale ||
        "Evolutionary synthesis: single-buffered 16x16 shared tile memory with 4-way unrolled fma() accumulation in registers.",
      workgroupSize: parsedWgSize,
      elementsPerThread: [1, 1, 1],
      wgslBody: sanitizedBody,
      generation: 2,
      parentWinnerId,
    },
  ];
}
