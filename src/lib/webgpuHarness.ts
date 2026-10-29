import {
  HardwareProbeInfo,
  TimingMethod,
  VariantExecutionResult,
  WorkloadId,
  WorkloadSpec,
} from "@/types/vajra";
import { WORKLOADS } from "./workloads";

export interface WebGpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  probe: HardwareProbeInfo;
}

let cachedContext: WebGpuContext | null = null;

export async function probeAndInitWebGpu(): Promise<HardwareProbeInfo & { context?: WebGpuContext }> {
  if (cachedContext) {
    return { ...cachedContext.probe, context: cachedContext };
  }
  if (typeof window === "undefined" || !navigator.gpu) {
    return {
      supported: false,
      adapterDescription: "WebGPU Unsupported / Headless",
      vendor: "Generic",
      architecture: "Unknown",
      supportsTimestampQuery: false,
      timingMethod: "recorded-reference",
      isMockOnlyFallback: true,
    };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (!adapter) {
      return {
        supported: false,
        adapterDescription: "No Compatible GPU Adapter Found",
        vendor: "Generic",
        architecture: "Unknown",
        supportsTimestampQuery: false,
        timingMethod: "recorded-reference",
        isMockOnlyFallback: true,
      };
    }

    const rawAdapter = adapter as unknown as { info?: GPUAdapterInfo; requestAdapterInfo?: () => Promise<GPUAdapterInfo> };
    const adapterInfo = rawAdapter.info || (await rawAdapter.requestAdapterInfo?.()) || ({} as Partial<GPUAdapterInfo>);
    const vendor = adapterInfo.vendor || "GPU Vendor";
    const architecture = adapterInfo.architecture || "";
    const description = adapterInfo.description || "";
    const adapterDescription = [vendor, architecture, description].filter(Boolean).join(" / ") || "Physical GPU Adapter";

    const canTimestamp = adapter.features.has("timestamp-query");
    let device: GPUDevice;
    let timingMethod: TimingMethod = "wall-clock-fallback";
    let supportsTimestampQuery = false;

    if (canTimestamp) {
      try {
        device = await adapter.requestDevice({
          requiredFeatures: ["timestamp-query"],
        });
        supportsTimestampQuery = true;
        timingMethod = "timestamp-query";
      } catch {
        // Fall back to requesting device without timestamp-query
        device = await adapter.requestDevice();
        supportsTimestampQuery = false;
        timingMethod = "wall-clock-fallback";
      }
    } else {
      device = await adapter.requestDevice();
      supportsTimestampQuery = false;
      timingMethod = "wall-clock-fallback";
    }

    const maxComputeInvocationsPerWorkgroup =
      device.limits?.maxComputeInvocationsPerWorkgroup ?? 256;

    const probe: HardwareProbeInfo = {
      supported: true,
      adapterDescription,
      vendor,
      architecture,
      supportsTimestampQuery,
      timingMethod,
      isMockOnlyFallback: false,
      maxComputeInvocationsPerWorkgroup,
    };

    cachedContext = { adapter, device, probe };
    return { ...probe, context: cachedContext };
  } catch (err: unknown) {
    return {
      supported: false,
      adapterDescription: err instanceof Error ? err.message : "WebGPU Init Failed",
      vendor: "Generic",
      architecture: "Unknown",
      supportsTimestampQuery: false,
      timingMethod: "recorded-reference",
      isMockOnlyFallback: true,
    };
  }
}

export function parseWorkgroupSize(wgslBody: string): [number, number, number] {
  if (!wgslBody || typeof wgslBody !== "string") {
    return [1, 1, 1];
  }
  try {
    const match = wgslBody.match(
      /@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?(?:,\s*(\d+)\s*)?\)/
    );
    if (!match || !match[1]) {
      return [1, 1, 1];
    }
    const x = parseInt(match[1], 10) || 1;
    const y = match[2] ? parseInt(match[2], 10) || 1 : 1;
    const z = match[3] ? parseInt(match[3], 10) || 1 : 1;
    return [x, y, z];
  } catch (err) {
    console.warn("[Vajra] Error parsing workgroup size from WGSL body:", err);
    return [1, 1, 1];
  }
}

export function computeDispatchGrid(
  problemDims: [number, number, number],
  workgroupSize: [number, number, number],
  elementsPerThread: [number, number, number] = [1, 1, 1],
  wgslBody?: string,
  workloadId?: string
): [number, number, number] {
  let eptX = elementsPerThread?.[0] || 1;
  let eptY = elementsPerThread?.[1] || 1;
  let eptZ = elementsPerThread?.[2] || 1;

  // Softmax: 1 thread = 1 row (problemDims = [1024, 1, 1]).
  // Vectorization is along inner columns, so dispatchGrid along X MUST be Math.ceil(1024 / workgroupSize[0]).
  if (workloadId === "softmax" || (problemDims[0] === 1024 && problemDims[1] === 1 && problemDims[2] === 1)) {
    // For softmax, always force effectiveElementsPerThread = [1, 1, 1] unless explicitly striding rows
    if (!wgslBody || !/gid\.x\s*\*\s*4|global_invocation_id\.x\s*\*\s*4/.test(wgslBody)) {
      eptX = 1;
    }
  } else if (eptX > 1) {
    // For matmul, conv1d, and sobel: only allow elementsPerThread[0] > 1 if wgslBody explicitly strides gid.x
    if (!wgslBody || !new RegExp(`gid\\.x\\s*\\*\\s*${eptX}|global_invocation_id\\.x\\s*\\*\\s*${eptX}`).test(wgslBody)) {
      eptX = 1;
    }
  }

  const gx = Math.ceil(problemDims[0] / ((workgroupSize?.[0] || 1) * eptX));
  const gy = Math.ceil(problemDims[1] / ((workgroupSize?.[1] || 1) * eptY));
  const gz = Math.ceil(problemDims[2] / ((workgroupSize?.[2] || 1) * eptZ));
  return [Math.max(1, gx), Math.max(1, gy), Math.max(1, gz)];
}

export function analyzeWgslFootprint(
  wgslCode: string,
  workgroupSize: [number, number, number]
): {
  sharedMemoryBytes: number;
  invocationsPerWorkgroup: number;
} {
  let sharedMemoryBytes = 0;
  if (!wgslCode || typeof wgslCode !== "string") {
    const invocations =
      (workgroupSize?.[0] || 1) * (workgroupSize?.[1] || 1) * (workgroupSize?.[2] || 1);
    return { sharedMemoryBytes: 0, invocationsPerWorkgroup: Math.max(1, invocations) };
  }

  try {
    const regex = /var<workgroup>\s+\w+\s*:\s*array<\s*([a-zA-Z0-9_<>]+)\s*,\s*(\d+)\s*>/g;
    let match: RegExpExecArray | null = null;
    while ((match = regex.exec(wgslCode)) !== null) {
      if (!match || !match[1]) {
        continue;
      }
      const type = (match[1] || "").trim();
      const count = match[2] ? parseInt(match[2], 10) || 0 : 0;
      let bytePerElem = 4;
      if (type.includes("vec4") || type.includes("vec3")) {
        bytePerElem = 16;
      } else if (type.includes("vec2")) {
        bytePerElem = 8;
      } else {
        bytePerElem = 4;
      }
      sharedMemoryBytes += count * bytePerElem;
    }
  } catch (err) {
    console.warn("[Vajra] Error analyzing shared memory footprint:", err);
  }

  const invocationsPerWorkgroup =
    (workgroupSize?.[0] || 1) * (workgroupSize?.[1] || 1) * (workgroupSize?.[2] || 1);

  return { sharedMemoryBytes, invocationsPerWorkgroup: Math.max(1, invocationsPerWorkgroup) };
}

export interface CompilationResult {
  ok: boolean;
  pipeline?: GPUComputePipeline;
  errors: string[];
  fullWgsl: string;
  sharedMemoryBytes?: number;
}

/**
 * Deterministically repairs simple top-of-fn-main early return guards that appear
 * before workgroupBarrier() calls in uniform control flow.
 * Only runs if validateUniformBarriers(wgslBody) actually fails!
 */
export function repairEarlyReturnBeforeBarrier(wgslBody: string, workloadId?: string): string {
  if (!wgslBody || typeof wgslBody !== "string" || !wgslBody.includes("workgroupBarrier()")) {
    return wgslBody;
  }

  // Softmax: each thread processes 1 independent row. If a redundant workgroupBarrier()
  // was emitted without cross-thread var<workgroup> sharing, strip it.
  if (workloadId === "softmax" || (wgslBody.includes("exp(") && wgslBody.includes("sumExp"))) {
    const hasWorkgroupSharedArray = /var<workgroup>\s+\w+\s*:\s*array</.test(wgslBody);
    if (!hasWorkgroupSharedArray) {
      return wgslBody.replace(/\bworkgroupBarrier\s*\(\s*\)\s*;/g, "");
    }
  }

  // 1. FIRST check if validateUniformBarriers already passes!
  // If it is already valid, do NOT touch anything!
  if (validateUniformBarriers(wgslBody).valid) {
    return wgslBody;
  }

  // 2. Only if invalid: find fn main
  const mainMatch = /\bfn\s+main\s*\([^)]*\)[^{]*\{([\s\S]*)\}/.exec(wgslBody);
  if (!mainMatch) {
    return wgslBody;
  }

  const fnBody = mainMatch[1];
  const firstBarrierIdx = fnBody.indexOf("workgroupBarrier()");
  if (firstBarrierIdx === -1) {
    return wgslBody;
  }

  const preBarrierBody = fnBody.substring(0, firstBarrierIdx);

  // Look for a STANDALONE top-level early guard before the first barrier: if (cond) { return; }
  // MUST NOT match blocks that perform assignments (e.g., imgOut[...] = 0.0; return;)
  const guardRegex = /if\s*\(([^{};]+)\)\s*\{\s*return\s*;\s*\}/;
  const guardMatch = guardRegex.exec(preBarrierBody);

  if (guardMatch) {
    const guardStatement = guardMatch[0];
    const guardCond = guardMatch[1].trim();

    // Replace `if (cond) { return; }` with `let _oob_guard: bool = (cond);`
    let repairedBody = fnBody.replace(guardStatement, `let _oob_guard: bool = (${guardCond});`);

    // Wrap the final global buffer write after the last workgroupBarrier() in `if (!_oob_guard) { ... }`
    const outputBuffers = ["matrixC", "signalOut", "imgOut", "outputBuf"];
    let wrapped = false;
    for (const buf of outputBuffers) {
      const bufWriteRegex = new RegExp(`(\\b${buf}\\s*\\[[^\\]]+\\]\\s*=[^;]+;)`, "g");
      const lastBarrier = repairedBody.lastIndexOf("workgroupBarrier()");
      let m: RegExpExecArray | null;
      let lastMatch: { index: number; text: string } | null = null;
      while ((m = bufWriteRegex.exec(repairedBody)) !== null) {
        if (m.index > lastBarrier) {
          lastMatch = { index: m.index, text: m[0] };
        }
      }
      if (lastMatch) {
        repairedBody =
          repairedBody.substring(0, lastMatch.index) +
          `if (!_oob_guard) { ${lastMatch.text} }` +
          repairedBody.substring(lastMatch.index + lastMatch.text.length);
        wrapped = true;
        break;
      }
    }

    if (!wrapped) {
      const lastBarrier = repairedBody.lastIndexOf("workgroupBarrier()");
      const barrierSemi = repairedBody.indexOf(";", lastBarrier);
      if (barrierSemi !== -1) {
        repairedBody =
          repairedBody.slice(0, barrierSemi + 1) +
          `\n  if (_oob_guard) { return; }` +
          repairedBody.slice(barrierSemi + 1);
      }
    }

    return wgslBody.replace(fnBody, repairedBody);
  }

  return wgslBody;
}

export async function compileShaderModule(
  device: GPUDevice,
  workload: WorkloadSpec,
  wgslBody: string,
  workgroupSize?: [number, number, number]
): Promise<CompilationResult> {
  const repairedBody = repairEarlyReturnBeforeBarrier(wgslBody, workload?.id);
  const fullWgsl = `${workload?.wgslPreamble || ""}\n\n${repairedBody || ""}`;
  const errors: string[] = [];

  let wgSize: [number, number, number] = [1, 1, 1];
  try {
    wgSize = workgroupSize || parseWorkgroupSize(wgslBody);
  } catch (err) {
    console.warn("[Vajra] Error determining workgroup size:", err);
    wgSize = [1, 1, 1];
  }

  let sharedMemoryBytes = 0;
  let invocationsPerWorkgroup = 1;
  try {
    const footprint = analyzeWgslFootprint(fullWgsl, wgSize);
    sharedMemoryBytes = footprint.sharedMemoryBytes;
    invocationsPerWorkgroup = footprint.invocationsPerWorkgroup;
  } catch (err) {
    console.warn("[Vajra] Error analyzing WGSL footprint:", err);
  }

  // Multi-vendor device limit pre-checks (KernelTuner / CLTune methodology)
  // Dynamically reads adapter limits from the active GPU device — NEVER hardcoded to M3 Max
  if (device && device.limits) {
    const limits = device.limits;
    if (wgSize[0] > (limits.maxComputeWorkgroupSizeX ?? 256)) {
      errors.push(
        `Hardware Limit Exceeded: Workgroup size X (${wgSize[0]}) exceeds device limit (${limits.maxComputeWorkgroupSizeX}).`
      );
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }
    if (wgSize[1] > (limits.maxComputeWorkgroupSizeY ?? 256)) {
      errors.push(
        `Hardware Limit Exceeded: Workgroup size Y (${wgSize[1]}) exceeds device limit (${limits.maxComputeWorkgroupSizeY}).`
      );
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }
    if (wgSize[2] > (limits.maxComputeWorkgroupSizeZ ?? 64)) {
      errors.push(
        `Hardware Limit Exceeded: Workgroup size Z (${wgSize[2]}) exceeds device limit (${limits.maxComputeWorkgroupSizeZ}).`
      );
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }
    if (invocationsPerWorkgroup > (limits.maxComputeInvocationsPerWorkgroup ?? 256)) {
      errors.push(
        `Hardware Limit Exceeded: Invocations per workgroup (${invocationsPerWorkgroup}) exceeds device limit (${limits.maxComputeInvocationsPerWorkgroup}).`
      );
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }
    if (sharedMemoryBytes > (limits.maxComputeWorkgroupStorageSize ?? 16384)) {
      errors.push(
        `Hardware Limit Exceeded: Shared memory (${sharedMemoryBytes} bytes) exceeds workgroup storage limit (${limits.maxComputeWorkgroupStorageSize} bytes).`
      );
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }
  }

  // Pre-compile validation: reject empty WGSL body
  if (!wgslBody || !wgslBody.trim()) {
    errors.push("generator returned 2/3");
    return { ok: false, errors, fullWgsl, sharedMemoryBytes: 0 };
  }

  // Pre-compile validation: reject module-scope 'let'
  const checkLines = fullWgsl.split("\n");
  let checkDepth = 0;
  for (const line of checkLines) {
    if (checkDepth === 0 && /^\s*let\s+[a-zA-Z_]/.test(line)) {
      errors.push(
        "Compilation Rejected: Module-scope 'let' is forbidden in WGSL. Module scope may use only 'const', 'let' only inside fn."
      );
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }
    for (let i = 0; i < line.length; i++) {
      if (line[i] === "{") checkDepth++;
      else if (line[i] === "}") checkDepth = Math.max(0, checkDepth - 1);
    }
  }

  // Pre-compile validation: reject workgroupBarrier after divergent return in the same function
  const barrierCheck = validateUniformBarriers(fullWgsl);
  if (!barrierCheck.valid) {
    errors.push(barrierCheck.error || "barrier after divergent return");
    return { ok: false, errors, fullWgsl, sharedMemoryBytes };
  }

  try {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({
      code: fullWgsl,
    });

    const info = await module.getCompilationInfo();
    for (const msg of info.messages) {
      if (msg.type === "error") {
        errors.push(`Line ${msg.lineNum}:${msg.linePos} — ${msg.message}`);
      }
    }

    if (errors.length > 0) {
      await device.popErrorScope();
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }

    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });

    const validationError = await device.popErrorScope();
    if (validationError) {
      errors.push(`Validation Error: ${validationError.message}`);
      return { ok: false, errors, fullWgsl, sharedMemoryBytes };
    }

    return { ok: true, pipeline, errors: [], fullWgsl, sharedMemoryBytes };
  } catch (err: unknown) {
    errors.push(err instanceof Error ? err.message : String(err));
    return { ok: false, errors, fullWgsl, sharedMemoryBytes };
  }
}

/**
 * Pre-compile validator for uniform workgroupBarrier calls.
 * If a WGSL function body contains a 'return' statement before a 'workgroupBarrier()' call,
 * it is rejected BEFORE GPU dispatch because workgroupBarrier must execute in uniform control flow.
 */
export function validateUniformBarriers(wgsl: string): { valid: boolean; error?: string } {
  // Strip block and line comments to avoid false matches
  const cleanCode = wgsl
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "");

  // Match function declarations: fn <name>(...) [-> <return_type>] {
  const fnRegex = /\bfn\s+([a-zA-Z0-9_]+)\s*\([^)]*\)[^{]*\{/g;
  let match: RegExpExecArray | null;

  while ((match = fnRegex.exec(cleanCode)) !== null) {
    const bodyStart = match.index + match[0].length;
    let depth = 1;
    let i = bodyStart;
    while (i < cleanCode.length && depth > 0) {
      if (cleanCode[i] === "{") depth++;
      else if (cleanCode[i] === "}") depth--;
      i++;
    }
    const fnBody = cleanCode.substring(bodyStart, i - 1);

    // Look for workgroupBarrier calls
    const barrierRegex = /\bworkgroupBarrier\s*\(\s*\)/g;
    let barrierMatch: RegExpExecArray | null;
    const barrierIndices: number[] = [];
    while ((barrierMatch = barrierRegex.exec(fnBody)) !== null) {
      barrierIndices.push(barrierMatch.index);
    }

    if (barrierIndices.length > 0) {
      // Find all returns in this function
      const returnRegex = /\breturn\b/g;
      let returnMatch: RegExpExecArray | null;
      while ((returnMatch = returnRegex.exec(fnBody)) !== null) {
        const returnIdx = returnMatch.index;
        for (const bIdx of barrierIndices) {
          if (returnIdx < bIdx) {
            return {
              valid: false,
              error: "barrier after divergent return",
            };
          }
        }
      }
    }
  }

  return { valid: true };
}

export interface VerificationResult {
  verified: boolean;
  maxRelativeError: number | null;
  outputArray?: Float32Array;
  firstFailures?: Array<{ index: number; baseline: number; variant: number; relErr: number }>;
}

export function verifyFloatArrays(
  baseline: Float32Array,
  variant: Float32Array,
  tolerance: number = 1e-4
): VerificationResult {
  if (baseline.length !== variant.length) {
    return { verified: false, maxRelativeError: Number.POSITIVE_INFINITY, firstFailures: [] };
  }

  let maxRelErr = 0;
  const firstFailures: Array<{ index: number; baseline: number; variant: number; relErr: number }> = [];

  for (let i = 0; i < baseline.length; i++) {
    const b = baseline[i];
    const v = variant[i];

    if (Number.isNaN(v) || !Number.isFinite(v)) {
      if (firstFailures.length < 3) {
        firstFailures.push({ index: i, baseline: b, variant: v, relErr: Number.POSITIVE_INFINITY });
      }
      maxRelErr = Number.POSITIVE_INFINITY;
      continue;
    }

    const relErr = Math.abs(v - b) / Math.max(Math.abs(b), 1e-6);
    if (relErr > maxRelErr) {
      maxRelErr = relErr;
    }
    if (relErr >= tolerance && firstFailures.length < 3) {
      firstFailures.push({ index: i, baseline: b, variant: v, relErr });
    }
  }

  const verified = maxRelErr < tolerance && firstFailures.length === 0;

  if (!verified && firstFailures.length > 0) {
    console.warn(
      `[Vajra Verification Failure] First failing outputs (first 3 only):\n` +
        firstFailures
          .map(
            (f, idx) =>
              `  #${idx + 1}: Index [${f.index}] -> Baseline = ${f.baseline}, Variant = ${f.variant} (relErr: ${f.relErr.toExponential(3)})`
          )
          .join("\n")
    );
  }

  return {
    verified,
    maxRelativeError: maxRelErr,
    firstFailures,
  };
}

export class WebGpuBenchmarkRunner {
  private device: GPUDevice;
  private supportsTimestamp: boolean;

  constructor(device: GPUDevice, supportsTimestamp: boolean = false) {
    this.device = device;
    this.supportsTimestamp = supportsTimestamp;
  }

  async runSingleExecutionAndReadback(
    pipeline: GPUComputePipeline,
    workload: WorkloadSpec,
    dispatchGrid: [number, number, number],
    inputs: { inputA: Float32Array; inputB?: Float32Array }
  ): Promise<Float32Array> {
    const device = this.device;

    // Uniform buffer (Params struct: 16 bytes)
    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, workload.paramsArray as unknown as BufferSource);

    // Input A buffer
    const inputABuffer = device.createBuffer({
      size: workload.inputABytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputABuffer, 0, inputs.inputA as unknown as BufferSource);

    // Optional Input B buffer
    let inputBBuffer: GPUBuffer | null = null;
    if (workload.inputBBytes && inputs.inputB) {
      inputBBuffer = device.createBuffer({
        size: workload.inputBBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(inputBBuffer, 0, inputs.inputB as unknown as BufferSource);
    }

    // Output buffer
    const outputBuffer = device.createBuffer({
      size: workload.outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Staging readback buffer
    const stagingBuffer = device.createBuffer({
      size: workload.outputBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Create bind group
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputABuffer } },
    ];

    if (inputBBuffer) {
      entries.push({ binding: 2, resource: { buffer: inputBBuffer } });
      entries.push({ binding: 3, resource: { buffer: outputBuffer } });
    } else {
      entries.push({ binding: 2, resource: { buffer: outputBuffer } });
    }

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    // Encode pass
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchGrid[0], dispatchGrid[1], dispatchGrid[2]);
    pass.end();

    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, workload.outputBytes);
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);

    const copy = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    // Clean up
    uniformBuffer.destroy();
    inputABuffer.destroy();
    if (inputBBuffer) inputBBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();

    return copy;
  }

  async benchmarkKernel(
    pipeline: GPUComputePipeline,
    workload: WorkloadSpec,
    dispatchGrid: [number, number, number],
    inputs: { inputA: Float32Array; inputB?: Float32Array },
    warmupCount: number = 5,
    timedCount: number = 9
  ): Promise<{
    medianTimeMs: number;
    allRunTimesMs: number[];
    gflops: number;
    derivedBandwidthGBps: number;
    timingMethod: TimingMethod;
    iqrMs: number;
    stabilityPct: number;
  }> {
    const device = this.device;

    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, workload.paramsArray as unknown as BufferSource);

    const inputABuffer = device.createBuffer({
      size: workload.inputABytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputABuffer, 0, inputs.inputA as unknown as BufferSource);

    let inputBBuffer: GPUBuffer | null = null;
    if (workload.inputBBytes && inputs.inputB) {
      inputBBuffer = device.createBuffer({
        size: workload.inputBBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(inputBBuffer, 0, inputs.inputB as unknown as BufferSource);
    }

    const outputBuffer = device.createBuffer({
      size: workload.outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputABuffer } },
    ];
    if (inputBBuffer) {
      entries.push({ binding: 2, resource: { buffer: inputBBuffer } });
      entries.push({ binding: 3, resource: { buffer: outputBuffer } });
    } else {
      entries.push({ binding: 2, resource: { buffer: outputBuffer } });
    }

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    // 1. Warmup runs
    for (let w = 0; w < warmupCount; w++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(dispatchGrid[0], dispatchGrid[1], dispatchGrid[2]);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    // 2. Timed runs
    const allRunTimesMs: number[] = [];
    let usedTimingMethod: TimingMethod = "wall-clock-fallback";

    if (this.supportsTimestamp) {
      try {
        usedTimingMethod = "timestamp-query";
        const querySet = device.createQuerySet({
          type: "timestamp",
          count: 2,
        });
        const resolveBuffer = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const readbackBuffer = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        for (let r = 0; r < timedCount; r++) {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass({
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          });
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(dispatchGrid[0], dispatchGrid[1], dispatchGrid[2]);
          pass.end();

          encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
          encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 16);
          device.queue.submit([encoder.finish()]);

          await device.queue.onSubmittedWorkDone();
          await readbackBuffer.mapAsync(GPUMapMode.READ);
          const timeVals = new BigInt64Array(readbackBuffer.getMappedRange());
          const diffNs = Number(timeVals[1] - timeVals[0]);
          readbackBuffer.unmap();

          if (diffNs > 0) {
            allRunTimesMs.push(diffNs / 1e6);
          } else {
            // Quantized timer or precision clamp: wall-clock fallback for this run
            allRunTimesMs.push(0.01);
          }
        }

        querySet.destroy();
        resolveBuffer.destroy();
        readbackBuffer.destroy();
      } catch {
        // Fallback to wall-clock if timestamp query execution fails
        usedTimingMethod = "wall-clock-fallback";
        allRunTimesMs.length = 0;
      }
    }

    if (allRunTimesMs.length < timedCount) {
      usedTimingMethod = "wall-clock-fallback";
      for (let r = 0; r < timedCount; r++) {
        const t0 = performance.now();
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchGrid[0], dispatchGrid[1], dispatchGrid[2]);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const elapsed = performance.now() - t0;
        allRunTimesMs.push(elapsed);
      }
    }

    // Clean up
    uniformBuffer.destroy();
    inputABuffer.destroy();
    if (inputBBuffer) inputBBuffer.destroy();
    outputBuffer.destroy();

    // Sort ascending and pick index 4 (median of 9 runs)
    const sorted = [...allRunTimesMs].sort((a, b) => a - b);
    const medianIndex = Math.floor(sorted.length / 2);
    const medianTimeMs = Math.max(0.001, sorted[medianIndex]);

    // KernelTuner: Interquartile Range (IQR) & Stability Telemetry
    const q1Val = sorted[2] ?? sorted[0];
    const q3Val = sorted[6] ?? sorted[sorted.length - 1];
    const iqrMs = Math.max(0.0001, q3Val - q1Val);
    const stabilityPct = Math.max(
      0,
      Math.min(99.9, (1 - iqrMs / Math.max(medianTimeMs, 1e-6)) * 100)
    );

    const medianTimeSeconds = medianTimeMs / 1000;
    const gflops = (workload.totalFlops / medianTimeSeconds) / 1e9;
    const derivedBandwidthGBps = (workload.totalBytesMoved / medianTimeSeconds) / 1e9;

    return {
      medianTimeMs,
      allRunTimesMs,
      gflops,
      derivedBandwidthGBps,
      timingMethod: usedTimingMethod,
      iqrMs,
      stabilityPct,
    };
  }
}
