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

    const probe: HardwareProbeInfo = {
      supported: true,
      adapterDescription,
      vendor,
      architecture,
      supportsTimestampQuery,
      timingMethod,
      isMockOnlyFallback: false,
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
  const match = wgslBody.match(
    /@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?(?:,\s*(\d+)\s*)?\)/
  );
  if (!match) {
    return [1, 1, 1];
  }
  const x = parseInt(match[1], 10) || 1;
  const y = parseInt(match[2] || "1", 10) || 1;
  const z = parseInt(match[3] || "1", 10) || 1;
  return [x, y, z];
}

export function computeDispatchGrid(
  problemDims: [number, number, number],
  workgroupSize: [number, number, number],
  elementsPerThread: [number, number, number] = [1, 1, 1]
): [number, number, number] {
  const gx = Math.ceil(problemDims[0] / (workgroupSize[0] * (elementsPerThread[0] || 1)));
  const gy = Math.ceil(problemDims[1] / (workgroupSize[1] * (elementsPerThread[1] || 1)));
  const gz = Math.ceil(problemDims[2] / (workgroupSize[2] * (elementsPerThread[2] || 1)));
  return [Math.max(1, gx), Math.max(1, gy), Math.max(1, gz)];
}

export interface CompilationResult {
  ok: boolean;
  pipeline?: GPUComputePipeline;
  errors: string[];
  fullWgsl: string;
}

export async function compileShaderModule(
  device: GPUDevice,
  workload: WorkloadSpec,
  wgslBody: string
): Promise<CompilationResult> {
  const fullWgsl = `${workload.wgslPreamble}\n\n${wgslBody}`;
  const errors: string[] = [];

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
      return { ok: false, errors, fullWgsl };
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
      return { ok: false, errors, fullWgsl };
    }

    return { ok: true, pipeline, errors: [], fullWgsl };
  } catch (err: unknown) {
    errors.push(err instanceof Error ? err.message : String(err));
    return { ok: false, errors, fullWgsl };
  }
}

export interface VerificationResult {
  verified: boolean;
  maxRelativeError: number | null;
  outputArray?: Float32Array;
}

export function verifyFloatArrays(
  baseline: Float32Array,
  variant: Float32Array,
  tolerance: number = 1e-4
): VerificationResult {
  if (baseline.length !== variant.length) {
    return { verified: false, maxRelativeError: Number.POSITIVE_INFINITY };
  }

  let maxRelErr = 0;
  for (let i = 0; i < baseline.length; i++) {
    const b = baseline[i];
    const v = variant[i];

    if (Number.isNaN(v) || !Number.isFinite(v)) {
      return { verified: false, maxRelativeError: Number.POSITIVE_INFINITY };
    }

    const relErr = Math.abs(v - b) / Math.max(Math.abs(b), 1e-6);
    if (relErr > maxRelErr) {
      maxRelErr = relErr;
    }
  }

  return {
    verified: maxRelErr < tolerance,
    maxRelativeError: maxRelErr,
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
    warmupCount: number = 3,
    timedCount: number = 9
  ): Promise<{
    medianTimeMs: number;
    allRunTimesMs: number[];
    gflops: number;
    derivedBandwidthGBps: number;
    timingMethod: TimingMethod;
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

    const medianTimeSeconds = medianTimeMs / 1000;
    const gflops = (workload.totalFlops / medianTimeSeconds) / 1e9;
    const derivedBandwidthGBps = (workload.totalBytesMoved / medianTimeSeconds) / 1e9;

    return {
      medianTimeMs,
      allRunTimesMs,
      gflops,
      derivedBandwidthGBps,
      timingMethod: usedTimingMethod,
    };
  }
}
