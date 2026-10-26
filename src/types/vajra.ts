/// <reference types="@webgpu/types" />

export type WorkloadId = "matmul" | "softmax" | "conv1d" | "sobel";

export type VariantStrategy =
  | "baseline"
  | "workgroup_tuning"
  | "shared_memory_tiling"
  | "vec4_vectorized";

export type VariantStatus =
  | "IDLE"
  | "GENERATING"
  | "COMPILING"
  | "COMPILE_FAILED"
  | "VERIFYING"
  | "VERIFY_FAILED"
  | "BENCHMARKING"
  | "VERIFIED_WINNER"
  | "VERIFIED_RUNNER_UP";

export type TimingMethod =
  | "timestamp-query"
  | "wall-clock-fallback"
  | "recorded-reference";

export interface WorkloadSpec {
  id: WorkloadId;
  name: string;
  shortName: string;
  dimensionsLabel: string;
  description: string;
  problemDims: [number, number, number]; // [width, height, depth]
  totalFlops: number;
  totalBytesMoved: number;
  paramsArray: Uint32Array; // 4 u32 uniform values passed to binding 0
  inputABytes: number;
  inputBBytes?: number; // for matmul (matrix B) and conv1d (filter)
  outputBytes: number;
  wgslPreamble: string; // Fixed @group(0) @binding(...) declarations + struct Params
  baselineWgslBody: string;
  baselineWorkgroupSize: [number, number, number];
  generateInputs: () => {
    inputA: Float32Array;
    inputB?: Float32Array;
  };
}

export interface GeneratedVariant {
  id: "variant-a" | "variant-b" | "variant-c";
  name: string;
  strategy: VariantStrategy;
  rationale: string;
  workgroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
  wgslBody: string;
}

export interface VariantExecutionResult {
  id: "baseline" | "variant-a" | "variant-b" | "variant-c";
  name: string;
  strategy: VariantStrategy;
  rationale: string;
  status: VariantStatus;
  workgroupSize: [number, number, number];
  dispatchGrid: [number, number, number];
  compileErrors: string[];
  verified: boolean;
  maxRelativeError: number | null;
  medianTimeMs: number | null;
  allRunTimesMs: number[];
  gflops: number | null;
  derivedBandwidthGBps: number | null;
  speedupVsBaseline: number | null;
  fullWgslCode: string;
  isWinner: boolean;
}

export interface GenerateRequest {
  workloadId: WorkloadId;
  mockMode?: boolean;
  simulateCompileError?: boolean;
}

export interface GenerateResponse {
  workloadId: WorkloadId;
  source: "mock" | "nebius-super" | "nebius-ultra-fallback";
  modelId: string;
  usedFallbackModel: boolean;
  latencyMs: number;
  variants: GeneratedVariant[];
}

export interface GenerateErrorResponse {
  error: string;
  code:
    | "INVALID_WORKLOAD"
    | "MISSING_API_KEY"
    | "UPSTREAM_SUPER_AND_ULTRA_FAILED"
    | "PARSE_ERROR";
  retryable: boolean;
  details?: string;
}

export interface HardwareProbeInfo {
  supported: boolean;
  adapterDescription: string;
  vendor: string;
  architecture: string;
  supportsTimestampQuery: boolean;
  timingMethod: TimingMethod;
  isMockOnlyFallback: boolean;
}

export interface RecordedBenchmarkEntry {
  workloadId: WorkloadId;
  workloadName: string;
  hardwareDescription: string;
  timingMethod: TimingMethod;
  baseline: VariantExecutionResult;
  variants: VariantExecutionResult[];
}
