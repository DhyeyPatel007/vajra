"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  HardwareProbeInfo,
  TimingMethod,
  VariantExecutionResult,
  WorkloadId,
  GenerateResponse,
  GenerateErrorResponse,
} from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";
import { RECORDED_BENCHMARKS } from "@/lib/recordedBenchmarks";
import {
  probeAndInitWebGpu,
  compileShaderModule,
  verifyFloatArrays,
  WebGpuBenchmarkRunner,
  computeDispatchGrid,
  WebGpuContext,
} from "@/lib/webgpuHarness";
import { TopBar } from "@/components/TopBar";
import { WorkloadSelector } from "@/components/WorkloadSelector";
import { GflopsChart } from "@/components/GflopsChart";
import { SiliconReportCard } from "@/components/SiliconReportCard";
import { VariantCardGrid } from "@/components/VariantCardGrid";
import { WgslInspector } from "@/components/WgslInspector";
import { AlertTriangle, RotateCcw, ShieldAlert, Cpu } from "lucide-react";

export default function VajraDashboard() {
  // Environment & Hardware state
  const [probe, setProbe] = useState<HardwareProbeInfo>({
    supported: false,
    adapterDescription: "Probing Silicon...",
    vendor: "Generic",
    architecture: "Unknown",
    supportsTimestampQuery: false,
    timingMethod: "recorded-reference",
    isMockOnlyFallback: true,
  });

  const webGpuCtxRef = useRef<WebGpuContext | null>(null);

  // Settings & Toggles
  const [selectedWorkloadId, setSelectedWorkloadId] = useState<WorkloadId>("matmul");
  const [isMockMode, setIsMockMode] = useState<boolean>(true);
  const [simulateCompileError, setSimulateCompileError] = useState<boolean>(false);
  const [previewNoWebGpu, setPreviewNoWebGpu] = useState<boolean>(false);

  // Pipeline execution state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [apiError, setApiError] = useState<GenerateErrorResponse | null>(null);

  // Results state
  const [baselineResult, setBaselineResult] = useState<VariantExecutionResult | null>(null);
  const [variantResults, setVariantResults] = useState<VariantExecutionResult[]>([]);
  const [winnerResult, setWinnerResult] = useState<VariantExecutionResult | null>(null);
  const [selectedInspectId, setSelectedInspectId] = useState<string>("baseline");

  // Metadata
  const [activeModelSource, setActiveModelSource] = useState<"mock" | "nebius-super" | "nebius-ultra-fallback">("mock");
  const [activeModelId, setActiveModelId] = useState<string>("mock-nemotron-local");
  const [usedFallbackModel, setUsedFallbackModel] = useState<boolean>(false);

  // Helper to load recorded fallback benchmarks
  const loadRecordedBenchmark = useCallback((workloadId: WorkloadId) => {
    const recorded = RECORDED_BENCHMARKS[workloadId];
    if (recorded) {
      setBaselineResult(recorded.baseline);
      setVariantResults(recorded.variants);
      const winner = recorded.variants.find((v) => v.isWinner) || recorded.baseline;
      setWinnerResult(winner);
      setSelectedInspectId(winner.id);
      setActiveModelSource("mock");
      setActiveModelId("Apple M3 Max Reference (Chrome 142)");
      setUsedFallbackModel(false);
      setApiError(null);
    }
  }, []);

  // Main closed-loop WebGPU benchmark pipeline
  const runPipeline = useCallback(
    async (workloadId: WorkloadId, mock: boolean, simError: boolean) => {
      // If WebGPU is not supported or user requested fallback preview
      if (!probe.supported || previewNoWebGpu || !webGpuCtxRef.current) {
        loadRecordedBenchmark(workloadId);
        return;
      }

      const ctx = webGpuCtxRef.current;
      const device = ctx.device;
      const workload = WORKLOADS[workloadId];
      const runner = new WebGpuBenchmarkRunner(device, probe.supportsTimestampQuery);

      setIsRunning(true);
      setApiError(null);
      setStatusMessage("Allocating buffers & inputs...");

      try {
        // Step 1: Generate deterministic inputs
        const inputs = workload.generateInputs();

        // Step 2: Compile & Benchmark Naive Baseline
        setStatusMessage("Compiling Naive Baseline...");
        const baselineCompilation = await compileShaderModule(
          device,
          workload,
          workload.baselineWgslBody
        );

        if (!baselineCompilation.ok || !baselineCompilation.pipeline) {
          throw new Error(
            `Baseline compilation failed: ${baselineCompilation.errors.join("; ")}`
          );
        }

        const baselineDispatchGrid = computeDispatchGrid(
          workload.problemDims,
          workload.baselineWorkgroupSize
        );

        setStatusMessage("Benchmarking Baseline (3 warmup + 9 timed)...");
        const baselineBench = await runner.benchmarkKernel(
          baselineCompilation.pipeline,
          workload,
          baselineDispatchGrid,
          inputs
        );

        // Read baseline output for float verification
        setStatusMessage("Reading baseline float reference...");
        const baselineOutput = await runner.runSingleExecutionAndReadback(
          baselineCompilation.pipeline,
          workload,
          baselineDispatchGrid,
          inputs
        );

        const initialBaselineResult: VariantExecutionResult = {
          id: "baseline",
          name: "Naive Baseline",
          strategy: "baseline",
          rationale: "Uncoalesced baseline reading directly from global DRAM on each instruction.",
          status: "IDLE",
          workgroupSize: workload.baselineWorkgroupSize,
          dispatchGrid: baselineDispatchGrid,
          compileErrors: [],
          verified: true,
          maxRelativeError: 0,
          medianTimeMs: baselineBench.medianTimeMs,
          allRunTimesMs: baselineBench.allRunTimesMs,
          gflops: baselineBench.gflops,
          derivedBandwidthGBps: baselineBench.derivedBandwidthGBps,
          speedupVsBaseline: 1.0,
          fullWgslCode: baselineCompilation.fullWgsl,
          isWinner: true, // Default winner until a verified variant beats it
        };

        setBaselineResult(initialBaselineResult);
        setWinnerResult(initialBaselineResult);
        setSelectedInspectId("baseline");

        // Step 3: Fetch 3 Variants from /api/generate
        setStatusMessage(
          mock
            ? "Loading Nemotron Verified Mock Kernels..."
            : "Querying NVIDIA Nemotron via Nebius Token Factory..."
        );

        // Pre-populate cards in GENERATING state
        setVariantResults([
          {
            id: "variant-a",
            name: "Variant A: Workgroup Geometry",
            strategy: "workgroup_tuning",
            rationale: "Tuning workgroup dimensions and unrolling loops for coalesced global access.",
            status: "GENERATING",
            workgroupSize: [16, 16, 1],
            dispatchGrid: [1, 1, 1],
            compileErrors: [],
            verified: false,
            maxRelativeError: null,
            medianTimeMs: null,
            allRunTimesMs: [],
            gflops: null,
            derivedBandwidthGBps: null,
            speedupVsBaseline: null,
            fullWgslCode: "",
            isWinner: false,
          },
          {
            id: "variant-b",
            name: "Variant B: Shared Memory Tiling",
            strategy: "shared_memory_tiling",
            rationale: "Allocating var<workgroup> tiles and barrier sync to minimize DRAM traffic.",
            status: "GENERATING",
            workgroupSize: [16, 16, 1],
            dispatchGrid: [1, 1, 1],
            compileErrors: [],
            verified: false,
            maxRelativeError: null,
            medianTimeMs: null,
            allRunTimesMs: [],
            gflops: null,
            derivedBandwidthGBps: null,
            speedupVsBaseline: null,
            fullWgslCode: "",
            isWinner: false,
          },
          {
            id: "variant-c",
            name: "Variant C: Vec4 Vectorized",
            strategy: "vec4_vectorized",
            rationale: "Combining shared memory caching with SIMD vector arithmetic.",
            status: "GENERATING",
            workgroupSize: [16, 16, 1],
            dispatchGrid: [1, 1, 1],
            compileErrors: [],
            verified: false,
            maxRelativeError: null,
            medianTimeMs: null,
            allRunTimesMs: [],
            gflops: null,
            derivedBandwidthGBps: null,
            speedupVsBaseline: null,
            fullWgslCode: "",
            isWinner: false,
          },
        ]);

        const genRes = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            workloadId,
            mockMode: mock,
            simulateCompileError: simError,
          }),
        });

        if (!genRes.ok) {
          const errData = (await genRes.json().catch(() => ({}))) as GenerateErrorResponse;
          setApiError({
            error: errData.error || `HTTP ${genRes.status} Error from /api/generate`,
            code: errData.code || "UPSTREAM_SUPER_AND_ULTRA_FAILED",
            retryable: true,
            details: errData.details,
          });
          setIsRunning(false);
          setStatusMessage("");
          return;
        }

        const genData = (await genRes.json()) as GenerateResponse;
        setActiveModelSource(genData.source);
        setActiveModelId(genData.modelId);
        setUsedFallbackModel(genData.usedFallbackModel);

        const currentVariants: VariantExecutionResult[] = [];

        // Step 4: Process Each Variant (Compile -> Verify -> Benchmark)
        for (let i = 0; i < genData.variants.length; i++) {
          const rawVariant = genData.variants[i];
          const variantId = rawVariant.id;

          setStatusMessage(`Compiling ${rawVariant.name}...`);

          // Update card state: COMPILING
          setVariantResults((prev) =>
            prev.map((v) =>
              v.id === variantId ? { ...v, status: "COMPILING" } : v
            )
          );

          const compilation = await compileShaderModule(
            device,
            workload,
            rawVariant.wgslBody
          );

          if (!compilation.ok || !compilation.pipeline) {
            // Drop compile failures cleanly!
            const failedResult: VariantExecutionResult = {
              id: variantId,
              name: rawVariant.name,
              strategy: rawVariant.strategy,
              rationale: rawVariant.rationale,
              status: "COMPILE_FAILED",
              workgroupSize: rawVariant.workgroupSize,
              dispatchGrid: [1, 1, 1],
              compileErrors: compilation.errors,
              verified: false,
              maxRelativeError: null,
              medianTimeMs: null,
              allRunTimesMs: [],
              gflops: null,
              derivedBandwidthGBps: null,
              speedupVsBaseline: null,
              fullWgslCode: compilation.fullWgsl,
              isWinner: false,
            };
            currentVariants.push(failedResult);
            setVariantResults((prev) =>
              prev.map((v) => (v.id === variantId ? failedResult : v))
            );
            continue; // Proceed to next variant without crashing
          }

          // Compilation passed: compute dispatch grid
          const dispatchGrid = computeDispatchGrid(
            workload.problemDims,
            rawVariant.workgroupSize,
            rawVariant.elementsPerThread
          );

          // Step 5: Numerical Tolerance Verification
          setStatusMessage(`Verifying floats for ${rawVariant.name}...`);
          setVariantResults((prev) =>
            prev.map((v) =>
              v.id === variantId ? { ...v, status: "VERIFYING" } : v
            )
          );

          const variantOutput = await runner.runSingleExecutionAndReadback(
            compilation.pipeline,
            workload,
            dispatchGrid,
            inputs
          );

          const verification = verifyFloatArrays(baselineOutput, variantOutput, 1e-4);

          if (!verification.verified) {
            // Drop verification failures cleanly!
            const verifyFailedResult: VariantExecutionResult = {
              id: variantId,
              name: rawVariant.name,
              strategy: rawVariant.strategy,
              rationale: rawVariant.rationale,
              status: "VERIFY_FAILED",
              workgroupSize: rawVariant.workgroupSize,
              dispatchGrid,
              compileErrors: [],
              verified: false,
              maxRelativeError: verification.maxRelativeError,
              medianTimeMs: null,
              allRunTimesMs: [],
              gflops: null,
              derivedBandwidthGBps: null,
              speedupVsBaseline: null,
              fullWgslCode: compilation.fullWgsl,
              isWinner: false,
            };
            currentVariants.push(verifyFailedResult);
            setVariantResults((prev) =>
              prev.map((v) => (v.id === variantId ? verifyFailedResult : v))
            );
            continue;
          }

          // Step 6: Hardware Benchmarking (3 warmup + 9 timed runs)
          setStatusMessage(`Benchmarking ${rawVariant.name} (9 timed runs)...`);
          setVariantResults((prev) =>
            prev.map((v) =>
              v.id === variantId ? { ...v, status: "BENCHMARKING" } : v
            )
          );

          const bench = await runner.benchmarkKernel(
            compilation.pipeline,
            workload,
            dispatchGrid,
            inputs
          );

          const speedup =
            initialBaselineResult.medianTimeMs && bench.medianTimeMs
              ? initialBaselineResult.medianTimeMs / bench.medianTimeMs
              : 1.0;

          const verifiedResult: VariantExecutionResult = {
            id: variantId,
            name: rawVariant.name,
            strategy: rawVariant.strategy,
            rationale: rawVariant.rationale,
            status: "VERIFIED_RUNNER_UP",
            workgroupSize: rawVariant.workgroupSize,
            dispatchGrid,
            compileErrors: [],
            verified: true,
            maxRelativeError: verification.maxRelativeError,
            medianTimeMs: bench.medianTimeMs,
            allRunTimesMs: bench.allRunTimesMs,
            gflops: bench.gflops,
            derivedBandwidthGBps: bench.derivedBandwidthGBps,
            speedupVsBaseline: speedup,
            fullWgslCode: compilation.fullWgsl,
            isWinner: false,
          };

          currentVariants.push(verifiedResult);
          setVariantResults((prev) =>
            prev.map((v) => (v.id === variantId ? verifiedResult : v))
          );
        }

        // Step 7: Winner Selection
        setStatusMessage("Crowning Fastest Kernel...");
        const verifiedCandidates = currentVariants.filter(
          (v) => v.verified && v.medianTimeMs !== null
        );

        let champion: VariantExecutionResult = initialBaselineResult;

        if (verifiedCandidates.length > 0) {
          // Sort by medianTimeMs ascending (lowest execution time wins)
          verifiedCandidates.sort((a, b) => (a.medianTimeMs || 0) - (b.medianTimeMs || 0));
          const bestVariant = verifiedCandidates[0];

          if (
            initialBaselineResult.medianTimeMs &&
            bestVariant.medianTimeMs &&
            bestVariant.medianTimeMs < initialBaselineResult.medianTimeMs
          ) {
            champion = bestVariant;
          }
        }

        // Update champion flags
        const finalVariants = currentVariants.map((v) => ({
          ...v,
          isWinner: v.id === champion.id,
          status:
            v.id === champion.id
              ? ("VERIFIED_WINNER" as const)
              : v.status === "COMPILE_FAILED" || v.status === "VERIFY_FAILED"
              ? v.status
              : ("VERIFIED_RUNNER_UP" as const),
        }));

        setVariantResults(finalVariants);
        setWinnerResult({ ...champion, isWinner: true });
        setSelectedInspectId(champion.id);
        setStatusMessage("");
      } catch (err: unknown) {
        console.error("[Vajra] Pipeline error:", err);
        setApiError({
          error: err instanceof Error ? err.message : String(err),
          code: "PARSE_ERROR",
          retryable: true,
        });
        setStatusMessage("");
      } finally {
        setIsRunning(false);
      }
    },
    [probe, previewNoWebGpu, loadRecordedBenchmark]
  );

  // Initialize WebGPU on mount
  useEffect(() => {
    let isMounted = true;

    async function init() {
      const probeRes = await probeAndInitWebGpu();
      if (!isMounted) return;

      setProbe(probeRes);
      if (probeRes.context) {
        webGpuCtxRef.current = probeRes.context;
      }

      // If WebGPU is supported, run the default matmul workload automatically on first paint!
      if (probeRes.supported && probeRes.context) {
        runPipeline("matmul", true, false);
      } else {
        // Fall back to recorded reference benchmark
        loadRecordedBenchmark("matmul");
      }
    }

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  // Handle Workload Change
  const handleSelectWorkload = (id: WorkloadId) => {
    setSelectedWorkloadId(id);
    if (!probe.supported || previewNoWebGpu) {
      loadRecordedBenchmark(id);
    } else {
      runPipeline(id, isMockMode, simulateCompileError);
    }
  };

  // Handle Tune Action
  const handleTune = () => {
    runPipeline(selectedWorkloadId, isMockMode, simulateCompileError);
  };

  // Handle Preview No-WebGPU toggle
  const handleTogglePreviewNoWebGpu = (preview: boolean) => {
    setPreviewNoWebGpu(preview);
    if (preview) {
      loadRecordedBenchmark(selectedWorkloadId);
    } else if (probe.supported && webGpuCtxRef.current) {
      runPipeline(selectedWorkloadId, isMockMode, simulateCompileError);
    }
  };

  const activeWorkloadSpec = WORKLOADS[selectedWorkloadId];

  return (
    <div className="min-h-screen bg-[#0B0C0E] text-[#F3F4F6]">
      {/* 1. Header & Navigation */}
      <TopBar
        probe={probe}
        isMockMode={isMockMode}
        onToggleMockMode={(mock) => {
          setIsMockMode(mock);
          runPipeline(selectedWorkloadId, mock, simulateCompileError);
        }}
        simulateCompileError={simulateCompileError}
        onToggleSimulateCompileError={(val) => {
          setSimulateCompileError(val);
          if (isMockMode) {
            runPipeline(selectedWorkloadId, isMockMode, val);
          }
        }}
        previewNoWebGpu={previewNoWebGpu}
        onTogglePreviewNoWebGpu={handleTogglePreviewNoWebGpu}
        liveModelSource={activeModelSource}
        usedFallbackModel={usedFallbackModel}
      />

      {/* 2. Workload Selector Bar */}
      <WorkloadSelector
        selectedId={selectedWorkloadId}
        onSelect={handleSelectWorkload}
        onTune={handleTune}
        isRunning={isRunning}
        statusLabel={statusMessage}
      />

      {/* 3. API Error Banner (if Live Model or Super/Ultra fails) */}
      {apiError && (
        <div className="mx-auto max-w-7xl px-4 pt-4 sm:px-6">
          <div className="rounded border border-[#EF4444] bg-[#181B22] p-4 text-xs text-[#F3F4F6]">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <ShieldAlert className="h-5 w-5 shrink-0 text-[#EF4444]" />
                <div>
                  <div className="font-bold text-[#EF4444]">
                    Generation Failed: {apiError.error}
                  </div>
                  {apiError.details && (
                    <div className="mt-1 font-mono text-[11px] text-[#9CA3AF]">
                      {apiError.details}
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => runPipeline(selectedWorkloadId, false, false)}
                  className="rounded border border-[#EF4444] bg-[#EF4444] px-3 py-1 text-xs font-bold text-[#F3F4F6] hover:bg-opacity-90"
                >
                  Retry Live Generation
                </button>
                <button
                  onClick={() => {
                    setIsMockMode(true);
                    runPipeline(selectedWorkloadId, true, false);
                  }}
                  className="rounded border border-[#262A34] bg-[#121418] px-3 py-1 text-xs font-semibold text-[#D4AF37] hover:border-[#D4AF37]"
                >
                  Fall Back to Mock Kernels
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 4. Hero Section: Live GFLOPS Chart + Silicon Report Card */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 space-y-6">
        <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Left: 7 Columns GFLOPS Chart */}
          <div className="lg:col-span-7">
            <GflopsChart
              baseline={baselineResult}
              variants={variantResults}
              workloadName={activeWorkloadSpec.name}
            />
          </div>

          {/* Right: 5 Columns Silicon Report Card */}
          <div className="lg:col-span-5">
            <SiliconReportCard
              winner={winnerResult}
              baseline={baselineResult}
              timingMethod={
                previewNoWebGpu || !probe.supported
                  ? "recorded-reference"
                  : probe.timingMethod
              }
              modelId={activeModelId}
              isMock={isMockMode}
              adapterName={
                previewNoWebGpu || !probe.supported
                  ? "Apple M3 Max (Reference)"
                  : probe.adapterDescription
              }
            />
          </div>
        </section>

        {/* 5. Pipeline Variant Status Cards Grid */}
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-[#9CA3AF]">
              Pipeline Execution & Numerical Verification (4 Variants)
            </h2>
            <span className="font-mono text-xs text-[#D4AF37]">
              Tolerance: maxRelativeError &lt; 1e-4
            </span>
          </div>

          <VariantCardGrid
            baseline={baselineResult}
            variants={variantResults}
            selectedInspectId={selectedInspectId}
            onSelectVariantForInspect={(variant) => setSelectedInspectId(variant.id)}
          />
        </section>

        {/* 6. Bottom Panel: Full WGSL Code Inspector */}
        <section>
          <WgslInspector
            variants={variantResults}
            baseline={baselineResult}
            winner={winnerResult}
            selectedVariantId={selectedInspectId}
            onSelectTab={(id) => setSelectedInspectId(id)}
          />
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#262A34] bg-[#07080A] py-6 text-center text-xs text-[#9CA3AF]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-2">
            <span className="font-bold text-[#D4AF37]">VAJRA</span>
            <span>—</span>
            <span>Zero-Install Browser-Native WebGPU Auto-Tuner</span>
          </div>
          <div>
            Built with NVIDIA Nemotron via Nebius Token Factory for the Global AI Hackathon.
          </div>
        </div>
      </footer>
    </div>
  );
}
