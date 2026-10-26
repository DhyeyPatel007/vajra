"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  HardwareProbeInfo,
  VariantExecutionResult,
  WorkloadId,
  GenerateResponse,
  GenerateErrorResponse,
} from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";
import { RECORDED_BENCHMARKS, getGen2RecordedResult } from "@/lib/recordedBenchmarks";
import {
  probeAndInitWebGpu,
  compileShaderModule,
  verifyFloatArrays,
  WebGpuBenchmarkRunner,
  computeDispatchGrid,
  analyzeWgslFootprint,
  WebGpuContext,
} from "@/lib/webgpuHarness";
import { TopBar } from "@/components/TopBar";
import { WorkloadSelector } from "@/components/WorkloadSelector";
import { GflopsChart } from "@/components/GflopsChart";
import { SiliconReportCard } from "@/components/SiliconReportCard";
import { VariantCardGrid } from "@/components/VariantCardGrid";
import { WgslInspector } from "@/components/WgslInspector";
import { FailureGraveyard } from "@/components/FailureGraveyard";
import { ShieldAlert } from "lucide-react";

export default function VajraDashboard() {
  // Environment & Hardware probe
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

  // Helper to load recorded fallback benchmarks statically or with simulation
  const loadRecordedBenchmark = useCallback(
    (workloadId: WorkloadId, simError: boolean = false) => {
      const recorded = RECORDED_BENCHMARKS[workloadId];
      if (!recorded) return;

      const base = { ...recorded.baseline };
      const rawVariants = recorded.variants.map((v) => ({ ...v }));

      if (simError) {
        // Variant C fails compilation in simulation
        const updatedVariants = rawVariants.map((v) => {
          if (v.id === "variant-c") {
            return {
              ...v,
              status: "COMPILE_FAILED" as const,
              verified: false,
              isWinner: false,
              gflops: null,
              medianTimeMs: null,
              derivedBandwidthGBps: null,
              speedupVsBaseline: null,
              compileErrors: [
                "Line 18:14 — error: unresolved call target 'unknown_intrinsic_fn(u32)'",
              ],
              fullWgslCode: `${v.fullWgslCode}\n\n// INJECTED FAULT FOR DEMO:\nlet broken: vec4<f32> = unknown_intrinsic_fn(gid.x);`,
            };
          }
          if (v.id === "variant-b") {
            return {
              ...v,
              status: "VERIFIED_WINNER" as const,
              isWinner: true,
            };
          }
          return {
            ...v,
            isWinner: false,
          };
        });

        setBaselineResult(base);
        setVariantResults(updatedVariants);
        const champion = updatedVariants.find((v) => v.id === "variant-b") || base;
        setWinnerResult(champion);
        setSelectedInspectId(champion.id);
      } else {
        setBaselineResult(base);
        setVariantResults(rawVariants);
        const champion = rawVariants.find((v) => v.isWinner) || base;
        setWinnerResult(champion);
        setSelectedInspectId(champion.id);
      }

      setActiveModelSource("mock");
      setActiveModelId("Apple M3 Max Reference (Chrome 142)");
      setUsedFallbackModel(false);
      setApiError(null);
    },
    []
  );

  // Replay recorded benchmark with visual state machine animations
  const replayRecordedPipeline = useCallback(
    async (workloadId: WorkloadId, simError: boolean) => {
      const recorded = RECORDED_BENCHMARKS[workloadId];
      if (!recorded) return;

      setIsRunning(true);
      setApiError(null);

      // Baseline setup
      setBaselineResult(recorded.baseline);
      setWinnerResult(recorded.baseline);
      setSelectedInspectId("baseline");

      // Stage 1: GENERATING
      setStatusMessage("GENERATING VARIANTS VIA RECORDED SPEC...");
      setVariantResults(
        recorded.variants.map((v) => ({
          ...v,
          status: "GENERATING",
          gflops: null,
          medianTimeMs: null,
          speedupVsBaseline: null,
          isWinner: false,
        }))
      );
      await new Promise((r) => setTimeout(r, 120));

      // Stage 2: COMPILING
      setStatusMessage("VALIDATING WGSL COMPILATION...");
      if (simError) {
        setVariantResults((prev) =>
          prev.map((v) => {
            if (v.id === "variant-c") {
              return {
                ...v,
                status: "COMPILE_FAILED",
                verified: false,
                isWinner: false,
                gflops: null,
                medianTimeMs: null,
                derivedBandwidthGBps: null,
                speedupVsBaseline: null,
                compileErrors: [
                  "Line 18:14 — error: unresolved call target 'unknown_intrinsic_fn(u32)'",
                ],
                fullWgslCode: `${v.fullWgslCode}\n\n// INJECTED FAULT:\nlet broken: vec4<f32> = unknown_intrinsic_fn(gid.x);`,
              };
            }
            return { ...v, status: "COMPILING" };
          })
        );
      } else {
        setVariantResults((prev) => prev.map((v) => ({ ...v, status: "COMPILING" })));
      }
      await new Promise((r) => setTimeout(r, 120));

      // Stage 3: VERIFYING
      setStatusMessage("VERIFYING NUMERICAL TOLERANCE (< 1e-4)...");
      setVariantResults((prev) =>
        prev.map((v) =>
          v.id === "variant-c" && simError ? v : { ...v, status: "VERIFYING" }
        )
      );
      await new Promise((r) => setTimeout(r, 120));

      // Stage 4: BENCHMARKING
      setStatusMessage("BENCHMARKING SILICON RUNS (9-RUN MEDIAN)...");
      setVariantResults((prev) =>
        prev.map((v) =>
          v.id === "variant-c" && simError ? v : { ...v, status: "BENCHMARKING" }
        )
      );
      await new Promise((r) => setTimeout(r, 140));

      // Final settling on recorded reference data
      setStatusMessage("");
      loadRecordedBenchmark(workloadId, simError);
      setIsRunning(false);
    },
    [loadRecordedBenchmark]
  );

  // Main closed-loop WebGPU benchmark pipeline
  const runPipeline = useCallback(
    async (workloadId: WorkloadId, mock: boolean, simError: boolean) => {
      const isRecorded = !probe.supported || previewNoWebGpu || !webGpuCtxRef.current;

      // Fix 1: When in recorded mode, do NOT return early as a no-op!
      if (isRecorded) {
        await replayRecordedPipeline(workloadId, simError);
        return;
      }

      const ctx = webGpuCtxRef.current!;
      const device = ctx.device;
      const workload = WORKLOADS[workloadId];
      const runner = new WebGpuBenchmarkRunner(device, probe.supportsTimestampQuery);

      setIsRunning(true);
      setApiError(null);
      setStatusMessage("ALLOCATING BUFFERS & DETERMINISTIC INPUTS...");

      try {
        // Step 1: Generate deterministic inputs
        const inputs = workload.generateInputs();

        // Step 2: Compile & Benchmark Naive Baseline
        setStatusMessage("COMPILING NAIVE BASELINE...");
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

        setStatusMessage("BENCHMARKING BASELINE (3 WARMUP + 9 TIMED)...");
        const baselineBench = await runner.benchmarkKernel(
          baselineCompilation.pipeline,
          workload,
          baselineDispatchGrid,
          inputs
        );

        // Read baseline output for float verification
        setStatusMessage("READING BASELINE FLOAT REFERENCE...");
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
          isWinner: true,
          sharedMemoryBytes: 0,
          iqrMs: baselineBench.iqrMs,
          stabilityPct: baselineBench.stabilityPct,
        };

        setBaselineResult(initialBaselineResult);
        setWinnerResult(initialBaselineResult);
        setSelectedInspectId("baseline");

        // Step 3: Fetch 3 Variants from /api/generate
        setStatusMessage(
          mock
            ? "LOADING NEMOTRON VERIFIED MOCK KERNELS..."
            : "QUERYING NVIDIA NEMOTRON VIA NEBIUS TOKEN FACTORY..."
        );

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

          setStatusMessage(`COMPILING ${rawVariant.name.toUpperCase()}...`);

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
            // Drop compile failures cleanly
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
            continue;
          }

          const dispatchGrid = computeDispatchGrid(
            workload.problemDims,
            rawVariant.workgroupSize,
            rawVariant.elementsPerThread
          );

          // Step 5: Numerical Tolerance Verification
          setStatusMessage(`VERIFYING FLOATS FOR ${rawVariant.name.toUpperCase()}...`);
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
          setStatusMessage(`BENCHMARKING ${rawVariant.name.toUpperCase()} (9 RUNS)...`);
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

          const { sharedMemoryBytes: variantSharedBytes } = analyzeWgslFootprint(
            compilation.fullWgsl,
            rawVariant.workgroupSize
          );

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
            sharedMemoryBytes: variantSharedBytes,
            iqrMs: bench.iqrMs,
            stabilityPct: bench.stabilityPct,
          };

          currentVariants.push(verifiedResult);
          setVariantResults((prev) =>
            prev.map((v) => (v.id === variantId ? verifiedResult : v))
          );
        }

        // Step 7: Winner Selection
        setStatusMessage("CROWNING FASTEST KERNEL...");
        const verifiedCandidates = currentVariants.filter(
          (v) => v.verified && v.medianTimeMs !== null
        );

        let champion: VariantExecutionResult = initialBaselineResult;

        if (verifiedCandidates.length > 0) {
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
    [probe, previewNoWebGpu, replayRecordedPipeline]
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

      if (probeRes.supported && probeRes.context) {
        runPipeline("matmul", true, false);
      } else {
        loadRecordedBenchmark("matmul", false);
      }
    }

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  // Handle Workload Tab Change
  const handleSelectWorkload = (id: WorkloadId) => {
    setSelectedWorkloadId(id);
    runPipeline(id, isMockMode, simulateCompileError);
  };

  // Handle Tune Action Trigger (Always resets to clean Gen 1 4-kernel state)
  const handleTune = () => {
    runPipeline(selectedWorkloadId, isMockMode, simulateCompileError);
  };

  // Handle Refine Winner (Gen 2) Action Trigger
  const handleRefineGen2 = useCallback(async () => {
    const isRecorded = !probe.supported || previewNoWebGpu || !webGpuCtxRef.current;
    const workloadId = selectedWorkloadId;

    if (isRecorded) {
      setIsRunning(true);
      setApiError(null);

      const parentWinnerId = simulateCompileError ? "variant-b" : "variant-c";

      // Stage 1: GENERATING
      setStatusMessage("SYNTHESIZING EVOLVED VARIANT D (GEN 2)...");
      const placeholderD: VariantExecutionResult = {
        id: "variant-d",
        name: "Variant D: Gen 2 Fused Tile + Register Coarsened",
        strategy: "gen2_evolved",
        rationale: "Synthesizing fused shared memory tiling with unrolled SIMD registers...",
        status: "GENERATING",
        workgroupSize: [16, 16, 1],
        dispatchGrid: [32, 32, 1],
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
        generation: 2,
        parentWinnerId,
      };

      setVariantResults((prev) => [
        ...prev.filter((v) => v.id !== "variant-d"),
        placeholderD,
      ]);
      setSelectedInspectId("variant-d");
      await new Promise((r) => setTimeout(r, 120));

      // Stage 2: COMPILING
      setStatusMessage("VALIDATING GEN 2 LIMITS & COMPILATION...");
      setVariantResults((prev) =>
        prev.map((v) => (v.id === "variant-d" ? { ...v, status: "COMPILING" } : v))
      );
      await new Promise((r) => setTimeout(r, 120));

      // Stage 3: VERIFYING
      setStatusMessage("VERIFYING GEN 2 TOLERANCE (< 1e-4)...");
      setVariantResults((prev) =>
        prev.map((v) => (v.id === "variant-d" ? { ...v, status: "VERIFYING" } : v))
      );
      await new Promise((r) => setTimeout(r, 120));

      // Stage 4: BENCHMARKING
      setStatusMessage("BENCHMARKING GEN 2 SILICON SPEEDUP (9 RUNS)...");
      setVariantResults((prev) =>
        prev.map((v) => (v.id === "variant-d" ? { ...v, status: "BENCHMARKING" } : v))
      );
      await new Promise((r) => setTimeout(r, 140));

      // Settle Gen 2 recorded result
      const gen2Winner = getGen2RecordedResult(workloadId, simulateCompileError);
      setVariantResults((prev) => [
        ...prev.filter((v) => v.id !== "variant-d").map((v) => ({
          ...v,
          isWinner: false,
          status:
            v.status === "COMPILE_FAILED" || v.status === "VERIFY_FAILED"
              ? v.status
              : ("VERIFIED_RUNNER_UP" as const),
        })),
        gen2Winner,
      ]);
      setWinnerResult(gen2Winner);
      setSelectedInspectId("variant-d");
      setStatusMessage("");
      setIsRunning(false);
      return;
    }

    // Physical WebGPU Execution for Gen 2
    const ctx = webGpuCtxRef.current!;
    const device = ctx.device;
    const workload = WORKLOADS[workloadId];
    const runner = new WebGpuBenchmarkRunner(device, probe.supportsTimestampQuery);

    setIsRunning(true);
    setApiError(null);
    setStatusMessage("QUERYING NEMOTRON FOR GEN 2 EVOLUTIONARY REFINEMENT...");

    try {
      const inputs = workload.generateInputs();

      const baselineCompilation = await compileShaderModule(
        device,
        workload,
        workload.baselineWgslBody
      );
      const baselineDispatchGrid = computeDispatchGrid(
        workload.problemDims,
        workload.baselineWorkgroupSize
      );
      const baselineOutput = await runner.runSingleExecutionAndReadback(
        baselineCompilation.pipeline!,
        workload,
        baselineDispatchGrid,
        inputs
      );

      const placeholderD: VariantExecutionResult = {
        id: "variant-d",
        name: "Variant D: Gen 2 Fused Tile + Register Coarsened",
        strategy: "gen2_evolved",
        rationale: "Nemotron evolutionary synthesis from Gen 1 winner...",
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
        generation: 2,
        parentWinnerId: winnerResult?.id,
      };

      setVariantResults((prev) => [
        ...prev.filter((v) => v.id !== "variant-d"),
        placeholderD,
      ]);
      setSelectedInspectId("variant-d");

      const genRes = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workloadId,
          mockMode: isMockMode,
          simulateCompileError,
          generation: 2,
          previousWinner: winnerResult,
          failedVariants: variantResults.filter(
            (v) => v.status === "COMPILE_FAILED" || v.status === "VERIFY_FAILED"
          ),
        }),
      });

      if (!genRes.ok) {
        const errData = (await genRes.json().catch(() => ({}))) as GenerateErrorResponse;
        setApiError({
          error: errData.error || `HTTP ${genRes.status} Error from /api/generate (Gen 2)`,
          code: errData.code || "UPSTREAM_SUPER_AND_ULTRA_FAILED",
          retryable: true,
          details: errData.details,
        });
        setIsRunning(false);
        setStatusMessage("");
        return;
      }

      const genData = (await genRes.json()) as GenerateResponse;
      const rawVariant = genData.variants[0];

      setStatusMessage("COMPILING EVOLVED VARIANT D ON PHYSICAL GPU...");
      setVariantResults((prev) =>
        prev.map((v) => (v.id === "variant-d" ? { ...v, status: "COMPILING" } : v))
      );

      const compilation = await compileShaderModule(
        device,
        workload,
        rawVariant.wgslBody,
        rawVariant.workgroupSize
      );

      if (!compilation.ok || !compilation.pipeline) {
        const failedResult: VariantExecutionResult = {
          id: "variant-d",
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
          generation: 2,
          parentWinnerId: rawVariant.parentWinnerId,
          sharedMemoryBytes: compilation.sharedMemoryBytes,
        };
        setVariantResults((prev) => [
          ...prev.filter((v) => v.id !== "variant-d"),
          failedResult,
        ]);
        setStatusMessage("");
        setIsRunning(false);
        return;
      }

      const dispatchGrid = computeDispatchGrid(
        workload.problemDims,
        rawVariant.workgroupSize,
        rawVariant.elementsPerThread
      );

      setStatusMessage("VERIFYING GEN 2 FLOATS (< 1e-4 TOLERANCE)...");
      setVariantResults((prev) =>
        prev.map((v) => (v.id === "variant-d" ? { ...v, status: "VERIFYING" } : v))
      );

      const variantOutput = await runner.runSingleExecutionAndReadback(
        compilation.pipeline,
        workload,
        dispatchGrid,
        inputs
      );

      const verification = verifyFloatArrays(baselineOutput, variantOutput, 1e-4);

      if (!verification.verified) {
        const verifyFailedResult: VariantExecutionResult = {
          id: "variant-d",
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
          generation: 2,
          parentWinnerId: rawVariant.parentWinnerId,
          sharedMemoryBytes: compilation.sharedMemoryBytes,
        };
        setVariantResults((prev) => [
          ...prev.filter((v) => v.id !== "variant-d"),
          verifyFailedResult,
        ]);
        setStatusMessage("");
        setIsRunning(false);
        return;
      }

      setStatusMessage("BENCHMARKING GEN 2 SILICON TIMINGS (9 RUNS)...");
      setVariantResults((prev) =>
        prev.map((v) => (v.id === "variant-d" ? { ...v, status: "BENCHMARKING" } : v))
      );

      const bench = await runner.benchmarkKernel(
        compilation.pipeline,
        workload,
        dispatchGrid,
        inputs
      );

      const speedup =
        baselineResult?.medianTimeMs && bench.medianTimeMs
          ? baselineResult.medianTimeMs / bench.medianTimeMs
          : 1.0;

      const gen2SuccessResult: VariantExecutionResult = {
        id: "variant-d",
        name: rawVariant.name,
        strategy: rawVariant.strategy,
        rationale: rawVariant.rationale,
        status: "VERIFIED_WINNER",
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
        isWinner: true,
        generation: 2,
        parentWinnerId: rawVariant.parentWinnerId,
        sharedMemoryBytes: compilation.sharedMemoryBytes,
        iqrMs: bench.iqrMs,
        stabilityPct: bench.stabilityPct,
      };

      setVariantResults((prev) => [
        ...prev.filter((v) => v.id !== "variant-d").map((v) => ({
          ...v,
          isWinner: false,
          status:
            v.status === "COMPILE_FAILED" || v.status === "VERIFY_FAILED"
              ? v.status
              : ("VERIFIED_RUNNER_UP" as const),
        })),
        gen2SuccessResult,
      ]);
      setWinnerResult(gen2SuccessResult);
      setSelectedInspectId("variant-d");
      setStatusMessage("");
    } catch (err: unknown) {
      console.error("[Vajra] Gen 2 Pipeline error:", err);
      setApiError({
        error: err instanceof Error ? err.message : String(err),
        code: "PARSE_ERROR",
        retryable: true,
      });
      setStatusMessage("");
    } finally {
      setIsRunning(false);
    }
  }, [
    probe,
    previewNoWebGpu,
    selectedWorkloadId,
    simulateCompileError,
    isMockMode,
    winnerResult,
    variantResults,
    baselineResult,
  ]);

  // Handle Preview No-WebGPU toggle
  const handleTogglePreviewNoWebGpu = (preview: boolean) => {
    setPreviewNoWebGpu(preview);
    if (preview) {
      loadRecordedBenchmark(selectedWorkloadId, simulateCompileError);
    } else if (probe.supported && webGpuCtxRef.current) {
      runPipeline(selectedWorkloadId, isMockMode, simulateCompileError);
    }
  };

  const activeWorkloadSpec = WORKLOADS[selectedWorkloadId];

  return (
    <div className="min-h-screen bg-[#0F1419] text-[#F1F5F9]">
      <div className="mx-auto max-w-[1400px] px-6 py-8 md:px-10 space-y-8">
        {/* 1. Industrial Header Strip */}
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
            runPipeline(selectedWorkloadId, isMockMode, val);
          }}
          previewNoWebGpu={previewNoWebGpu}
          onTogglePreviewNoWebGpu={handleTogglePreviewNoWebGpu}
          liveModelSource={activeModelSource}
          usedFallbackModel={usedFallbackModel}
        />

        {/* 2. Architectural Workload Selector Bar */}
        <WorkloadSelector
          selectedId={selectedWorkloadId}
          onSelect={handleSelectWorkload}
          onTune={handleTune}
          onRefineGen2={handleRefineGen2}
          isRunning={isRunning}
          statusLabel={statusMessage}
        />

        {/* 3. API Error Banner */}
        {apiError && (
          <div className="rounded-lg border-2 border-[#F43F5E] bg-[#160B0E] p-5 font-mono text-xs text-[#F1F5F9]">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <ShieldAlert className="h-5 w-5 shrink-0 text-[#F43F5E]" />
                <div>
                  <div className="font-bold text-[#F43F5E]">
                    GENERATION FAILED: {apiError.error}
                  </div>
                  {apiError.details && (
                    <div className="mt-1 text-[11px] text-[#94A3B8]">
                      {apiError.details}
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => runPipeline(selectedWorkloadId, false, false)}
                  className="rounded-md bg-[#F43F5E] px-4 py-2 font-mono text-xs font-semibold uppercase text-[#0F1419]"
                >
                  Retry Live Generation
                </button>
                <button
                  onClick={() => {
                    setIsMockMode(true);
                    runPipeline(selectedWorkloadId, true, false);
                  }}
                  className="rounded-md border border-[#2A3542] bg-[#11171D] px-4 py-2 font-mono text-xs font-semibold uppercase text-[#F59E0B] hover:bg-[#1E2730]"
                >
                  Fall Back to Mock Kernels
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 4. Hero Section: Throughput Chart + Silicon Report Card */}
        <section className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-stretch">
          {/* Left: 7 Columns GFLOPS Plot */}
          <div className="lg:col-span-7">
            <GflopsChart
              baseline={baselineResult}
              variants={variantResults}
              workloadName={activeWorkloadSpec.name}
            />
          </div>

          {/* Right: 5 Columns Locked Winner Telemetry Sheet */}
          <div className="lg:col-span-5">
            <SiliconReportCard
              winner={winnerResult}
              baseline={baselineResult}
              variants={variantResults}
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
              workloadId={selectedWorkloadId}
            />
          </div>
        </section>

        {/* 5. Pipeline Telemetry Cards Grid */}
        <section className="space-y-4">
          <VariantCardGrid
            baseline={baselineResult}
            variants={variantResults}
            selectedInspectId={selectedInspectId}
            onSelectVariantForInspect={(variant) => setSelectedInspectId(variant.id)}
            workloadId={selectedWorkloadId}
          />

          {/* Failure Graveyard & Guardrail Diagnostics Log */}
          <FailureGraveyard variants={variantResults} />
        </section>

        {/* 6. Terminal Shader Deck (WGSL Inspector) */}
        <section>
          <WgslInspector
            variants={variantResults}
            baseline={baselineResult}
            winner={winnerResult}
            selectedVariantId={selectedInspectId}
            onSelectTab={(id) => setSelectedInspectId(id)}
            workloadId={selectedWorkloadId}
            adapterDescription={
              previewNoWebGpu || !probe.supported
                ? "Apple M3 Max Reference (Chrome 142)"
                : probe.adapterDescription
            }
          />
        </section>
      </div>

      {/* Industrial Footer */}
      <footer className="border-t border-[#2A3542] bg-[#161D24] py-6 font-mono text-xs text-[#94A3B8]">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4 px-6 md:px-10">
          <div className="flex items-center gap-2">
            <span className="font-bold text-[#F59E0B]">VAJRA</span>
            <span className="text-[#64748B]">//</span>
            <span>ZERO-INSTALL BROWSER-NATIVE WEBGPU AUTO-TUNER</span>
          </div>
          <div className="text-[#64748B]">
            BUILT WITH NVIDIA NEMOTRON VIA NEBIUS TOKEN FACTORY FOR THE GLOBAL AI HACKATHON.
          </div>
        </div>
      </footer>
    </div>
  );
}
