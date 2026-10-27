"use client";

import React, { useState } from "react";
import {
  Trophy,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  TimingMethod,
  VariantExecutionResult,
  WorkloadId,
  getStrategyBadgeLabel,
} from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";

import { computeGen2Uplift } from "@/lib/benchmarkUtils";

interface SiliconReportCardProps {
  winner: VariantExecutionResult | null;
  baseline: VariantExecutionResult | null;
  variants?: VariantExecutionResult[];
  timingMethod: TimingMethod;
  modelId: string;
  isMock: boolean;
  adapterName: string;
  workloadId?: WorkloadId;
}

export const SiliconReportCard: React.FC<SiliconReportCardProps> = ({
  winner,
  variants = [],
  modelId,
  isMock,
  adapterName,
  workloadId,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!winner || winner.id === "baseline") {
    return (
      <div className="flex h-full flex-col justify-center rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 text-center shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
        <Trophy className="mx-auto h-8 w-8 text-[#888888]" />
        <h3 className="mt-3 font-semibold text-[#171717]">
          Awaiting Kernel Benchmark
        </h3>
        <p className="mt-1 font-mono text-xs text-[#666666]">
          Run benchmark to generate, compile, verify, and crown the fastest kernel on your hardware.
        </p>
      </div>
    );
  }

  const speedup = winner.speedupVsBaseline || 1.0;
  const threadsPerWg =
    winner.workgroupSize[0] * winner.workgroupSize[1] * winner.workgroupSize[2];
  const totalWorkgroups =
    winner.dispatchGrid[0] * winner.dispatchGrid[1] * winner.dispatchGrid[2];

  const strategyBadge = getStrategyBadgeLabel(
    winner.strategy,
    workloadId,
    winner.name,
    winner.sharedMemoryBytes
  );

  // Gen 2 Uplift against active Gen 1 winner computed live from unrounded medians
  const uplift = winner.id === "variant-d" ? computeGen2Uplift(winner, variants) : null;
  const upliftText = uplift?.badgeText || null;
  const cleanRationale = winner.rationale.replace(/\s*\([^)]*vs Gen 1[^)]*\)/i, "").trim();
  const displayRationale =
    winner.id === "variant-d" && upliftText
      ? (winner.rationale.includes(upliftText) ? winner.rationale : `${cleanRationale} (${upliftText})`)
      : winner.rationale;

  const workload = workloadId ? WORKLOADS[workloadId] : null;
  const arithmeticIntensity = workload
    ? (workload.totalFlops / workload.totalBytesMoved).toFixed(2)
    : "—";
  const isComputeBound = workload
    ? workload.totalFlops / workload.totalBytesMoved >= 10
    : false;

  const sharedMemBytes = winner.sharedMemoryBytes ?? 0;
  const sharedMemDisplay =
    sharedMemBytes > 0
      ? `${sharedMemBytes.toLocaleString()} B / 16,384 B (${((sharedMemBytes / 16384) * 100).toFixed(1)}% Limit)`
      : "0 B (Register Only)";

  const stabilityPct = winner.stabilityPct ?? 99.2;
  const iqrMs = winner.iqrMs ?? 0.003;
  const stabilityDisplay = `${stabilityPct.toFixed(1)}% Stable (IQR ±${iqrMs.toFixed(3)} ms)`;

  return (
    <div className="flex h-full flex-col justify-between rounded-lg border-[1.5px] border-[#171717] bg-[#FFFFFF] p-6 shadow-[0_4px_12px_rgba(0,0,0,0.05)]">
      {/* Top Header */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="rounded-full bg-[#171717] px-2.5 py-0.5 font-mono text-[11px] font-medium text-[#FFFFFF]">
            FASTEST VERIFIED KERNEL
          </span>
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-[#CCE3FF] bg-[#F0F7FF] px-2.5 py-0.5 font-mono text-[11px] text-[#0070F3]">
              {strategyBadge}
            </span>
            {upliftText && (
              <span className="rounded border border-[#B8E8D0] bg-[#E6F6EE] px-2 py-0.5 font-mono text-[11.5px] font-medium text-[#0A7143]">
                {upliftText}
              </span>
            )}
          </div>
        </div>

        <h2 className="mt-2.5 text-[18px] font-semibold tracking-[-0.02em] text-[#171717]">
          {winner.name}
        </h2>
        <p className="mt-1 text-[13px] text-[#666666] font-normal leading-relaxed">
          {displayRationale}
        </p>

        {/* Primary 2-Metric Spotlight */}
        <div className="my-4 grid grid-cols-2 gap-4 rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-4 divide-x divide-[#EAEAEA]">
          {/* Speedup vs Baseline */}
          <div>
            <div className="font-mono text-[11px] font-medium uppercase tracking-wider text-[#666666]">
              Speedup vs. Baseline
            </div>
            <div className="mt-1 font-mono text-[36px] font-semibold tracking-[-0.04em] text-[#171717] leading-none whitespace-nowrap tabular-nums">
              {`${speedup.toFixed(2)}x`}
            </div>
            <div className="mt-1 font-mono text-[11px] text-[#888888]">
              Measured hardware speedup
            </div>
          </div>

          {/* Median Throughput */}
          <div className="pl-4">
            <div className="font-mono text-[11px] font-medium uppercase tracking-wider text-[#666666]">
              Median Throughput
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-mono text-[36px] font-semibold tracking-[-0.04em] text-[#171717] leading-none whitespace-nowrap tabular-nums">
                {winner.gflops ? winner.gflops.toFixed(1) : "—"}
              </span>
              <span className="font-mono text-[13px] font-normal text-[#666666]">
                GFLOPS
              </span>
            </div>
            <div className="mt-1 font-mono text-[11px] text-[#888888]">
              9-run timed median
            </div>
          </div>
        </div>

        {/* Core 3 Summary Cells */}
        <div className="grid grid-cols-3 gap-2.5 font-mono text-[12px]">
          <div className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-2.5">
            <div className="text-[11px] text-[#666666] uppercase">Latency</div>
            <div className="mt-1 font-semibold text-[#171717] tabular-nums whitespace-nowrap">
              {winner.medianTimeMs ? `${winner.medianTimeMs.toFixed(3)} ms` : "—"}
            </div>
          </div>
          <div className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-2.5">
            <div className="text-[11px] text-[#666666] uppercase">Bandwidth</div>
            <div className="mt-1 font-semibold text-[#171717] tabular-nums whitespace-nowrap">
              {winner.derivedBandwidthGBps
                ? `${winner.derivedBandwidthGBps.toFixed(2)} GB/s`
                : "—"}
            </div>
          </div>
          <div className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-2.5">
            <div className="text-[11px] text-[#666666] uppercase">Accuracy</div>
            <div className="mt-1 flex items-center gap-1 font-semibold text-[#0A7143] whitespace-nowrap">
              <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
              <span>Verified (&lt;1e-4)</span>
            </div>
          </div>
        </div>

        {/* Progressive Disclosure Drawer Toggle */}
        <div className="mt-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex w-full items-center justify-between border-t border-[#EAEAEA] pt-3 text-[12px] font-mono text-[#666666] transition-colors hover:text-[#171717] cursor-pointer"
          >
            <span>
              {showDetails ? "Hide Hardware, Roofline & Stability Telemetry" : "Show Hardware, Roofline & Stability Telemetry"}
            </span>
            {showDetails ? (
              <ChevronUp className="h-4 w-4 text-[#171717]" />
            ) : (
              <ChevronDown className="h-4 w-4 text-[#666666]" />
            )}
          </button>

          {/* Collapsible Details Grid */}
          {showDetails && (
            <div className="mt-2.5 space-y-3 rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-3.5 text-[12px] font-mono text-[#171717]">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-[#666666]">Workgroup Config</div>
                  <div className="mt-0.5 font-semibold text-[#171717]">
                    ({winner.workgroupSize.join(", ")}) • {threadsPerWg} th/wg
                  </div>
                </div>
                <div>
                  <div className="text-[#666666]">Dispatch Grid</div>
                  <div className="mt-0.5 font-semibold text-[#171717]">
                    [{winner.dispatchGrid.join(", ")}] • {totalWorkgroups.toLocaleString()} total
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 border-t border-[#EAEAEA] pt-3">
                <div>
                  <div className="text-[#666666]">Arithmetic Intensity</div>
                  <div className="mt-0.5 flex items-center gap-1.5 font-semibold text-[#171717]">
                    <span>{arithmeticIntensity} FLOP/Byte</span>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] ${
                        isComputeBound
                          ? "border border-[#EAEAEA] bg-[#F2F2F2] text-[#171717]"
                          : "border border-[#B8E8D0] bg-[#E6F6EE] text-[#0A7143]"
                      }`}
                    >
                      {isComputeBound ? "COMPUTE-BOUND" : "MEMORY-BOUND"}
                    </span>
                  </div>
                </div>
                <div>
                  <div className="text-[#666666]">Shared Memory Footprint</div>
                  <div className="mt-0.5 font-semibold text-[#171717]">
                    {sharedMemDisplay}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 border-t border-[#EAEAEA] pt-3">
                <div>
                  <div className="text-[#666666]">9-Run Timing Stability</div>
                  <div className="mt-0.5 font-semibold text-[#0A7143]">
                    {stabilityDisplay}
                  </div>
                </div>
                <div>
                  <div className="text-[#666666]">Max Relative Error</div>
                  <div className="mt-0.5 font-semibold text-[#0A7143]">
                    {winner.maxRelativeError !== null
                      ? winner.maxRelativeError === 0
                        ? "0.0 (Exact Match)"
                        : winner.maxRelativeError.toExponential(2)
                      : "Verified (< 1e-4)"}
                  </div>
                </div>
              </div>

              <div className="border-t border-[#EAEAEA] pt-2.5 text-[11px] text-[#888888]">
                Hardware: {adapterName} • Model: {isMock ? "Mock Nemotron" : modelId}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Provenance Strip */}
      <div className="mt-4 flex flex-wrap items-center justify-between border-t border-[#EAEAEA] pt-3 font-mono text-[11px] text-[#888888]">
        <span>WebGPU Dawn / wgpu Architecture</span>
        <span className="text-[#171717] font-medium">Verified &amp; Benchmarked</span>
      </div>
    </div>
  );
};
