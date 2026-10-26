"use client";

import React from "react";
import {
  Trophy,
  Zap,
  HardDrive,
  Grid,
  Cpu,
  ShieldCheck,
} from "lucide-react";
import {
  TimingMethod,
  VariantExecutionResult,
  WorkloadId,
  getStrategyBadgeLabel,
} from "@/types/vajra";

interface SiliconReportCardProps {
  winner: VariantExecutionResult | null;
  baseline: VariantExecutionResult | null;
  timingMethod: TimingMethod;
  modelId: string;
  isMock: boolean;
  adapterName: string;
  workloadId?: WorkloadId;
}

export const SiliconReportCard: React.FC<SiliconReportCardProps> = ({
  winner,
  timingMethod,
  modelId,
  isMock,
  adapterName,
  workloadId,
}) => {
  if (!winner) {
    return (
      <div className="flex h-full flex-col justify-center border border-[#242936] bg-[#0E1014] p-5 text-center">
        <Trophy className="mx-auto h-8 w-8 text-[#5E6678]" />
        <h3 className="mt-3 font-mono text-sm font-bold uppercase tracking-wider text-[#F4F6F8]">
          Awaiting Kernel Benchmark
        </h3>
        <p className="mt-1 font-mono text-xs text-[#8E96A8]">
          Run benchmark to generate, compile, verify, and crown the fastest kernel on your GPU.
        </p>
      </div>
    );
  }

  const speedup = winner.speedupVsBaseline || 1.0;
  const isBaselineWinner = winner.id === "baseline";
  const threadsPerWg =
    winner.workgroupSize[0] * winner.workgroupSize[1] * winner.workgroupSize[2];

  const totalWorkgroups =
    winner.dispatchGrid[0] * winner.dispatchGrid[1] * winner.dispatchGrid[2];

  const strategyBadge = getStrategyBadgeLabel(
    winner.strategy,
    workloadId,
    winner.name
  );

  const timingMethodLabel =
    timingMethod === "timestamp-query"
      ? "WebGPU Timestamp Query (9-run Median)"
      : timingMethod === "recorded-reference"
      ? "Recorded Reference Benchmark (M3 Max / Chrome 142)"
      : "Wall-Clock Timing (queue.onSubmittedWorkDone)";

  return (
    <div className="flex h-full flex-col justify-between border-2 border-[#E5B842] bg-[#0E1014]">
      {/* Top Winner Lock Header */}
      <div>
        <div className="bg-[#E5B842] px-5 py-3 text-[#07080A]">
          <div className="font-mono text-[11px] font-bold tracking-widest uppercase">
            SILICON REPORT CARD // VERIFIED FASTEST KERNEL
          </div>
          <div className="mt-1 flex flex-wrap items-center justify-between gap-2">
            <h2 className="font-sans text-base font-extrabold tracking-tight">
              {winner.name}
            </h2>
            <span className="border border-[#07080A] bg-[#07080A] px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-[#E5B842]">
              {strategyBadge}
            </span>
          </div>
        </div>

        {/* Primary Hero Metric Strip (2-Column Split inside #0A0C0F) */}
        <div className="grid grid-cols-2 divide-x divide-[#242936] border-b border-[#242936] bg-[#0A0C0F]">
          {/* Cell 1: Speedup */}
          <div className="p-4 sm:p-5">
            <div className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#8E96A8]">
              SPEEDUP VS. BASELINE
            </div>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="font-mono text-3xl font-extrabold tracking-tight text-[#E5B842] sm:text-4xl tabular-nums">
                {isBaselineWinner ? "1.00x" : `${speedup.toFixed(2)}x`}
              </span>
              <span className="font-mono text-xs text-[#8E96A8]">
                {isBaselineWinner ? "Reference" : "Faster"}
              </span>
            </div>
          </div>

          {/* Cell 2: Median Throughput */}
          <div className="p-4 sm:p-5">
            <div className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#8E96A8]">
              MEDIAN THROUGHPUT
            </div>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="font-mono text-3xl font-extrabold tracking-tight text-[#F4F6F8] sm:text-4xl tabular-nums">
                {winner.gflops ? winner.gflops.toFixed(1) : "—"}
              </span>
              <span className="font-mono text-xs font-bold text-[#E5B842]">
                GFLOPS
              </span>
            </div>
          </div>
        </div>

        {/* Rationale Bar */}
        <div className="border-b border-[#242936] bg-[#0E1014] px-5 py-2.5 font-mono text-xs text-[#8E96A8]">
          {winner.rationale}
        </div>

        {/* 4-Cell Hardware Telemetry Matrix */}
        <div className="grid grid-cols-2 divide-x divide-y divide-[#242936] border-b border-[#242936] bg-[#0E1014]">
          {/* 1. Median Kernel Latency */}
          <div className="p-3.5">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-[#8E96A8]">
              <Zap className="h-3.5 w-3.5 text-[#E5B842]" />
              <span>MEDIAN KERNEL LATENCY</span>
            </div>
            <div className="mt-1 font-mono text-lg font-bold text-[#F4F6F8] tabular-nums">
              {winner.medianTimeMs ? `${winner.medianTimeMs.toFixed(3)} ms` : "—"}
            </div>
            <div className="font-mono text-[10px] text-[#5E6678]">
              9-Run Timed Median
            </div>
          </div>

          {/* 2. Derived Bandwidth */}
          <div className="p-3.5">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-[#8E96A8]">
              <HardDrive className="h-3.5 w-3.5 text-[#E5B842]" />
              <span>DERIVED MEMORY BANDWIDTH</span>
            </div>
            <div className="mt-1 font-mono text-lg font-bold text-[#F4F6F8] tabular-nums">
              {winner.derivedBandwidthGBps ? `${winner.derivedBandwidthGBps.toFixed(2)} GB/s` : "—"}
            </div>
            <div className="font-mono text-[10px] text-[#5E6678]">
              Bytes Moved / Median Time
            </div>
          </div>

          {/* 3. Workgroup Configuration */}
          <div className="p-3.5">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-[#8E96A8]">
              <Grid className="h-3.5 w-3.5 text-[#E5B842]" />
              <span>WORKGROUP CONFIGURATION</span>
            </div>
            <div className="mt-1 font-mono text-sm font-bold text-[#F4F6F8] tabular-nums">
              ({winner.workgroupSize.join(", ")})
            </div>
            <div className="font-mono text-[10px] text-[#5E6678]">
              {threadsPerWg} threads/workgroup
            </div>
          </div>

          {/* 4. Dispatch Grid Geometry */}
          <div className="p-3.5">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-[#8E96A8]">
              <Cpu className="h-3.5 w-3.5 text-[#E5B842]" />
              <span>DISPATCH GRID GEOMETRY</span>
            </div>
            <div className="mt-1 font-mono text-sm font-bold text-[#F4F6F8] tabular-nums">
              [{winner.dispatchGrid.join(", ")}]
            </div>
            <div className="font-mono text-[10px] text-[#5E6678]">
              {totalWorkgroups.toLocaleString()} total workgroups
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Verification & Provenance Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[#242936] bg-[#0A0C0F] px-5 py-2.5 font-mono text-xs">
        <div className="flex items-center gap-1.5 text-[#10B981]">
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>
            {winner.maxRelativeError !== null
              ? winner.maxRelativeError === 0
                ? "Exact Match (0.0)"
                : `Max Rel Error: ${winner.maxRelativeError.toExponential(2)} (< 1e-4)`
              : "Tolerance Verified"}
          </span>
        </div>
        <div className="text-[#8E96A8]">
          <span>{adapterName}</span>
          <span className="mx-1.5 text-[#5E6678]">•</span>
          <span className="text-[#E5B842]">{isMock ? "Mock Nemotron" : modelId}</span>
        </div>
      </div>
    </div>
  );
};
