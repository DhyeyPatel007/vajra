"use client";

import React from "react";
import {
  Trophy,
  Zap,
  HardDrive,
  Grid,
  CheckCircle2,
  Clock,
  Cpu,
  ShieldCheck,
} from "lucide-react";
import { TimingMethod, VariantExecutionResult } from "@/types/vajra";

interface SiliconReportCardProps {
  winner: VariantExecutionResult | null;
  baseline: VariantExecutionResult | null;
  timingMethod: TimingMethod;
  modelId: string;
  isMock: boolean;
  adapterName: string;
}

export const SiliconReportCard: React.FC<SiliconReportCardProps> = ({
  winner,
  baseline,
  timingMethod,
  modelId,
  isMock,
  adapterName,
}) => {
  if (!winner) {
    return (
      <div className="flex h-full flex-col justify-center rounded border border-[#262A34] bg-[#121418] p-5 text-center">
        <Trophy className="mx-auto h-8 w-8 text-[#9CA3AF] opacity-40" />
        <h3 className="mt-3 text-sm font-bold text-[#F3F4F6]">
          Awaiting Kernel Benchmark
        </h3>
        <p className="mt-1 text-xs text-[#9CA3AF]">
          Run benchmark to generate, compile, verify, and crown the fastest kernel on your GPU.
        </p>
      </div>
    );
  }

  const speedup = winner.speedupVsBaseline || 1.0;
  const isBaselineWinner = winner.id === "baseline";
  const threadsPerWg =
    winner.workgroupSize[0] * winner.workgroupSize[1] * winner.workgroupSize[2];

  const formatStrategy = (strat: string) => {
    switch (strat) {
      case "workgroup_tuning":
        return "Workgroup Geometry Tuning";
      case "shared_memory_tiling":
        return "Shared Memory Tiling";
      case "vec4_vectorized":
        return "Vec4 Vectorized SIMD";
      default:
        return "Baseline Reference";
    }
  };

  const timingMethodLabel =
    timingMethod === "timestamp-query"
      ? "WebGPU Timestamp Query (9-run Median)"
      : timingMethod === "recorded-reference"
      ? "Recorded Reference Benchmark (M3 Max / Chrome 142)"
      : "Wall-Clock Timing (queue.onSubmittedWorkDone)";

  return (
    <div className="flex h-full flex-col justify-between rounded border border-[#D4AF37] bg-[#121418] p-5 shadow-sm">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between border-b border-[#262A34] pb-3">
          <div className="flex items-center gap-2">
            <Trophy className="h-5 w-5 text-[#D4AF37]" />
            <div>
              <span className="text-[10px] font-bold tracking-wider uppercase text-[#D4AF37]">
                Silicon Report Card
              </span>
              <h2 className="text-base font-bold text-[#F3F4F6]">
                {winner.name}
              </h2>
            </div>
          </div>
          <span className="rounded bg-[#181B22] border border-[#D4AF37] px-2 py-0.5 text-[10px] font-bold text-[#D4AF37]">
            {formatStrategy(winner.strategy)}
          </span>
        </div>

        {/* Hero Speedup Display */}
        <div className="my-4 rounded border border-[#262A34] bg-[#181B22] p-3 text-center">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-[#9CA3AF]">
            Measured Silicon Speedup
          </div>
          <div className="mt-1 flex items-baseline justify-center gap-1">
            <span className="font-mono text-3xl font-extrabold text-[#D4AF37]">
              {isBaselineWinner ? "1.00x" : `${speedup.toFixed(2)}x`}
            </span>
            <span className="text-xs font-semibold text-[#F3F4F6]">
              {isBaselineWinner ? "Reference Speed" : "Faster than Baseline"}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-[#9CA3AF] line-clamp-2">
            {winner.rationale}
          </p>
        </div>

        {/* 6 Core Hardware Telemetry Grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          {/* 1. Compute */}
          <div className="rounded border border-[#262A34] bg-[#0B0C0E] p-2.5">
            <div className="flex items-center gap-1.5 text-[#9CA3AF]">
              <Zap className="h-3.5 w-3.5 text-[#D4AF37]" />
              <span className="text-[11px]">Compute Throughput</span>
            </div>
            <div className="mt-1 font-mono text-base font-bold text-[#F3F4F6]">
              {winner.gflops ? `${winner.gflops.toFixed(1)} GFLOPS` : "—"}
            </div>
            <div className="text-[10px] text-[#9CA3AF]">
              Median Latency: {winner.medianTimeMs?.toFixed(3)} ms
            </div>
          </div>

          {/* 2. Derived Bandwidth */}
          <div className="rounded border border-[#262A34] bg-[#0B0C0E] p-2.5">
            <div className="flex items-center gap-1.5 text-[#9CA3AF]">
              <HardDrive className="h-3.5 w-3.5 text-[#D4AF37]" />
              <span className="text-[11px]">Derived Memory BW</span>
            </div>
            <div className="mt-1 font-mono text-base font-bold text-[#F3F4F6]">
              {winner.derivedBandwidthGBps
                ? `${winner.derivedBandwidthGBps.toFixed(2)} GB/s`
                : "—"}
            </div>
            <div className="text-[10px] text-[#9CA3AF]" title="Derived strictly from data moved divided by median execution time">
              Derived (Bytes / Time)
            </div>
          </div>

          {/* 3. Workgroup Geometry */}
          <div className="rounded border border-[#262A34] bg-[#0B0C0E] p-2.5">
            <div className="flex items-center gap-1.5 text-[#9CA3AF]">
              <Grid className="h-3.5 w-3.5 text-[#D4AF37]" />
              <span className="text-[11px]">Workgroup Size</span>
            </div>
            <div className="mt-1 font-mono text-xs font-bold text-[#F3F4F6]">
              ({winner.workgroupSize.join(", ")})
            </div>
            <div className="text-[10px] text-[#9CA3AF]">
              {threadsPerWg} threads/workgroup
            </div>
          </div>

          {/* 4. Dispatch Grid */}
          <div className="rounded border border-[#262A34] bg-[#0B0C0E] p-2.5">
            <div className="flex items-center gap-1.5 text-[#9CA3AF]">
              <Cpu className="h-3.5 w-3.5 text-[#D4AF37]" />
              <span className="text-[11px]">Dispatch Grid</span>
            </div>
            <div className="mt-1 font-mono text-xs font-bold text-[#F3F4F6]">
              [{winner.dispatchGrid.join(", ")}]
            </div>
            <div className="text-[10px] text-[#9CA3AF]">
              {winner.dispatchGrid[0] * winner.dispatchGrid[1] * winner.dispatchGrid[2]} workgroups
            </div>
          </div>

          {/* 5. Numerical Accuracy */}
          <div className="rounded border border-[#262A34] bg-[#0B0C0E] p-2.5">
            <div className="flex items-center gap-1.5 text-[#9CA3AF]">
              <ShieldCheck className="h-3.5 w-3.5 text-[#10B981]" />
              <span className="text-[11px]">Float Accuracy</span>
            </div>
            <div className="mt-1 font-mono text-xs font-bold text-[#10B981]">
              {winner.maxRelativeError !== null
                ? winner.maxRelativeError === 0
                  ? "Exact Match (0.0)"
                  : `${winner.maxRelativeError.toExponential(2)}`
                : "Verified"}
            </div>
            <div className="text-[10px] text-[#9CA3AF]">
              Threshold &lt; 1e-4 relative
            </div>
          </div>

          {/* 6. Hardware Adapter */}
          <div className="rounded border border-[#262A34] bg-[#0B0C0E] p-2.5">
            <div className="flex items-center gap-1.5 text-[#9CA3AF]">
              <Cpu className="h-3.5 w-3.5 text-[#D4AF37]" />
              <span className="text-[11px]">Target Silicon</span>
            </div>
            <div className="mt-1 truncate font-mono text-xs font-bold text-[#F3F4F6]" title={adapterName}>
              {adapterName}
            </div>
            <div className="text-[10px] text-[#9CA3AF]">
              Physical Device
            </div>
          </div>
        </div>
      </div>

      {/* Footer Provenance */}
      <div className="mt-3 border-t border-[#262A34] pt-2 text-[10px] text-[#9CA3AF]">
        <div className="flex items-center justify-between">
          <span>{timingMethodLabel}</span>
          <span className="font-mono text-[#D4AF37]">
            {isMock ? "Mock Nemotron" : modelId}
          </span>
        </div>
      </div>
    </div>
  );
};
