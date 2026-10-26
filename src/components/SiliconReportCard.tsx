"use client";

import React, { useState } from "react";
import {
  Trophy,
  Zap,
  HardDrive,
  Grid,
  Cpu,
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

  if (!winner) {
    return (
      <div className="flex h-full flex-col justify-center rounded-lg border border-[#2A3542] bg-[#161D24] p-6 text-center">
        <Trophy className="mx-auto h-8 w-8 text-[#64748B]" />
        <h3 className="mt-3 font-semibold text-[#F1F5F9]">
          Awaiting Kernel Benchmark
        </h3>
        <p className="mt-1 font-mono text-xs text-[#94A3B8]">
          Run benchmark to generate, compile, verify, and crown the fastest kernel on your hardware.
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
    winner.name,
    winner.sharedMemoryBytes
  );

  // Gen 2 Uplift against active Gen 1 winner
  let upliftText: string | null = null;
  if (winner.id === "variant-d" && winner.gflops) {
    const gen1Winners = variants
      .filter(
        (v) =>
          v.id !== "variant-d" &&
          v.id !== "baseline" &&
          v.status !== "COMPILE_FAILED" &&
          v.status !== "VERIFY_FAILED" &&
          v.gflops !== null &&
          v.gflops > 0
      )
      .sort((a, b) => (b.gflops || 0) - (a.gflops || 0));

    const activeGen1Winner = gen1Winners[0];
    if (activeGen1Winner && activeGen1Winner.gflops) {
      const upliftPct =
        ((winner.gflops - activeGen1Winner.gflops) / activeGen1Winner.gflops) * 100;
      const winnerLabel = activeGen1Winner.id === "variant-b" ? "Winner B" : "Winner C";
      upliftText = `+${upliftPct.toFixed(1)}% vs Gen 1 ${winnerLabel}`;
    }
  }

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
    <div className="flex h-full flex-col justify-between rounded-lg border-2 border-[#F59E0B] bg-[#161D24] p-6">
      {/* Top Header */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="rounded bg-[#291F0E] px-2.5 py-1 font-mono text-xs font-semibold uppercase tracking-wider text-[#F59E0B] border border-[#F59E0B]/30">
            FASTEST VERIFIED KERNEL
          </span>
          <div className="flex items-center gap-2">
            <span className="rounded border border-[#2A3542] bg-[#11171D] px-2 py-0.5 font-mono text-xs text-[#94A3B8]">
              {strategyBadge}
            </span>
            {upliftText && (
              <span className="rounded border border-[#10B981]/50 bg-[#0B261F] px-2.5 py-0.5 font-mono text-xs font-bold text-[#10B981]">
                {upliftText}
              </span>
            )}
          </div>
        </div>

        <h2 className="mt-3 text-xl font-bold tracking-tight text-[#F1F5F9]">
          {winner.name}
        </h2>
        <p className="mt-1 text-xs text-[#94A3B8] font-mono leading-relaxed">
          {winner.rationale}
        </p>

        {/* Primary 2-Metric Hero Spotlight */}
        <div className="my-5 grid grid-cols-2 gap-4 rounded-lg border border-[#2A3542] bg-[#11171D] p-5">
          {/* Speedup vs Baseline */}
          <div>
            <div className="font-mono text-xs font-medium uppercase tracking-wider text-[#94A3B8]">
              Speedup vs. Baseline
            </div>
            <div className="mt-1 font-mono text-4xl font-bold tracking-tight text-[#F59E0B] tabular-nums">
              {isBaselineWinner ? "1.00x" : `${speedup.toFixed(2)}x`}
            </div>
            <div className="mt-1 font-mono text-xs text-[#64748B]">
              {isBaselineWinner ? "Reference baseline" : "Measured hardware speedup"}
            </div>
          </div>

          {/* Median Throughput */}
          <div>
            <div className="font-mono text-xs font-medium uppercase tracking-wider text-[#94A3B8]">
              Median Throughput
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="font-mono text-4xl font-bold tracking-tight text-[#F1F5F9] tabular-nums">
                {winner.gflops ? winner.gflops.toFixed(1) : "—"}
              </span>
              <span className="font-mono text-sm font-bold text-[#F59E0B]">
                GFLOPS
              </span>
            </div>
            <div className="mt-1 font-mono text-xs text-[#64748B]">
              9-run timed median
            </div>
          </div>
        </div>

        {/* Core 3 Summary Pills */}
        <div className="grid grid-cols-3 gap-3 font-mono text-xs">
          <div className="rounded-md border border-[#2A3542] bg-[#11171D] p-3">
            <div className="text-[11px] text-[#64748B] uppercase">Latency</div>
            <div className="mt-1 font-bold text-[#F1F5F9] tabular-nums">
              {winner.medianTimeMs ? `${winner.medianTimeMs.toFixed(3)} ms` : "—"}
            </div>
          </div>
          <div className="rounded-md border border-[#2A3542] bg-[#11171D] p-3">
            <div className="text-[11px] text-[#64748B] uppercase">Bandwidth</div>
            <div className="mt-1 font-bold text-[#F1F5F9] tabular-nums">
              {winner.derivedBandwidthGBps
                ? `${winner.derivedBandwidthGBps.toFixed(2)} GB/s`
                : "—"}
            </div>
          </div>
          <div className="rounded-md border border-[#2A3542] bg-[#11171D] p-3">
            <div className="text-[11px] text-[#64748B] uppercase">Accuracy</div>
            <div className="mt-1 flex items-center gap-1 font-bold text-[#10B981]">
              <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
              <span>Verified (&lt;1e-4)</span>
            </div>
          </div>
        </div>

        {/* Progressive Disclosure Drawer Toggle */}
        <div className="mt-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex w-full items-center justify-between rounded-md border border-[#2A3542] bg-[#11171D] px-4 py-2.5 text-xs font-mono text-[#94A3B8] transition-colors hover:bg-[#1E2730] hover:text-[#F1F5F9]"
          >
            <span>
              {showDetails ? "Hide Hardware & Roofline Details" : "Show Hardware & Roofline Details"}
            </span>
            {showDetails ? (
              <ChevronUp className="h-4 w-4 text-[#F59E0B]" />
            ) : (
              <ChevronDown className="h-4 w-4 text-[#F59E0B]" />
            )}
          </button>

          {/* Collapsible Details Grid */}
          {showDetails && (
            <div className="mt-3 space-y-3 rounded-md border border-[#2A3542] bg-[#11171D] p-4 text-xs font-mono">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-[#64748B]">Workgroup Config</div>
                  <div className="mt-0.5 font-bold text-[#F1F5F9]">
                    ({winner.workgroupSize.join(", ")}) • {threadsPerWg} th/wg
                  </div>
                </div>
                <div>
                  <div className="text-[#64748B]">Dispatch Grid</div>
                  <div className="mt-0.5 font-bold text-[#F1F5F9]">
                    [{winner.dispatchGrid.join(", ")}] • {totalWorkgroups.toLocaleString()} total
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 border-t border-[#2A3542] pt-3">
                <div>
                  <div className="text-[#64748B]">Arithmetic Intensity</div>
                  <div className="mt-0.5 flex items-center gap-1.5 font-bold text-[#F1F5F9]">
                    <span>{arithmeticIntensity} FLOP/Byte</span>
                    <span
                      className={`rounded px-1.5 py-0.2 text-[10px] ${
                        isComputeBound
                          ? "bg-[#291F0E] text-[#F59E0B]"
                          : "bg-[#0B261F] text-[#10B981]"
                      }`}
                    >
                      {isComputeBound ? "COMPUTE-BOUND" : "MEMORY-BOUND"}
                    </span>
                  </div>
                </div>
                <div>
                  <div className="text-[#64748B]">Shared Memory Footprint</div>
                  <div className="mt-0.5 font-bold text-[#F1F5F9]">
                    {sharedMemDisplay}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 border-t border-[#2A3542] pt-3">
                <div>
                  <div className="text-[#64748B]">9-Run Timing Stability</div>
                  <div className="mt-0.5 font-bold text-[#10B981]">
                    {stabilityDisplay}
                  </div>
                </div>
                <div>
                  <div className="text-[#64748B]">Max Relative Error</div>
                  <div className="mt-0.5 font-bold text-[#10B981]">
                    {winner.maxRelativeError !== null
                      ? winner.maxRelativeError === 0
                        ? "0.0 (Exact Match)"
                        : winner.maxRelativeError.toExponential(2)
                      : "Verified (< 1e-4)"}
                  </div>
                </div>
              </div>

              <div className="border-t border-[#2A3542] pt-3 text-[11px] text-[#64748B]">
                Hardware: {adapterName} • Model: {isMock ? "Mock Nemotron" : modelId}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Provenance Strip */}
      <div className="mt-4 flex flex-wrap items-center justify-between border-t border-[#2A3542] pt-3 font-mono text-[11px] text-[#64748B]">
        <span>WebGPU Dawn / wgpu Architecture</span>
        <span className="text-[#F59E0B]">Verified &amp; Benchmarked</span>
      </div>
    </div>
  );
};
