"use client";

import React from "react";
import { VariantExecutionResult, getStrategyBadgeLabel } from "@/types/vajra";
import { Trophy, AlertCircle } from "lucide-react";

interface GflopsChartProps {
  baseline: VariantExecutionResult | null;
  variants: VariantExecutionResult[];
  workloadName: string;
}

export const GflopsChart: React.FC<GflopsChartProps> = ({
  baseline,
  variants,
  workloadName,
}) => {
  const hasVariantD = variants.some((v) => v.id === "variant-d");
  const allItems: Array<VariantExecutionResult | null> = [
    baseline,
    variants.find((v) => v.id === "variant-a") || null,
    variants.find((v) => v.id === "variant-b") || null,
    variants.find((v) => v.id === "variant-c") || null,
  ];

  if (hasVariantD) {
    allItems.push(variants.find((v) => v.id === "variant-d") || null);
  }

  const validGflops = allItems
    .map((item) => item?.gflops || 0)
    .filter((g) => g > 0);
  const maxGflops = Math.max(10, ...validGflops);

  return (
    <div className="flex h-full flex-col justify-between rounded-lg border border-[#2A3542] bg-[#161D24] p-6">
      {/* Header */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#2A3542] pb-4">
          <div>
            <h2 className="text-lg font-semibold text-[#F1F5F9]">
              Compute Throughput Comparison (GFLOPS)
            </h2>
            <p className="text-sm text-[#94A3B8]">
              9-run hardware median — higher is faster • {workloadName}
            </p>
          </div>
          {/* Subtle Legend */}
          <div className="flex items-center gap-3 font-mono text-xs text-[#94A3B8]">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[#F59E0B]"></span>
              <span className="text-[#F59E0B]">Winner</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[#2A3B4C]"></span>
              <span>Verified</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[#F43F5E]"></span>
              <span className="text-[#F43F5E]">Dropped</span>
            </span>
          </div>
        </div>

        {/* Spacious Horizontal Bar Rows */}
        <div className="my-5 space-y-5">
          {allItems.map((item, idx) => {
            if (!item) return null;
            const isWinner = item.isWinner;
            const isFailed = item.status === "COMPILE_FAILED";
            const isVerFailed = item.status === "VERIFY_FAILED";
            const isBusy =
              item.status === "GENERATING" ||
              item.status === "COMPILING" ||
              item.status === "VERIFYING" ||
              item.status === "BENCHMARKING";

            const widthPct =
              item.gflops && maxGflops > 0
                ? Math.max(14, Math.min(100, (item.gflops / maxGflops) * 100))
                : 0;

            const badge = getStrategyBadgeLabel(
              item.strategy,
              undefined,
              item.name,
              item.sharedMemoryBytes
            );

            return (
              <div key={item.id || idx} className="space-y-1.5">
                {/* Variant Label & Strategy */}
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-[#F1F5F9]">
                      {item.name}
                    </span>
                    <span className="rounded border border-[#2A3542] bg-[#11171D] px-2 py-0.5 font-mono text-[10px] text-[#94A3B8]">
                      {badge}
                    </span>
                  </div>
                  {isWinner && (
                    <span className="flex items-center gap-1 rounded border border-[#F59E0B] bg-[#291F0E] px-2.5 py-0.5 font-mono text-xs font-semibold text-[#F59E0B]">
                      <Trophy className="h-3 w-3" />
                      FASTEST WINNER
                    </span>
                  )}
                </div>

                {/* Track */}
                <div className="relative h-11 w-full rounded-md border border-[#2A3542] bg-[#11171D] p-1">
                  {isFailed ? (
                    <div className="flex h-full w-full items-center gap-2 rounded border border-[#F43F5E]/60 bg-[#2A1219] px-3 font-mono text-xs text-[#F43F5E]">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      <span>Dropped — WGSL Compile Error (See Diagnostics Below)</span>
                    </div>
                  ) : isVerFailed ? (
                    <div className="flex h-full w-full items-center gap-2 rounded border border-[#F43F5E]/60 bg-[#2A1219] px-3 font-mono text-xs text-[#F43F5E]">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      <span>Verification Failed (&gt; 1e-4 tolerance)</span>
                    </div>
                  ) : isBusy ? (
                    <div className="flex h-full w-full items-center px-3 font-mono text-xs text-[#F59E0B] animate-pulse">
                      <span>{item.status}...</span>
                    </div>
                  ) : item.gflops ? (
                    <div
                      style={{ width: `${widthPct}%` }}
                      className={`flex h-full items-center justify-between rounded px-3 transition-all duration-500 ${
                        isWinner
                          ? "bg-[#F59E0B] text-[#0F1419]"
                          : "bg-[#2A3B4C] text-[#F1F5F9]"
                      }`}
                    >
                      <span className="font-mono text-sm font-bold tracking-tight">
                        {item.gflops.toFixed(1)} GFLOPS
                      </span>
                      <span
                        className={`font-mono text-xs font-semibold ${
                          isWinner ? "text-[#0F1419]" : "text-[#94A3B8]"
                        }`}
                      >
                        {item.id === "baseline"
                          ? "1.00x (Ref)"
                          : `${item.speedupVsBaseline?.toFixed(2) || "1.00"}x`}
                      </span>
                    </div>
                  ) : (
                    <div className="flex h-full w-full items-center px-3 font-mono text-xs text-[#64748B]">
                      Awaiting run...
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Subtext */}
      <div className="border-t border-[#2A3542] pt-3 font-mono text-xs text-[#64748B]">
        IEEE-754 FP32 Throughput • 3 Warmup + 9 Timed Dispatches • Dawn/wgpu Architecture
      </div>
    </div>
  );
};
