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
    <div className="flex h-full flex-col justify-between rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
      {/* Header */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#EAEAEA] pb-4">
          <div>
            <h2 className="text-[16px] font-semibold tracking-[-0.02em] text-[#171717]">
              Compute Throughput Comparison (GFLOPS)
            </h2>
            <p className="text-[13px] text-[#666666] mt-0.5">
              9-run hardware median — higher is faster • {workloadName}
            </p>
          </div>
          {/* Subtle Legend */}
          <div className="flex items-center gap-3 font-mono text-[12px] text-[#666666]">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[#171717] border-r-2 border-[#0070F3]" />
              <span className="text-[#171717] font-medium">Winner</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[#666666]" />
              <span>Verified</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[#E00000]" />
              <span className="text-[#C81E1E]">Dropped</span>
            </span>
          </div>
        </div>

        {/* Spacious Horizontal Bar Rows */}
        <div className="my-5 space-y-4">
          {allItems.map((item, idx) => {
            if (!item) return null;
            const isWinner = item.id !== "baseline" && Boolean(item.isWinner);
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
                {/* Variant Label & Strategy & Readout */}
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-[#171717]">
                      {item.name}
                    </span>
                    <span className="rounded border border-[#EAEAEA] bg-[#F2F2F2] px-2 py-0.5 font-mono text-[11px] text-[#666666]">
                      {badge}
                    </span>
                    {isWinner && (
                      <span className="flex items-center gap-1 rounded-full bg-[#171717] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#FFFFFF]">
                        <Trophy className="h-3 w-3" />
                        FASTEST WINNER
                      </span>
                    )}
                  </div>
                  {/* Right Tabular Readout (Guaranteed no truncation) */}
                  {item.gflops && !isFailed && !isVerFailed ? (
                    <div className="font-mono text-[14px] font-semibold text-[#171717] whitespace-nowrap tabular-nums">
                      {item.gflops.toFixed(1)} GFLOPS{" "}
                      <span className="text-[12px] font-normal text-[#666666]">
                        ({item.id === "baseline"
                          ? "1.00x Ref"
                          : `${item.speedupVsBaseline ? item.speedupVsBaseline.toFixed(2) : "1.00"}x`})
                      </span>
                    </div>
                  ) : null}
                </div>

                {/* Bar Track */}
                <div className="relative h-8 w-full rounded-md border border-[#EAEAEA] bg-[#F2F2F2] overflow-hidden">
                  {isFailed ? (
                    <div className="flex h-full w-full items-center gap-2 rounded bg-[#FFF0F0] px-3 font-mono text-[12px] text-[#C81E1E]">
                      <AlertCircle className="h-4 w-4 shrink-0 text-[#E00000]" />
                      <span>Dropped — WGSL Compile Error (See Diagnostics Below)</span>
                    </div>
                  ) : isVerFailed ? (
                    <div className="flex h-full w-full items-center gap-2 rounded bg-[#FFF0F0] px-3 font-mono text-[12px] text-[#C81E1E]">
                      <AlertCircle className="h-4 w-4 shrink-0 text-[#E00000]" />
                      <span>Verification Failed (&gt; 1e-4 tolerance)</span>
                    </div>
                  ) : isBusy ? (
                    <div className="flex h-full w-full items-center px-3 font-mono text-[12px] text-[#0070F3] animate-pulse">
                      <span>{item.status}...</span>
                    </div>
                  ) : item.gflops ? (
                    <div
                      style={{ width: `${widthPct}%` }}
                      className={`flex h-full items-center justify-between rounded-sm px-3 transition-all duration-500 ${
                        isWinner
                          ? "bg-[#171717] text-[#FFFFFF] border-r-4 border-[#0070F3]"
                          : item.id === "baseline"
                          ? "bg-[#A3A3A3] text-[#FFFFFF]"
                          : "bg-[#666666] text-[#FFFFFF]"
                      }`}
                    >
                      <span className="font-mono text-[12.5px] font-semibold tracking-tight whitespace-nowrap tabular-nums">
                        {item.gflops.toFixed(1)} GFLOPS
                      </span>
                      <span className="font-mono text-[11px] font-medium whitespace-nowrap opacity-90">
                        {item.id === "baseline"
                          ? "1.00x"
                          : `${item.speedupVsBaseline?.toFixed(2) || "1.00"}x`}
                      </span>
                    </div>
                  ) : (
                    <div className="flex h-full w-full items-center px-3 font-mono text-[12px] text-[#888888]">
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
      <div className="border-t border-[#EAEAEA] pt-3 font-mono text-xs text-[#888888]">
        IEEE-754 FP32 Throughput • 3 Warmup + 9 Timed Dispatches • Dawn/wgpu Architecture
      </div>
    </div>
  );
};
