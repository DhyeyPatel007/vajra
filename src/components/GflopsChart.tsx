"use client";

import React from "react";
import { Award, AlertOctagon, CheckCircle2, Zap } from "lucide-react";
import { VariantExecutionResult } from "@/types/vajra";

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
  // Combine baseline + variants into 4 ordered slots
  const allItems: Array<VariantExecutionResult | null> = [
    baseline,
    variants.find((v) => v.id === "variant-a") || null,
    variants.find((v) => v.id === "variant-b") || null,
    variants.find((v) => v.id === "variant-c") || null,
  ];

  // Calculate max GFLOPS for scaling
  const validGflops = allItems
    .map((item) => item?.gflops || 0)
    .filter((g) => g > 0);
  const maxGflops = Math.max(10, ...validGflops) * 1.18; // 18% headroom

  const baselineGflops = baseline?.gflops || 0;

  // Chart dimensions for SVG
  const chartHeight = 240;
  const barWidth = 56;
  const gap = 38;
  const chartWidth = 4 * barWidth + 3 * gap + 80;
  const originX = 60;
  const originY = 200;
  const plotHeight = 160;

  return (
    <div className="flex h-full flex-col justify-between rounded border border-[#262A34] bg-[#121418] p-5">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#262A34] pb-3">
        <div>
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-[#D4AF37]" />
            <h2 className="text-sm font-bold tracking-wide uppercase text-[#F3F4F6]">
              Real Physical Compute Throughput
            </h2>
          </div>
          <p className="mt-0.5 text-xs text-[#9CA3AF]">
            Comparing Naive Baseline vs. Nemotron Optimized Variants (GFLOPS)
          </p>
        </div>

        {/* Legend */}
        <div className="hidden items-center gap-3 text-[11px] sm:flex">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[#383E4D]"></span>
            <span className="text-[#9CA3AF]">Baseline</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[#6B7280]"></span>
            <span className="text-[#9CA3AF]">Verified</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[#D4AF37]"></span>
            <span className="font-semibold text-[#D4AF37]">Fastest Winner</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm border border-[#EF4444] bg-[#262A34]"></span>
            <span className="text-[#EF4444]">Dropped</span>
          </div>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative my-4 flex w-full items-center justify-center overflow-x-auto">
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="h-64 w-full max-w-xl"
        >
          {/* Background Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const y = originY - ratio * plotHeight;
            const val = (ratio * maxGflops).toFixed(0);
            return (
              <g key={ratio}>
                <line
                  x1={originX}
                  y1={y}
                  x2={chartWidth - 20}
                  y2={y}
                  stroke="#1C1F26"
                  strokeWidth="1"
                  strokeDasharray={ratio > 0 ? "3 3" : undefined}
                />
                <text
                  x={originX - 8}
                  y={y + 4}
                  textAnchor="end"
                  fill="#6B7280"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Baseline Reference Dash Line */}
          {baselineGflops > 0 && (
            <g>
              <line
                x1={originX}
                y1={originY - (baselineGflops / maxGflops) * plotHeight}
                x2={chartWidth - 20}
                y2={originY - (baselineGflops / maxGflops) * plotHeight}
                stroke="#D4AF37"
                strokeWidth="1.5"
                strokeDasharray="4 4"
                opacity="0.6"
              />
              <text
                x={chartWidth - 22}
                y={originY - (baselineGflops / maxGflops) * plotHeight - 4}
                textAnchor="end"
                fill="#D4AF37"
                fontSize="9"
                fontFamily="monospace"
                opacity="0.9"
              >
                Baseline: {baselineGflops.toFixed(1)} GFLOPS
              </text>
            </g>
          )}

          {/* Bars */}
          {allItems.map((item, index) => {
            const x = originX + index * (barWidth + gap) + 12;
            const gflops = item?.gflops || 0;
            const barHeight =
              maxGflops > 0 && gflops > 0
                ? (gflops / maxGflops) * plotHeight
                : 0;
            const y = originY - barHeight;

            const isBaseline = index === 0;
            const isWinner = item?.isWinner ?? false;
            const isFailed =
              item?.status === "COMPILE_FAILED" ||
              item?.status === "VERIFY_FAILED";
            const isPending =
              !item ||
              item.status === "IDLE" ||
              item.status === "GENERATING" ||
              item.status === "COMPILING" ||
              item.status === "VERIFYING" ||
              item.status === "BENCHMARKING";

            let barFill = "#383E4D";
            if (isWinner) barFill = "#D4AF37";
            else if (isBaseline) barFill = "#262A34";
            else if (!isFailed && !isPending && item?.verified) barFill = "#4B5563";

            const label =
              index === 0
                ? "Baseline"
                : index === 1
                ? "Variant A"
                : index === 2
                ? "Variant B"
                : "Variant C";

            return (
              <g key={index}>
                {/* Winner Crown / Badge Above Bar */}
                {isWinner && gflops > 0 && (
                  <g>
                    <rect
                      x={x - 14}
                      y={y - 28}
                      width={barWidth + 28}
                      height={18}
                      fill="#D4AF37"
                      rx="2"
                    />
                    <text
                      x={x + barWidth / 2}
                      y={y - 16}
                      textAnchor="middle"
                      fill="#0B0C0E"
                      fontSize="9"
                      fontWeight="bold"
                      fontFamily="monospace"
                    >
                      FASTEST GPU
                    </text>
                  </g>
                )}

                {/* Bar Body */}
                {!isFailed && !isPending && gflops > 0 ? (
                  <g>
                    <rect
                      x={x}
                      y={y}
                      width={barWidth}
                      height={barHeight}
                      fill={barFill}
                      rx="2"
                      stroke={isWinner ? "#F5C542" : "#383E4D"}
                      strokeWidth={isWinner ? 2 : 1}
                    />
                    {/* GFLOPS label on top of bar */}
                    <text
                      x={x + barWidth / 2}
                      y={isWinner ? y - 34 : y - 6}
                      textAnchor="middle"
                      fill={isWinner ? "#D4AF37" : "#F3F4F6"}
                      fontSize="11"
                      fontWeight="bold"
                      fontFamily="monospace"
                    >
                      {gflops.toFixed(1)}
                    </text>
                    {/* Speedup label */}
                    {item?.speedupVsBaseline && item.speedupVsBaseline > 1.05 && (
                      <text
                        x={x + barWidth / 2}
                        y={Math.min(originY - 8, y + 16)}
                        textAnchor="middle"
                        fill={isWinner ? "#0B0C0E" : "#F3F4F6"}
                        fontSize="10"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        {item.speedupVsBaseline.toFixed(2)}x
                      </text>
                    )}
                  </g>
                ) : isFailed ? (
                  /* Failed / Dropped Slot */
                  <g>
                    <rect
                      x={x}
                      y={originY - 40}
                      width={barWidth}
                      height={40}
                      fill="#181B22"
                      stroke="#EF4444"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                      rx="2"
                    />
                    <text
                      x={x + barWidth / 2}
                      y={originY - 22}
                      textAnchor="middle"
                      fill="#EF4444"
                      fontSize="9"
                      fontWeight="bold"
                      fontFamily="monospace"
                    >
                      DROPPED
                    </text>
                    <text
                      x={x + barWidth / 2}
                      y={originY - 10}
                      textAnchor="middle"
                      fill="#9CA3AF"
                      fontSize="8"
                      fontFamily="monospace"
                    >
                      {item?.status === "COMPILE_FAILED" ? "Syntax" : "Verify"}
                    </text>
                  </g>
                ) : (
                  /* Loading / Pending placeholder */
                  <g>
                    <rect
                      x={x}
                      y={originY - 24}
                      width={barWidth}
                      height={24}
                      fill="#181B22"
                      stroke="#262A34"
                      strokeWidth="1"
                      strokeDasharray="2 2"
                      rx="2"
                    />
                    <text
                      x={x + barWidth / 2}
                      y={originY - 8}
                      textAnchor="middle"
                      fill="#6B7280"
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      {item?.status || "Waiting"}
                    </text>
                  </g>
                )}

                {/* X-Axis Category Label */}
                <text
                  x={x + barWidth / 2}
                  y={originY + 16}
                  textAnchor="middle"
                  fill={isWinner ? "#D4AF37" : "#9CA3AF"}
                  fontSize="11"
                  fontWeight={isWinner ? "bold" : "normal"}
                  fontFamily="monospace"
                >
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Footer Notes */}
      <div className="flex flex-wrap items-center justify-between border-t border-[#262A34] pt-2 text-[11px] text-[#9CA3AF]">
        <span>Hardware metric: Median of 9 timed GPU passes</span>
        <span>Relative numerical tolerance: &lt; 1e-4</span>
      </div>
    </div>
  );
};
