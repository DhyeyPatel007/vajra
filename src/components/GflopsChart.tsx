"use client";

import React from "react";
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
  const allItems: Array<VariantExecutionResult | null> = [
    baseline,
    variants.find((v) => v.id === "variant-a") || null,
    variants.find((v) => v.id === "variant-b") || null,
    variants.find((v) => v.id === "variant-c") || null,
  ];

  const validGflops = allItems
    .map((item) => item?.gflops || 0)
    .filter((g) => g > 0);
  const maxGflops = Math.max(10, ...validGflops) * 1.15; // 15% headroom

  const baselineGflops = baseline?.gflops || 0;

  // Chart dimensions for SVG
  const chartHeight = 240;
  const barWidth = 60;
  const gap = 38;
  const chartWidth = 4 * barWidth + 3 * gap + 80;
  const originX = 64;
  const originY = 195;
  const plotHeight = 155;

  return (
    <div className="flex h-full flex-col justify-between border border-[#242936] bg-[#0E1014]">
      {/* Top Header Strip */}
      <div className="flex flex-wrap items-center justify-between border-b border-[#242936] bg-[#0A0C0F] px-5 py-3">
        <div>
          <div className="font-mono text-[11px] font-bold tracking-widest text-[#E5B842]">
            01 // REAL PHYSICAL COMPUTE THROUGHPUT
          </div>
          <p className="mt-0.5 text-xs text-[#8E96A8]">
            Comparing Naive Baseline vs. Nemotron Optimized Variants (GFLOPS)
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 font-mono text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 bg-[#1F2430]"></span>
            <span className="text-[#8E96A8]">Baseline</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 bg-[#2E3646]"></span>
            <span className="text-[#8E96A8]">Verified</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 bg-[#E5B842]"></span>
            <span className="font-bold text-[#E5B842]">Winner</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 border border-[#F43F5E] bg-[#280B12]"></span>
            <span className="text-[#F43F5E]">Dropped</span>
          </div>
        </div>
      </div>

      {/* SVG Canvas inside recessed well */}
      <div className="p-4 sm:p-5">
        <div className="border border-[#242936] bg-[#0A0C0F] p-4">
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            className="h-60 w-full"
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
                    stroke="#181C26"
                    strokeWidth="1"
                    strokeDasharray={ratio > 0 ? "3 3" : undefined}
                  />
                  <text
                    x={originX - 10}
                    y={y + 4}
                    textAnchor="end"
                    fill="#5E6678"
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
                  stroke="#8A6D1F"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                />
                <text
                  x={chartWidth - 22}
                  y={originY - (baselineGflops / maxGflops) * plotHeight - 5}
                  textAnchor="end"
                  fill="#E5B842"
                  fontSize="9"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  Baseline: {baselineGflops.toFixed(1)} GFLOPS
                </text>
              </g>
            )}

            {/* Bars */}
            {allItems.map((item, index) => {
              const x = originX + index * (barWidth + gap) + 10;
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

              let barFill = "#2E3646";
              if (isWinner) barFill = "#E5B842";
              else if (isBaseline) barFill = "#1F2430";
              else if (!isFailed && !isPending && item?.verified) barFill = "#2E3646";

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
                  {/* Winner Label Above Bar */}
                  {isWinner && gflops > 0 && (
                    <g>
                      <rect
                        x={x - 10}
                        y={y - 26}
                        width={barWidth + 20}
                        height={18}
                        fill="#E5B842"
                      />
                      <text
                        x={x + barWidth / 2}
                        y={y - 14}
                        textAnchor="middle"
                        fill="#07080A"
                        fontSize="9"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        FASTEST WINNER
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
                        stroke={isWinner ? "#F5C542" : "#242936"}
                        strokeWidth={1}
                      />
                      {/* GFLOPS label on top */}
                      <text
                        x={x + barWidth / 2}
                        y={isWinner ? y - 32 : y - 6}
                        textAnchor="middle"
                        fill={isWinner ? "#E5B842" : "#F4F6F8"}
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
                          fill={isWinner ? "#07080A" : "#F4F6F8"}
                          fontSize="10"
                          fontWeight="bold"
                          fontFamily="monospace"
                        >
                          {item.speedupVsBaseline.toFixed(2)}x
                        </text>
                      )}
                    </g>
                  ) : isFailed ? (
                    /* Flat Crimson Hatched Dropped Slot */
                    <g>
                      <rect
                        x={x}
                        y={originY - 38}
                        width={barWidth}
                        height={38}
                        fill="#280B12"
                        stroke="#F43F5E"
                        strokeWidth="1"
                      />
                      <text
                        x={x + barWidth / 2}
                        y={originY - 22}
                        textAnchor="middle"
                        fill="#F43F5E"
                        fontSize="9"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        DROPPED
                      </text>
                      <text
                        x={x + barWidth / 2}
                        y={originY - 9}
                        textAnchor="middle"
                        fill="#8E96A8"
                        fontSize="8"
                        fontFamily="monospace"
                      >
                        {item?.status === "COMPILE_FAILED" ? "COMPILE FAULT" : "VERIFY FAULT"}
                      </text>
                    </g>
                  ) : (
                    /* Waiting / Pending placeholder */
                    <g>
                      <rect
                        x={x}
                        y={originY - 24}
                        width={barWidth}
                        height={24}
                        fill="#0E1014"
                        stroke="#242936"
                        strokeWidth="1"
                        strokeDasharray="2 2"
                      />
                      <text
                        x={x + barWidth / 2}
                        y={originY - 8}
                        textAnchor="middle"
                        fill="#5E6678"
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
                    fill={isWinner ? "#E5B842" : "#8E96A8"}
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
      </div>

      {/* Footer Notes */}
      <div className="flex flex-wrap items-center justify-between border-t border-[#242936] bg-[#0A0C0F] px-5 py-2.5 font-mono text-[11px] text-[#5E6678]">
        <span>Hardware metric: Median of 9 timed GPU passes</span>
        <span>Relative numerical tolerance: &lt; 1e-4</span>
      </div>
    </div>
  );
};
