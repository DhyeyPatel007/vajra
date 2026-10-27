"use client";

import React from "react";
import { Play, Loader2 } from "lucide-react";
import { WorkloadId } from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";

interface WorkloadSelectorProps {
  selectedId: WorkloadId;
  onSelect: (id: WorkloadId) => void;
  onTune: () => void;
  onRefineGen2?: () => void;
  isRunning: boolean;
  statusLabel?: string;
  hasGen2Run?: boolean;
}

export const WorkloadSelector: React.FC<WorkloadSelectorProps> = ({
  selectedId,
  onSelect,
  onTune,
  onRefineGen2,
  isRunning,
  statusLabel,
}) => {
  const workloadIds: WorkloadId[] = ["matmul", "softmax", "conv1d", "sobel"];
  const activeSpec = WORKLOADS[selectedId] || WORKLOADS.matmul;

  const formatFlops = (flops: number): string => {
    if (flops >= 1e9) return `${(flops / 1e9).toFixed(1)}G FLOPs`;
    if (flops >= 1e6) return `${(flops / 1e6).toFixed(1)}M FLOPs`;
    return `${flops} FLOPs`;
  };

  const formatBytes = (bytes: number): string => {
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  };

  return (
    <div className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-4 shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
      {/* Row 1: Controls Row — ONLY the 4 tabs + Action Buttons (No status label here) */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* 4 Workload Tabs in grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 flex-1 min-w-0">
          {workloadIds.map((id) => {
            const spec = WORKLOADS[id];
            const isSelected = id === selectedId;

            return (
              <button
                key={id}
                onClick={() => !isRunning && onSelect(id)}
                disabled={isRunning}
                className={`rounded-md px-3.5 py-2.5 text-left transition-all border min-w-0 ${
                  isSelected
                    ? "bg-[#FAFAFA] border-[#171717] ring-1 ring-[#171717] text-[#171717]"
                    : "bg-[#FFFFFF] border-[#EAEAEA] text-[#666666] hover:border-[#999999] hover:text-[#171717]"
                } ${isRunning ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="text-[14px] font-semibold text-[#171717]">
                    {spec.shortName}
                  </span>
                  <span className="font-mono text-[11.5px] text-[#666666] truncate">
                    {spec.dimensionsLabel}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-1.5 font-mono text-[11px] text-[#888888]">
                  <span>{formatFlops(spec.totalFlops)}</span>
                  <span>•</span>
                  <span>{formatBytes(spec.totalBytesMoved)}</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Action Buttons Cluster */}
        <div className="flex items-center justify-end gap-2.5 shrink-0">
          {/* Secondary CTA: Refine Winner (Gen 2) */}
          <button
            onClick={onRefineGen2}
            disabled={isRunning}
            className={`rounded-md border border-[#0070F3] bg-[#FFFFFF] px-4 py-2.5 font-mono text-[13px] font-medium text-[#0070F3] whitespace-nowrap transition-colors hover:bg-[#F0F7FF] ${
              isRunning ? "cursor-not-allowed opacity-50" : "cursor-pointer active:translate-y-0.5"
            }`}
          >
            Refine Winner (Gen 2)
          </button>

          {/* Primary CTA: Tune on My GPU */}
          <button
            onClick={onTune}
            disabled={isRunning}
            className={`flex items-center justify-center gap-2 rounded-md px-5 py-2.5 font-mono text-[13px] font-medium whitespace-nowrap transition-colors border ${
              isRunning
                ? "cursor-not-allowed bg-[#F2F2F2] text-[#999999] border-[#EAEAEA]"
                : "bg-[#171717] hover:bg-[#333333] active:bg-[#000000] text-[#FFFFFF] border-[#171717] cursor-pointer"
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-[#999999]" />
                <span>BENCHMARKING...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current text-[#FFFFFF]" />
                <span>Tune on My GPU</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Row 2: MANDATORY Dedicated Full-Width Pipeline Status Strip */}
      <div className="mt-3 w-full min-h-[38px] px-3.5 py-2 rounded-md bg-[#FAFAFA] border border-[#EAEAEA] flex flex-wrap items-center justify-between gap-2 font-mono text-[12px] text-[#666666]">
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full shrink-0 ${
              isRunning ? "bg-[#0070F3] animate-pulse" : "bg-[#10B981]"
            }`}
          />
          {statusLabel ? (
            <span className={isRunning ? "font-medium text-[#171717]" : "text-[#666666]"}>
              {statusLabel}
            </span>
          ) : (
            <span>Pipeline Ready — Select a workload or click Tune on My GPU</span>
          )}
        </div>
        <div className="text-[11px] text-[#888888] whitespace-nowrap">
          {activeSpec.name}: {formatFlops(activeSpec.totalFlops)} • {formatBytes(activeSpec.totalBytesMoved)}
        </div>
      </div>
    </div>
  );
};
