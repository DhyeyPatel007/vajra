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
  const indexTags = ["W.01", "W.02", "W.03", "W.04"];

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
    <div className="rounded-lg border border-[#2A3542] bg-[#161D24] p-3">
      <div className="flex flex-col items-stretch justify-between gap-4 lg:flex-row lg:items-center">
        {/* 4 Workload Tabs in grid */}
        <div className="grid flex-1 grid-cols-2 gap-2 sm:grid-cols-4">
          {workloadIds.map((id) => {
            const spec = WORKLOADS[id];
            const isSelected = id === selectedId;

            return (
              <button
                key={id}
                onClick={() => !isRunning && onSelect(id)}
                disabled={isRunning}
                className={`flex flex-col justify-between rounded-md px-4 py-3 text-left transition-colors ${
                  isSelected
                    ? "border-2 border-[#F59E0B] bg-[#1E2730] text-[#F1F5F9]"
                    : "border border-[#2A3542] bg-[#11171D] text-[#94A3B8] hover:bg-[#1E2730] hover:text-[#F1F5F9]"
                } ${isRunning ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="text-[15px] font-semibold text-[#F1F5F9]">
                    {spec.shortName}
                  </span>
                  <span className="font-mono text-xs text-[#94A3B8]">
                    {spec.dimensionsLabel}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-1.5 font-mono text-[11px] text-[#64748B]">
                  <span>{formatFlops(spec.totalFlops)}</span>
                  <span>•</span>
                  <span>{formatBytes(spec.totalBytesMoved)}</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Action Triggers (Refine Winner Gen 2 + Tune on My GPU) */}
        <div className="flex shrink-0 flex-wrap items-center gap-3">
          {statusLabel && (
            <div className="font-mono text-xs text-[#F59E0B] animate-pulse">
              {statusLabel}
            </div>
          )}

          {/* Secondary CTA: Clean Outline Button */}
          <button
            onClick={onRefineGen2}
            disabled={isRunning}
            className={`rounded-md border border-[#F59E0B] bg-[#11171D] px-5 py-3 font-mono text-sm font-medium text-[#F59E0B] transition-colors hover:bg-[#1E2730] ${
              isRunning ? "cursor-not-allowed opacity-60" : "cursor-pointer active:translate-y-0.5"
            }`}
          >
            Refine Winner (Gen 2)
          </button>

          {/* Primary CTA: Solid Bright Terra-Amber Button */}
          <button
            onClick={onTune}
            disabled={isRunning}
            className={`flex items-center justify-center gap-2 rounded-md px-6 py-3 font-mono text-sm font-semibold transition-all ${
              isRunning
                ? "cursor-not-allowed bg-[#2A3542] text-[#94A3B8]"
                : "bg-[#F59E0B] text-[#0F1419] hover:bg-[#FBBF24] active:translate-y-0.5"
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-[#0F1419]" />
                <span>BENCHMARKING...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current text-[#0F1419]" />
                <span>Tune on My GPU</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
