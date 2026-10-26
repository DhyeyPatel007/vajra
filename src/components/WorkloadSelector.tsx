"use client";

import React from "react";
import { Play, Loader2 } from "lucide-react";
import { WorkloadId } from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";

interface WorkloadSelectorProps {
  selectedId: WorkloadId;
  onSelect: (id: WorkloadId) => void;
  onTune: () => void;
  isRunning: boolean;
  statusLabel?: string;
}

export const WorkloadSelector: React.FC<WorkloadSelectorProps> = ({
  selectedId,
  onSelect,
  onTune,
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
    <div className="border-b border-[#242936] bg-[#07080A] px-4 py-3 sm:px-6">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        {/* 4 Workload Cells in 1px Hairline Grid */}
        <div className="grid flex-1 grid-cols-2 divide-x divide-y divide-[#242936] border border-[#242936] bg-[#0E1014] sm:grid-cols-4 sm:divide-y-0">
          {workloadIds.map((id, index) => {
            const spec = WORKLOADS[id];
            const isSelected = id === selectedId;

            return (
              <button
                key={id}
                onClick={() => !isRunning && onSelect(id)}
                disabled={isRunning}
                className={`relative flex flex-col justify-between p-3.5 text-left transition-colors ${
                  isSelected
                    ? "bg-[#151921] text-[#F4F6F8]"
                    : "bg-[#0E1014] text-[#8E96A8] hover:bg-[#12151C] hover:text-[#F4F6F8]"
                } ${isRunning ? "opacity-60 cursor-not-allowed" : "cursor-pointer"}`}
              >
                {/* Active Top Accent Line */}
                {isSelected && (
                  <span className="absolute inset-x-0 top-0 h-[3px] bg-[#E5B842]" />
                )}

                <div className="flex items-center justify-between gap-1">
                  <span className={`font-mono text-[10px] font-bold tracking-wider ${
                    isSelected ? "text-[#E5B842]" : "text-[#5E6678]"
                  }`}>
                    {indexTags[index]}
                  </span>
                  <span className="font-mono text-[11px] text-[#8E96A8]">
                    {spec.dimensionsLabel}
                  </span>
                </div>

                <div className="mt-1 font-sans text-sm font-bold text-[#F4F6F8]">
                  {spec.shortName}
                </div>

                <div className="mt-1 flex items-center gap-1.5 font-mono text-[10px] text-[#8E96A8]">
                  <span>{formatFlops(spec.totalFlops)}</span>
                  <span>•</span>
                  <span>{formatBytes(spec.totalBytesMoved)}</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Primary Action Button (Tune on My GPU) */}
        <div className="flex shrink-0 items-center gap-3">
          {statusLabel && (
            <div className="font-mono text-xs text-[#E5B842] animate-pulse">
              {statusLabel}
            </div>
          )}
          <button
            onClick={onTune}
            disabled={isRunning}
            className={`flex items-center justify-center gap-2 px-6 py-4 font-mono text-xs font-bold uppercase tracking-[0.12em] transition-all ${
              isRunning
                ? "bg-[#242936] text-[#8E96A8] cursor-not-allowed"
                : "bg-[#E5B842] text-[#07080A] hover:bg-[#F5C542] active:translate-y-0.5"
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-[#07080A]" />
                <span>BENCHMARKING...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current text-[#07080A]" />
                <span>TUNE ON MY GPU</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
