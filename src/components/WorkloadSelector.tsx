"use client";

import React from "react";
import { Play, Loader2, Sparkles } from "lucide-react";
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

  const formatFlops = (flops: number): string => {
    if (flops >= 1e9) return `${(flops / 1e9).toFixed(1)}G FLOPs`;
    if (flops >= 1e6) return `${(flops / 1e6).toFixed(1)}M FLOPs`;
    return `${flops} FLOPs`;
  };

  const formatBytes = (bytes: number): string => {
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB moved`;
    return `${(bytes / 1024).toFixed(1)} KB moved`;
  };

  return (
    <div className="border-b border-[#262A34] bg-[#0B0C0E] px-4 py-3 sm:px-6">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4">
        {/* Tabs */}
        <div className="flex flex-wrap items-center gap-2">
          {workloadIds.map((id) => {
            const spec = WORKLOADS[id];
            const isSelected = id === selectedId;

            return (
              <button
                key={id}
                onClick={() => !isRunning && onSelect(id)}
                disabled={isRunning}
                className={`flex flex-col items-start rounded border px-3 py-2 text-left transition-all ${
                  isSelected
                    ? "border-[#D4AF37] bg-[#181B22] text-[#F3F4F6]"
                    : "border-[#262A34] bg-[#121418] text-[#9CA3AF] hover:border-[#383E4D] hover:text-[#F3F4F6]"
                } ${isRunning ? "opacity-60 cursor-not-allowed" : "cursor-pointer"}`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs font-bold ${
                      isSelected ? "text-[#D4AF37]" : "text-[#F3F4F6]"
                    }`}
                  >
                    {spec.shortName}
                  </span>
                  <span className="font-mono text-[11px] text-[#9CA3AF]">
                    {spec.dimensionsLabel}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-2 font-mono text-[10px] text-[#9CA3AF]">
                  <span>{formatFlops(spec.totalFlops)}</span>
                  <span>•</span>
                  <span>{formatBytes(spec.totalBytesMoved)}</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* CTA Button */}
        <div className="flex items-center gap-3">
          {statusLabel && (
            <span className="font-mono text-xs text-[#D4AF37] animate-pulse">
              {statusLabel}
            </span>
          )}
          <button
            onClick={onTune}
            disabled={isRunning}
            className={`flex items-center gap-2 rounded px-4 py-2.5 text-xs font-bold tracking-wide uppercase transition-all ${
              isRunning
                ? "bg-[#262A34] text-[#9CA3AF] cursor-not-allowed"
                : "bg-[#D4AF37] text-[#0B0C0E] hover:bg-[#F5C542] active:scale-95 shadow-sm"
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Benchmarking...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>Tune on My GPU</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
