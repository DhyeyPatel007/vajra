"use client";

import React from "react";
import { Cpu, AlertTriangle, Radio } from "lucide-react";
import { HardwareProbeInfo, VajraMode } from "@/types/vajra";

interface TopBarProps {
  probe: HardwareProbeInfo;
  mode?: VajraMode;
  onSelectMode?: (mode: VajraMode) => void;
  isMockMode?: boolean;
  onToggleMockMode?: (mock: boolean) => void;
  simulateCompileError: boolean;
  onToggleSimulateCompileError: (val: boolean) => void;
  previewNoWebGpu: boolean;
  onTogglePreviewNoWebGpu: (val: boolean) => void;
  liveModelSource?:
    | "mock"
    | "nebius-super"
    | "nebius-ultra-fallback"
    | "nvidia-super"
    | "nvidia-nano-fallback";
  usedFallbackModel?: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  probe,
  mode,
  onSelectMode,
  isMockMode = true,
  onToggleMockMode,
  simulateCompileError,
  onToggleSimulateCompileError,
  previewNoWebGpu,
  onTogglePreviewNoWebGpu,
  usedFallbackModel,
}) => {
  const isWebGpuActive = probe.supported && !previewNoWebGpu;
  const activeMode: VajraMode = mode || (isMockMode ? "mock" : "nebius");

  const handleSelectMode = (nextMode: VajraMode) => {
    if (onSelectMode) {
      onSelectMode(nextMode);
    } else if (onToggleMockMode) {
      onToggleMockMode(nextMode === "mock");
    }
  };

  return (
    <header className="w-full">
      {/* Vercel Light Header Bar */}
      <div className="w-full border-b border-[#EAEAEA] bg-[#FFFFFF]">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4 px-6 py-3.5 lg:px-8">
          {/* Left: Brand & Identity */}
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#171717] text-sm font-bold text-[#FFFFFF]">
              V
            </div>
            <div className="h-4 w-px bg-[#EAEAEA]" />
            <div className="flex items-center gap-2">
              <span className="text-[18px] font-semibold tracking-[-0.03em] text-[#171717]">
                VAJRA
              </span>
              <span className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] px-2 py-0.5 font-mono text-[11px] font-medium tracking-wide text-[#666666]">
                WEBGPU AUTO-TUNER
              </span>
              {activeMode === "mock" && (
                <span className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] px-2 py-0.5 font-mono text-[11px] font-medium tracking-wide text-[#666666]">
                  Mock Mode
                </span>
              )}
              {activeMode === "nebius" && (
                <span className="rounded-md border border-[#B8E8D0] bg-[#E6F6EE] px-2 py-0.5 font-mono text-[11px] font-semibold tracking-wide text-[#0A7143]">
                  Live Token Factory
                </span>
              )}
              {activeMode === "nvidia" && (
                <span className="rounded-md border border-[#93C5FD] bg-[#EFF6FF] px-2 py-0.5 font-mono text-[11px] font-semibold tracking-wide text-[#1D4ED8]">
                  Live NVIDIA
                </span>
              )}
              {usedFallbackModel && (
                <span className="rounded-md border border-[#FDE68A] bg-[#FFFBEB] px-2 py-0.5 font-mono text-[10px] font-bold text-[#9A5B13]">
                  FALLBACK ACTIVE
                </span>
              )}
            </div>
            <p className="hidden text-[13px] text-[#666666] lg:block ml-2 border-l border-[#EAEAEA] pl-3">
              Tunes per-device on any modern WebGPU browser, and falls back to recorded Preview where WebGPU is unavailable.
            </p>
          </div>

          {/* Right: Controls Cluster */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Adapter Status Pill */}
            <div className="flex items-center gap-2 rounded-full border border-[#EAEAEA] bg-[#FAFAFA] px-3 py-1 font-mono text-[12px] text-[#171717]">
              <span
                className={`h-2 w-2 rounded-full shrink-0 ${
                  isWebGpuActive ? "bg-[#10B981]" : "bg-[#F5A623]"
                }`}
              />
              <Cpu className="h-3.5 w-3.5 text-[#666666]" />
              <span className="max-w-[140px] truncate sm:max-w-[200px] font-medium">
                {isWebGpuActive ? probe.adapterDescription : "Apple M3 Max (Reference)"}
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                  !isWebGpuActive
                    ? "border border-[#FDE68A] bg-[#FFFBEB] text-[#9A5B13]"
                    : probe.supportsTimestampQuery
                    ? "border border-[#B8E8D0] bg-[#E6F6EE] text-[#0A7143]"
                    : "border border-[#EAEAEA] bg-[#F2F2F2] text-[#666666]"
                }`}
              >
                {!isWebGpuActive
                  ? "RECORDED PREVIEW"
                  : probe.supportsTimestampQuery
                  ? "TIMESTAMP QUERIES: ON"
                  : "WALL-CLOCK FALLBACK"}
              </span>
            </div>

            {/* Segmented Mode Switch */}
            <div className="flex items-center rounded-md border border-[#EAEAEA] bg-[#F2F2F2] p-0.5 font-mono">
              <button
                type="button"
                onClick={() => handleSelectMode("mock")}
                className={`rounded-[5px] px-3 py-1 text-[12px] font-medium transition-all cursor-pointer ${
                  activeMode === "mock"
                    ? "bg-[#FFFFFF] text-[#171717] shadow-[0_1px_2px_rgba(0,0,0,0.06)] font-semibold"
                    : "text-[#666666] hover:text-[#171717]"
                }`}
                title="Runs verified local mock kernels with zero API calls"
              >
                Mock Mode
              </button>
              <button
                type="button"
                onClick={() => handleSelectMode("nebius")}
                className={`flex items-center gap-1.5 rounded-[5px] px-3 py-1 text-[12px] font-medium transition-all cursor-pointer ${
                  activeMode === "nebius"
                    ? "bg-[#FFFFFF] text-[#171717] shadow-[0_1px_2px_rgba(0,0,0,0.06)] font-semibold"
                    : "text-[#666666] hover:text-[#171717]"
                }`}
                title="Queries NVIDIA Nemotron via Nebius Token Factory with automatic Ultra fallback"
              >
                <Radio className="h-3 w-3" />
                <span>Live Token Factory</span>
              </button>
              <button
                type="button"
                onClick={() => handleSelectMode("nvidia")}
                className={`flex items-center gap-1.5 rounded-[5px] px-3 py-1 text-[12px] font-medium transition-all cursor-pointer ${
                  activeMode === "nvidia"
                    ? "bg-[#FFFFFF] text-[#171717] shadow-[0_1px_2px_rgba(0,0,0,0.06)] font-semibold"
                    : "text-[#666666] hover:text-[#171717]"
                }`}
                title="Queries NVIDIA Nemotron via NVIDIA NIM API directly"
              >
                <Radio className="h-3 w-3" />
                <span>Live NVIDIA</span>
              </button>
            </div>

            {/* Simulate Compile Fail Checkbox */}
            <label className="flex cursor-pointer items-center gap-2 rounded-md border border-[#EAEAEA] bg-[#FFFFFF] px-3 py-1.5 text-[12px] font-medium text-[#171717] transition-colors hover:border-[#999999]">
              <input
                type="checkbox"
                checked={simulateCompileError}
                onChange={(e) => onToggleSimulateCompileError(e.target.checked)}
                className="h-3.5 w-3.5 accent-[#171717] rounded cursor-pointer"
              />
              <span className="font-mono text-[11px] uppercase tracking-wider">Simulate Compile Fail</span>
            </label>

            {/* Preview No-WebGPU / Exit Reference Mode Button */}
            <button
              onClick={() => onTogglePreviewNoWebGpu(!previewNoWebGpu)}
              className={`rounded-md border px-3 py-1.5 font-mono text-[12px] font-medium uppercase tracking-wider transition-colors ${
                previewNoWebGpu
                  ? "border-[#FDE68A] bg-[#FFFBEB] text-[#9A5B13] hover:bg-[#FEF3C7]"
                  : "border-[#EAEAEA] bg-[#FFFFFF] text-[#171717] hover:border-[#999999] hover:bg-[#FAFAFA]"
              }`}
              title="Simulates WebGPU being unavailable on this system to preview recorded reference benchmarks"
            >
              {previewNoWebGpu ? "Exit Reference Mode" : "Preview No-WebGPU"}
            </button>
          </div>
        </div>
      </div>

      {/* Reference Mode Banner */}
      {(!probe.supported || previewNoWebGpu) && (
        <div className="w-full border-b border-[#FDE68A] bg-[#FFFBEB]">
          <div className="mx-auto flex max-w-[1400px] items-center gap-2.5 px-6 py-2 lg:px-8 font-mono text-[12px] text-[#9A5B13]">
            <AlertTriangle className="h-4 w-4 shrink-0 text-[#9A5B13]" />
            <span className="font-semibold">
              {previewNoWebGpu
                ? "PREVIEW MODE (RECORDED REFERENCE — NOT LIVE GPU RUN)"
                : "WebGPU Unavailable on this Browser/Device — Showing Recorded Reference (NOT Live GPU Run)"}
            </span>
            <span className="text-[#666666]">
              • Apple M3 Max 40-core GPU / Chrome 142 Recorded Baseline
            </span>
          </div>
        </div>
      )}
    </header>
  );
};
