"use client";

import React from "react";
import { Cpu, AlertTriangle, Radio } from "lucide-react";
import { HardwareProbeInfo } from "@/types/vajra";

interface TopBarProps {
  probe: HardwareProbeInfo;
  isMockMode: boolean;
  onToggleMockMode: (mock: boolean) => void;
  simulateCompileError: boolean;
  onToggleSimulateCompileError: (val: boolean) => void;
  previewNoWebGpu: boolean;
  onTogglePreviewNoWebGpu: (val: boolean) => void;
  liveModelSource?: "mock" | "nebius-super" | "nebius-ultra-fallback";
  usedFallbackModel?: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  probe,
  isMockMode,
  onToggleMockMode,
  simulateCompileError,
  onToggleSimulateCompileError,
  previewNoWebGpu,
  onTogglePreviewNoWebGpu,
  usedFallbackModel,
}) => {
  const isWebGpuActive = probe.supported && !previewNoWebGpu;

  return (
    <header className="space-y-4">
      {/* Clean Spacious Horizontal Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-[#2A3542] bg-[#161D24] px-6 py-4 text-[#F1F5F9]">
        {/* Left: Brand & Identity */}
        <div className="flex items-center gap-3.5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[#F59E0B] text-lg font-bold text-[#0F1419]">
            V
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-tight text-[#F1F5F9]">
                VAJRA
              </span>
              <span className="rounded border border-[#2A3542] bg-[#11171D] px-2 py-0.5 font-mono text-[11px] font-semibold tracking-wider text-[#94A3B8]">
                WEBGPU AUTO-TUNER
              </span>
              {usedFallbackModel && (
                <span className="rounded bg-[#F59E0B] px-2 py-0.5 font-mono text-[10px] font-bold text-[#0F1419]">
                  ULTRA FALLBACK ACTIVE
                </span>
              )}
            </div>
            <p className="hidden text-xs text-[#94A3B8] sm:block">
              WebGPU Kernel Auto-Tuner — Verify &amp; benchmark Nemotron WGSL on your hardware
            </p>
          </div>
        </div>

        {/* Right: Compact Control Cluster */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Hardware Adapter Pill */}
          <div className="flex items-center gap-2 rounded-md border border-[#2A3542] bg-[#11171D] px-3 py-1.5 font-mono text-xs text-[#F1F5F9]">
            <Cpu className="h-3.5 w-3.5 text-[#F59E0B]" />
            <span className="max-w-[140px] truncate sm:max-w-[200px]">
              {isWebGpuActive ? probe.adapterDescription : "Apple M3 Max (Reference)"}
            </span>
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
                !isWebGpuActive
                  ? "bg-[#291F0E] text-[#F59E0B]"
                  : probe.supportsTimestampQuery
                  ? "bg-[#0B261F] text-[#10B981]"
                  : "bg-[#1E2730] text-[#94A3B8]"
              }`}
            >
              {!isWebGpuActive
                ? "RECORDED"
                : probe.supportsTimestampQuery
                ? "TIMESTAMP QUERIES: ON"
                : "WALL-CLOCK FALLBACK"}
            </span>
          </div>

          {/* Mode Segmented Toggle */}
          <div className="flex rounded-md border border-[#2A3542] bg-[#11171D] p-1">
            <button
              onClick={() => onToggleMockMode(true)}
              className={`rounded px-3 py-1.5 font-mono text-xs font-medium transition-colors ${
                isMockMode
                  ? "border border-[#F59E0B]/40 bg-[#1E2730] text-[#F59E0B]"
                  : "text-[#94A3B8] hover:text-[#F1F5F9]"
              }`}
              title="Runs verified local mock kernels with zero API calls"
            >
              Mock Mode
            </button>
            <button
              onClick={() => onToggleMockMode(false)}
              className={`flex items-center gap-1.5 rounded px-3 py-1.5 font-mono text-xs font-medium transition-colors ${
                !isMockMode
                  ? "border border-[#F59E0B]/40 bg-[#1E2730] text-[#F59E0B]"
                  : "text-[#94A3B8] hover:text-[#F1F5F9]"
              }`}
              title="Queries NVIDIA Nemotron via Nebius Token Factory with automatic Ultra fallback"
            >
              <Radio className="h-3 w-3" />
              <span>Live Token Factory</span>
            </button>
          </div>

          {/* Secondary Controls: Simulate Compile Fail & Reference Preview */}
          <label className="flex cursor-pointer items-center gap-2 rounded-md border border-[#2A3542] bg-[#11171D] px-3 py-1.5 font-mono text-xs text-[#94A3B8] transition-colors hover:border-[#F59E0B]/40 hover:text-[#F1F5F9]">
            <input
              type="checkbox"
              checked={simulateCompileError}
              onChange={(e) => onToggleSimulateCompileError(e.target.checked)}
              className="h-3.5 w-3.5 accent-[#F59E0B]"
            />
            <span className="text-[11px] uppercase tracking-wider">Simulate Compile Fail</span>
          </label>

          <button
            onClick={() => onTogglePreviewNoWebGpu(!previewNoWebGpu)}
            className={`rounded-md border px-3 py-1.5 font-mono text-xs font-medium uppercase tracking-wider transition-colors ${
              previewNoWebGpu
                ? "border-[#F59E0B] bg-[#F59E0B] text-[#0F1419]"
                : "border-[#2A3542] bg-[#11171D] text-[#94A3B8] hover:border-[#F59E0B]/40 hover:text-[#F1F5F9]"
            }`}
            title="Simulates WebGPU being unavailable on this system to preview recorded reference benchmarks"
          >
            {previewNoWebGpu ? "Exit Reference Mode" : "Preview No-WebGPU"}
          </button>
        </div>
      </div>

      {/* Fallback Reference Mode Banner */}
      {(!probe.supported || previewNoWebGpu) && (
        <div className="flex items-center gap-2.5 rounded-lg border border-[#F59E0B]/50 bg-[#291F0E] px-4 py-2.5 font-mono text-xs text-[#F59E0B]">
          <AlertTriangle className="h-4 w-4 shrink-0 text-[#F59E0B]" />
          <span className="font-bold">
            {previewNoWebGpu
              ? "PREVIEW MODE: Showing Recorded Reference Benchmark"
              : "WebGPU Unavailable on this Browser/Device — Showing Recorded Reference Benchmark"}
          </span>
          <span className="text-[#94A3B8]">
            (Apple M3 Max 40-Core GPU / Chrome 142 Reference Datasets)
          </span>
        </div>
      )}
    </header>
  );
};
