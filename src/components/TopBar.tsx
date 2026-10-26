"use client";

import React from "react";
import {
  Cpu,
  Clock,
  Zap,
  AlertTriangle,
  Flame,
  Radio,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { HardwareProbeInfo, TimingMethod } from "@/types/vajra";

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
  liveModelSource,
  usedFallbackModel,
}) => {
  const isWebGpuActive = probe.supported && !previewNoWebGpu;

  return (
    <header className="border-b border-[#262A34] bg-[#121418] px-4 py-3 sm:px-6">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4">
        {/* Left: Brand + Tagline */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded border border-[#D4AF37] bg-[#181B22] text-[#D4AF37]">
            <Zap className="h-5 w-5 fill-[#D4AF37]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-wider text-[#D4AF37]">
                VAJRA
              </span>
              <span className="rounded bg-[#181B22] px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-[#9CA3AF] border border-[#262A34]">
                WebGPU AUTO-TUNER
              </span>
              {usedFallbackModel && (
                <span className="rounded bg-[#D4AF37] px-2 py-0.5 text-[10px] font-bold text-[#0B0C0E]">
                  ULTRA FALLBACK ACTIVE
                </span>
              )}
            </div>
            <p className="hidden text-xs text-[#9CA3AF] sm:block">
              AI writes GPU shaders, but only your hardware knows the truth.
            </p>
          </div>
        </div>

        {/* Right: Hardware telemetry pill & Mode controls */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* Hardware Badge */}
          <div className="flex items-center gap-2 rounded border border-[#262A34] bg-[#0B0C0E] px-2.5 py-1 text-xs">
            <Cpu className="h-3.5 w-3.5 text-[#D4AF37]" />
            <span className="max-w-[180px] truncate font-mono text-[#F3F4F6] sm:max-w-[260px]">
              {isWebGpuActive ? probe.adapterDescription : "Apple M3 Max (Reference)"}
            </span>
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-mono font-semibold ${
                !isWebGpuActive
                  ? "bg-[#262A34] text-[#D4AF37]"
                  : probe.supportsTimestampQuery
                  ? "bg-[#10B981] text-[#0B0C0E]"
                  : "bg-[#262A34] text-[#9CA3AF]"
              }`}
            >
              {!isWebGpuActive
                ? "RECORDED"
                : probe.supportsTimestampQuery
                ? "TIMESTAMPS: ON"
                : "WALL-CLOCK"}
            </span>
          </div>

          {/* Mode Switch (Mock vs Live) */}
          <div className="flex items-center rounded border border-[#262A34] bg-[#0B0C0E] p-0.5 text-xs">
            <button
              onClick={() => onToggleMockMode(true)}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors ${
                isMockMode
                  ? "bg-[#D4AF37] font-semibold text-[#0B0C0E]"
                  : "text-[#9CA3AF] hover:text-[#F3F4F6]"
              }`}
              title="Runs verified local mock kernels with zero API calls"
            >
              <span>Mock Mode</span>
            </button>
            <button
              onClick={() => onToggleMockMode(false)}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors ${
                !isMockMode
                  ? "bg-[#D4AF37] font-semibold text-[#0B0C0E]"
                  : "text-[#9CA3AF] hover:text-[#F3F4F6]"
              }`}
              title="Queries NVIDIA Nemotron via Nebius Token Factory with automatic Ultra fallback"
            >
              <Radio className="h-3 w-3" />
              <span>Live Token Factory</span>
            </button>
          </div>

          {/* Test Toggles */}
          <div className="flex items-center gap-2">
            <label className="flex cursor-pointer items-center gap-1.5 rounded border border-[#262A34] bg-[#181B22] px-2 py-1 text-xs text-[#9CA3AF] hover:border-[#D4AF37] hover:text-[#F3F4F6]">
              <input
                type="checkbox"
                checked={simulateCompileError}
                onChange={(e) => onToggleSimulateCompileError(e.target.checked)}
                className="h-3.5 w-3.5 accent-[#D4AF37]"
              />
              <span className="text-[11px]">Simulate Compile Fail</span>
            </label>

            <button
              onClick={() => onTogglePreviewNoWebGpu(!previewNoWebGpu)}
              className={`rounded border px-2 py-1 text-[11px] font-medium transition-colors ${
                previewNoWebGpu
                  ? "border-[#D4AF37] bg-[#D4AF37] text-[#0B0C0E]"
                  : "border-[#262A34] bg-[#181B22] text-[#9CA3AF] hover:text-[#F3F4F6]"
              }`}
              title="Simulates WebGPU being unavailable on this system to preview recorded reference benchmarks"
            >
              {previewNoWebGpu ? "Exit Reference Mode" : "Preview No-WebGPU"}
            </button>
          </div>
        </div>
      </div>

      {/* Fallback Reference Mode Banner */}
      {(!probe.supported || previewNoWebGpu) && (
        <div className="mt-3 rounded border border-[#D4AF37] bg-[#181B22] px-4 py-2.5 text-xs text-[#F3F4F6]">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="h-4 w-4 shrink-0 text-[#D4AF37]" />
            <span className="font-semibold text-[#D4AF37]">
              {previewNoWebGpu
                ? "PREVIEW MODE: Showing Recorded Reference Benchmark"
                : "WebGPU Unavailable on this Browser/Device — Showing Recorded Reference Benchmark"}
            </span>
            <span className="text-[#9CA3AF]">
              (Apple M3 Max 40-Core GPU / Chrome 142 Reference Datasets)
            </span>
          </div>
        </div>
      )}
    </header>
  );
};
