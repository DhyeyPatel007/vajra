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
    <header className="border-b border-[#242936] bg-[#0E1014] text-[#F4F6F8]">
      {/* 4-Cell Industrial Instrument Command Strip */}
      <div className="mx-auto flex max-w-7xl flex-wrap items-stretch divide-y divide-[#242936] border-x border-[#242936] lg:divide-y-0 lg:divide-x">
        {/* Cell 1: Brand & Identity */}
        <div className="flex flex-1 items-center gap-3.5 p-3.5 sm:px-5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center bg-[#E5B842] text-[#07080A] font-mono text-base font-extrabold tracking-tighter">
            V
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-sans text-xl font-bold tracking-[0.12em] text-[#E5B842]">
                VAJRA
              </span>
              <span className="border border-[#242936] bg-[#0A0C0F] px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wider text-[#8E96A8]">
                WEBGPU AUTO-TUNER
              </span>
              {usedFallbackModel && (
                <span className="bg-[#E5B842] px-2 py-0.5 font-mono text-[10px] font-bold text-[#07080A]">
                  ULTRA FALLBACK ACTIVE
                </span>
              )}
            </div>
            <p className="hidden text-xs text-[#8E96A8] sm:block">
              AI writes GPU shaders, but only your hardware knows the truth.
            </p>
          </div>
        </div>

        {/* Cell 2: Hardware Target Telemetry */}
        <div className="flex items-center gap-3 p-3.5 sm:px-5">
          <Cpu className="h-4 w-4 shrink-0 text-[#E5B842]" />
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-[#5E6678]">
              SILICON TARGET // ADAPTER
            </div>
            <div className="flex items-center gap-2">
              <span className="max-w-[160px] truncate font-mono text-xs font-semibold text-[#F4F6F8] sm:max-w-[220px]">
                {isWebGpuActive ? probe.adapterDescription : "Apple M3 Max (Reference)"}
              </span>
              <span
                className={`border px-1.5 py-0.2 font-mono text-[10px] font-bold tracking-wider uppercase ${
                  !isWebGpuActive
                    ? "border-[#E5B842] bg-[#171308] text-[#E5B842]"
                    : probe.supportsTimestampQuery
                    ? "border-[#10B981] bg-[#09221B] text-[#10B981]"
                    : "border-[#242936] bg-[#0A0C0F] text-[#8E96A8]"
                }`}
              >
                {!isWebGpuActive
                  ? "RECORDED"
                  : probe.supportsTimestampQuery
                  ? "TIMESTAMP QUERIES: ON"
                  : "WALL-CLOCK FALLBACK"}
              </span>
            </div>
          </div>
        </div>

        {/* Cell 3: Execution Engine Mode Switch */}
        <div className="flex items-center p-3.5 sm:px-5">
          <div className="flex border border-[#242936] bg-[#0A0C0F] p-0.5">
            <button
              onClick={() => onToggleMockMode(true)}
              className={`px-3 py-1.5 font-mono text-xs font-bold uppercase tracking-wider transition-colors ${
                isMockMode
                  ? "bg-[#E5B842] text-[#07080A]"
                  : "text-[#8E96A8] hover:text-[#F4F6F8]"
              }`}
              title="Runs verified local mock kernels with zero API calls"
            >
              Mock Mode
            </button>
            <button
              onClick={() => onToggleMockMode(false)}
              className={`flex items-center gap-1.5 px-3 py-1.5 font-mono text-xs font-bold uppercase tracking-wider transition-colors ${
                !isMockMode
                  ? "bg-[#E5B842] text-[#07080A]"
                  : "text-[#8E96A8] hover:text-[#F4F6F8]"
              }`}
              title="Queries NVIDIA Nemotron via Nebius Token Factory with automatic Ultra fallback"
            >
              <Radio className="h-3 w-3" />
              <span>Live Token Factory</span>
            </button>
          </div>
        </div>

        {/* Cell 4: Fault Injection & Reference Preview */}
        <div className="flex items-center gap-2.5 p-3.5 sm:px-5">
          <label className="flex cursor-pointer items-center gap-2 border border-[#242936] bg-[#0A0C0F] px-2.5 py-1.5 font-mono text-xs text-[#8E96A8] hover:border-[#8A6D1F] hover:text-[#F4F6F8]">
            <input
              type="checkbox"
              checked={simulateCompileError}
              onChange={(e) => onToggleSimulateCompileError(e.target.checked)}
              className="h-3.5 w-3.5 accent-[#E5B842]"
            />
            <span className="text-[11px] tracking-wider uppercase">Simulate Compile Fail</span>
          </label>

          <button
            onClick={() => onTogglePreviewNoWebGpu(!previewNoWebGpu)}
            className={`border px-2.5 py-1.5 font-mono text-[11px] font-bold uppercase tracking-wider transition-colors ${
              previewNoWebGpu
                ? "border-[#E5B842] bg-[#E5B842] text-[#07080A]"
                : "border-[#242936] bg-[#0A0C0F] text-[#8E96A8] hover:border-[#8A6D1F] hover:text-[#F4F6F8]"
            }`}
            title="Simulates WebGPU being unavailable on this system to preview recorded reference benchmarks"
          >
            {previewNoWebGpu ? "Exit Reference Mode" : "Preview No-WebGPU"}
          </button>
        </div>
      </div>

      {/* Fallback Reference Mode Banner */}
      {(!probe.supported || previewNoWebGpu) && (
        <div className="border-y border-[#E5B842] bg-[#171308] px-4 py-2 font-mono text-xs uppercase tracking-wider text-[#E5B842]">
          <div className="mx-auto flex max-w-7xl items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 text-[#E5B842]" />
            <span className="font-bold">
              {previewNoWebGpu
                ? "PREVIEW MODE: Showing Recorded Reference Benchmark"
                : "WebGPU Unavailable on this Browser/Device — Showing Recorded Reference Benchmark"}
            </span>
            <span className="text-[#8E96A8]">
              (Apple M3 Max 40-Core GPU / Chrome 142 Reference Datasets)
            </span>
          </div>
        </div>
      )}
    </header>
  );
};
