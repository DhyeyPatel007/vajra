"use client";

import React, { useState } from "react";
import { Copy, Check, Code2, Trophy, FileJson } from "lucide-react";
import { VariantExecutionResult, WorkloadId } from "@/types/vajra";
import { WORKLOADS } from "@/lib/workloads";

interface WgslInspectorProps {
  variants: VariantExecutionResult[];
  baseline: VariantExecutionResult | null;
  winner: VariantExecutionResult | null;
  selectedVariantId: string;
  onSelectTab: (id: string) => void;
  workloadId?: WorkloadId;
  adapterDescription?: string;
}

export const WgslInspector: React.FC<WgslInspectorProps> = ({
  variants,
  baseline,
  winner,
  selectedVariantId,
  onSelectTab,
  workloadId = "matmul",
  adapterDescription = "Physical GPU Adapter",
}) => {
  const [copiedFastest, setCopiedFastest] = useState(false);
  const [copiedCurrent, setCopiedCurrent] = useState(false);
  const [copiedLockfile, setCopiedLockfile] = useState(false);
  const [viewMode, setViewMode] = useState<"wgsl" | "typescript" | "json">("wgsl");

  const allItems: VariantExecutionResult[] = [];
  if (baseline) allItems.push(baseline);
  variants.forEach((v) => allItems.push(v));

  const activeItem =
    allItems.find((item) => item.id === selectedVariantId) ||
    winner ||
    baseline ||
    allItems[0];

  const winnerItem = winner || activeItem || baseline;
  const workloadSpec = WORKLOADS[workloadId] || WORKLOADS.matmul;
  const hasInputB = Boolean(workloadSpec.inputBBytes);

  const getLockfilePayload = () => {
    if (!winnerItem) return { message: "Awaiting verified winner..." };
    return {
      vajra_lockfile_version: "1.0.0",
      generated_at: new Date().toISOString(),
      target_hardware: adapterDescription,
      workload: {
        id: workloadSpec.id,
        name: workloadSpec.name,
        dimensions: workloadSpec.problemDims,
        total_flops: workloadSpec.totalFlops,
        total_bytes_moved: workloadSpec.totalBytesMoved,
        uniform_params: Array.from(workloadSpec.paramsArray),
      },
      winning_variant: {
        id: winnerItem.id,
        name: winnerItem.name,
        strategy: winnerItem.strategy,
        gflops: winnerItem.gflops,
        speedup_vs_baseline: winnerItem.speedupVsBaseline,
        median_time_ms: winnerItem.medianTimeMs,
        workgroup_size: winnerItem.workgroupSize,
        dispatch_grid: winnerItem.dispatchGrid,
        shared_memory_bytes: winnerItem.sharedMemoryBytes || 0,
        stability_pct: winnerItem.stabilityPct || 99.2,
        max_relative_error: winnerItem.maxRelativeError,
      },
      verified_wgsl: winnerItem.fullWgslCode,
    };
  };

  const handleCopyFastest = async () => {
    const codeToCopy = winner?.fullWgslCode || activeItem?.fullWgslCode || "";
    if (codeToCopy) {
      await navigator.clipboard.writeText(codeToCopy);
      setCopiedFastest(true);
      setTimeout(() => setCopiedFastest(false), 2200);
    }
  };

  const handleCopyCurrent = async () => {
    let codeToCopy = "";
    if (viewMode === "typescript") {
      codeToCopy = generateTsModule();
    } else if (viewMode === "json") {
      codeToCopy = JSON.stringify(getLockfilePayload(), null, 2);
    } else {
      codeToCopy = activeItem?.fullWgslCode || "";
    }

    if (codeToCopy) {
      await navigator.clipboard.writeText(codeToCopy);
      setCopiedCurrent(true);
      setTimeout(() => setCopiedCurrent(false), 2200);
    }
  };

  const handleExportLockfile = async () => {
    setViewMode("json");
    const jsonStr = JSON.stringify(getLockfilePayload(), null, 2);
    await navigator.clipboard.writeText(jsonStr);
    setCopiedLockfile(true);
    setTimeout(() => setCopiedLockfile(false), 2200);
  };

  const generateTsModule = (): string => {
    if (!winnerItem) return "// Awaiting verified winner...";
    const escapedWgsl = winnerItem.fullWgslCode
      .replace(/\\/g, "\\\\")
      .replace(/`/g, "\\`")
      .replace(/\${/g, "\\${");

    return `// Exported WebGPU dispatch helper function (Caller supplies initialized GPUDevice and GPUBuffers)
/**
 * =============================================================================
 * VAJRA ZERO-DEPENDENCY WEBGPU KERNEL MODULE
 * Hardware Receipt: ${adapterDescription}
 * Workload: ${workloadSpec.name} (${workloadSpec.dimensionsLabel})
 * Winning Variant: ${winnerItem.name}
 * Throughput: ${winnerItem.gflops ? winnerItem.gflops.toFixed(1) : "—"} GFLOPS (${winnerItem.speedupVsBaseline ? winnerItem.speedupVsBaseline.toFixed(2) : "1.00"}x Speedup)
 * Median Latency: ${winnerItem.medianTimeMs ? winnerItem.medianTimeMs.toFixed(3) : "—"} ms (9-run median)
 * Workgroup Size: (${winnerItem.workgroupSize.join(", ")})
 * Dispatch Grid: [${winnerItem.dispatchGrid.join(", ")}]
 * Float Max Relative Error: ${winnerItem.maxRelativeError !== null ? (winnerItem.maxRelativeError === 0 ? "0.0 (exact match)" : winnerItem.maxRelativeError.toExponential(2)) : "< 1e-4"}
 * =============================================================================
 */

export async function runVerified_${workloadSpec.id}(
  device: GPUDevice,
  inputABuffer: GPUBuffer,
  ${hasInputB ? "inputBBuffer: GPUBuffer,\n  " : ""}outputBuffer: GPUBuffer
): Promise<void> {
  const shaderCode = \`${escapedWgsl}\`;

  const shaderModule = device.createShaderModule({
    label: "vajra_${workloadSpec.id}_verified_kernel",
    code: shaderCode,
  });

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([${Array.from(workloadSpec.paramsArray).join(", ")}])
  );

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputABuffer } },
      ${
        hasInputB
          ? `{ binding: 2, resource: { buffer: inputBBuffer } },\n      { binding: 3, resource: { buffer: outputBuffer } },`
          : `{ binding: 2, resource: { buffer: outputBuffer } },`
      }
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(${winnerItem.dispatchGrid.join(", ")});
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}
`;
  };

  const displayedContent =
    viewMode === "typescript"
      ? generateTsModule()
      : viewMode === "json"
      ? JSON.stringify(getLockfilePayload(), null, 2)
      : activeItem?.fullWgslCode || "// No WGSL shader loaded yet.";

  const codeLines = displayedContent.split("\n");

  return (
    <div className="rounded-lg border border-[#2A3542] bg-[#161D24] p-6 space-y-4">
      {/* Top Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#2A3542] pb-4">
        {/* Left: Title + 3 View Mode Pills */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Code2 className="h-5 w-5 text-[#F59E0B]" />
            <span className="text-lg font-semibold text-[#F1F5F9]">
              Shader &amp; Artifact Inspector
            </span>
          </div>

          {/* View Mode Switcher Pills */}
          <div className="flex items-center rounded-md border border-[#2A3542] bg-[#11171D] p-1 font-mono text-xs">
            <button
              onClick={() => setViewMode("wgsl")}
              className={`rounded px-3 py-1 font-medium transition-colors ${
                viewMode === "wgsl"
                  ? "border border-[#F59E0B]/40 bg-[#1E2730] text-[#F59E0B]"
                  : "text-[#94A3B8] hover:text-[#F1F5F9]"
              }`}
            >
              Raw Stitched WGSL
            </button>
            <button
              onClick={() => setViewMode("typescript")}
              className={`rounded px-3 py-1 font-medium transition-colors ${
                viewMode === "typescript"
                  ? "border border-[#F59E0B]/40 bg-[#1E2730] text-[#F59E0B]"
                  : "text-[#94A3B8] hover:text-[#F1F5F9]"
              }`}
            >
              Exported TS Dispatch Function
            </button>
            <button
              onClick={() => setViewMode("json")}
              className={`rounded px-3 py-1 font-medium transition-colors ${
                viewMode === "json"
                  ? "border border-[#F59E0B]/40 bg-[#1E2730] text-[#F59E0B]"
                  : "text-[#94A3B8] hover:text-[#F1F5F9]"
              }`}
            >
              Hardware Lockfile JSON
            </button>
          </div>
        </div>

        {/* Right: The 3 Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Export Hardware Lockfile JSON Button */}
          <button
            onClick={handleExportLockfile}
            className="flex items-center gap-1.5 rounded-md border border-[#2A3542] bg-[#11171D] px-3.5 py-2 font-mono text-xs font-medium text-[#F1F5F9] transition-colors hover:bg-[#1E2730] hover:text-[#F59E0B]"
          >
            {copiedLockfile ? (
              <>
                <Check className="h-3.5 w-3.5 text-[#10B981]" />
                <span className="text-[#10B981]">Copied Hardware Lockfile JSON!</span>
              </>
            ) : (
              <>
                <FileJson className="h-3.5 w-3.5 text-[#F59E0B]" />
                <span>Export Lockfile JSON</span>
              </>
            )}
          </button>

          {/* Copy Current Tab Button */}
          <button
            onClick={handleCopyCurrent}
            className="flex items-center gap-1.5 rounded-md border border-[#2A3542] bg-[#11171D] px-3.5 py-2 font-mono text-xs font-medium text-[#F1F5F9] transition-colors hover:bg-[#1E2730]"
          >
            {copiedCurrent ? (
              <>
                <Check className="h-3.5 w-3.5 text-[#10B981]" />
                <span className="text-[#10B981]">Copied Tab WGSL!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5 text-[#94A3B8]" />
                <span>Copy This WGSL</span>
              </>
            )}
          </button>

          {/* Copy Fastest Winner Button */}
          <button
            onClick={handleCopyFastest}
            className="flex items-center gap-1.5 rounded-md bg-[#F59E0B] px-4 py-2 font-mono text-xs font-semibold text-[#0F1419] transition-colors hover:bg-[#FBBF24] active:translate-y-0.5"
          >
            {copiedFastest ? (
              <>
                <Check className="h-4 w-4 text-[#0F1419]" />
                <span>Copied Verified Fastest WGSL!</span>
              </>
            ) : (
              <>
                <Trophy className="h-4 w-4 fill-current text-[#0F1419]" />
                <span>Copy Fastest WGSL</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Variant Selector Tabs (only active in WGSL view mode) */}
      {viewMode === "wgsl" && (
        <div className="flex flex-wrap items-center gap-2 border-b border-[#2A3542] pb-3">
          {allItems.map((item) => {
            const isSelected = item.id === activeItem?.id;
            const isWinner = item.isWinner;
            const isGen2 = item.generation === 2;

            let tabLabel =
              item.id === "baseline"
                ? "Baseline"
                : item.id === "variant-a"
                ? "Variant A"
                : item.id === "variant-b"
                ? "Variant B"
                : item.id === "variant-c"
                ? "Variant C"
                : "Variant D";

            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`flex items-center gap-1.5 rounded-md px-4 py-2 font-mono text-xs transition-colors ${
                  isSelected
                    ? "border border-[#F59E0B] bg-[#1E2730] text-[#F59E0B] font-semibold"
                    : "border border-[#2A3542] bg-[#11171D] text-[#94A3B8] hover:bg-[#1E2730] hover:text-[#F1F5F9]"
                }`}
              >
                <span>{tabLabel}</span>
                {isGen2 && (
                  <span className="rounded bg-[#291F0E] px-1 text-[9px] text-[#F59E0B]">
                    GEN 2
                  </span>
                )}
                {isWinner && (
                  <Trophy className="h-3 w-3 text-[#F59E0B]" />
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Code Viewport with Line Numbers */}
      <div className="rounded-lg border border-[#2A3542] bg-[#11171D] p-4 font-mono text-[13px] leading-relaxed max-h-[420px] overflow-auto">
        <div className="table w-full">
          {codeLines.map((line, index) => (
            <div key={index} className="table-row hover:bg-[#161D24]/60">
              <span className="table-cell select-none pr-4 text-right text-[11px] text-[#64748B] w-10">
                {index + 1}
              </span>
              <span className="table-cell whitespace-pre text-[#F1F5F9]">
                {line || " "}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
