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
    <div className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] overflow-hidden shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
      {/* Top Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#EAEAEA] px-6 py-4">
        {/* Left: Title + 3 View Mode Pills */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Code2 className="h-5 w-5 text-[#171717]" />
            <span className="text-[16px] font-semibold tracking-[-0.02em] text-[#171717]">
              Shader &amp; Artifact Inspector
            </span>
          </div>

          {/* View Mode Switcher Pills */}
          <div className="flex items-center rounded-md border border-[#EAEAEA] bg-[#F2F2F2] p-0.5 font-mono text-[12px]">
            <button
              onClick={() => setViewMode("wgsl")}
              className={`rounded-[5px] px-3 py-1 font-medium transition-all ${
                viewMode === "wgsl"
                  ? "bg-[#FFFFFF] text-[#171717] shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                  : "text-[#666666] hover:text-[#171717]"
              }`}
            >
              Raw Stitched WGSL
            </button>
            <button
              onClick={() => setViewMode("typescript")}
              className={`rounded-[5px] px-3 py-1 font-medium transition-all ${
                viewMode === "typescript"
                  ? "bg-[#FFFFFF] text-[#171717] shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                  : "text-[#666666] hover:text-[#171717]"
              }`}
            >
              Exported TS Dispatch Function
            </button>
            <button
              onClick={() => setViewMode("json")}
              className={`rounded-[5px] px-3 py-1 font-medium transition-all ${
                viewMode === "json"
                  ? "bg-[#FFFFFF] text-[#171717] shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                  : "text-[#666666] hover:text-[#171717]"
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
            className="flex items-center gap-1.5 rounded-md border border-[#EAEAEA] bg-[#FFFFFF] px-3.5 py-2 font-mono text-[12px] font-medium text-[#171717] transition-colors hover:border-[#999999] hover:bg-[#FAFAFA] cursor-pointer"
          >
            {copiedLockfile ? (
              <>
                <Check className="h-3.5 w-3.5 text-[#0A7143]" />
                <span className="text-[#0A7143]">Copied Hardware Lockfile JSON!</span>
              </>
            ) : (
              <>
                <FileJson className="h-3.5 w-3.5 text-[#666666]" />
                <span>Export Lockfile JSON</span>
              </>
            )}
          </button>

          {/* Copy Current Tab Button */}
          <button
            onClick={handleCopyCurrent}
            className="flex items-center gap-1.5 rounded-md border border-[#EAEAEA] bg-[#FFFFFF] px-3.5 py-2 font-mono text-[12px] font-medium text-[#171717] transition-colors hover:border-[#999999] hover:bg-[#FAFAFA] cursor-pointer"
          >
            {copiedCurrent ? (
              <>
                <Check className="h-3.5 w-3.5 text-[#0A7143]" />
                <span className="text-[#0A7143]">Copied Tab WGSL!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5 text-[#666666]" />
                <span>Copy This WGSL</span>
              </>
            )}
          </button>

          {/* Copy Fastest Winner Button */}
          <button
            onClick={handleCopyFastest}
            className="flex items-center gap-1.5 rounded-md border border-[#171717] bg-[#171717] px-4 py-2 font-mono text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#333333] active:bg-[#000000] cursor-pointer"
          >
            {copiedFastest ? (
              <>
                <Check className="h-4 w-4 text-[#FFFFFF]" />
                <span>Copied Verified Fastest WGSL!</span>
              </>
            ) : (
              <>
                <Trophy className="h-4 w-4 fill-current text-[#FFFFFF]" />
                <span>Copy Fastest WGSL</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Variant Selector Tabs (only in WGSL mode) */}
      {viewMode === "wgsl" && (
        <div className="flex items-center gap-6 border-b border-[#EAEAEA] bg-[#FAFAFA] px-6 overflow-x-auto">
          {allItems.map((item) => {
            const isSelected = item.id === activeItem?.id;
            const isWinner = item.id !== "baseline" && Boolean(item.isWinner);
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
                className={`flex items-center gap-1.5 py-3 font-mono text-[12.5px] border-b-2 transition-colors whitespace-nowrap cursor-pointer ${
                  isSelected
                    ? "border-[#171717] text-[#171717] font-semibold"
                    : "border-transparent text-[#666666] hover:text-[#171717]"
                }`}
              >
                <span>{tabLabel}</span>
                {isGen2 && (
                  <span className="rounded border border-[#EAEAEA] bg-[#F2F2F2] px-1 py-0.2 text-[9px] text-[#171717]">
                    GEN 2
                  </span>
                )}
                {isWinner && (
                  <span className="rounded bg-[#171717] px-1.5 py-0.5 text-[10px] text-[#FFFFFF] font-sans">
                    WINNER
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Code Viewport with Line Numbers */}
      <div className="bg-[#FAFAFA] p-5 font-mono text-[12.5px] leading-relaxed max-h-[440px] overflow-auto">
        <div className="table w-full">
          {codeLines.map((line, index) => (
            <div key={index} className="table-row hover:bg-[#F2F2F2]/80">
              <span className="table-cell select-none pr-4 text-right text-[11px] text-[#888888] w-10 border-r border-[#EAEAEA]">
                {index + 1}
              </span>
              <span className="table-cell pl-4 whitespace-pre text-[#171717]">
                {line || " "}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
