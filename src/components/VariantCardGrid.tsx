"use client";

import React, { useState } from "react";
import {
  Trophy,
  CheckCircle2,
  XCircle,
  AlertOctagon,
  ChevronDown,
  ChevronUp,
  Cpu,
  Zap,
  HardDrive,
  Clock,
  Code2,
} from "lucide-react";
import { VariantExecutionResult, VariantStatus } from "@/types/vajra";

interface VariantCardGridProps {
  baseline: VariantExecutionResult | null;
  variants: VariantExecutionResult[];
  onSelectVariantForInspect: (variant: VariantExecutionResult) => void;
  selectedInspectId?: string;
}

export const VariantCardGrid: React.FC<VariantCardGridProps> = ({
  baseline,
  variants,
  onSelectVariantForInspect,
  selectedInspectId,
}) => {
  const allCards: Array<VariantExecutionResult | null> = [
    baseline,
    variants.find((v) => v.id === "variant-a") || null,
    variants.find((v) => v.id === "variant-b") || null,
    variants.find((v) => v.id === "variant-c") || null,
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {allCards.map((card, index) => {
        const defaultName =
          index === 0
            ? "Naive Baseline"
            : index === 1
            ? "Variant A: Workgroup Geometry"
            : index === 2
            ? "Variant B: Shared Memory Tiling"
            : "Variant C: Vec4 Vectorized";

        return (
          <VariantCardItem
            key={card?.id || `slot-${index}`}
            card={card}
            defaultName={defaultName}
            isBaseline={index === 0}
            isSelectedForInspect={card?.id === selectedInspectId}
            onInspect={() => card && onSelectVariantForInspect(card)}
          />
        );
      })}
    </div>
  );
};

interface VariantCardItemProps {
  card: VariantExecutionResult | null;
  defaultName: string;
  isBaseline: boolean;
  isSelectedForInspect: boolean;
  onInspect: () => void;
}

const VariantCardItem: React.FC<VariantCardItemProps> = ({
  card,
  defaultName,
  isBaseline,
  isSelectedForInspect,
  onInspect,
}) => {
  const [showErrors, setShowErrors] = useState(false);

  if (!card) {
    return (
      <div className="flex h-72 flex-col justify-between rounded border border-[#262A34] bg-[#121418] p-4 opacity-50">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-[#9CA3AF]">
            Pending Generation
          </span>
          <h3 className="mt-1 text-sm font-semibold text-[#F3F4F6]">
            {defaultName}
          </h3>
        </div>
        <div className="text-center font-mono text-xs text-[#9CA3AF]">
          Waiting for dispatch...
        </div>
      </div>
    );
  }

  const isWinner = card.isWinner;
  const isCompileFailed = card.status === "COMPILE_FAILED";
  const isVerifyFailed = card.status === "VERIFY_FAILED";
  const isFailed = isCompileFailed || isVerifyFailed;

  const renderStatusBadge = (status: VariantStatus) => {
    switch (status) {
      case "VERIFIED_WINNER":
        return (
          <span className="flex items-center gap-1 rounded bg-[#D4AF37] px-2 py-0.5 text-[10px] font-bold text-[#0B0C0E]">
            <Trophy className="h-3 w-3" />
            FASTEST WINNER
          </span>
        );
      case "VERIFIED_RUNNER_UP":
        return (
          <span className="flex items-center gap-1 rounded bg-[#10B981] px-2 py-0.5 text-[10px] font-bold text-[#0B0C0E]">
            <CheckCircle2 className="h-3 w-3" />
            VERIFIED
          </span>
        );
      case "COMPILE_FAILED":
        return (
          <span className="flex items-center gap-1 rounded bg-[#EF4444] px-2 py-0.5 text-[10px] font-bold text-[#F3F4F6]">
            <XCircle className="h-3 w-3" />
            COMPILE FAILED
          </span>
        );
      case "VERIFY_FAILED":
        return (
          <span className="flex items-center gap-1 rounded bg-[#EF4444] px-2 py-0.5 text-[10px] font-bold text-[#F3F4F6]">
            <AlertOctagon className="h-3 w-3" />
            VERIFY FAILED
          </span>
        );
      case "GENERATING":
        return (
          <span className="rounded bg-[#181B22] border border-[#D4AF37] px-2 py-0.5 text-[10px] font-bold text-[#D4AF37] animate-pulse">
            GENERATING...
          </span>
        );
      case "COMPILING":
        return (
          <span className="rounded bg-[#181B22] border border-[#383E4D] px-2 py-0.5 text-[10px] font-bold text-[#F3F4F6]">
            COMPILING...
          </span>
        );
      case "VERIFYING":
        return (
          <span className="rounded bg-[#181B22] border border-[#383E4D] px-2 py-0.5 text-[10px] font-bold text-[#F3F4F6]">
            VERIFYING FLOATS...
          </span>
        );
      case "BENCHMARKING":
        return (
          <span className="rounded bg-[#181B22] border border-[#D4AF37] px-2 py-0.5 text-[10px] font-bold text-[#D4AF37] animate-pulse">
            BENCHMARKING (9 RUNS)...
          </span>
        );
      default:
        return (
          <span className="rounded bg-[#181B22] border border-[#262A34] px-2 py-0.5 text-[10px] font-bold text-[#9CA3AF]">
            {isBaseline ? "BASELINE" : "IDLE"}
          </span>
        );
    }
  };

  return (
    <div
      className={`flex flex-col justify-between rounded border p-4 transition-all ${
        isWinner
          ? "border-[#D4AF37] bg-[#181B22] shadow-sm"
          : isFailed
          ? "border-[#EF4444] bg-[#121418]"
          : isSelectedForInspect
          ? "border-[#D4AF37] bg-[#121418]"
          : "border-[#262A34] bg-[#121418]"
      }`}
    >
      <div>
        {/* Top Badges */}
        <div className="flex items-center justify-between gap-1 border-b border-[#262A34] pb-2.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[#9CA3AF]">
            {isBaseline ? "REFERENCE HARNESS" : card.id.toUpperCase()}
          </span>
          {renderStatusBadge(card.status)}
        </div>

        {/* Title */}
        <h3 className="mt-2 text-xs font-bold text-[#F3F4F6] line-clamp-1">
          {card.name}
        </h3>
        <p className="mt-0.5 text-[11px] text-[#9CA3AF] line-clamp-2">
          {card.rationale}
        </p>

        {/* Metrics Section */}
        {!isFailed ? (
          <div className="mt-3 space-y-1.5 font-mono text-xs">
            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Compute:</span>
              <span className={`font-bold ${isWinner ? "text-[#D4AF37]" : "text-[#F3F4F6]"}`}>
                {card.gflops ? `${card.gflops.toFixed(1)} GFLOPS` : "—"}
              </span>
            </div>

            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Speedup:</span>
              <span className={`font-bold ${isWinner ? "text-[#D4AF37]" : "text-[#F3F4F6]"}`}>
                {card.speedupVsBaseline ? `${card.speedupVsBaseline.toFixed(2)}x` : "1.00x"}
              </span>
            </div>

            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Median Latency:</span>
              <span className="text-[#F3F4F6]">
                {card.medianTimeMs ? `${card.medianTimeMs.toFixed(3)} ms` : "—"}
              </span>
            </div>

            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Workgroup Size:</span>
              <span className="text-[#F3F4F6]">
                ({card.workgroupSize.join(", ")})
              </span>
            </div>

            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Dispatch Grid:</span>
              <span className="text-[#F3F4F6]">
                [{card.dispatchGrid.join(", ")}]
              </span>
            </div>

            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Derived BW:</span>
              <span className="text-[#F3F4F6]">
                {card.derivedBandwidthGBps ? `${card.derivedBandwidthGBps.toFixed(2)} GB/s` : "—"}
              </span>
            </div>

            <div className="flex items-center justify-between rounded bg-[#0B0C0E] px-2.5 py-1">
              <span className="text-[#9CA3AF]">Float Error:</span>
              <span className="text-[#10B981]">
                {card.maxRelativeError !== null
                  ? card.maxRelativeError === 0
                    ? "0.0 (exact)"
                    : card.maxRelativeError.toExponential(2)
                  : "—"}
              </span>
            </div>
          </div>
        ) : (
          /* Error Drawer for Failed Shader */
          <div className="mt-3 rounded border border-[#EF4444] bg-[#0B0C0E] p-2.5 text-xs">
            <div className="flex items-center justify-between text-[#EF4444] font-semibold">
              <span>{isCompileFailed ? "WebGPU Compiler Rejection" : "Tolerance Verification Fault"}</span>
              {card.compileErrors.length > 0 && (
                <button
                  onClick={() => setShowErrors(!showErrors)}
                  className="flex items-center gap-1 text-[10px] text-[#9CA3AF] hover:text-[#F3F4F6]"
                >
                  {showErrors ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  <span>{showErrors ? "Hide" : "Details"}</span>
                </button>
              )}
            </div>
            <p className="mt-1 text-[11px] text-[#9CA3AF]">
              {isCompileFailed
                ? "Dropped from benchmark podium. Pipeline did not pass device.createShaderModule() validation."
                : `Relative error (${card.maxRelativeError?.toExponential(2)}) exceeded strict 1e-4 tolerance.`}
            </p>

            {(showErrors || isCompileFailed) && card.compileErrors.length > 0 && (
              <div className="mt-2 max-h-28 overflow-y-auto rounded bg-[#181B22] p-2 font-mono text-[10px] text-[#EF4444]">
                {card.compileErrors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Inspect WGSL Button */}
      <button
        onClick={onInspect}
        className={`mt-4 flex w-full items-center justify-center gap-1.5 rounded border py-1.5 text-xs font-semibold transition-colors ${
          isSelectedForInspect
            ? "border-[#D4AF37] bg-[#D4AF37] text-[#0B0C0E]"
            : "border-[#262A34] bg-[#0B0C0E] text-[#9CA3AF] hover:border-[#383E4D] hover:text-[#F3F4F6]"
        }`}
      >
        <Code2 className="h-3.5 w-3.5" />
        <span>Inspect WGSL</span>
      </button>
    </div>
  );
};
