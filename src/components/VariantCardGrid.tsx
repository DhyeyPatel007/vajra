"use client";

import React, { useState } from "react";
import {
  Code2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  VariantExecutionResult,
  VariantStatus,
  WorkloadId,
  getStrategyBadgeLabel,
} from "@/types/vajra";

interface VariantCardGridProps {
  baseline: VariantExecutionResult | null;
  variants: VariantExecutionResult[];
  onSelectVariantForInspect: (variant: VariantExecutionResult) => void;
  selectedInspectId?: string;
  workloadId?: WorkloadId;
}

export const VariantCardGrid: React.FC<VariantCardGridProps> = ({
  baseline,
  variants,
  onSelectVariantForInspect,
  selectedInspectId,
  workloadId,
}) => {
  const allCards: Array<VariantExecutionResult | null> = [
    baseline,
    variants.find((v) => v.id === "variant-a") || null,
    variants.find((v) => v.id === "variant-b") || null,
    variants.find((v) => v.id === "variant-c") || null,
  ];

  const indexLabels = [
    "00 // NAIVE BASELINE",
    "01 // VARIANT A",
    "02 // VARIANT B",
    "03 // VARIANT C",
  ];

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
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
            indexLabel={indexLabels[index]}
            isBaseline={index === 0}
            isSelectedForInspect={card?.id === selectedInspectId}
            onInspect={() => card && onSelectVariantForInspect(card)}
            workloadId={workloadId}
          />
        );
      })}
    </div>
  );
};

interface VariantCardItemProps {
  card: VariantExecutionResult | null;
  defaultName: string;
  indexLabel: string;
  isBaseline: boolean;
  isSelectedForInspect: boolean;
  onInspect: () => void;
  workloadId?: WorkloadId;
}

const VariantCardItem: React.FC<VariantCardItemProps> = ({
  card,
  defaultName,
  indexLabel,
  isBaseline,
  isSelectedForInspect,
  onInspect,
  workloadId,
}) => {
  const [showErrors, setShowErrors] = useState(false);

  if (!card) {
    return (
      <div className="flex h-80 flex-col justify-between border border-[#242936] bg-[#0E1014] p-4 opacity-50">
        <div>
          <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#5E6678]">
            {indexLabel}
          </span>
          <h3 className="mt-1 font-sans text-sm font-semibold text-[#F4F6F8]">
            {defaultName}
          </h3>
        </div>
        <div className="text-center font-mono text-xs text-[#5E6678]">
          Waiting for dispatch...
        </div>
      </div>
    );
  }

  const isWinner = card.isWinner;
  const isCompileFailed = card.status === "COMPILE_FAILED";
  const isVerifyFailed = card.status === "VERIFY_FAILED";
  const isFailed = isCompileFailed || isVerifyFailed;

  const strategyBadge = getStrategyBadgeLabel(
    card.strategy,
    workloadId,
    card.name
  );

  return (
    <div
      className={`flex flex-col justify-between border transition-all ${
        isWinner
          ? "border-2 border-[#E5B842] bg-[#0E1014]"
          : isFailed
          ? "border-2 border-[#F43F5E] bg-[#160B0E]"
          : isSelectedForInspect
          ? "border border-[#E5B842] bg-[#0E1014]"
          : "border border-[#242936] bg-[#0E1014]"
      }`}
    >
      <div>
        {/* Top Status Strip */}
        <div
          className={`flex items-center justify-between px-3.5 py-2 font-mono text-xs font-bold tracking-wider ${
            isWinner
              ? "bg-[#E5B842] text-[#07080A]"
              : isCompileFailed
              ? "bg-[#F43F5E] text-[#07080A]"
              : isVerifyFailed
              ? "bg-[#F43F5E] text-[#07080A]"
              : "border-b border-[#242936] bg-[#0A0C0F] text-[#8E96A8]"
          }`}
        >
          <span>{indexLabel}</span>
          <span>
            {isWinner
              ? "FASTEST WINNER"
              : isCompileFailed
              ? "COMPILE FAILED"
              : isVerifyFailed
              ? "VERIFY FAILED"
              : card.status === "VERIFIED_RUNNER_UP"
              ? "VERIFIED"
              : card.status}
          </span>
        </div>

        {/* Card Content Body */}
        <div className="p-4">
          <div className="flex items-start justify-between gap-2">
            <h3 className="font-sans text-sm font-bold text-[#F4F6F8] line-clamp-1">
              {card.name}
            </h3>
            <span className="border border-[#242936] bg-[#0A0C0F] px-1.5 py-0.5 font-mono text-[9px] font-semibold text-[#E5B842] shrink-0">
              {strategyBadge}
            </span>
          </div>

          <p className="mt-1 font-mono text-xs text-[#8E96A8] line-clamp-2">
            {card.rationale}
          </p>

          {/* Telemetry Display */}
          {!isFailed ? (
            <div className="mt-3.5 border border-[#242936] bg-[#0A0C0F] p-3 font-mono">
              <div className="grid grid-cols-2 gap-2 text-xs">
                {/* Compute */}
                <div>
                  <div className="text-[10px] uppercase text-[#5E6678]">COMPUTE</div>
                  <div className={`text-base font-bold tabular-nums ${isWinner ? "text-[#E5B842]" : "text-[#F4F6F8]"}`}>
                    {card.gflops ? `${card.gflops.toFixed(1)}` : "—"}
                    <span className="text-[10px] ml-1 font-normal text-[#8E96A8]">GFLOPS</span>
                  </div>
                </div>

                {/* Speedup */}
                <div>
                  <div className="text-[10px] uppercase text-[#5E6678]">SPEEDUP</div>
                  <div className={`text-base font-bold tabular-nums ${isWinner ? "text-[#E5B842]" : "text-[#F4F6F8]"}`}>
                    {card.speedupVsBaseline ? `${card.speedupVsBaseline.toFixed(2)}x` : "1.00x"}
                  </div>
                </div>

                {/* Latency */}
                <div className="border-t border-[#181C26] pt-1.5">
                  <div className="text-[10px] uppercase text-[#5E6678]">LATENCY</div>
                  <div className="text-xs text-[#F4F6F8] tabular-nums">
                    {card.medianTimeMs ? `${card.medianTimeMs.toFixed(3)} ms` : "—"}
                  </div>
                </div>

                {/* Workgroup */}
                <div className="border-t border-[#181C26] pt-1.5">
                  <div className="text-[10px] uppercase text-[#5E6678]">WORKGROUP</div>
                  <div className="text-xs text-[#F4F6F8] tabular-nums">
                    ({card.workgroupSize.join(", ")})
                  </div>
                </div>
              </div>

              {/* Derived Bandwidth & Float Error Sub-line */}
              <div className="mt-2.5 flex items-center justify-between border-t border-[#181C26] pt-2 text-[10px] text-[#8E96A8]">
                <span>BW: {card.derivedBandwidthGBps ? `${card.derivedBandwidthGBps.toFixed(2)} GB/s` : "—"}</span>
                <span className="text-[#10B981]">
                  Err: {card.maxRelativeError !== null
                    ? card.maxRelativeError === 0
                      ? "0.0 (exact)"
                      : card.maxRelativeError.toExponential(1)
                    : "—"}
                </span>
              </div>
            </div>
          ) : (
            /* Diagnostic Error Drawer for Failed Shader */
            <div className="mt-3.5 border border-[#F43F5E] bg-[#0A0C0F] p-3 font-mono text-xs">
              <div className="flex items-center justify-between text-[#F43F5E] font-bold">
                <span>{isCompileFailed ? "WEBGPU COMPILER REJECTION" : "VERIFICATION ERROR"}</span>
                {card.compileErrors.length > 0 && (
                  <button
                    onClick={() => setShowErrors(!showErrors)}
                    className="flex items-center gap-1 text-[10px] text-[#8E96A8] hover:text-[#F4F6F8]"
                  >
                    {showErrors ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    <span>{showErrors ? "LESS" : "EXPAND"}</span>
                  </button>
                )}
              </div>
              <p className="mt-1 text-[11px] text-[#8E96A8]">
                {isCompileFailed
                  ? "Dropped from benchmark podium. Pipeline rejected by device.createShaderModule()."
                  : `Relative float error (${card.maxRelativeError?.toExponential(2)}) exceeded strict 1e-4 tolerance.`}
              </p>

              {(showErrors || isCompileFailed) && card.compileErrors.length > 0 && (
                <div className="mt-2 max-h-28 overflow-y-auto border border-[#F43F5E] bg-[#160B0E] p-2 text-[10px] text-[#F43F5E]">
                  {card.compileErrors.map((err, i) => (
                    <div key={i}>{err}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer Action Button */}
      <div className="p-4 pt-0">
        <button
          onClick={onInspect}
          className={`flex w-full items-center justify-center gap-2 border py-2.5 font-mono text-xs font-bold uppercase tracking-wider transition-colors ${
            isSelectedForInspect
              ? "border-[#E5B842] bg-[#E5B842] text-[#07080A]"
              : "border-[#242936] bg-[#0A0C0F] text-[#8E96A8] hover:border-[#8A6D1F] hover:text-[#F4F6F8]"
          }`}
        >
          <Code2 className="h-3.5 w-3.5" />
          <span>INSPECT WGSL</span>
        </button>
      </div>
    </div>
  );
};
