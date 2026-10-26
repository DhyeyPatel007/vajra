"use client";

import React, { useState } from "react";
import {
  Code2,
  ChevronDown,
  ChevronUp,
  Trophy,
  AlertCircle,
  ShieldCheck,
} from "lucide-react";
import {
  VariantExecutionResult,
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
  const [expandAll, setExpandAll] = useState(false);
  const hasVariantD = variants.some((v) => v.id === "variant-d");

  const allCards: Array<VariantExecutionResult | null> = [
    baseline,
    variants.find((v) => v.id === "variant-a") || null,
    variants.find((v) => v.id === "variant-b") || null,
    variants.find((v) => v.id === "variant-c") || null,
  ];

  if (hasVariantD) {
    allCards.push(variants.find((v) => v.id === "variant-d") || null);
  }

  const indexLabels = [
    "Baseline",
    "Variant A",
    "Variant B",
    "Variant C",
    "Variant D",
  ];

  return (
    <div className="space-y-4">
      {/* Section Header with Expand All Toggle */}
      <div className="flex items-center justify-between border-b border-[#2A3542] pb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[#F1F5F9]">
            Kernel Candidates
          </h2>
          <span className="rounded border border-[#2A3542] bg-[#11171D] px-2 py-0.5 font-mono text-xs text-[#94A3B8]">
            {allCards.filter(Boolean).length} Variants Tested
          </span>
        </div>

        <button
          onClick={() => setExpandAll(!expandAll)}
          className="flex items-center gap-1.5 rounded-md border border-[#2A3542] bg-[#11171D] px-3 py-1.5 font-mono text-xs text-[#94A3B8] transition-colors hover:bg-[#1E2730] hover:text-[#F1F5F9]"
        >
          {expandAll ? (
            <>
              <ChevronUp className="h-3.5 w-3.5 text-[#F59E0B]" />
              <span>Collapse All Card Details</span>
            </>
          ) : (
            <>
              <ChevronDown className="h-3.5 w-3.5 text-[#F59E0B]" />
              <span>Expand All Card Details</span>
            </>
          )}
        </button>
      </div>

      {/* Spacious Grid */}
      <div
        className={`grid grid-cols-1 gap-6 md:grid-cols-2 ${
          hasVariantD ? "xl:grid-cols-5" : "xl:grid-cols-4"
        }`}
      >
        {allCards.map((card, index) => {
          const defaultName =
            index === 0
              ? "Naive Baseline"
              : index === 1
              ? "Variant A: Workgroup Geometry"
              : index === 2
              ? "Variant B: Shared Memory Tiling"
              : index === 3
              ? "Variant C: Vec4 Vectorized"
              : "Variant D: Gen 2 Shared Tile + Unrolled FMA Pipeline";

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
              forceExpand={expandAll}
            />
          );
        })}
      </div>
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
  forceExpand: boolean;
}

const VariantCardItem: React.FC<VariantCardItemProps> = ({
  card,
  defaultName,
  indexLabel,
  isSelectedForInspect,
  onInspect,
  workloadId,
  forceExpand,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!card) {
    return (
      <div className="flex h-80 flex-col justify-between rounded-lg border border-[#2A3542] bg-[#161D24] p-5 opacity-50">
        <div>
          <span className="font-mono text-xs font-semibold text-[#64748B]">
            {indexLabel}
          </span>
          <h3 className="mt-2 text-base font-semibold text-[#F1F5F9]">
            {defaultName}
          </h3>
        </div>
        <div className="text-center font-mono text-xs text-[#64748B]">
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
    card.name,
    card.sharedMemoryBytes
  );

  const isDetailsOpen = forceExpand || showDetails || isFailed;

  return (
    <div
      className={`flex flex-col justify-between rounded-lg border p-5 transition-all ${
        isWinner
          ? "border-2 border-[#F59E0B] bg-[#161D24]"
          : isFailed
          ? "border-2 border-[#F43F5E] bg-[#1A1215]"
          : isSelectedForInspect
          ? "border-2 border-[#F59E0B]/60 bg-[#161D24]"
          : "border border-[#2A3542] bg-[#161D24]"
      }`}
    >
      <div>
        {/* Top Row: Variant Label + Status Badge */}
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs font-bold uppercase tracking-wider text-[#94A3B8]">
            {indexLabel}
          </span>
          {isWinner ? (
            <span className="flex items-center gap-1 rounded bg-[#291F0E] px-2 py-0.5 font-mono text-[11px] font-semibold text-[#F59E0B] border border-[#F59E0B]/40">
              <Trophy className="h-3 w-3" />
              FASTEST WINNER
            </span>
          ) : isCompileFailed ? (
            <span className="rounded bg-[#2A1219] px-2 py-0.5 font-mono text-[11px] font-semibold text-[#F43F5E] border border-[#F43F5E]/40">
              COMPILE_FAILED
            </span>
          ) : isVerifyFailed ? (
            <span className="rounded bg-[#2A1219] px-2 py-0.5 font-mono text-[11px] font-semibold text-[#F43F5E] border border-[#F43F5E]/40">
              VERIFY_FAILED
            </span>
          ) : card.status === "VERIFIED_RUNNER_UP" ? (
            <span className="flex items-center gap-1 rounded bg-[#0B261F] px-2 py-0.5 font-mono text-[11px] font-semibold text-[#10B981] border border-[#10B981]/40">
              <ShieldCheck className="h-3 w-3" />
              VERIFIED
            </span>
          ) : (
            <span className="rounded bg-[#11171D] px-2 py-0.5 font-mono text-[11px] text-[#94A3B8]">
              {card.status}
            </span>
          )}
        </div>

        {/* Title + Strategy Badge */}
        <div className="mt-3">
          <h3 className="text-base font-semibold text-[#F1F5F9] line-clamp-1">
            {card.name}
          </h3>
          <span className="mt-1 inline-block rounded border border-[#2A3542] bg-[#11171D] px-2 py-0.5 font-mono text-[10px] text-[#94A3B8]">
            {strategyBadge}
          </span>
        </div>

        {/* Big Scannable Numbers */}
        {!isFailed ? (
          <div className="my-4 flex items-baseline justify-between rounded-md border border-[#2A3542] bg-[#11171D] p-3.5 font-mono">
            <div>
              <div className="text-[10px] text-[#64748B] uppercase">THROUGHPUT</div>
              <div className="text-[22px] font-bold text-[#F1F5F9] tabular-nums leading-tight">
                {card.gflops ? `${card.gflops.toFixed(1)}` : "—"}{" "}
                <span className="text-xs font-normal text-[#94A3B8]">GFLOPS</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-[#64748B] uppercase">SPEEDUP</div>
              <div className="text-[18px] font-semibold text-[#F59E0B] tabular-nums leading-tight">
                {card.id === "baseline"
                  ? "1.00x"
                  : `${card.speedupVsBaseline ? card.speedupVsBaseline.toFixed(2) : "1.00"}x`}
              </div>
            </div>
          </div>
        ) : (
          <div className="my-4 rounded-md border border-[#F43F5E] bg-[#2A1219] p-3 font-mono text-xs text-[#F43F5E]">
            <div className="flex items-center gap-1.5 font-bold">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{isCompileFailed ? "COMPILER REJECTION" : "VERIFY REJECTION"}</span>
            </div>
            {card.compileErrors.length > 0 && (
              <div className="mt-2 max-h-24 overflow-y-auto rounded bg-[#11171D] p-2 text-[11px] text-[#F43F5E] border border-[#F43F5E]/40">
                {card.compileErrors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Progressive Disclosure Toggle */}
        <div className="mb-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 text-xs font-mono text-[#94A3B8] hover:text-[#F1F5F9] transition-colors"
          >
            <span>{isDetailsOpen ? "Hide Telemetry & Rationale" : "Show Telemetry & Rationale"}</span>
            {isDetailsOpen ? (
              <ChevronUp className="h-3 w-3 text-[#F59E0B]" />
            ) : (
              <ChevronDown className="h-3 w-3 text-[#F59E0B]" />
            )}
          </button>

          {isDetailsOpen && (
            <div className="mt-3 space-y-2 rounded-md border border-[#2A3542] bg-[#11171D] p-3 text-xs font-mono">
              <p className="text-[#94A3B8] text-[11px] leading-relaxed border-b border-[#2A3542] pb-2">
                {card.rationale}
              </p>
              <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                <div>
                  <span className="text-[#64748B]">Latency: </span>
                  <span className="text-[#F1F5F9] font-bold">
                    {card.medianTimeMs ? `${card.medianTimeMs.toFixed(3)} ms` : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-[#64748B]">Bandwidth: </span>
                  <span className="text-[#F1F5F9] font-bold">
                    {card.derivedBandwidthGBps ? `${card.derivedBandwidthGBps.toFixed(2)} GB/s` : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-[#64748B]">Workgroup: </span>
                  <span className="text-[#F1F5F9]">({card.workgroupSize.join(", ")})</span>
                </div>
                <div>
                  <span className="text-[#64748B]">Shared Mem: </span>
                  <span className="text-[#F1F5F9]">{card.sharedMemoryBytes ? `${card.sharedMemoryBytes} B` : "0 B"}</span>
                </div>
                <div className="col-span-2">
                  <span className="text-[#64748B]">Rel Error: </span>
                  <span className="text-[#10B981]">
                    {card.maxRelativeError !== null
                      ? card.maxRelativeError === 0
                        ? "0.0 (Exact)"
                        : card.maxRelativeError.toExponential(2)
                      : "Verified (< 1e-4)"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Action: Inspect WGSL Button */}
      <button
        onClick={onInspect}
        className={`flex w-full items-center justify-center gap-2 rounded-md py-2 font-mono text-xs font-medium transition-colors border ${
          isSelectedForInspect
            ? "border-[#F59E0B] bg-[#1E2730] text-[#F59E0B]"
            : "border-[#2A3542] bg-[#11171D] text-[#94A3B8] hover:border-[#F59E0B]/50 hover:text-[#F1F5F9]"
        }`}
      >
        <Code2 className="h-3.5 w-3.5" />
        <span>Inspect WGSL</span>
      </button>
    </div>
  );
};
