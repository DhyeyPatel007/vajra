"use client";

import React, { useState } from "react";
import {
  Code2,
  ChevronDown,
  ChevronUp,
  Trophy,
  AlertCircle,
  ShieldCheck,
  Check,
} from "lucide-react";
import {
  VariantExecutionResult,
  WorkloadId,
  getStrategyBadgeLabel,
} from "@/types/vajra";
import { computeGen2Uplift, copyToClipboard, formatGflops, getDeterministicErrorSummary } from "@/lib/benchmarkUtils";

interface VariantCardGridProps {
  baseline: VariantExecutionResult | null;
  variants: VariantExecutionResult[];
  onSelectVariantForInspect: (variant: VariantExecutionResult) => void;
  selectedInspectId?: string;
  workloadId?: WorkloadId;
  isRunning?: boolean;
}

export const VariantCardGrid: React.FC<VariantCardGridProps> = ({
  baseline,
  variants,
  onSelectVariantForInspect,
  selectedInspectId,
  workloadId,
  isRunning = false,
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
      <div className="flex items-center justify-between border-b border-[#EAEAEA] pb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-[16px] font-semibold tracking-[-0.02em] text-[#171717]">
            Kernel Candidates
          </h2>
          <span className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] px-2 py-0.5 font-mono text-[11px] text-[#666666]">
            {allCards.filter(Boolean).length} Variants Tested
          </span>
        </div>

        <button
          onClick={() => setExpandAll(!expandAll)}
          className="flex items-center gap-1.5 rounded-md border border-[#EAEAEA] bg-[#FFFFFF] px-3 py-1.5 font-mono text-[12px] text-[#666666] transition-colors hover:border-[#999999] hover:text-[#171717] cursor-pointer"
        >
          {expandAll ? (
            <>
              <ChevronUp className="h-3.5 w-3.5 text-[#171717]" />
              <span>Collapse All Card Details</span>
            </>
          ) : (
            <>
              <ChevronDown className="h-3.5 w-3.5 text-[#666666]" />
              <span>Expand All Card Details</span>
            </>
          )}
        </button>
      </div>

      {/* Spacious Grid */}
      <div
        className={`grid grid-cols-1 gap-5 md:grid-cols-2 ${
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
              variants={variants}
              defaultName={defaultName}
              indexLabel={indexLabels[index]}
              isSelectedForInspect={card?.id === selectedInspectId}
              onInspect={() => card && onSelectVariantForInspect(card)}
              workloadId={workloadId}
              forceExpand={expandAll}
              isRunning={isRunning}
            />
          );
        })}
      </div>
    </div>
  );
};

interface VariantCardItemProps {
  card: VariantExecutionResult | null;
  variants: VariantExecutionResult[];
  defaultName: string;
  indexLabel: string;
  isSelectedForInspect: boolean;
  onInspect: () => void;
  workloadId?: WorkloadId;
  forceExpand: boolean;
  isRunning?: boolean;
}

const VariantCardItem: React.FC<VariantCardItemProps> = ({
  card,
  variants,
  defaultName,
  indexLabel,
  isSelectedForInspect,
  onInspect,
  workloadId,
  forceExpand,
  isRunning = false,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [copiedInspect, setCopiedInspect] = useState(false);

  if (!card) {
    if (isRunning) {
      return (
        <div className="flex h-80 flex-col justify-between rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-5 opacity-50 shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
          <div>
            <span className="font-mono text-xs font-semibold text-[#888888]">
              {indexLabel}
            </span>
            <h3 className="mt-2 text-base font-semibold text-[#171717]">
              {defaultName}
            </h3>
          </div>
          <div className="text-center font-mono text-xs text-[#888888]">
            Waiting for dispatch...
          </div>
        </div>
      );
    }

    return (
      <div className="flex h-80 flex-col justify-between rounded-lg border border-[#FFD6D6] bg-[#FFF0F0]/40 p-5 shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
        <div>
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#666666]">
              {indexLabel}
            </span>
            <span className="rounded-full border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#C81E1E]">
              DROPPED
            </span>
          </div>
          <h3 className="mt-2 text-[14px] font-semibold text-[#171717]">
            {defaultName}
          </h3>
          <div className="mt-1">
            <span className="inline-block rounded border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 font-mono text-[11px] text-[#C81E1E]">
              Not Dispatched
            </span>
          </div>
          <p className="mt-2 text-[11.5px] font-mono text-[#C81E1E] leading-relaxed">
            Dropped: Variant not dispatched or pipeline completed without return.
          </p>
        </div>
        <div className="rounded border border-[#FFD6D6] bg-[#FFFFFF] p-2 text-center font-mono text-[11px] text-[#C81E1E]">
          Quarantined in Failure Graveyard
        </div>
      </div>
    );
  }

  const isWinner =
    card.id !== "baseline" &&
    Boolean(card.isWinner) &&
    (card.speedupVsBaseline === null || card.speedupVsBaseline > 1.0);
  const isCompileFailed = card.status === "COMPILE_FAILED";
  const isVerifyFailed = card.status === "VERIFY_FAILED";
  const isFailed = isCompileFailed || isVerifyFailed;

  const strategyBadge = getStrategyBadgeLabel(
    card.strategy,
    workloadId,
    card.name,
    card.sharedMemoryBytes
  );

  const uplift = card.id === "variant-d" ? computeGen2Uplift(card, variants) : null;
  const cleanRationale = card.rationale.replace(/\s*\([^)]*vs Gen 1[^)]*\)/i, "").trim();
  const displayRationale =
    card.id === "variant-d" && uplift
      ? (card.rationale.includes(uplift.badgeText) ? card.rationale : `${cleanRationale} (${uplift.badgeText})`)
      : card.rationale;

  const isDetailsOpen = forceExpand || showDetails || isFailed;

  return (
    <div
      className={`flex flex-col justify-between rounded-lg p-5 transition-all ${
        isWinner
          ? "border-[1.5px] border-[#171717] bg-[#FFFFFF] shadow-[0_4px_12px_rgba(0,0,0,0.05)]"
          : isFailed
          ? "border border-[#E00000] bg-[#FFF0F0]/50"
          : "border border-[#EAEAEA] bg-[#FFFFFF] shadow-[0_1px_2px_rgba(0,0,0,0.03)]"
      }`}
    >
      <div>
        {/* Top Row: Variant Label + Status Badge */}
        <div className="flex items-center justify-between">
          <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#666666]">
            {indexLabel}
          </span>
          {isWinner ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="flex items-center gap-1 rounded-full bg-[#171717] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#FFFFFF]">
                <Trophy className="h-3 w-3" />
                FASTEST WINNER
              </span>
              {card.speedupVsBaseline !== null && card.speedupVsBaseline < 1.0 && (
                <span className="rounded-full border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#C81E1E]">
                  Slower than baseline
                </span>
              )}
            </div>
          ) : card.id === "baseline" ? (
            <span className="rounded-full border border-[#EAEAEA] bg-[#F2F2F2] px-2 py-0.5 font-mono text-[10.5px] text-[#666666]">
              BASELINE
            </span>
          ) : isCompileFailed ? (
            <span className="rounded-full border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#C81E1E]">
              COMPILE_FAILED
            </span>
          ) : isVerifyFailed ? (
            <span className="rounded-full border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#C81E1E]">
              VERIFY_FAILED
            </span>
          ) : card.status === "VERIFIED_RUNNER_UP" ? (
            <span className="flex items-center gap-1 rounded-full border border-[#B8E8D0] bg-[#E6F6EE] px-2 py-0.5 font-mono text-[10.5px] font-medium text-[#0A7143]">
              <ShieldCheck className="h-3 w-3" />
              VERIFIED
            </span>
          ) : (
            <span className="rounded-full border border-[#EAEAEA] bg-[#F2F2F2] px-2 py-0.5 font-mono text-[10.5px] text-[#666666]">
              {card.status}
            </span>
          )}
        </div>

        {/* Title + Strategy Badge + Gen 2 Uplift Badge */}
        <div className="mt-2.5">
          <h3 className="text-[14px] font-semibold text-[#171717] line-clamp-1">
            {card.name}
          </h3>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <span className="inline-block rounded border border-[#EAEAEA] bg-[#F2F2F2] px-2 py-0.5 font-mono text-[11px] text-[#666666]">
              {strategyBadge}
            </span>
            {uplift && (
              <span
                className={`inline-block rounded px-2 py-0.5 font-mono text-[11px] font-medium border ${
                  uplift.upliftPct >= 0
                    ? "border-[#B8E8D0] bg-[#E6F6EE] text-[#0A7143]"
                    : "border-[#EAEAEA] bg-[#F2F2F2] text-[#666666]"
                }`}
              >
                {uplift.badgeText}
              </span>
            )}
          </div>
        </div>

        {/* Big Scannable Numbers */}
        {!isFailed ? (
          <div className="my-3.5 flex items-baseline justify-between rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-3 font-mono">
            <div>
              <div className="text-[10px] text-[#888888] uppercase">THROUGHPUT</div>
              <div className="text-[22px] font-semibold text-[#171717] tabular-nums leading-tight whitespace-nowrap">
                {formatGflops(card.gflops)}{" "}
                <span className="text-xs font-normal text-[#666666]">GFLOPS</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-[#888888] uppercase">SPEEDUP</div>
              <div className="text-[16px] font-semibold text-[#0070F3] tabular-nums leading-tight whitespace-nowrap">
                {card.id === "baseline"
                  ? "1.00x"
                  : `${card.speedupVsBaseline ? card.speedupVsBaseline.toFixed(2) : "1.00"}x`}
              </div>
            </div>
          </div>
        ) : (
          <div className="my-3.5 rounded-md border border-[#FFD6D6] bg-[#FFF0F0] p-3 font-mono text-xs text-[#C81E1E]">
            <div className="flex items-center gap-1.5 font-semibold">
              <AlertCircle className="h-4 w-4 shrink-0 text-[#E00000]" />
              <span>{isCompileFailed ? "COMPILER REJECTION" : "QUARANTINED — WRONG ANSWER"}</span>
            </div>
            <div className="mt-2 text-[11.5px] leading-snug font-sans text-[#C81E1E] font-medium">
              {getDeterministicErrorSummary(card)}
            </div>
            {card.compileErrors.length > 0 && (
              <div className="mt-2 max-h-24 overflow-y-auto rounded bg-[#FFFFFF] p-2 text-[11px] text-[#C81E1E] border border-[#FFD6D6]">
                {card.compileErrors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Progressive Disclosure Toggle */}
        <div className="mb-3">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 text-[12px] font-mono text-[#666666] hover:text-[#171717] transition-colors cursor-pointer"
          >
            <span>{isDetailsOpen ? "Hide Telemetry & Rationale" : "Show Telemetry & Rationale"}</span>
            {isDetailsOpen ? (
              <ChevronUp className="h-3 w-3 text-[#171717]" />
            ) : (
              <ChevronDown className="h-3 w-3 text-[#666666]" />
            )}
          </button>

          {isDetailsOpen && (
            <div className="mt-2.5 space-y-2 rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-[12px] font-mono text-[#171717]">
              <p className="text-[#666666] text-[11.5px] leading-relaxed border-b border-[#EAEAEA] pb-2">
                {displayRationale}
              </p>
              <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                <div>
                  <span className="text-[#888888]">Latency: </span>
                  <span className="text-[#171717] font-semibold">
                    {card.medianTimeMs ? `${card.medianTimeMs.toFixed(3)} ms` : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-[#888888]">Bandwidth: </span>
                  <span className="text-[#171717] font-semibold">
                    {card.derivedBandwidthGBps ? `${card.derivedBandwidthGBps.toFixed(2)} GB/s` : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-[#888888]">Workgroup: </span>
                  <span className="text-[#171717]">({card.workgroupSize.join(", ")})</span>
                </div>
                <div>
                  <span className="text-[#888888]">Shared Mem: </span>
                  <span className="text-[#171717]">{card.sharedMemoryBytes ? `${card.sharedMemoryBytes} B` : "0 B"}</span>
                </div>
                <div className="col-span-2">
                  <span className="text-[#888888]">Rel Error: </span>
                  <span className={`font-semibold ${
                    card.status === "COMPILE_FAILED"
                      ? "text-[#666666]"
                      : card.status === "VERIFY_FAILED"
                      ? "text-[#C81E1E]"
                      : "text-[#0A7143]"
                  }`}>
                    {card.status === "COMPILE_FAILED"
                      ? "—"
                      : card.status === "VERIFY_FAILED"
                      ? `${card.maxRelativeError !== null ? card.maxRelativeError.toExponential(2) : "—"} (Exceeds 1e-4)`
                      : card.maxRelativeError !== null
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
        onClick={async () => {
          if (card?.fullWgslCode) {
            await copyToClipboard(card.fullWgslCode);
            setCopiedInspect(true);
            setTimeout(() => setCopiedInspect(false), 2200);
          }
          onInspect();
          if (typeof document !== "undefined") {
            const el = document.getElementById("wgsl-inspector");
            if (el) {
              el.scrollIntoView({ behavior: "smooth", block: "start" });
            }
          }
        }}
        className={`flex w-full items-center justify-center gap-2 rounded-md py-2 font-mono text-[12px] font-medium transition-colors border cursor-pointer ${
          copiedInspect
            ? "border-[#059669] bg-[#ECFDF5] text-[#065F46]"
            : isSelectedForInspect
            ? "border-[#171717] bg-[#171717] text-[#FFFFFF]"
            : "border-[#EAEAEA] bg-[#FFFFFF] text-[#171717] hover:border-[#999999] hover:bg-[#FAFAFA]"
        }`}
      >
        {copiedInspect ? (
          <>
            <Check className="h-3.5 w-3.5 text-[#059669]" />
            <span className="text-[#065F46] font-semibold">Copied &amp; Inspecting WGSL ↓</span>
          </>
        ) : isSelectedForInspect ? (
          <>
            <Code2 className="h-3.5 w-3.5" />
            <span>Inspecting in Editor ↓</span>
          </>
        ) : (
          <>
            <Code2 className="h-3.5 w-3.5" />
            <span>Inspect WGSL</span>
          </>
        )}
      </button>
    </div>
  );
};
