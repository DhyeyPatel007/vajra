"use client";

import React, { useState } from "react";
import { Copy, Check, Code2, Trophy, ShieldCheck, AlertTriangle } from "lucide-react";
import { VariantExecutionResult } from "@/types/vajra";

interface WgslInspectorProps {
  variants: VariantExecutionResult[];
  baseline: VariantExecutionResult | null;
  winner: VariantExecutionResult | null;
  selectedVariantId: string;
  onSelectTab: (id: string) => void;
}

export const WgslInspector: React.FC<WgslInspectorProps> = ({
  variants,
  baseline,
  winner,
  selectedVariantId,
  onSelectTab,
}) => {
  const [copiedFastest, setCopiedFastest] = useState(false);
  const [copiedCurrent, setCopiedCurrent] = useState(false);

  const allItems: VariantExecutionResult[] = [];
  if (baseline) allItems.push(baseline);
  variants.forEach((v) => allItems.push(v));

  const activeItem =
    allItems.find((item) => item.id === selectedVariantId) ||
    winner ||
    baseline ||
    allItems[0];

  const handleCopyFastest = async () => {
    const codeToCopy = winner?.fullWgslCode || activeItem?.fullWgslCode || "";
    if (codeToCopy) {
      await navigator.clipboard.writeText(codeToCopy);
      setCopiedFastest(true);
      setTimeout(() => setCopiedFastest(false), 2200);
    }
  };

  const handleCopyCurrent = async () => {
    if (activeItem?.fullWgslCode) {
      await navigator.clipboard.writeText(activeItem.fullWgslCode);
      setCopiedCurrent(true);
      setTimeout(() => setCopiedCurrent(false), 2200);
    }
  };

  const codeLines = (activeItem?.fullWgslCode || "// No WGSL shader loaded yet.").split("\n");

  return (
    <div className="rounded border border-[#262A34] bg-[#121418] p-5">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#262A34] pb-4">
        <div className="flex items-center gap-2">
          <Code2 className="h-5 w-5 text-[#D4AF37]" />
          <div>
            <h2 className="text-sm font-bold tracking-wide uppercase text-[#F3F4F6]">
              WGSL Compute Shader Inspector
            </h2>
            <p className="text-xs text-[#9CA3AF]">
              Stitched Harness Binding Preamble + Physical GPU Executed Kernel Body
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopyCurrent}
            className="flex items-center gap-1.5 rounded border border-[#262A34] bg-[#181B22] px-3 py-1.5 text-xs font-semibold text-[#F3F4F6] transition-colors hover:border-[#383E4D]"
          >
            {copiedCurrent ? (
              <>
                <Check className="h-3.5 w-3.5 text-[#10B981]" />
                <span className="text-[#10B981]">Copied Tab WGSL!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5 text-[#9CA3AF]" />
                <span>Copy This WGSL</span>
              </>
            )}
          </button>

          <button
            onClick={handleCopyFastest}
            className="flex items-center gap-1.5 rounded bg-[#D4AF37] px-3.5 py-1.5 text-xs font-bold text-[#0B0C0E] transition-all hover:bg-[#F5C542] shadow-sm active:scale-95"
          >
            {copiedFastest ? (
              <>
                <Check className="h-4 w-4 text-[#0B0C0E]" />
                <span>Copied Verified Fastest WGSL!</span>
              </>
            ) : (
              <>
                <Trophy className="h-4 w-4 fill-current text-[#0B0C0E]" />
                <span>Copy Fastest WGSL</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="mt-3 flex flex-wrap items-center gap-2 border-b border-[#262A34] pb-2">
        {allItems.map((item) => {
          const isSelected = item.id === activeItem?.id;
          const isWinner = item.isWinner;
          const isFailed =
            item.status === "COMPILE_FAILED" || item.status === "VERIFY_FAILED";

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-semibold transition-all ${
                isSelected
                  ? "border border-[#D4AF37] bg-[#181B22] text-[#D4AF37]"
                  : "border border-transparent bg-[#0B0C0E] text-[#9CA3AF] hover:text-[#F3F4F6]"
              }`}
            >
              {isWinner ? (
                <Trophy className="h-3 w-3 text-[#D4AF37]" />
              ) : isFailed ? (
                <AlertTriangle className="h-3 w-3 text-[#EF4444]" />
              ) : (
                <ShieldCheck className="h-3 w-3 text-[#10B981]" />
              )}
              <span>{item.name}</span>
              {isWinner && (
                <span className="rounded bg-[#D4AF37] px-1.5 py-0.2 text-[9px] font-bold text-[#0B0C0E]">
                  WINNER
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Code Viewer */}
      <div className="mt-3 relative max-h-96 overflow-y-auto rounded border border-[#262A34] bg-[#07080A] p-4 font-mono text-xs text-[#F3F4F6]">
        <table className="w-full border-collapse">
          <tbody>
            {codeLines.map((line, idx) => (
              <tr key={idx} className="hover:bg-[#121418]">
                <td className="w-10 select-none text-right pr-4 text-[#4B5563] text-[11px] align-top">
                  {idx + 1}
                </td>
                <td className="whitespace-pre font-mono text-[#F3F4F6] align-top">
                  {line}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
