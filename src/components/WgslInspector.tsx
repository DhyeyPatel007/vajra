"use client";

import React, { useState } from "react";
import { Copy, Check, Code2, Trophy } from "lucide-react";
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
    <div className="border border-[#242936] bg-[#0E1014]">
      {/* Top Header Command Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#242936] bg-[#0A0C0F] px-5 py-3.5">
        <div className="flex items-center gap-2.5">
          <Code2 className="h-4 w-4 text-[#E5B842]" />
          <div>
            <div className="font-mono text-[11px] font-bold tracking-widest uppercase text-[#E5B842]">
              03 // WGSL COMPUTE SHADER INSPECTOR
            </div>
            <p className="mt-0.5 text-xs text-[#8E96A8]">
              Stitched Harness Binding Preamble + Physical GPU Executed Kernel Body
            </p>
          </div>
        </div>

        {/* Action Triggers */}
        <div className="flex items-center gap-2">
          {/* Copy Current Tab Button */}
          <button
            onClick={handleCopyCurrent}
            className="flex items-center gap-1.5 border border-[#242936] bg-[#151921] px-3.5 py-2 font-mono text-xs font-bold uppercase tracking-wider text-[#F4F6F8] transition-colors hover:border-[#8A6D1F]"
          >
            {copiedCurrent ? (
              <>
                <Check className="h-3.5 w-3.5 text-[#10B981]" />
                <span className="text-[#10B981]">Copied Tab WGSL!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5 text-[#8E96A8]" />
                <span>Copy This WGSL</span>
              </>
            )}
          </button>

          {/* Copy Fastest Winner Button */}
          <button
            onClick={handleCopyFastest}
            className="flex items-center gap-1.5 bg-[#E5B842] px-4 py-2 font-mono text-xs font-bold uppercase tracking-wider text-[#07080A] transition-all hover:bg-[#F5C542] active:translate-y-0.5"
          >
            {copiedFastest ? (
              <>
                <Check className="h-4 w-4 text-[#07080A]" />
                <span>Copied Verified Fastest WGSL!</span>
              </>
            ) : (
              <>
                <Trophy className="h-4 w-4 fill-current text-[#07080A]" />
                <span>Copy Fastest WGSL</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Tab Deck */}
      <div className="flex overflow-x-auto border-b border-[#242936] bg-[#0A0C0F]">
        {allItems.map((item) => {
          const isSelected = item.id === activeItem?.id;
          const isWinner = item.isWinner;

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center gap-2 border-r border-[#242936] px-4 py-2.5 font-mono text-xs font-bold uppercase tracking-wider transition-all ${
                isSelected
                  ? "border-b-2 border-b-[#E5B842] bg-[#151921] text-[#F4F6F8]"
                  : "bg-[#0A0C0F] text-[#8E96A8] hover:bg-[#12151C] hover:text-[#F4F6F8]"
              }`}
            >
              <span>{item.name}</span>
              {isWinner && (
                <span className="bg-[#E5B842] px-1.5 py-0.2 text-[9px] font-bold text-[#07080A]">
                  WINNER
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Code Viewport with Line-number Gutter */}
      <div className="max-h-[460px] overflow-auto bg-[#07080A] p-4 font-mono text-[13px] leading-relaxed">
        <table className="w-full border-collapse">
          <tbody>
            {codeLines.map((line, idx) => (
              <tr key={idx} className="hover:bg-[#0E1014]">
                <td className="w-12 select-none border-r border-[#181C26] pr-4 text-right font-mono text-[11px] text-[#5E6678] align-top">
                  {idx + 1}
                </td>
                <td className="whitespace-pre pl-4 font-mono text-[#F4F6F8] align-top">
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
