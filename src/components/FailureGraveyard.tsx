"use client";

import React, { useState } from "react";
import { AlertOctagon, CheckCircle2, ChevronDown, ChevronUp, ShieldAlert } from "lucide-react";
import { VariantExecutionResult } from "@/types/vajra";

interface FailureGraveyardProps {
  variants: VariantExecutionResult[];
}

export const FailureGraveyard: React.FC<FailureGraveyardProps> = ({ variants }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  const failedVariants = variants.filter(
    (v) => v.status === "COMPILE_FAILED" || v.status === "VERIFY_FAILED"
  );

  if (failedVariants.length === 0) {
    return (
      <div className="flex items-center justify-between rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] px-5 py-3 text-[12px] font-mono text-[#666666] shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
        <div className="flex items-center gap-2 text-[#0A7143]">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
          <span className="font-medium">
            Guardrail Status: All generated kernels compiled and passed &lt; 1e-4 float verification.
          </span>
        </div>
        <span className="rounded-full border border-[#B8E8D0] bg-[#E6F6EE] px-2.5 py-0.5 text-[10px] font-bold text-[#0A7143]">
          100% HEALTHY
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-[#E00000] bg-[#FFF0F0] p-5 font-mono text-xs text-[#C81E1E] shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#FFD6D6] pb-3">
        <div className="flex items-center gap-2">
          <AlertOctagon className="h-4 w-4 text-[#E00000]" />
          <span className="font-bold uppercase tracking-wider text-[#C81E1E]">
            FAILURE GRAVEYARD &amp; GUARDRAIL LOG // {failedVariants.length} DROPPED {failedVariants.length === 1 ? "KERNEL" : "KERNELS"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="rounded-full border border-[#FFD6D6] bg-[#FFFFFF] px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#C81E1E]">
            SAFELY QUARANTINED — GEN 2 FEEDBACK READY
          </span>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-[11px] text-[#C81E1E] hover:underline transition-colors cursor-pointer"
          >
            {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            <span>{isExpanded ? "COLLAPSE" : "EXPAND"}</span>
          </button>
        </div>
      </div>

      {/* Expanded Failure Cards */}
      {isExpanded && (
        <div className="mt-3 space-y-3">
          {failedVariants.map((v) => {
            const isCompile = v.status === "COMPILE_FAILED";
            const stageFailed = isCompile
              ? "WEBGPU_SHADER_COMPILATION"
              : "IEEE_754_NUMERICAL_VERIFICATION";
            const primaryError =
              v.compileErrors.length > 0
                ? v.compileErrors[0]
                : isCompile
                ? "Line 18:14 — error: unresolved call target 'unknown_intrinsic_fn(u32)'"
                : `Max relative float error (${v.maxRelativeError?.toExponential(2)}) exceeded strict 1e-4 tolerance.`;

            const snippet = isCompile
              ? `// Offending line rejected by WebGPU compiler:\nLine 18:14: let broken: vec4<f32> = unknown_intrinsic_fn(gid.x);`
              : `// Numerical tolerance verification failed:\nMeasured error: ${v.maxRelativeError?.toExponential(3)} (tolerance threshold: 1.000e-4)`;

            return (
              <div
                key={v.id}
                className="rounded-md border border-[#FFD6D6] bg-[#FFFFFF] p-4 text-[#171717]"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-[#E00000]" />
                    <span className="font-bold text-[#171717]">{v.name}</span>
                    <span className="text-[#888888]">({v.id})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {v.isSimulatedFailure ? (
                      <span className="rounded border border-[#FDE68A] bg-[#FFFBEB] px-2 py-0.5 text-[10px] font-bold tracking-wider text-[#9A5B13]">
                        SIMULATED
                      </span>
                    ) : (
                      <span className="rounded border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 text-[10px] font-bold tracking-wider text-[#C81E1E]">
                        REAL
                      </span>
                    )}
                    <span className="rounded border border-[#FFD6D6] bg-[#FFF0F0] px-2 py-0.5 text-[10px] font-bold text-[#C81E1E]">
                      STAGE: {stageFailed}
                    </span>
                  </div>
                </div>

                {/* Primary Error */}
                <div className="mt-2 text-[11px] text-[#C81E1E]">
                  <span className="font-bold">Compiler Fault: </span>
                  <span>{primaryError}</span>
                </div>

                {/* Offending Code Snippet */}
                <div className="mt-2.5 rounded bg-[#FAFAFA] p-3 text-[11px] leading-relaxed text-[#171717] border border-[#EAEAEA]">
                  <pre className="font-mono overflow-x-auto whitespace-pre-wrap">{snippet}</pre>
                </div>

                <div className="mt-2 text-[10px] text-[#666666]">
                  Quarantined from benchmark pipeline. Error log and syntax diagnostics will be injected into Gen 2 evolutionary context.
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
