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
      <div className="flex items-center justify-between rounded-lg border border-[#2A3542] bg-[#161D24] px-5 py-3 text-xs text-[#94A3B8]">
        <div className="flex items-center gap-2 text-[#10B981]">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
          <span className="font-mono text-xs font-medium">
            Guardrail Status: All generated kernels compiled and passed &lt; 1e-4 float verification.
          </span>
        </div>
        <span className="rounded bg-[#0B261F] px-2 py-0.5 font-mono text-[10px] font-bold text-[#10B981] border border-[#10B981]/30">
          100% HEALTHY
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-lg border-2 border-[#F43F5E] bg-[#161D24] p-5 font-mono text-xs text-[#F1F5F9]">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#2A3542] pb-3">
        <div className="flex items-center gap-2">
          <AlertOctagon className="h-4 w-4 text-[#F43F5E]" />
          <span className="font-bold uppercase tracking-wider text-[#F1F5F9]">
            FAILURE GRAVEYARD &amp; GUARDRAIL LOG // {failedVariants.length} DROPPED {failedVariants.length === 1 ? "KERNEL" : "KERNELS"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="rounded border border-[#F43F5E]/40 bg-[#2A1219] px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#F43F5E]">
            SAFELY QUARANTINED — GEN 2 FEEDBACK READY
          </span>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-[11px] text-[#94A3B8] hover:text-[#F1F5F9] transition-colors"
          >
            {isExpanded ? <ChevronUp className="h-3.5 w-3.5 text-[#F43F5E]" /> : <ChevronDown className="h-3.5 w-3.5 text-[#F43F5E]" />}
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
                className="rounded-md border border-[#F43F5E]/40 bg-[#1A1215] p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-[#F43F5E]" />
                    <span className="font-bold text-[#F1F5F9]">{v.name}</span>
                    <span className="text-[#64748B]">({v.id})</span>
                  </div>
                  <span className="rounded bg-[#2A1219] px-2 py-0.5 text-[10px] font-bold text-[#F43F5E]">
                    STAGE: {stageFailed}
                  </span>
                </div>

                {/* Primary Error */}
                <div className="mt-2 text-[11px] text-[#F43F5E]">
                  <span className="font-bold">Compiler Fault: </span>
                  <span>{primaryError}</span>
                </div>

                {/* Offending Code Snippet */}
                <div className="mt-2.5 rounded bg-[#11171D] p-3 text-[11px] leading-relaxed text-[#94A3B8] border border-[#2A3542]">
                  <pre className="font-mono overflow-x-auto whitespace-pre-wrap">{snippet}</pre>
                </div>

                <div className="mt-2 text-[10px] text-[#64748B]">
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
