import { VariantExecutionResult } from "@/types/vajra";

export interface Gen2UpliftInfo {
  upliftPct: number;
  formattedPct: string;
  winnerLabel: string;
  badgeText: string;
  activeGen1Winner: VariantExecutionResult;
}

/**
 * Computes Gen 2 uplift live from unrounded medians:
 * (D - active Gen 1 winner) / winner * 100
 */
export function computeGen2Uplift(
  variantD: VariantExecutionResult | null | undefined,
  variants: Array<VariantExecutionResult | null | undefined>
): Gen2UpliftInfo | null {
  if (
    !variantD ||
    variantD.id !== "variant-d" ||
    !variantD.verified ||
    variantD.gflops === null ||
    variantD.gflops <= 0
  ) {
    return null;
  }

  // Active Gen 1 winner: highest verified non-baseline, non-variant-d variant
  const gen1Candidates = variants
    .filter(
      (v): v is VariantExecutionResult =>
        Boolean(
          v &&
            v.id !== "variant-d" &&
            v.id !== "baseline" &&
            v.status !== "COMPILE_FAILED" &&
            v.status !== "VERIFY_FAILED" &&
            v.gflops !== null &&
            v.gflops > 0
        )
    )
    .sort((a, b) => (b.gflops || 0) - (a.gflops || 0));

  const activeGen1Winner = gen1Candidates[0];
  if (!activeGen1Winner || !activeGen1Winner.gflops) {
    return null;
  }

  // Compute live from unrounded medians: (D - active Gen 1 winner) / winner * 100
  const upliftPct =
    ((variantD.gflops - activeGen1Winner.gflops) / activeGen1Winner.gflops) * 100;
  const sign = upliftPct >= 0 ? "+" : "";
  const formattedPct = `${sign}${upliftPct.toFixed(1)}%`;
  const winnerLabel =
    activeGen1Winner.id === "variant-b"
      ? "Winner B"
      : activeGen1Winner.id === "variant-a"
      ? "Winner A"
      : "Winner C";
  const badgeText = `${formattedPct} vs Gen 1 ${winnerLabel}`;

  return {
    upliftPct,
    formattedPct,
    winnerLabel,
    badgeText,
    activeGen1Winner,
  };
}

/**
 * Robust cross-browser clipboard copy with textarea execCommand fallback.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (!text) return false;
  let copied = false;
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch (err) {
      console.warn("navigator.clipboard.writeText failed, attempting execCommand fallback:", err);
    }
  }

  if (!copied && typeof document !== "undefined") {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      textArea.setAttribute("readonly", "");
      document.body.appendChild(textArea);
      textArea.select();
      copied = document.execCommand("copy");
      document.body.removeChild(textArea);
    } catch (fallbackErr) {
      console.error("execCommand fallback copy failed:", fallbackErr);
    }
  }

  return copied;
}

/**
 * Generates a one-line plain-language error summary deterministically from measured data.
 * Uses: stage (compile / verification / generation), measured error vs 1e-4 limit where present,
 * first line of compiler message where present, and what Vajra did (quarantined, excluded from winner, fed to Gen 2).
 */
export function getDeterministicErrorSummary(variant: VariantExecutionResult): string {
  // 1. Generation Stage Failure (omitted/dropped by generator)
  const isGenerationFailure =
    variant.rationale?.toLowerCase().includes("generator returned") ||
    variant.rationale?.toLowerCase().includes("variant dropped") ||
    variant.compileErrors.some(
      (e) =>
        e.toLowerCase().includes("dropped or not returned") ||
        e.toLowerCase().includes("generator returned")
    );

  if (isGenerationFailure) {
    return "Failed at generation stage: generator omitted candidate code (quarantined, excluded from winner, fed to Gen 2).";
  }

  // 2. Compile Stage Failure
  if (variant.status === "COMPILE_FAILED") {
    const rawError =
      variant.compileErrors.length > 0
        ? variant.compileErrors[0]
        : "Shader compilation error";
    const firstLine = rawError.split("\n")[0].trim().replace(/\s+/g, " ");
    return `Failed at compile stage: ${firstLine} (quarantined, excluded from winner, fed to Gen 2).`;
  }

  // 3. Verification Stage Failure
  if (variant.status === "VERIFY_FAILED" || !variant.verified) {
    const errVal =
      variant.maxRelativeError !== null &&
      variant.maxRelativeError !== undefined &&
      !isNaN(variant.maxRelativeError)
        ? variant.maxRelativeError.toExponential(2)
        : "> 1.00e-4";
    return `Failed at verification stage: measured error ${errVal} vs 1e-4 limit (quarantined, excluded from winner, fed to Gen 2).`;
  }

  return "Passed verification.";
}

