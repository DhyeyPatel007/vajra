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
