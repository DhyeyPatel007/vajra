"use client";

import React from "react";
import Link from "next/link";
import {
  ArrowLeft,
  CheckCircle2,
  Cpu,
  ShieldCheck,
  AlertOctagon,
  Copy,
  Terminal,
  Zap,
  HelpCircle,
  Layers,
  ArrowRight,
} from "lucide-react";

export default function GuidePage() {
  const [copiedSnippet, setCopiedSnippet] = React.useState<string | null>(null);

  const handleCopy = (text: string, id: string) => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopiedSnippet(id);
      setTimeout(() => setCopiedSnippet(null), 2000);
    }
  };

  const naiveWgslSnippet = `@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  let N = uniforms.N;
  if (row >= N || col >= N) { return; }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < N; k = k + 1u) {
    sum = sum + A[row * N + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}`;

  const tiledWgslSnippet = `var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>
) {
  let N = uniforms.N;
  let row = group_id.x * 16u + local_id.x;
  let col = group_id.y * 16u + local_id.y;
  var sum: f32 = 0.0;

  for (var t: u32 = 0u; t < N; t = t + 16u) {
    if (row < N && (t + local_id.y) < N) {
      tileA[local_id.x][local_id.y] = A[row * N + (t + local_id.y)];
    } else {
      tileA[local_id.x][local_id.y] = 0.0;
    }
    if ((t + local_id.x) < N && col < N) {
      tileB[local_id.x][local_id.y] = B[(t + local_id.x) * N + col];
    } else {
      tileB[local_id.x][local_id.y] = 0.0;
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      sum = sum + tileA[local_id.x][k] * tileB[k][local_id.y];
    }
    workgroupBarrier();
  }
  if (row < N && col < N) {
    C[row * N + col] = sum;
  }
}`;

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-[#171717] font-sans antialiased selection:bg-[#EAEAEA]">
      {/* Top Header */}
      <header className="sticky top-0 z-20 border-b border-[#EAEAEA] bg-[#FFFFFF]/90 backdrop-blur-sm">
        <div className="mx-auto flex max-w-[1000px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              onClick={(e) => {
                if (typeof window !== "undefined" && window.history.length > 1) {
                  e.preventDefault();
                  window.history.back();
                }
              }}
              className="inline-flex items-center gap-1.5 font-mono text-xs font-semibold text-[#666666] hover:text-[#171717] transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>← Back to Tuner</span>
            </Link>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded bg-[#171717] text-xs font-bold text-[#FFFFFF]">
              V
            </div>
            <span className="font-mono text-xs font-semibold tracking-wider text-[#171717]">
              VAJRA // USER GUIDE
            </span>
          </div>
        </div>
      </header>

      {/* Main Guide Content */}
      <main className="mx-auto max-w-[1000px] px-6 py-10 space-y-12">
        {/* Title & Introduction Banner */}
        <section className="space-y-3 border-b border-[#EAEAEA] pb-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-[#EAEAEA] bg-[#FFFFFF] px-3 py-1 font-mono text-[11px] font-medium text-[#666666]">
            <span>ZERO API CALLS • STATIC REFERENCE • NO CREDITS USED</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-[#171717] sm:text-4xl">
            How to use Vajra
          </h1>
          <p className="text-sm text-[#666666]">
            A complete guide to hardware-verified WebGPU kernel optimization with NVIDIA Nemotron.
          </p>
        </section>

        {/* 1. What is Vajra */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-3">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            01 / Definition
          </div>
          <h2 className="text-xl font-bold text-[#171717]">What is Vajra</h2>
          <p className="text-base font-medium text-[#171717] leading-relaxed">
            &ldquo;AI writes GPU shaders, but only your hardware knows the truth. Vajra benchmarks every Nemotron kernel live on your GPU and keeps the fastest.&rdquo;
          </p>
        </section>

        {/* 2. What is WebGPU / WGSL */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-3">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            02 / Core Technology
          </div>
          <h2 className="text-xl font-bold text-[#171717]">What is WebGPU / WGSL</h2>
          <div className="space-y-1.5 text-sm text-[#444444] leading-relaxed">
            <p>
              WebGPU is the browser API that lets a web page run code on the user&apos;s GPU, no install.
            </p>
            <p>
              WGSL is the shader language for that code.
            </p>
            <p>
              Vajra tunes WGSL kernels.
            </p>
          </div>
        </section>

        {/* 3. The Problem */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-4">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            03 / Motivation
          </div>
          <h2 className="text-xl font-bold text-[#171717]">The Problem</h2>
          <p className="text-sm font-medium text-[#171717] leading-relaxed">
            The same shader is fast on one GPU and slow on another (AMD vs Apple vs NVIDIA). Hand-tuning takes hours and guessing risks wrong maths. AI alone can write code that is slow or wrong — with no check.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="rounded border border-[#EAEAEA] bg-[#FAFAFA] p-4 text-xs space-y-1.5">
              <span className="font-mono font-bold text-[#171717]">Hardware Divergence</span>
              <p className="text-[#666666]">
                Different warp/wavefront widths (32 threads on NVIDIA vs 64 threads on AMD RDNA), register pressure limits, and L1 cache sizes cause identical code to perform wildly differently across chipsets.
              </p>
            </div>
            <div className="rounded border border-[#EAEAEA] bg-[#FAFAFA] p-4 text-xs space-y-1.5">
              <span className="font-mono font-bold text-[#171717]">Exhausting Manual Search</span>
              <p className="text-[#666666]">
                Tuning workgroup sizes, unroll factors, vectorization, and shared-memory tile padding requires testing dozens of subtle permutations per hardware target.
              </p>
            </div>
            <div className="rounded border border-[#EAEAEA] bg-[#FAFAFA] p-4 text-xs space-y-1.5">
              <span className="font-mono font-bold text-[#171717]">Silent Hallucination</span>
              <p className="text-[#666666]">
                LLMs write convincing shader code that introduces data races, missing barriers, out-of-bounds indexing, or subtle precision drift that crashes the pipeline or corrupts neural weights.
              </p>
            </div>
          </div>
        </section>

        {/* 4. How Vajra Works */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-6">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            04 / Pipeline Architecture
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#171717]">How Vajra Works</h2>
            <p className="text-xs text-[#666666] mt-1">
              Five automated steps from prompt to verified, hardware-native speedup:
            </p>
          </div>

          {/* Simple Diagram */}
          <div className="rounded-md border border-[#EAEAEA] bg-[#FAFAFA] p-4">
            <div className="font-mono text-[11px] font-semibold text-[#888888] mb-3">
              PIPELINE EXECUTION FLOW
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 text-center font-mono text-xs">
              <div className="rounded border border-[#EAEAEA] bg-[#FFFFFF] p-2.5 flex flex-col justify-center items-center shadow-xs">
                <span className="text-[10px] text-[#888888]">STEP 1</span>
                <span className="font-bold text-[#171717] mt-0.5">Select Workload</span>
              </div>
              <div className="rounded border border-[#EAEAEA] bg-[#FFFFFF] p-2.5 flex flex-col justify-center items-center shadow-xs">
                <span className="text-[10px] text-[#888888]">STEP 2</span>
                <span className="font-bold text-[#171717] mt-0.5">Naive Baseline</span>
              </div>
              <div className="rounded border border-[#EAEAEA] bg-[#FFFFFF] p-2.5 flex flex-col justify-center items-center shadow-xs">
                <span className="text-[10px] text-[#888888]">STEP 3</span>
                <span className="font-bold text-[#171717] mt-0.5">Nemotron Variants</span>
              </div>
              <div className="rounded border border-[#EAEAEA] bg-[#FFFFFF] p-2.5 flex flex-col justify-center items-center shadow-xs">
                <span className="text-[10px] text-[#888888]">STEP 4</span>
                <span className="font-bold text-[#171717] mt-0.5">Verify &amp; Benchmark</span>
              </div>
              <div className="rounded border border-[#059669]/30 bg-[#ECFDF5] p-2.5 flex flex-col justify-center items-center shadow-xs">
                <span className="text-[10px] text-[#059669] font-bold">STEP 5</span>
                <span className="font-bold text-[#065F46] mt-0.5">Fastest Wins</span>
              </div>
            </div>
          </div>

          {/* 5 Numbered Steps */}
          <ol className="space-y-3 font-sans text-sm text-[#333333]">
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-xs font-bold text-[#FFFFFF]">
                1
              </span>
              <div>
                <strong className="text-[#171717]">Select workload:</strong> Pick a computationally intensive primitive (SGEMM, Softmax, Conv1D, or Sobel Filter) to optimize.
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-xs font-bold text-[#FFFFFF]">
                2
              </span>
              <div>
                <strong className="text-[#171717]">Run naive baseline:</strong> Vajra compiles the reference unoptimized shader and benchmarks it live on your local GPU to establish the ground-truth timing and floating-point reference.
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-xs font-bold text-[#FFFFFF]">
                3
              </span>
              <div>
                <strong className="text-[#171717]">Nemotron writes 3 variants:</strong> NVIDIA Nemotron (via Nebius Token Factory or NVIDIA NIM) generates 3 diverse WGSL optimization hypotheses (e.g. shared memory tiling, register caching, loop unrolling).
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-xs font-bold text-[#FFFFFF]">
                4
              </span>
              <div>
                <strong className="text-[#171717]">Compile + verify to 1e-4 + benchmark 9-run median on YOUR GPU:</strong> Each generated kernel is compiled directly in your browser. Vajra compares computed output against baseline ground truth (max allowed diff &le; 1e-4) and measures execution time across 9 GPU passes using high-resolution timestamp queries.
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#059669] font-mono text-xs font-bold text-[#FFFFFF]">
                5
              </span>
              <div>
                <strong className="text-[#171717]">Fastest verified wins, slow/wrong variants go to Failure Graveyard:</strong> The fastest numerically verified variant is declared champion with 1-click &ldquo;Copy Fastest WGSL&rdquo;. Any code that failed compilation or exceeded 1e-4 tolerance is isolated in the Failure Graveyard.
              </div>
            </li>
          </ol>
        </section>

        {/* 5. Modes Explained */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-4">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            05 / System Modes
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#171717]">Modes Explained</h2>
            <p className="text-xs text-[#666666] mt-1">
              Vajra provides four distinct operating modes to accommodate live hackathon submissions, local testing, and non-WebGPU devices:
            </p>
          </div>

          <div className="overflow-x-auto rounded border border-[#EAEAEA]">
            <table className="w-full border-collapse font-sans text-xs text-left">
              <thead>
                <tr className="border-b border-[#EAEAEA] bg-[#FAFAFA] font-mono text-[11px] text-[#666666]">
                  <th className="p-3">Mode</th>
                  <th className="p-3">How it works</th>
                  <th className="p-3">Benchmarking</th>
                  <th className="p-3">Status / Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#EAEAEA] text-[#333333]">
                <tr>
                  <td className="p-3 font-mono font-bold text-[#171717]">Mock</td>
                  <td className="p-3">Pre-written variants, benchmarking is live.</td>
                  <td className="p-3 text-[#059669] font-medium">Live on your GPU (9-run median)</td>
                  <td className="p-3 text-[#666666]">Default demonstration, instant, no API key required.</td>
                </tr>
                <tr>
                  <td className="p-3 font-mono font-bold text-[#171717]">Live Token Factory</td>
                  <td className="p-3">Nemotron generates variants via Nebius Token Factory.</td>
                  <td className="p-3 text-[#059669] font-medium">Live on your GPU (9-run median)</td>
                  <td className="p-3 text-[#171717] font-semibold">Official submission path for hackathon evaluation.</td>
                </tr>
                <tr>
                  <td className="p-3 font-mono font-bold text-[#171717]">Live NVIDIA</td>
                  <td className="p-3">Nemotron generates directly via NVIDIA NIM endpoint.</td>
                  <td className="p-3 text-[#059669] font-medium">Live on your GPU (9-run median)</td>
                  <td className="p-3 text-[#666666]">Check only, not submission proof.</td>
                </tr>
                <tr>
                  <td className="p-3 font-mono font-bold text-[#171717]">Preview</td>
                  <td className="p-3">No WebGPU, recorded reference only.</td>
                  <td className="p-3 text-[#888888]">Pre-recorded reference (Apple M3 Max)</td>
                  <td className="p-3 text-[#666666]">For headless servers or browsers without WebGPU.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 6. Larger-Project Worked Example */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-6">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            06 / Technical Integration
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#171717]">Larger-Project Worked Example: In-Browser LLM / Video Processing</h2>
            <p className="text-xs text-[#666666] mt-1">
              How a production engineering team drops Vajra-tuned kernels into a transformer inference engine or video processing pipeline.
            </p>
          </div>

          <div className="rounded border border-[#EAEAEA] bg-[#FAFAFA] p-4 text-xs text-[#444444] space-y-2 leading-relaxed">
            <p>
              In client-side LLM inference (e.g. running Llama or Whisper in WebGPU) or realtime canvas video processing, matrix multiplication (SGEMM) executes hundreds of times per token or frame.
            </p>
            <p>
              A <strong>2&times; to 3&times; SGEMM speedup</strong> compounds directly into reduced time-to-first-token, higher token throughput, and reduced thermal throttling on client laptops and workstations.
            </p>
          </div>

          {/* Snippets Side-by-Side / Stacked */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Baseline */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold text-[#888888]">
                  BASELINE WGSL (NAIVE GLOBAL READS)
                </span>
                <button
                  onClick={() => handleCopy(naiveWgslSnippet, "naive")}
                  className="inline-flex items-center gap-1 font-mono text-[10px] text-[#666666] hover:text-[#171717]"
                >
                  <Copy className="h-3 w-3" />
                  <span>{copiedSnippet === "naive" ? "Copied" : "Copy"}</span>
                </button>
              </div>
              <pre className="overflow-x-auto rounded border border-[#EAEAEA] bg-[#171717] p-3 font-mono text-[11px] leading-relaxed text-[#D4D4D4]">
                <code>{naiveWgslSnippet}</code>
              </pre>
            </div>

            {/* Winner */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold text-[#059669]">
                  WINNER WGSL (TILED SHARED-MEMORY)
                </span>
                <button
                  onClick={() => handleCopy(tiledWgslSnippet, "tiled")}
                  className="inline-flex items-center gap-1 font-mono text-[10px] text-[#666666] hover:text-[#171717]"
                >
                  <Copy className="h-3 w-3" />
                  <span>{copiedSnippet === "tiled" ? "Copied" : "Copy"}</span>
                </button>
              </div>
              <pre className="overflow-x-auto rounded border border-[#059669]/30 bg-[#171717] p-3 font-mono text-[11px] leading-relaxed text-[#D4D4D4]">
                <code>{tiledWgslSnippet}</code>
              </pre>
            </div>
          </div>

          {/* Integration Location & Labelled Numbers */}
          <div className="space-y-3 pt-2">
            <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-[#171717]">
              Exact Paste Location in Your Codebase
            </h3>
            <div className="rounded border border-[#EAEAEA] bg-[#FAFAFA] p-4 text-xs space-y-2">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-[#666666]" />
                <span className="font-mono font-semibold text-[#171717]">
                  src/gpu/matmul.wgsl (or compute pipeline shader module)
                </span>
              </div>
              <p className="text-[#666666]">
                Replace the compute kernel entrypoint function in your project file. Keep the exact same <code className="rounded bg-[#FFFFFF] px-1 py-0.5 border border-[#EAEAEA] font-mono">@group</code> and <code className="rounded bg-[#FFFFFF] px-1 py-0.5 border border-[#EAEAEA] font-mono">@binding</code> buffer harness so your WebGPU pipeline layout, bind groups, and dispatch dimensions remain unchanged. Rebuild your application and run the target inference pipeline.
              </p>
            </div>

            {/* Labelled Example Numbers */}
            <div className="rounded border border-[#EAEAEA] bg-[#FFFFFF] p-4 text-xs space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#EAEAEA] pb-2">
                <span className="font-mono font-semibold text-[#171717]">
                  Benchmark Comparison
                </span>
                <span className="font-mono text-[11px] rounded bg-[#FAFAFA] px-2 py-0.5 border border-[#EAEAEA] text-[#666666]">
                  Example, AMD RDNA-3, Oct 27 — your numbers will differ per GPU
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                <div className="rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3">
                  <span className="text-[11px] text-[#888888] block">Naive Baseline (Global Memory)</span>
                  <span className="font-mono text-base font-bold text-[#666666]">4.22 ms</span>
                  <span className="font-mono text-xs text-[#888888] ml-2">~12.2 GFLOPS</span>
                </div>
                <div className="rounded border border-[#059669]/30 bg-[#ECFDF5] p-3">
                  <span className="text-[11px] text-[#065F46] font-semibold block">Winner: Tiled Shared-Memory (Gen 1)</span>
                  <span className="font-mono text-base font-bold text-[#059669]">1.83 ms</span>
                  <span className="font-mono text-xs text-[#065F46] ml-2 font-semibold">~28.1 GFLOPS (2.31&times; speedup)</span>
                </div>
              </div>
              <p className="text-[11px] text-[#888888] pt-1">
                Note: The above figures are from an individual test run on an AMD RDNA-3 GPU; they are not medians across all devices. Vajra always measures live on your exact silicon.
              </p>
            </div>
          </div>
        </section>

        {/* 7. How to Use Vajra (8 Steps) */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-4">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            07 / Workflow Checklist
          </div>
          <h2 className="text-xl font-bold text-[#171717]">How to Use Vajra (8 Steps)</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">1</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Open Demo</strong>
                <span className="text-[#666666]">Load Vajra in any modern WebGPU-capable browser (Chrome 113+, Edge 113+).</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">2</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Check Adapter / Timestamps</strong>
                <span className="text-[#666666]">Verify hardware detection and timestamp query availability in the header status indicator.</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">3</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Choose Mode</strong>
                <span className="text-[#666666]">Select &ldquo;Mock Mode&rdquo; (instant offline benchmark) or &ldquo;Live Token Factory&rdquo; (submission generation).</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">4</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Select Workload</strong>
                <span className="text-[#666666]">Choose the target compute kernel: SGEMM, Softmax, Conv1D, or Sobel Filter.</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">5</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Tune</strong>
                <span className="text-[#666666]">Click &ldquo;Tune [Workload]&rdquo; to trigger live kernel generation and automatic GPU verification.</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">6</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Read Chart + Guardrail</strong>
                <span className="text-[#666666]">Examine the GFLOPS speedup bar chart, execution times, and floating-point verification badges.</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#171717] font-mono text-[10px] font-bold text-white">7</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Refine Winner (Gen 2)</strong>
                <span className="text-[#666666]">If wanted, click &ldquo;Refine Winner (Gen 2)&rdquo; to feed the champion back to Nemotron for additional gains.</span>
              </div>
            </div>

            <div className="flex items-start gap-3 rounded border border-[#EAEAEA] bg-[#FAFAFA] p-3 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#059669] font-mono text-[10px] font-bold text-white">8</span>
              <div>
                <strong className="text-[#171717] block mb-0.5">Copy Fastest WGSL &amp; Rebuild</strong>
                <span className="text-[#666666]">Click &ldquo;Copy Fastest WGSL&rdquo;, paste into your project shader file, and rebuild your application.</span>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Guardrail in Plain English */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-4">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            08 / Correctness Engine
          </div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-[#059669]" />
            <h2 className="text-xl font-bold text-[#171717]">Guardrail, in Plain English</h2>
          </div>
          <p className="text-sm font-semibold text-[#171717] leading-relaxed">
            &ldquo;Compile fail, verify fail, and Graveyard mean Vajra refused to ship a wrong result — dropping is a feature, not a bug.&rdquo;
          </p>
          <div className="space-y-2 text-xs text-[#555555] leading-relaxed">
            <p>
              In machine learning and graphics computation, a shader that runs at 100 GFLOPS but calculates <code className="font-mono bg-[#FAFAFA] px-1 py-0.5 border border-[#EAEAEA]">NaN</code> or introduces a <code className="font-mono bg-[#FAFAFA] px-1 py-0.5 border border-[#EAEAEA]">0.01</code> error will corrupt inference weights, create NaN cascades, and destroy model outputs.
            </p>
            <p>
              Vajra executes an uncompromising mathematical guardrail: every single variant is compiled on the local WebGPU device and verified against CPU IEEE 754 reference arrays. If the maximum absolute difference exceeds <code className="font-mono font-bold text-[#171717]">1e-4</code>, or if the compiler reports a syntax or resource error, the variant is instantly quarantined in the <strong>Failure Graveyard</strong>.
            </p>
            <p>
              Under no circumstances will Vajra promote a broken or inaccurate kernel to winner, regardless of how fast it executes.
            </p>
          </div>
        </section>

        {/* 9. Limitations Honestly */}
        <section className="rounded-lg border border-[#EAEAEA] bg-[#FFFFFF] p-6 shadow-sm space-y-4">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#888888]">
            09 / Scope &amp; Transparency
          </div>
          <h2 className="text-xl font-bold text-[#171717]">Limitations, Honestly</h2>
          <div className="space-y-3 text-xs text-[#555555]">
            <div className="flex items-start gap-2.5">
              <span className="font-mono font-bold text-[#171717]">•</span>
              <div>
                <strong className="text-[#171717]">Preset workloads only:</strong> Vajra currently ships with four curated compute kernels: Matrix Multiplication (SGEMM), Softmax Normalization, 1D Convolution (Conv1D), and Image Edge Detection (Sobel Filter). Arbitrary user shader ingestion is planned for future versions.
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <span className="font-mono font-bold text-[#171717]">•</span>
              <div>
                <strong className="text-[#171717]">Results are per-device:</strong> A kernel that achieves 2.4&times; on an Apple M3 Max unified memory architecture might only achieve 1.3&times; on an AMD RDNA-3 discrete card, or vice versa. GPU execution characteristics depend directly on hardware memory bus width, cache sizes, and driver compilers.
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <span className="font-mono font-bold text-[#171717]">•</span>
              <div>
                <strong className="text-[#171717]">Speeds up GPU maths, not downloading or loading a page:</strong> Vajra optimizes execution latency for compute kernels on the physical GPU. It does not compress web assets, accelerate network download bandwidth, or reduce initial bundle parsing time.
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#EAEAEA] bg-[#FFFFFF] py-6 font-mono text-xs text-[#666666]">
        <div className="mx-auto flex max-w-[1000px] flex-wrap items-center justify-between gap-4 px-6">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[#171717]">VAJRA</span>
            <span className="text-[#888888]">//</span>
            <span>HARDWARE-VERIFIED WEBGPU AUTO-TUNER</span>
          </div>
          <Link
            href="/"
            onClick={(e) => {
              if (typeof window !== "undefined" && window.history.length > 1) {
                e.preventDefault();
                window.history.back();
              }
            }}
            className="text-[#171717] hover:underline font-semibold"
          >
            ← Back to Tuner
          </Link>
        </div>
      </footer>
    </div>
  );
}
