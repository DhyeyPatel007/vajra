# Vajra — Browser-Native WebGPU Kernel Auto-Tuner

> **Pitch:** *"AI writes GPU shaders, but only your hardware knows the truth. Vajra benchmarks every Nemotron kernel live on your GPU and keeps the fastest."*  
> Built for the **Nebius x NVIDIA Global AI Hackathon** (Track: Coding and Agentic Engineering).

---

## ⚡ What is Vajra?

**Vajra** is a zero-install, browser-native WebGPU compute shader auto-tuner powered by **NVIDIA Nemotron** via **Nebius Token Factory**.

Traditional GPU kernel tuning (TVM, Triton, CUDA) requires local Python/C++ toolchains, driver headers, and minutes of offline compilation. Vajra runs entirely inside a single browser tab with zero dependencies. It closes the agentic loop between cloud LLM code generation and physical silicon truth in **under 10 seconds**:

1. **Deterministic Local Baseline:** Executes a naive baseline WGSL shader on the user's physical GPU to establish ground truth execution latency, GFLOPS, and reference output buffers.
2. **Nemotron Kernel Generation:** Queries NVIDIA Nemotron (via Nebius Token Factory) to generate 3 hardware-optimized variants:
   - **Variant A (Workgroup Geometry):** Tuned `@workgroup_size` and loop unrolling for memory coalescing.
   - **Variant B (Shared Memory Tiling):** `var<workgroup>` tile cache and `workgroupBarrier()` to minimize global DRAM reads.
   - **Variant C (Vec4 Vectorized):** `vec4<f32>` vector arithmetic and SIMD dot products for maximum ALU density.
3. **In-Browser Compilation & Error Dropping:** Compiles each variant using `device.createShaderModule()` and `module.getCompilationInfo()`. If any variant contains syntax or type errors, it is cleanly dropped with diagnostic error drawers while the remaining variants proceed without crashing.
4. **Tolerance-Based Float Verification:** Executes each compiled variant on identical input buffers and verifies against baseline using element-wise relative error:
   $$\text{relErr}_i = \frac{|v_i - b_i|}{\max(|b_i|, 10^{-6})}$$
   Kernels pass if and only if $\max_i(\text{relErr}_i) < 10^{-4}$ with zero NaNs or Infinities.
5. **Hardware Timestamp Benchmarking:** Executes 3 warmup dispatches followed by 9 timed dispatches using WebGPU hardware timestamp queries (or wall-clock fallback via `queue.onSubmittedWorkDone()`), reporting the **median run** (never best/min).
6. **Silicon Report Card & One-Click Copy:** Surfaces the fastest verified kernel on a live SVG GFLOPS chart and hardware scorecard, with a one-click button to copy the complete working WGSL shader.

---

## 🔬 4 Supported Compute Workloads

| Workload | Domain / Dimensions | Total FLOPs | Bytes Moved | Optimization Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **SGEMM** | Matrix Multiply ($512 \times 512$ FP32) | $268,435,456$ | $3,145,728$ | DRAM read coalescing & shared tile reuse |
| **Softmax** | Row-Wise Softmax ($1024 \times 1024$ FP32) | $5,242,880$ | $8,388,608$ | Reduction synchronization & exponential vectorization |
| **Conv1D** | 1D Signal Convolution ($N=1\text{M}, K=64$ FP32) | $134,217,728$ | $8,388,864$ | Tap filter caching & signal halo loading |
| **Sobel** | 2D Edge Detection ($2048 \times 2048$ FP32) | $75,497,472$ | $33,554,432$ | 2D halo memory apron & 3x3 stencil evaluation |

---

## 🛡️ Robust Fallback Architecture

- **Mock-First Architecture (`VAJRA_MOCK_MODE=true`):** Ships with 12 verified-fast WGSL mock kernels (3 per workload) that run locally on your GPU with **zero external API calls**.
- **Super ➔ Ultra Automatic Fallback Chain:** In live mode, requests are routed to **Nemotron Super** (`nvidia/nemotron-3-super-120b-a12b`) with a 25s timeout. If Super times out or fails, requests automatically retry against **Nemotron Ultra** (`nvidia/Nemotron-3-Ultra-550b-a55b`).
- **No-WebGPU Fallback Mode:** If WebGPU is not supported (or toggled via **"Preview No-WebGPU"**), Vajra automatically loads reference benchmark data recorded on an **Apple M3 Max 40-core GPU (Chrome 142)** with a gold reference banner.
- **Simulate Compile Fail:** Interactive toggle in the top bar injects an invalid intrinsic into Variant C in Mock Mode so judges can observe Vajra cleanly dropping a faulty shader while variants A and B succeed.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/vajra.git
cd vajra
npm install
```

### 2. Configure Environment (Optional for Live Mode)
Copy `.env.example` to `.env.local`:
```bash
cp .env.example .env.local
```
Add your Nebius Token Factory API key:
```env
NEBIUS_API_KEY=your_token_factory_key
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_MODEL_SUPER=nvidia/nemotron-3-super-120b-a12b
NEBIUS_MODEL_ULTRA=nvidia/Nemotron-3-Ultra-550b-a55b
VAJRA_MOCK_MODE=true
```

*(Note: In Mock Mode, `VAJRA_MOCK_MODE=true` allows full local execution and testing without any API key).*

### 3. Run Locally
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in any WebGPU-capable browser (Chrome 113+, Edge 113+, or Safari 18+).

### 4. Build for Production
```bash
npm run build
npm run start
```

---

## 🎨 Design System

Vajra adheres to a strict flat charcoal and metallic gold theme:
- **Background:** `#0B0C0E`
- **Elevated Surface:** `#121418`
- **Inner Panel:** `#181B22`
- **Border:** `#262A34`
- **Metallic Gold Primary:** `#D4AF37`
- **Gold Highlight:** `#F5C542`
- **Verified Green:** `#10B981`
- **Error Red:** `#EF4444`
- **Zero CSS Gradients:** Strictly flat, crisp, technical UI.
- **Zero Purple/Indigo:** Purpose-built for high-contrast technical instrumentation.
