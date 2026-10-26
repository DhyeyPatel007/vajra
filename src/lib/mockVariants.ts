import { GeneratedVariant, WorkloadId } from "@/types/vajra";

export function getMockVariants(
  workloadId: WorkloadId,
  simulateCompileError: boolean = false
): GeneratedVariant[] {
  const baseVariants = ALL_MOCK_VARIANTS[workloadId];

  if (!simulateCompileError) {
    return baseVariants;
  }

  // Inject a real syntax error into variant-c to verify the compile failure fallback
  return baseVariants.map((v) => {
    if (v.id === "variant-c") {
      return {
        ...v,
        wgslBody: `@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Intentional compiler fault for hackathon demo testing
  let x = gid.x;
  let y = gid.y;
  let faultyVal: vec4<f32> = unknown_intrinsic_device_call_xyz(x, y);
  let brokenMatrix: matrix_type_invalid = 999.0;
  if (x >= params.width) {
    return;
  }
}`,
      };
    }
    return v;
  });
}

const ALL_MOCK_VARIANTS: Record<WorkloadId, GeneratedVariant[]> = {
  matmul: [
    {
      id: "variant-a",
      name: "Variant A: 16×16 Workgroup + 4× Loop Unrolling",
      strategy: "workgroup_tuning",
      rationale:
        "Tuned 16×16 workgroup geometry (256 threads) with 4-way loop unrolling to maximize instruction-level parallelism and memory coalescing.",
      workgroupSize: [16, 16, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  if (row >= params.M || col >= params.N) {
    return;
  }
  var sum: f32 = 0.0;
  let rowOffset = row * params.K;
  let kLimit = params.K & (~3u);
  var k: u32 = 0u;
  for (; k < kLimit; k = k + 4u) {
    sum = sum + matrixA[rowOffset + k] * matrixB[k * params.N + col];
    sum = sum + matrixA[rowOffset + k + 1u] * matrixB[(k + 1u) * params.N + col];
    sum = sum + matrixA[rowOffset + k + 2u] * matrixB[(k + 2u) * params.N + col];
    sum = sum + matrixA[rowOffset + k + 3u] * matrixB[(k + 3u) * params.N + col];
  }
  for (; k < params.K; k = k + 1u) {
    sum = sum + matrixA[rowOffset + k] * matrixB[k * params.N + col];
  }
  matrixC[row * params.N + col] = sum;
}`,
    },
    {
      id: "variant-b",
      name: "Variant B: 16×16 Shared Memory Tiling",
      strategy: "shared_memory_tiling",
      rationale:
        "Cooperative 16×16 workgroup shared memory tiles (var<workgroup>) with barrier sync, slashing DRAM reads by 16×.",
      workgroupSize: [16, 16, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let lx = lid.x;
  let ly = lid.y;
  let numTiles = (params.K + 15u) / 16u;
  var sum: f32 = 0.0;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * 16u + lx;
    if (row < params.M && aCol < params.K) {
      tileA[ly * 16u + lx] = matrixA[row * params.K + aCol];
    } else {
      tileA[ly * 16u + lx] = 0.0;
    }

    let bRow = t * 16u + ly;
    if (bRow < params.K && col < params.N) {
      tileB[ly * 16u + lx] = matrixB[bRow * params.N + col];
    } else {
      tileB[ly * 16u + lx] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      sum = sum + tileA[ly * 16u + k] * tileB[k * 16u + lx];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    matrixC[row * params.N + col] = sum;
  }
}`,
    },
    {
      id: "variant-c",
      name: "Variant C: 16×16 Shared Tiling + Vec4 Vectorized Dot",
      strategy: "vec4_vectorized",
      rationale:
        "16×16 shared memory tile cache combined with 4-wide vec4<f32> dot-product intrinsic accumulation for maximum ALU throughput.",
      workgroupSize: [16, 16, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `var<workgroup> sTileA: array<f32, 256>;
var<workgroup> sTileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let lx = lid.x;
  let ly = lid.y;
  let numTiles = (params.K + 15u) / 16u;
  var sum: f32 = 0.0;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * 16u + lx;
    if (row < params.M && aCol < params.K) {
      sTileA[ly * 16u + lx] = matrixA[row * params.K + aCol];
    } else {
      sTileA[ly * 16u + lx] = 0.0;
    }

    let bRow = t * 16u + ly;
    if (bRow < params.K && col < params.N) {
      sTileB[ly * 16u + lx] = matrixB[bRow * params.N + col];
    } else {
      sTileB[ly * 16u + lx] = 0.0;
    }

    workgroupBarrier();

    // 4 iterations of 4-wide dot product
    for (var k: u32 = 0u; k < 16u; k = k + 4u) {
      let va = vec4<f32>(
        sTileA[ly * 16u + k],
        sTileA[ly * 16u + k + 1u],
        sTileA[ly * 16u + k + 2u],
        sTileA[ly * 16u + k + 3u]
      );
      let vb = vec4<f32>(
        sTileB[k * 16u + lx],
        sTileB[(k + 1u) * 16u + lx],
        sTileB[(k + 2u) * 16u + lx],
        sTileB[(k + 3u) * 16u + lx]
      );
      sum = sum + dot(va, vb);
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    matrixC[row * params.N + col] = sum;
  }
}`,
    },
  ],

  softmax: [
    {
      id: "variant-a",
      name: "Variant A: 128-Thread Workgroup + 4× Unrolled Reductions",
      strategy: "workgroup_tuning",
      rationale:
        "Higher workgroup occupancy (128 threads/workgroup) with 4-way unrolled loops for max-finder and exponential accumulator passes.",
      workgroupSize: [128, 1, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols;
  var maxVal: f32 = -3.402823466e+38;

  let colsLimit = params.cols & (~3u);
  var c: u32 = 0u;
  for (; c < colsLimit; c = c + 4u) {
    let v0 = inputBuf[base + c];
    let v1 = inputBuf[base + c + 1u];
    let v2 = inputBuf[base + c + 2u];
    let v3 = inputBuf[base + c + 3u];
    maxVal = max(maxVal, max(max(v0, v1), max(v2, v3)));
  }
  for (; c < params.cols; c = c + 1u) {
    maxVal = max(maxVal, inputBuf[base + c]);
  }

  var sumExp: f32 = 0.0;
  c = 0u;
  for (; c < colsLimit; c = c + 4u) {
    sumExp = sumExp + exp(inputBuf[base + c] - maxVal);
    sumExp = sumExp + exp(inputBuf[base + c + 1u] - maxVal);
    sumExp = sumExp + exp(inputBuf[base + c + 2u] - maxVal);
    sumExp = sumExp + exp(inputBuf[base + c + 3u] - maxVal);
  }
  for (; c < params.cols; c = c + 1u) {
    sumExp = sumExp + exp(inputBuf[base + c] - maxVal);
  }

  let invSum = 1.0 / sumExp;
  c = 0u;
  for (; c < colsLimit; c = c + 4u) {
    outputBuf[base + c] = exp(inputBuf[base + c] - maxVal) * invSum;
    outputBuf[base + c + 1u] = exp(inputBuf[base + c + 1u] - maxVal) * invSum;
    outputBuf[base + c + 2u] = exp(inputBuf[base + c + 2u] - maxVal) * invSum;
    outputBuf[base + c + 3u] = exp(inputBuf[base + c + 3u] - maxVal) * invSum;
  }
  for (; c < params.cols; c = c + 1u) {
    outputBuf[base + c] = exp(inputBuf[base + c] - maxVal) * invSum;
  }
}`,
    },
    {
      id: "variant-b",
      name: "Variant B: 256-Thread Hoisted Reciprocal",
      strategy: "shared_memory_tiling",
      rationale:
        "Expands workgroup to 256 threads and hoists reciprocal invSum = 1.0 / sumExp outside the normalization pass to eliminate per-element FP32 division.",
      workgroupSize: [256, 1, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols;
  var maxVal: f32 = -3.402823466e+38;

  for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
    let v = inputBuf[base + c];
    if (v > maxVal) {
      maxVal = v;
    }
  }

  var sumExp: f32 = 0.0;
  for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
    sumExp = sumExp + exp(inputBuf[base + c] - maxVal);
  }

  let invSum = 1.0 / sumExp;
  for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
    outputBuf[base + c] = exp(inputBuf[base + c] - maxVal) * invSum;
  }
}`,
    },
    {
      id: "variant-c",
      name: "Variant C: 256-Thread Vec4 Vectorized Softmax",
      strategy: "vec4_vectorized",
      rationale:
        "Vectorized 4-wide vec4<f32> SIMD loads, vector max reduction, and vector exp() arithmetic for 4× instruction density.",
      workgroupSize: [256, 1, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols;
  var maxVal: f32 = -3.402823466e+38;

  for (var c: u32 = 0u; c < params.cols; c = c + 4u) {
    let v = vec4<f32>(
      inputBuf[base + c],
      inputBuf[base + c + 1u],
      inputBuf[base + c + 2u],
      inputBuf[base + c + 3u]
    );
    let m = max(max(v.x, v.y), max(v.z, v.w));
    if (m > maxVal) {
      maxVal = m;
    }
  }

  var sumExp: f32 = 0.0;
  for (var c: u32 = 0u; c < params.cols; c = c + 4u) {
    let v = vec4<f32>(
      inputBuf[base + c],
      inputBuf[base + c + 1u],
      inputBuf[base + c + 2u],
      inputBuf[base + c + 3u]
    );
    let ev = exp(v - vec4<f32>(maxVal));
    sumExp = sumExp + ev.x + ev.y + ev.z + ev.w;
  }

  let invSum = 1.0 / sumExp;
  for (var c: u32 = 0u; c < params.cols; c = c + 4u) {
    let v = vec4<f32>(
      inputBuf[base + c],
      inputBuf[base + c + 1u],
      inputBuf[base + c + 2u],
      inputBuf[base + c + 3u]
    );
    let outVec = exp(v - vec4<f32>(maxVal)) * invSum;
    outputBuf[base + c] = outVec.x;
    outputBuf[base + c + 1u] = outVec.y;
    outputBuf[base + c + 2u] = outVec.z;
    outputBuf[base + c + 3u] = outVec.w;
  }
}`,
    },
  ],

  conv1d: [
    {
      id: "variant-a",
      name: "Variant A: 256-Thread Workgroup + 8× Loop Unrolling",
      strategy: "workgroup_tuning",
      rationale:
        "Wide 256-thread workgroup sizing with 8-tap unrolled inner convolution loop to saturate dual-issue compute units.",
      workgroupSize: [256, 1, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.signalLen) {
    return;
  }
  var acc: f32 = 0.0;
  let kLimit = params.kernelLen & (~7u);
  var k: u32 = 0u;
  for (; k < kLimit; k = k + 8u) {
    acc = acc + signalIn[min(idx + k, params.signalLen - 1u)] * filterWeight[k];
    acc = acc + signalIn[min(idx + k + 1u, params.signalLen - 1u)] * filterWeight[k + 1u];
    acc = acc + signalIn[min(idx + k + 2u, params.signalLen - 1u)] * filterWeight[k + 2u];
    acc = acc + signalIn[min(idx + k + 3u, params.signalLen - 1u)] * filterWeight[k + 3u];
    acc = acc + signalIn[min(idx + k + 4u, params.signalLen - 1u)] * filterWeight[k + 4u];
    acc = acc + signalIn[min(idx + k + 5u, params.signalLen - 1u)] * filterWeight[k + 5u];
    acc = acc + signalIn[min(idx + k + 6u, params.signalLen - 1u)] * filterWeight[k + 6u];
    acc = acc + signalIn[min(idx + k + 7u, params.signalLen - 1u)] * filterWeight[k + 7u];
  }
  for (; k < params.kernelLen; k = k + 1u) {
    if (idx + k < params.signalLen) {
      acc = acc + signalIn[idx + k] * filterWeight[k];
    }
  }
  signalOut[idx] = acc;
}`,
    },
    {
      id: "variant-b",
      name: "Variant B: Cooperative Shared Memory Signal Halo Tiling",
      strategy: "shared_memory_tiling",
      rationale:
        "Cooperative preloading of the 64-tap filter and 320-element signal halo into workgroup shared memory (var<workgroup>).",
      workgroupSize: [256, 1, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `var<workgroup> sharedFilter: array<f32, 64>;
var<workgroup> sharedSignal: array<f32, 320>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let baseIdx = wid.x * 256u;

  // Preload 64-tap filter into shared memory cooperatively
  if (tid < 64u) {
    sharedFilter[tid] = filterWeight[tid];
  }

  // Preload 256 elements of main signal
  let globalSignalIdx = baseIdx + tid;
  if (globalSignalIdx < params.signalLen) {
    sharedSignal[tid] = signalIn[globalSignalIdx];
  } else {
    sharedSignal[tid] = 0.0;
  }

  // Preload 64-element halo
  if (tid < 64u) {
    let haloIdx = baseIdx + 256u + tid;
    if (haloIdx < params.signalLen) {
      sharedSignal[256u + tid] = signalIn[haloIdx];
    } else {
      sharedSignal[256u + tid] = 0.0;
    }
  }

  workgroupBarrier();

  if (globalSignalIdx >= params.signalLen) {
    return;
  }

  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < 64u; k = k + 1u) {
    acc = acc + sharedSignal[tid + k] * sharedFilter[k];
  }

  signalOut[globalSignalIdx] = acc;
}`,
    },
    {
      id: "variant-c",
      name: "Variant C: Shared Memory Tiling + Vec4 Vectorized Taps",
      strategy: "vec4_vectorized",
      rationale:
        "Shared memory halo caching paired with 4-wide vec4<f32> dot products, executing 16 vector multiply-adds per thread.",
      workgroupSize: [256, 1, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `var<workgroup> sharedFilt: array<f32, 64>;
var<workgroup> sharedSig: array<f32, 320>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let baseIdx = wid.x * 256u;

  if (tid < 64u) {
    sharedFilt[tid] = filterWeight[tid];
  }

  let globalSignalIdx = baseIdx + tid;
  if (globalSignalIdx < params.signalLen) {
    sharedSig[tid] = signalIn[globalSignalIdx];
  } else {
    sharedSig[tid] = 0.0;
  }

  if (tid < 64u) {
    let haloIdx = baseIdx + 256u + tid;
    if (haloIdx < params.signalLen) {
      sharedSig[256u + tid] = signalIn[haloIdx];
    } else {
      sharedSig[256u + tid] = 0.0;
    }
  }

  workgroupBarrier();

  if (globalSignalIdx >= params.signalLen) {
    return;
  }

  var acc: f32 = 0.0;
  // 64 taps in 16 vec4 dot products
  for (var k: u32 = 0u; k < 64u; k = k + 4u) {
    let sVec = vec4<f32>(
      sharedSig[tid + k],
      sharedSig[tid + k + 1u],
      sharedSig[tid + k + 2u],
      sharedSig[tid + k + 3u]
    );
    let fVec = vec4<f32>(
      sharedFilt[k],
      sharedFilt[k + 1u],
      sharedFilt[k + 2u],
      sharedFilt[k + 3u]
    );
    acc = acc + dot(sVec, fVec);
  }

  signalOut[globalSignalIdx] = acc;
}`,
    },
  ],

  sobel: [
    {
      id: "variant-a",
      name: "Variant A: 16×16 Workgroup + Coalesced Row Offsets",
      strategy: "workgroup_tuning",
      rationale:
        "16×16 workgroup mapping with precomputed linear row strides (topRow, midRow, botRow) to reduce indexing instructions.",
      workgroupSize: [16, 16, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;

  if (x >= w || y >= h) {
    return;
  }
  if (x == 0u || y == 0u || x == w - 1u || y == h - 1u) {
    imgOut[y * w + x] = 0.0;
    return;
  }

  let topRow = (y - 1u) * w;
  let midRow = y * w;
  let botRow = (y + 1u) * w;

  let p00 = imgIn[topRow + (x - 1u)];
  let p02 = imgIn[topRow + (x + 1u)];
  let p10 = imgIn[midRow + (x - 1u)];
  let p12 = imgIn[midRow + (x + 1u)];
  let p20 = imgIn[botRow + (x - 1u)];
  let p22 = imgIn[botRow + (x + 1u)];

  let p01 = imgIn[topRow + x];
  let p21 = imgIn[botRow + x];

  let gx = (p02 - p00) + 2.0 * (p12 - p10) + (p22 - p20);
  let gy = (p20 - p00) + 2.0 * (p21 - p01) + (p22 - p02);

  imgOut[midRow + x] = sqrt(gx * gx + gy * gy);
}`,
    },
    {
      id: "variant-b",
      name: "Variant B: 18×18 Shared Memory Halo Tile",
      strategy: "shared_memory_tiling",
      rationale:
        "18×18 workgroup shared tile buffer caching 16×16 pixels plus 1-pixel surrounding halo, eliminating redundant global DRAM reads.",
      workgroupSize: [16, 16, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `var<workgroup> tile: array<f32, 324>; // 18 x 18 = 324 elements

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * 16u + lx; // 0 .. 255
  let baseGX = wid.x * 16u;
  let baseGY = wid.y * 16u;
  let w = params.width;
  let h = params.height;

  // 256 threads load 324 shared memory elements cooperatively
  for (var i: u32 = tid; i < 324u; i = i + 256u) {
    let ty = i / 18u;
    let tx = i % 18u;
    let gx = min(max(i32(baseGX + tx) - 1, 0), i32(w - 1u));
    let gy = min(max(i32(baseGY + ty) - 1, 0), i32(h - 1u));
    tile[i] = imgIn[u32(gy) * w + u32(gx)];
  }

  workgroupBarrier();

  let gx = gid.x;
  let gy = gid.y;
  if (gx >= w || gy >= h) {
    return;
  }
  if (gx == 0u || gy == 0u || gx == w - 1u || gy == h - 1u) {
    imgOut[gy * w + gx] = 0.0;
    return;
  }

  let tx = lx + 1u;
  let ty = ly + 1u;

  let p00 = tile[(ty - 1u) * 18u + (tx - 1u)];
  let p01 = tile[(ty - 1u) * 18u + tx];
  let p02 = tile[(ty - 1u) * 18u + (tx + 1u)];
  let p10 = tile[ty * 18u + (tx - 1u)];
  let p12 = tile[ty * 18u + (tx + 1u)];
  let p20 = tile[(ty + 1u) * 18u + (tx - 1u)];
  let p21 = tile[(ty + 1u) * 18u + tx];
  let p22 = tile[(ty + 1u) * 18u + (tx + 1u)];

  let sx = (p02 - p00) + 2.0 * (p12 - p10) + (p22 - p20);
  let sy = (p20 - p00) + 2.0 * (p21 - p01) + (p22 - p02);

  imgOut[gy * w + gx] = sqrt(sx * sx + sy * sy);
}`,
    },
    {
      id: "variant-c",
      name: "Variant C: 18×18 Halo Tile + Vec3 Vectorized Stencil",
      strategy: "vec4_vectorized",
      rationale:
        "18×18 shared halo memory paired with vec3<f32> dot products on rows [-1, 0, 1] and columns [-1, -2, -1].",
      workgroupSize: [16, 16, 1],
      elementsPerThread: [1, 1, 1],
      wgslBody: `var<workgroup> tileHalo: array<f32, 324>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * 16u + lx;
  let baseGX = wid.x * 16u;
  let baseGY = wid.y * 16u;
  let w = params.width;
  let h = params.height;

  for (var i: u32 = tid; i < 324u; i = i + 256u) {
    let ty = i / 18u;
    let tx = i % 18u;
    let gx = min(max(i32(baseGX + tx) - 1, 0), i32(w - 1u));
    let gy = min(max(i32(baseGY + ty) - 1, 0), i32(h - 1u));
    tileHalo[i] = imgIn[u32(gy) * w + u32(gx)];
  }

  workgroupBarrier();

  let gx = gid.x;
  let gy = gid.y;
  if (gx >= w || gy >= h) {
    return;
  }
  if (gx == 0u || gy == 0u || gx == w - 1u || gy == h - 1u) {
    imgOut[gy * w + gx] = 0.0;
    return;
  }

  let tx = lx + 1u;
  let ty = ly + 1u;

  let r0 = vec3<f32>(
    tileHalo[(ty - 1u) * 18u + (tx - 1u)],
    tileHalo[(ty - 1u) * 18u + tx],
    tileHalo[(ty - 1u) * 18u + (tx + 1u)]
  );
  let r1 = vec3<f32>(
    tileHalo[ty * 18u + (tx - 1u)],
    0.0,
    tileHalo[ty * 18u + (tx + 1u)]
  );
  let r2 = vec3<f32>(
    tileHalo[(ty + 1u) * 18u + (tx - 1u)],
    tileHalo[(ty + 1u) * 18u + tx],
    tileHalo[(ty + 1u) * 18u + (tx + 1u)]
  );

  let kx = vec3<f32>(-1.0, 0.0, 1.0);
  let sx = dot(r0, kx) + 2.0 * dot(r1, kx) + dot(r2, kx);

  let c0 = vec3<f32>(r0.x, r1.x, r2.x);
  let c1 = vec3<f32>(r0.y, tileHalo[ty * 18u + tx], r2.y);
  let c2 = vec3<f32>(r0.z, r1.z, r2.z);
  let ky = vec3<f32>(-1.0, 0.0, 1.0);
  let sy = dot(vec3<f32>(r0.x, r0.y, r0.z), vec3<f32>(-1.0, -2.0, -1.0))
         + dot(vec3<f32>(r2.x, r2.y, r2.z), vec3<f32>(1.0, 2.0, 1.0));

  imgOut[gy * w + gx] = sqrt(sx * sx + sy * sy);
}`,
    },
  ],
};
