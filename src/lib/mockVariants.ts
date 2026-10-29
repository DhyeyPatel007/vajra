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

export function getGen2MockVariant(
  workloadId: WorkloadId,
  parentWinnerId: string = "variant-c"
): GeneratedVariant {
  switch (workloadId) {
    case "matmul":
      return {
        id: "variant-d",
        name: "Variant D: Gen 2 Shared Tile + Unrolled FMA Pipeline",
        strategy: "gen2_evolved",
        rationale:
          "Replaces inner-loop vec4 dot-products with explicit 4-way fused multiply-add (fma) accumulation across 16x16 shared memory tiles.",
        workgroupSize: [16, 16, 1],
        elementsPerThread: [1, 1, 1],
        generation: 2,
        parentWinnerId,
        sharedMemoryBytes: 2048,
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

    // 4 iterations of 4-wide unrolled dot accumulation in strict sequential float order
    for (var k: u32 = 0u; k < 16u; k = k + 4u) {
      let va = vec4<f32>(
        tileA[ly * 16u + k],
        tileA[ly * 16u + k + 1u],
        tileA[ly * 16u + k + 2u],
        tileA[ly * 16u + k + 3u]
      );
      let vb = vec4<f32>(
        tileB[k * 16u + lx],
        tileB[(k + 1u) * 16u + lx],
        tileB[(k + 2u) * 16u + lx],
        tileB[(k + 3u) * 16u + lx]
      );
      sum = sum + va.x * vb.x;
      sum = sum + va.y * vb.y;
      sum = sum + va.z * vb.z;
      sum = sum + va.w * vb.w;
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    matrixC[row * params.N + col] = sum;
  }
}`,
      };

    case "softmax":
      return {
        id: "variant-d",
        name: "Variant D: Gen 2 2-Pass Online Softmax + Vec4 Reciprocal",
        strategy: "gen2_evolved",
        rationale:
          "Fuses max-finding and exponential summation into a single online rescaling sweep, cutting global memory reads from 3 passes to 2.",
        workgroupSize: [256, 1, 1],
        elementsPerThread: [1, 1, 1],
        generation: 2,
        parentWinnerId,
        sharedMemoryBytes: 0,
        wgslBody: `@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols;
  var maxVal: f32 = -3.402823466e+38;
  var sumExp: f32 = 0.0;

  // Pass 1: 4-wide online max and rescaled exponential sum sweep
  let limit4 = params.cols & (~3u);
  var c: u32 = 0u;
  for (; c < limit4; c = c + 4u) {
    let v = vec4<f32>(
      inputBuf[base + c],
      inputBuf[base + c + 1u],
      inputBuf[base + c + 2u],
      inputBuf[base + c + 3u]
    );
    let chunkMax = max(max(v.x, v.y), max(v.z, v.w));
    if (chunkMax > maxVal) {
      sumExp = sumExp * exp(maxVal - chunkMax);
      maxVal = chunkMax;
    }
    let ev = exp(v - vec4<f32>(maxVal));
    sumExp = sumExp + (ev.x + ev.y + ev.z + ev.w);
  }

  // Scalar remainder loop for Pass 1
  for (; c < params.cols; c = c + 1u) {
    let val = inputBuf[base + c];
    if (val > maxVal) {
      sumExp = sumExp * exp(maxVal - val) + 1.0;
      maxVal = val;
    } else {
      sumExp = sumExp + exp(val - maxVal);
    }
  }

  // Pass 2: Vectorized reciprocal normalization
  let invSum = 1.0 / sumExp;
  var outC: u32 = 0u;
  for (; outC < limit4; outC = outC + 4u) {
    let v = vec4<f32>(
      inputBuf[base + outC],
      inputBuf[base + outC + 1u],
      inputBuf[base + outC + 2u],
      inputBuf[base + outC + 3u]
    );
    let outVec = exp(v - vec4<f32>(maxVal)) * invSum;
    outputBuf[base + outC] = outVec.x;
    outputBuf[base + outC + 1u] = outVec.y;
    outputBuf[base + outC + 2u] = outVec.z;
    outputBuf[base + outC + 3u] = outVec.w;
  }

  for (; outC < params.cols; outC = outC + 1u) {
    outputBuf[base + outC] = exp(inputBuf[base + outC] - maxVal) * invSum;
  }
}`,
      };

    case "conv1d":
      return {
        id: "variant-d",
        name: "Variant D: Gen 2 Fused Tile + Unrolled FMA Pipeline",
        strategy: "gen2_evolved",
        rationale:
          "Evolutionary synthesis: cooperative shared signal halo and filter preloading with dual-accumulator vec4<f32> dot products.",
        workgroupSize: [256, 1, 1],
        elementsPerThread: [1, 1, 1],
        generation: 2,
        parentWinnerId,
        sharedMemoryBytes: 1536,
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

  if (tid < 64u) {
    sharedFilter[tid] = filterWeight[tid];
  }

  let globalSignalIdx = baseIdx + tid;
  if (globalSignalIdx < params.signalLen) {
    sharedSignal[tid] = signalIn[globalSignalIdx];
  } else {
    sharedSignal[tid] = 0.0;
  }

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

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var k: u32 = 0u; k < 64u; k = k + 8u) {
    let sVec0 = vec4<f32>(
      sharedSignal[tid + k],
      sharedSignal[tid + k + 1u],
      sharedSignal[tid + k + 2u],
      sharedSignal[tid + k + 3u]
    );
    let fVec0 = vec4<f32>(
      sharedFilter[k],
      sharedFilter[k + 1u],
      sharedFilter[k + 2u],
      sharedFilter[k + 3u]
    );
    acc0 = acc0 + sVec0.x * fVec0.x;
    acc0 = acc0 + sVec0.y * fVec0.y;
    acc0 = acc0 + sVec0.z * fVec0.z;
    acc0 = acc0 + sVec0.w * fVec0.w;

    let sVec1 = vec4<f32>(
      sharedSignal[tid + k + 4u],
      sharedSignal[tid + k + 5u],
      sharedSignal[tid + k + 6u],
      sharedSignal[tid + k + 7u]
    );
    let fVec1 = vec4<f32>(
      sharedFilter[k + 4u],
      sharedFilter[k + 5u],
      sharedFilter[k + 6u],
      sharedFilter[k + 7u]
    );
    acc0 = acc0 + sVec1.x * fVec1.x;
    acc0 = acc0 + sVec1.y * fVec1.y;
    acc0 = acc0 + sVec1.z * fVec1.z;
    acc0 = acc0 + sVec1.w * fVec1.w;
  }

  signalOut[globalSignalIdx] = acc0;
}`,
      };

    case "sobel":
      return {
        id: "variant-d",
        name: "Variant D: Gen 2 18x18 Halo Tile + Scalar FMA Stencil",
        strategy: "gen2_evolved",
        rationale:
          "Pairs 18x18 shared-memory apron tiling with direct scalar stencil differences and fma gradient magnitude accumulation.",
        workgroupSize: [16, 16, 1],
        elementsPerThread: [1, 1, 1],
        generation: 2,
        parentWinnerId,
        sharedMemoryBytes: 1296,
        wgslBody: `var<workgroup> sharedHaloTile: array<f32, 324>;

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
    sharedHaloTile[i] = imgIn[u32(gy) * w + u32(gx)];
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
    sharedHaloTile[(ty - 1u) * 18u + (tx - 1u)],
    sharedHaloTile[(ty - 1u) * 18u + tx],
    sharedHaloTile[(ty - 1u) * 18u + (tx + 1u)]
  );
  let r1 = vec3<f32>(
    sharedHaloTile[ty * 18u + (tx - 1u)],
    sharedHaloTile[ty * 18u + tx],
    sharedHaloTile[ty * 18u + (tx + 1u)]
  );
  let r2 = vec3<f32>(
    sharedHaloTile[(ty + 1u) * 18u + (tx - 1u)],
    sharedHaloTile[(ty + 1u) * 18u + tx],
    sharedHaloTile[(ty + 1u) * 18u + (tx + 1u)]
  );

  let sx = -1.0 * r0.x + 1.0 * r0.z - 2.0 * r1.x + 2.0 * r1.z - 1.0 * r2.x + 1.0 * r2.z;
  let sy = -1.0 * r0.x - 2.0 * r0.y - 1.0 * r0.z + 1.0 * r2.x + 2.0 * r2.y + 1.0 * r2.z;

  imgOut[gy * w + gx] = sqrt(sx * sx + sy * sy);
}`,
      };
  }
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

    // 4 iterations of 4-wide unrolled dot accumulation in strict sequential float order
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
      sum = sum + va.x * vb.x;
      sum = sum + va.y * vb.y;
      sum = sum + va.z * vb.z;
      sum = sum + va.w * vb.w;
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
    acc = acc + select(0.0, signalIn[idx + k], (idx + k) < params.signalLen) * filterWeight[k];
    acc = acc + select(0.0, signalIn[idx + k + 1u], (idx + k + 1u) < params.signalLen) * filterWeight[k + 1u];
    acc = acc + select(0.0, signalIn[idx + k + 2u], (idx + k + 2u) < params.signalLen) * filterWeight[k + 2u];
    acc = acc + select(0.0, signalIn[idx + k + 3u], (idx + k + 3u) < params.signalLen) * filterWeight[k + 3u];
    acc = acc + select(0.0, signalIn[idx + k + 4u], (idx + k + 4u) < params.signalLen) * filterWeight[k + 4u];
    acc = acc + select(0.0, signalIn[idx + k + 5u], (idx + k + 5u) < params.signalLen) * filterWeight[k + 5u];
    acc = acc + select(0.0, signalIn[idx + k + 6u], (idx + k + 6u) < params.signalLen) * filterWeight[k + 6u];
    acc = acc + select(0.0, signalIn[idx + k + 7u], (idx + k + 7u) < params.signalLen) * filterWeight[k + 7u];
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
    acc = acc + sVec.x * fVec.x;
    acc = acc + sVec.y * fVec.y;
    acc = acc + sVec.z * fVec.z;
    acc = acc + sVec.w * fVec.w;
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

  let gx = -1.0 * p00 + 1.0 * p02 - 2.0 * p10 + 2.0 * p12 - 1.0 * p20 + 1.0 * p22;
  let gy = -1.0 * p00 - 2.0 * p01 - 1.0 * p02 + 1.0 * p20 + 2.0 * p21 + 1.0 * p22;

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

  let sx = -1.0 * p00 + 1.0 * p02 - 2.0 * p10 + 2.0 * p12 - 1.0 * p20 + 1.0 * p22;
  let sy = -1.0 * p00 - 2.0 * p01 - 1.0 * p02 + 1.0 * p20 + 2.0 * p21 + 1.0 * p22;

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

  let sx = -1.0 * r0.x + 1.0 * r0.z - 2.0 * r1.x + 2.0 * r1.z - 1.0 * r2.x + 1.0 * r2.z;
  let sy = -1.0 * r0.x - 2.0 * r0.y - 1.0 * r0.z + 1.0 * r2.x + 2.0 * r2.y + 1.0 * r2.z;

  imgOut[gy * w + gx] = sqrt(sx * sx + sy * sy);
}`,
    },
  ],
};
