import { WorkloadId, WorkloadSpec } from "@/types/vajra";

// Mulberry32 fast deterministic PRNG with seed 0x56414A52 ("VAJR")
export function createPrng(seed: number = 0x56414a52) {
  let s = seed >>> 0;
  return function nextFloat(): number {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    // Map to [-1.0, 1.0] range
    return ((t >>> 0) / 4294967296) * 2.0 - 1.0;
  };
}

export const WORKLOADS: Record<WorkloadId, WorkloadSpec> = {
  matmul: {
    id: "matmul",
    name: "Matrix Multiply (SGEMM)",
    shortName: "SGEMM",
    dimensionsLabel: "512 × 512 FP32",
    description: "Dense matrix multiplication C = A × B for 512×512 FP32 matrices",
    problemDims: [512, 512, 1],
    totalFlops: 2 * 512 * 512 * 512, // 268,435,456 FLOPs
    totalBytesMoved: 3 * 512 * 512 * 4, // 3,145,728 bytes (Read A, Read B, Write C)
    paramsArray: new Uint32Array([512, 512, 512, 0]),
    inputABytes: 512 * 512 * 4,
    inputBBytes: 512 * 512 * 4,
    outputBytes: 512 * 512 * 4,
    baselineWorkgroupSize: [8, 8, 1],
    wgslPreamble: `struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;
@group(0) @binding(2) var<storage, read> matrixB: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;`,
    baselineWgslBody: `@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  if (row >= params.M || col >= params.N) {
    return;
  }
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + matrixA[row * params.K + k] * matrixB[k * params.N + col];
  }
  matrixC[row * params.N + col] = sum;
}`,
    generateInputs: () => {
      const prng = createPrng(0x56414a52);
      const count = 512 * 512;
      const inputA = new Float32Array(count);
      const inputB = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        // Small values in [-0.25, 0.25] to ensure safe FP32 dot product sum
        inputA[i] = prng() * 0.25;
        inputB[i] = prng() * 0.25;
      }
      return { inputA, inputB };
    },
  },

  softmax: {
    id: "softmax",
    name: "Row-Wise Softmax",
    shortName: "Softmax",
    dimensionsLabel: "1024 × 1024 FP32",
    description: "Row-wise softmax activation across 1024 rows with 1024 elements each",
    problemDims: [1024, 1, 1],
    totalFlops: 5 * 1024 * 1024, // 5,242,880 FLOPs (max + sub + exp + sum + div)
    totalBytesMoved: 2 * 1024 * 1024 * 4, // 8,388,608 bytes (Read input + Write output)
    paramsArray: new Uint32Array([1024, 1024, 0, 0]),
    inputABytes: 1024 * 1024 * 4,
    outputBytes: 1024 * 1024 * 4,
    baselineWorkgroupSize: [32, 1, 1],
    wgslPreamble: `struct Params {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputBuf: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuf: array<f32>;`,
    baselineWgslBody: `@compute @workgroup_size(32, 1, 1)
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
  for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
    outputBuf[base + c] = exp(inputBuf[base + c] - maxVal) / sumExp;
  }
}`,
    generateInputs: () => {
      const prng = createPrng(0x56414a53);
      const count = 1024 * 1024;
      const inputA = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        // Values in [-2.0, 2.0]
        inputA[i] = prng() * 2.0;
      }
      return { inputA };
    },
  },

  conv1d: {
    id: "conv1d",
    name: "1D Convolution",
    shortName: "Conv1D",
    dimensionsLabel: "N=1,048,576, K=64",
    description: "1D audio/signal filter over 1M samples with 64-tap kernel",
    problemDims: [1048576, 1, 1],
    totalFlops: 2 * 1048576 * 64, // 134,217,728 FLOPs
    totalBytesMoved: (2 * 1048576 + 64) * 4, // 8,388,864 bytes
    paramsArray: new Uint32Array([1048576, 64, 0, 0]),
    inputABytes: 1048576 * 4,
    inputBBytes: 64 * 4,
    outputBytes: 1048576 * 4,
    baselineWorkgroupSize: [64, 1, 1],
    wgslPreamble: `struct Params {
  signalLen: u32,
  kernelLen: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> signalIn: array<f32>;
@group(0) @binding(2) var<storage, read> filterWeight: array<f32>;
@group(0) @binding(3) var<storage, read_write> signalOut: array<f32>;`,
    baselineWgslBody: `@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.signalLen) {
    return;
  }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < params.kernelLen; k = k + 1u) {
    if (idx + k < params.signalLen) {
      acc = acc + signalIn[idx + k] * filterWeight[k];
    }
  }
  signalOut[idx] = acc;
}`,
    generateInputs: () => {
      const prng = createPrng(0x56414a54);
      const signal = new Float32Array(1048576);
      const filter = new Float32Array(64);
      for (let i = 0; i < signal.length; i++) {
        signal[i] = prng() * 0.5;
      }
      for (let k = 0; k < 64; k++) {
        filter[k] = prng() * 0.05;
      }
      return { inputA: signal, inputB: filter };
    },
  },

  sobel: {
    id: "sobel",
    name: "2D Sobel Edge Filter",
    shortName: "Sobel",
    dimensionsLabel: "2048 × 2048 FP32",
    description: "3×3 Sobel edge detection gradient filter on 2K resolution image",
    problemDims: [2048, 2048, 1],
    totalFlops: 18 * 2048 * 2048, // 75,497,472 FLOPs
    totalBytesMoved: 2 * 2048 * 2048 * 4, // 33,554,432 bytes
    paramsArray: new Uint32Array([2048, 2048, 0, 0]),
    inputABytes: 2048 * 2048 * 4,
    outputBytes: 2048 * 2048 * 4,
    baselineWorkgroupSize: [8, 8, 1],
    wgslPreamble: `struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> imgIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> imgOut: array<f32>;`,
    baselineWgslBody: `@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) {
    return;
  }
  if (x == 0u || y == 0u || x == params.width - 1u || y == params.height - 1u) {
    imgOut[y * params.width + x] = 0.0;
    return;
  }
  let w = params.width;
  let p00 = imgIn[(y - 1u) * w + (x - 1u)];
  let p01 = imgIn[(y - 1u) * w + x];
  let p02 = imgIn[(y - 1u) * w + (x + 1u)];
  let p10 = imgIn[y * w + (x - 1u)];
  let p12 = imgIn[y * w + (x + 1u)];
  let p20 = imgIn[(y + 1u) * w + (x - 1u)];
  let p21 = imgIn[(y + 1u) * w + x];
  let p22 = imgIn[(y + 1u) * w + (x + 1u)];
  let gx = -1.0 * p00 + 1.0 * p02 - 2.0 * p10 + 2.0 * p12 - 1.0 * p20 + 1.0 * p22;
  let gy = -1.0 * p00 - 2.0 * p01 - 1.0 * p02 + 1.0 * p20 + 2.0 * p21 + 1.0 * p22;
  imgOut[y * w + x] = sqrt(gx * gx + gy * gy);
}`,
    generateInputs: () => {
      const prng = createPrng(0x56414a55);
      const count = 2048 * 2048;
      const img = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        img[i] = Math.abs(prng());
      }
      return { inputA: img };
    },
  },
};
