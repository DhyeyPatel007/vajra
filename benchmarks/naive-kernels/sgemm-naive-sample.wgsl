// SAMPLE ONLY — naive SGEMM baseline for Tavily BYOK import testing.
// Not optimized.
// Test file only.

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;
@group(0) @binding(2) var<storage, read> matrixB: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;

@compute @workgroup_size(8, 8, 1)
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
}
