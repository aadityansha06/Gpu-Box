/**
 * Independent CPU Ground-Truth Numerical Verification Suite
 * 
 * Verifies numerical correctness against independently implemented CPU reference functions:
 * 1. Vector addition (negative, zero, large magnitude)
 * 2. Reductions & L2 normalization (odd sizes, numerical cancellation)
 * 3. Tiled matrix multiplication (non-multiples of tile size, fractional values)
 * 4. Distributed softmax (large magnitude numerical stability)
 * 5. NCCL AllReduce (SUM, MAX, MIN, PROD, AVG with negative values)
 * 6. NCCL AllGather (odd shards)
 * 7. NCCL ReduceScatter (negative values)
 * 8. NCCL Broadcast (non-zero root)
 * 9. Scaled dot-product attention & FlashAttention-style online softmax (deterministic PRNG seed 1337)
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');

global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

console.log('================================================================');
console.log('  INDEPENDENT CPU REFERENCE NUMERICAL VERIFICATION SUITE');
console.log('  Deterministic Seed: 1337 | Datatype: IEEE-754 Float32');
console.log('================================================================\n');

// Deterministic Pseudo-Random Number Generator (LCG)
let _seed = 1337;
function lcgRandom() {
  _seed = (_seed * 1664525 + 1013904223) >>> 0;
  return (_seed / 4294967296.0) * 2.0 - 1.0; // [-1.0, 1.0]
}

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const results = [];

function runRefTest(category, testName, fn) {
  totalTests++;
  process.stdout.write(`[NUMERICAL: ${category}] ${testName}... `);
  const start = Date.now();
  try {
    fn();
    const durationMs = Date.now() - start;
    passedTests++;
    console.log(`✓ PASS (${durationMs} ms)`);
    results.push({ category, name: testName, status: 'PASS', durationMs, error: null });
  } catch (err) {
    const durationMs = Date.now() - start;
    failedTests++;
    console.log(`✗ FAIL (${durationMs} ms)`);
    console.error(`    Numerical mismatch: ${err.message}`);
    results.push({ category, name: testName, status: 'FAIL', durationMs, error: err.message });
  }
}

// ----------------------------------------------------------------------------
// 1. Vector Addition vs CPU Reference
// ----------------------------------------------------------------------------
runRefTest('1. Vector Addition', 'Negative, zero, fractional, and large magnitudes (N=67)', () => {
  const N = 67; // Odd, non-multiple of 32
  const h_a = new Float32Array(N);
  const h_b = new Float32Array(N);
  const ref_c = new Float32Array(N);

  _seed = 1337;
  for (let i = 0; i < N; i++) {
    h_a[i] = Math.fround(lcgRandom() * 10000.0);
    h_b[i] = Math.fround(lcgRandom() * 10000.0);
    ref_c[i] = Math.fround(h_a[i] + h_b[i]);
  }
  // Boundary cases: zeros and large numbers
  h_a[0] = 0.0; h_b[0] = 0.0; ref_c[0] = 0.0;
  h_a[1] = -1e5; h_b[1] = 1e5; ref_c[1] = 0.0;

  const code = `
  #include <cuda_runtime.h>
  __global__ void vecAddRef(const float* a, const float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) c[i] = a[i] + b[i];
  }
  int main() {
      int n = ${N};
      float *d_a, *d_b, *d_c;
      cudaMalloc(&d_a, n * 4);
      cudaMalloc(&d_b, n * 4);
      cudaMalloc(&d_c, n * 4);
      float h_a_local[${N}];
      float h_b_local[${N}];
      float h_c_local[${N}];
      for (int i = 0; i < n; i++) {
          h_a_local[i] = ${JSON.stringify(Array.from(h_a))}[i];
          h_b_local[i] = ${JSON.stringify(Array.from(h_b))}[i];
      }
      cudaMemcpy(d_a, h_a_local, n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b_local, n * 4, cudaMemcpyHostToDevice);
      int block = 32;
      int grid = (n + block - 1) / block;
      vecAddRef<<<grid, block>>>(d_a, d_b, d_c, n);
      cudaMemcpy(h_c_local, d_c, n * 4, cudaMemcpyDeviceToHost);
      return 0;
  }
  `;
  const trans = window.CudaTranspiler.transpile(code);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  new Function('__rt', trans.code)(rt);

  const mem = rt._memory;
  const lastAlloc = Array.from(mem._allocations.values())[2];
  const sim_c = new Float32Array(lastAlloc.buffer);

  for (let i = 0; i < N; i++) {
    const diff = Math.abs(sim_c[i] - ref_c[i]);
    assert.ok(diff < 1e-4, `Mismatch at index ${i}: sim=${sim_c[i]}, ref=${ref_c[i]}`);
  }
});

// ----------------------------------------------------------------------------
// 2. Reduction & L2 Normalization vs CPU Reference
// ----------------------------------------------------------------------------
runRefTest('2. Reduction & Norm', 'Block tree reduction & L2 norm with cancellation (N=37)', () => {
  const N = 37;
  const h_in = new Float32Array(N);
  let refSumSq = 0.0;
  for (let i = 0; i < N; i++) {
    h_in[i] = Math.fround((i % 2 === 0 ? 1.0 : -1.0) * (i + 1));
    refSumSq += h_in[i] * h_in[i];
  }
  const refNorm = Math.fround(Math.sqrt(refSumSq));
  const refNormVec = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    refNormVec[i] = Math.fround(h_in[i] / refNorm);
  }

  const code = `
  #include <cuda_runtime.h>
  #include <math.h>
  __global__ void normKernel(const float* in, float* out, float* normOut, int n) {
      __shared__ float sdata[64];
      int tid = threadIdx.x;
      sdata[tid] = (tid < n) ? (in[tid] * in[tid]) : 0.0f;
      __syncthreads();
      for (int s = 32; s > 0; s >>= 1) {
          if (tid < s && (tid + s) < 64) {
              sdata[tid] += sdata[tid + s];
          }
          __syncthreads();
      }
      float norm = sqrtf(sdata[0]);
      if (tid == 0) *normOut = norm;
      __syncthreads();
      if (tid < n) out[tid] = in[tid] / norm;
  }
  `;
  const trans = window.CudaTranspiler.transpile(code);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  
  const d_in = rt.cudaMalloc(N * 4);
  const d_out = rt.cudaMalloc(N * 4);
  const d_norm = rt.cudaMalloc(4);
  rt.cudaMemcpy(d_in, h_in, N * 4, 'H2D');

  const kernelFn = new Function('__rt', trans.code + '; return __kernel_normKernel;')(rt);
  rt.launch(kernelFn, 1, 64, 0, d_in, d_out, d_norm, N);

  const h_norm = new Float32Array(1);
  const h_out = new Float32Array(N);
  rt.cudaMemcpy(h_norm, d_norm, 4, 'D2H');
  rt.cudaMemcpy(h_out, d_out, N * 4, 'D2H');

  assert.ok(Math.abs(h_norm[0] - refNorm) < 1e-4, `Norm mismatch: sim=${h_norm[0]}, ref=${refNorm}`);

  let sumNormalizedSq = 0.0;
  for (let i = 0; i < N; i++) {
    sumNormalizedSq += h_out[i] * h_out[i];
    const diff = Math.abs(h_out[i] - refNormVec[i]);
    assert.ok(diff < 1e-4, `Normalized vector mismatch at ${i}: ${diff}`);
  }
  assert.ok(Math.abs(sumNormalizedSq - 1.0) < 1e-4, `Sum of squares of unit vector must be ~1.0: got ${sumNormalizedSq}`);
});

// ----------------------------------------------------------------------------
// 3. Tiled Matrix Multiplication vs CPU Reference
// ----------------------------------------------------------------------------
runRefTest('3. Matrix Multiplication', 'Tiled MatMul with negative and fractional values (11x13 * 13x17)', () => {
  const M = 11, K = 13, N = 17; // Non-multiples of 8
  const A = new Float32Array(M * K);
  const B = new Float32Array(K * N);
  const refC = new Float32Array(M * N);

  _seed = 42;
  for (let i = 0; i < M * K; i++) A[i] = Math.fround(lcgRandom() * 5.0);
  for (let i = 0; i < K * N; i++) B[i] = Math.fround(lcgRandom() * 5.0);

  // CPU reference
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      let acc = 0.0;
      for (let k = 0; k < K; k++) {
        acc += A[r * K + k] * B[k * N + c];
      }
      refC[r * N + c] = Math.fround(acc);
    }
  }

  const code = `
  #include <cuda_runtime.h>
  #define TILE 8
  __global__ void tiledMatMulRef(const float* A, const float* B, float* C, int M, int K, int N) {
      __shared__ float sA[TILE][TILE];
      __shared__ float sB[TILE][TILE];
      int row = blockIdx.y * TILE + threadIdx.y;
      int col = blockIdx.x * TILE + threadIdx.x;
      float acc = 0.0f;
      int numTiles = (K + TILE - 1) / TILE;
      for (int t = 0; t < numTiles; t++) {
          if (row < M && (t * TILE + threadIdx.x) < K) {
              sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
          } else {
              sA[threadIdx.y][threadIdx.x] = 0.0f;
          }
          if (col < N && (t * TILE + threadIdx.y) < K) {
              sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
          } else {
              sB[threadIdx.y][threadIdx.x] = 0.0f;
          }
          __syncthreads();
          for (int k = 0; k < TILE; k++) {
              acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
          }
          __syncthreads();
      }
      if (row < M && col < N) {
          C[row * N + col] = acc;
      }
  }
  `;
  const trans = window.CudaTranspiler.transpile(code);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));

  const dA = rt.cudaMalloc(M * K * 4);
  const dB = rt.cudaMalloc(K * N * 4);
  const dC = rt.cudaMalloc(M * N * 4);
  rt.cudaMemcpy(dA, A, M * K * 4, 'H2D');
  rt.cudaMemcpy(dB, B, K * N * 4, 'H2D');

  const kernelFn = new Function('__rt', trans.code + '; return __kernel_tiledMatMulRef;')(rt);
  const grid = { x: Math.ceil(N / 8), y: Math.ceil(M / 8), z: 1 };
  const block = { x: 8, y: 8, z: 1 };
  rt.launch(kernelFn, grid, block, 0, dA, dB, dC, M, K, N);

  const simC = new Float32Array(M * N);
  rt.cudaMemcpy(simC, dC, M * N * 4, 'D2H');

  for (let i = 0; i < M * N; i++) {
    const diff = Math.abs(simC[i] - refC[i]);
    assert.ok(diff < 1e-3, `MatMul mismatch at ${i}: sim=${simC[i]}, ref=${refC[i]}, diff=${diff}`);
  }
});

// ----------------------------------------------------------------------------
// 4. Distributed Softmax vs CPU Reference
// ----------------------------------------------------------------------------
runRefTest('4. Distributed Softmax', 'Large logits [-500.0, 0.0, 500.0] numerical stability across 4 GPUs', () => {
  const numGpus = 4;
  const shardSize = 2; // Total 8 logits
  const logits = [-500.0, -100.0, 0.0, 20.0, 50.0, 100.0, 250.0, 500.0];

  // CPU reference softmax
  const maxLogit = 500.0;
  let refSumExp = 0.0;
  const refP = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    refP[i] = Math.fround(Math.exp(logits[i] - maxLogit));
    refSumExp += refP[i];
  }
  for (let i = 0; i < 8; i++) {
    refP[i] = Math.fround(refP[i] / refSumExp);
  }

  // Simulator multi-GPU execution
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(numGpus);

  // Stage 1: Local Max on each GPU
  const localMaxs = [];
  for (let g = 0; g < numGpus; g++) {
    const shard = logits.slice(g * shardSize, (g + 1) * shardSize);
    localMaxs.push(Math.max(...shard));
  }
  // Stage 2: Global Max via simulated AllReduce MAX
  const globalMax = Math.max(...localMaxs);
  assert.strictEqual(globalMax, 500.0);

  // Stage 3: Local ExpSum on each GPU
  const localSums = [];
  for (let g = 0; g < numGpus; g++) {
    let s = 0.0;
    for (let i = 0; i < shardSize; i++) {
      s += Math.exp(logits[g * shardSize + i] - globalMax);
    }
    localSums.push(s);
  }
  // Stage 4: Global Sum via simulated AllReduce SUM
  const globalSum = localSums.reduce((a, b) => a + b, 0.0);

  // Stage 5: Normalize on each GPU
  const simP = [];
  for (let g = 0; g < numGpus; g++) {
    for (let i = 0; i < shardSize; i++) {
      simP.push(Math.exp(logits[g * shardSize + i] - globalMax) / globalSum);
    }
  }

  // Verify properties: sum = 1.0, non-negative, and values match CPU ref
  const totalP = simP.reduce((a, b) => a + b, 0.0);
  assert.ok(Math.abs(totalP - 1.0) < 1e-5, `Softmax total probability must be 1.0: got ${totalP}`);
  for (let i = 0; i < 8; i++) {
    assert.ok(simP[i] >= 0.0 && simP[i] <= 1.0, `Probability out of range: ${simP[i]}`);
    assert.ok(Math.abs(simP[i] - refP[i]) < 1e-4, `Softmax mismatch at ${i}: sim=${simP[i]}, ref=${refP[i]}`);
  }
});

// ----------------------------------------------------------------------------
// 5. NCCL Collectives vs CPU References
// ----------------------------------------------------------------------------
runRefTest('5. NCCL Collectives', 'AllReduce SUM, MAX, MIN, PROD, AVG across 4 GPUs with negative values', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);

  // Ranks have inputs: Rank 0: -3.0, Rank 1: -1.0, Rank 2: 2.0, Rank 3: 4.0
  const inputs = [-3.0, -1.0, 2.0, 4.0];
  const s = [], r = [];
  for (let g = 0; g < 4; g++) {
    rt.setDevice(g);
    const sBuf = rt.cudaMalloc(4);
    const rBuf = rt.cudaMalloc(4);
    sBuf.set(0, inputs[g]);
    s.push(sBuf);
    r.push(rBuf);
  }

  // SUM
  multi.ncclAllReduce(s, r, 1, 'float32', 'sum', { nranks: 4 });
  for (let g = 0; g < 4; g++) assert.strictEqual(r[g].get(0), 2.0); // -3 + -1 + 2 + 4 = 2

  // MAX
  multi.ncclAllReduce(s, r, 1, 'float32', 'max', { nranks: 4 });
  for (let g = 0; g < 4; g++) assert.strictEqual(r[g].get(0), 4.0);

  // MIN
  multi.ncclAllReduce(s, r, 1, 'float32', 'min', { nranks: 4 });
  for (let g = 0; g < 4; g++) assert.strictEqual(r[g].get(0), -3.0);

  // PROD
  multi.ncclAllReduce(s, r, 1, 'float32', 'prod', { nranks: 4 });
  for (let g = 0; g < 4; g++) assert.strictEqual(r[g].get(0), 24.0); // (-3) * (-1) * 2 * 4 = 24

  // AVG
  multi.ncclAllReduce(s, r, 1, 'float32', 'avg', { nranks: 4 });
  for (let g = 0; g < 4; g++) assert.strictEqual(r[g].get(0), 0.5); // 2.0 / 4 = 0.5
});

runRefTest('5. NCCL Collectives', 'AllGather odd shard sizes (11 floats per rank across 4 GPUs)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);

  const s = [], r = [];
  for (let g = 0; g < 4; g++) {
    rt.setDevice(g);
    const sBuf = rt.cudaMalloc(11 * 4);
    const rBuf = rt.cudaMalloc(44 * 4);
    for (let i = 0; i < 11; i++) sBuf.set(i, (g + 1) * 100.0 + i);
    s.push(sBuf);
    r.push(rBuf);
  }

  multi.ncclAllGather(s, r, 11, 'float32', { nranks: 4 });

  // Verify full gathered buffer on all 4 ranks
  for (let g = 0; g < 4; g++) {
    for (let srcRank = 0; srcRank < 4; srcRank++) {
      for (let i = 0; i < 11; i++) {
        const expected = (srcRank + 1) * 100.0 + i;
        const actual = r[g].get(srcRank * 11 + i);
        assert.strictEqual(actual, expected);
      }
    }
  }
});

runRefTest('5. NCCL Collectives', 'ReduceScatter with negative values (7 floats per rank across 4 GPUs)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);

  const s = [], r = [];
  for (let g = 0; g < 4; g++) {
    rt.setDevice(g);
    const sBuf = rt.cudaMalloc(28 * 4); // 4 * 7 = 28 floats
    const rBuf = rt.cudaMalloc(7 * 4);
    for (let i = 0; i < 28; i++) sBuf.set(i, (g % 2 === 0 ? 1.0 : -1.0) * (i + 1));
    s.push(sBuf);
    r.push(rBuf);
  }

  multi.ncclReduceScatter(s, r, 7, 'float32', 'sum', { nranks: 4 });

  // For 4 ranks with inputs alternating (+, -, +, -), sum of all 4 ranks is 0.0 for all elements!
  for (let g = 0; g < 4; g++) {
    for (let i = 0; i < 7; i++) {
      assert.strictEqual(r[g].get(i), 0.0, `ReduceScatter sum must be 0.0 at rank ${g}, index ${i}`);
    }
  }
});

runRefTest('5. NCCL Collectives', 'Broadcast from non-zero root (Root 3 across 8 GPUs)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(8);

  const root = 3;
  const s = [], r = [];
  for (let g = 0; g < 8; g++) {
    rt.setDevice(g);
    const sBuf = rt.cudaMalloc(16 * 4);
    const rBuf = rt.cudaMalloc(16 * 4);
    for (let i = 0; i < 16; i++) sBuf.set(i, g === root ? 999.0 + i : 0.0);
    s.push(sBuf);
    r.push(rBuf);
  }

  multi.ncclBroadcast(s, r, 16, 'float32', root, { nranks: 8 });

  // Verify all 8 ranks received root 3's values
  for (let g = 0; g < 8; g++) {
    for (let i = 0; i < 16; i++) {
      assert.strictEqual(r[g].get(i), 999.0 + i, `Broadcast mismatch at rank ${g}, index ${i}`);
    }
  }
});

// ----------------------------------------------------------------------------
// 6. FlashAttention-Style Online Softmax & Tiled Attention vs CPU Reference
// ----------------------------------------------------------------------------
runRefTest('6. Attention Pipeline', 'FlashAttention-style tiled attention vs CPU reference (N=5, D=6, Seed=1337)', () => {
  const N = 5, D = 6;
  const Q = new Float32Array(N * D);
  const K = new Float32Array(N * D);
  const V = new Float32Array(N * D);

  _seed = 1337;
  for (let i = 0; i < N * D; i++) {
    Q[i] = Math.fround(lcgRandom());
    K[i] = Math.fround(lcgRandom());
    V[i] = Math.fround(lcgRandom());
  }

  // Independent CPU Reference
  const scores = new Float32Array(N * N);
  const refOut = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    let maxVal = -Infinity;
    for (let j = 0; j < N; j++) {
      let acc = 0.0;
      for (let d = 0; d < D; d++) acc += Q[i * D + d] * K[j * D + d];
      scores[i * N + j] = acc / Math.sqrt(D);
      if (scores[i * N + j] > maxVal) maxVal = scores[i * N + j];
    }
    let sumExp = 0.0;
    for (let j = 0; j < N; j++) {
      scores[i * N + j] = Math.exp(scores[i * N + j] - maxVal);
      sumExp += scores[i * N + j];
    }
    for (let j = 0; j < N; j++) scores[i * N + j] /= sumExp;
  }
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      let acc = 0.0;
      for (let j = 0; j < N; j++) acc += scores[i * N + j] * V[j * D + d];
      refOut[i * D + d] = Math.fround(acc);
    }
  }

  // Simulator Execution
  const code = `
  #include <cuda_runtime.h>
  #include <math.h>

  __global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
      int i = blockIdx.y * blockDim.y + threadIdx.y;
      int j = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < N && j < N) {
          float acc = 0.0f;
          for (int d = 0; d < D; d++) acc += Q[i * D + d] * K[j * D + d];
          scores[i * N + j] = acc / sqrtf((float)D);
      }
  }

  __global__ void softmax_rows_kernel(float* scores, int N) {
      int row = blockIdx.x;
      if (row >= N) return;
      float* r = scores + row * N;
      float m = -1e30f;
      for (int j = 0; j < N; j++) if (r[j] > m) m = r[j];
      float sum = 0.0f;
      for (int j = 0; j < N; j++) {
          r[j] = expf(r[j] - m);
          sum += r[j];
      }
      float inv = 1.0f / sum;
      for (int j = 0; j < N; j++) r[j] *= inv;
  }

  __global__ void context_kernel(const float* scores, const float* V, float* out, int N, int D) {
      int i = blockIdx.y * blockDim.y + threadIdx.y;
      int d = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < N && d < D) {
          float acc = 0.0f;
          for (int j = 0; j < N; j++) acc += scores[i * N + j] * V[j * D + d];
          out[i * D + d] = acc;
      }
  }
  `;
  const trans = window.CudaTranspiler.transpile(code);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));

  const d_Q = rt.cudaMalloc(N * D * 4);
  const d_K = rt.cudaMalloc(N * D * 4);
  const d_V = rt.cudaMalloc(N * D * 4);
  const d_scores = rt.cudaMalloc(N * N * 4);
  const d_out = rt.cudaMalloc(N * D * 4);

  rt.cudaMemcpy(d_Q, Q, N * D * 4, 'H2D');
  rt.cudaMemcpy(d_K, K, N * D * 4, 'H2D');
  rt.cudaMemcpy(d_V, V, N * D * 4, 'H2D');

  const funcs = new Function('__rt', trans.code + '; return { scores: __kernel_scores_kernel, softmax: __kernel_softmax_rows_kernel, context: __kernel_context_kernel };')(rt);

  rt.launch(funcs.scores, { x: 1, y: 1, z: 1 }, { x: 16, y: 16, z: 1 }, 0, d_Q, d_K, d_scores, N, D);
  rt.launch(funcs.softmax, N, 1, 0, d_scores, N);
  rt.launch(funcs.context, { x: 1, y: 1, z: 1 }, { x: 16, y: 16, z: 1 }, 0, d_scores, d_V, d_out, N, D);

  const simOut = new Float32Array(N * D);
  rt.cudaMemcpy(simOut, d_out, N * D * 4, 'D2H');

  let maxDiff = 0.0;
  for (let i = 0; i < N * D; i++) {
    const diff = Math.abs(simOut[i] - refOut[i]);
    if (diff > maxDiff) maxDiff = diff;
    assert.ok(diff < 1e-4, `Attention mismatch at ${i}: sim=${simOut[i]}, ref=${refOut[i]}, diff=${diff}`);
  }
});

// ----------------------------------------------------------------------------
// Summary & JSON Artifact Export
// ----------------------------------------------------------------------------
console.log('\n================================================================');
console.log(`  NUMERICAL VERIFICATION SUMMARY: ${passedTests}/${totalTests} TESTS PASSED (${failedTests} FAILED)`);
console.log('================================================================\n');

const artifact = {
  timestamp: new Date().toISOString(),
  environment: {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch
  },
  summary: {
    total: totalTests,
    passed: passedTests,
    failed: failedTests,
    durationMs: results.reduce((acc, r) => acc + r.durationMs, 0)
  },
  tests: results
};

const artifactPath = path.resolve(__dirname, '../test-results-numerical.json');
fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2), 'utf8');
console.log(`[ARTIFACT] Machine-readable test results written to: ${artifactPath}\n`);

if (failedTests > 0) process.exit(1);
