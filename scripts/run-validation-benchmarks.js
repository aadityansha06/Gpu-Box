/**
 * CUDA Validation Benchmark Suite
 * Automated regression testing across 1, 2, 4, 8, 16 GPU nodes.
 */

const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

console.log('================================================================');
console.log('  CUDA VALIDATION BENCHMARK SUITE — RUNNER');
console.log('  WebAssembly & SIMT JavaScript GPU Simulator');
console.log('================================================================\n');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const results = [];

function runBenchmark(suiteName, testName, fn) {
  totalTests++;
  process.stdout.write(`[SUITE: ${suiteName}] ${testName}... `);
  const start = Date.now();
  try {
    fn();
    const durationMs = Date.now() - start;
    passedTests++;
    console.log(`✓ PASS (${durationMs} ms)`);
    results.push({ suite: suiteName, name: testName, status: 'PASS', durationMs, error: null });
  } catch (err) {
    const durationMs = Date.now() - start;
    failedTests++;
    console.log(`✗ FAIL (${durationMs} ms)`);
    console.error(`    Error: ${err.message}`);
    results.push({ suite: suiteName, name: testName, status: 'FAIL', durationMs, error: err.message });
  }
}

// ============================================================================
// SUITE 1: Vector Addition (Boundary conditions, odd sizes, memory metrics)
// ============================================================================
const vecSizes = [1, 7, 31, 33, 63, 64, 127, 255, 1023, 2048];
vecSizes.forEach(n => {
  runBenchmark('1. Vector Addition', `Size N=${n} (Boundary & Odd Dim)`, () => {
    const code = `
    #include <cuda_runtime.h>
    __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    int main() {
        int n = ${n};
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n * 4);
        cudaMalloc(&d_b, n * 4);
        cudaMalloc(&d_c, n * 4);
        float h_a[${Math.max(n, 1)}];
        float h_b[${Math.max(n, 1)}];
        for (int i = 0; i < n; i++) { h_a[i] = 1.5f * i; h_b[i] = 2.5f * i; }
        cudaMemcpy(d_a, h_a, n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, n * 4, cudaMemcpyHostToDevice);
        int block = 32;
        int grid = (n + block - 1) / block;
        vecAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        float h_c[${Math.max(n, 1)}];
        cudaMemcpy(h_c, d_c, n * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) {
            float exp = 4.0f * i;
            if (h_c[i] !== exp) {
                printf("FAIL %d: %f != %f\\n", i, h_c[i], exp);
                break;
            }
        }
        return 0;
    }
    `;
    const trans = window.CudaTranspiler.transpile(code);
    assert.strictEqual(trans.success, true, 'Transpilation must succeed');

    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    new Function('__rt', trans.code)(rt);

    // Verify diagnostics had no errors
    const diags = rt.getDiagnostics().filter(d => d.message.includes('FAIL'));
    assert.strictEqual(diags.length, 0, 'No mismatches');

    // Verify metric instrumentation
    const c = rt.getCounters().toJSON();
    assert.strictEqual(c.globalLoads, 2 * n, `Expected ${2*n} global loads`);
    assert.strictEqual(c.globalStores, n, `Expected ${n} global stores`);
    assert.strictEqual(c.globalLoadBytes, 2 * n * 4, `Expected ${8*n} load bytes`);
    assert.strictEqual(c.globalStoreBytes, n * 4, `Expected ${4*n} store bytes`);
    assert.ok(c.globalTransactions > 0, 'Transactions must be recorded');
  });
});

// ============================================================================
// SUITE 2: Reduction and Normalization (Warp & Block, Odd & Non-Power-of-Two)
// ============================================================================
const redSizes = [1, 5, 16, 17, 32, 33, 64, 127, 128, 255];
redSizes.forEach(n => {
  runBenchmark('2. Reduction & Norm', `L2 Normalization N=${n}`, () => {
    const code = `
    #include <cuda_runtime.h>
    #include <math.h>
    __global__ void l2Norm(float* data, float* normOut, int n) {
        __shared__ float s[256];
        int tid = threadIdx.x;
        float val = (tid < n) ? data[tid] : 0.0f;
        s[tid] = val * val;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            normOut[0] = sqrtf(s[0]);
        }
        __syncthreads();
        if (tid < n && normOut[0] > 0.0f) {
            data[tid] = data[tid] / normOut[0];
        }
    }
    int main() {
        int n = ${n};
        float *d_data, *d_norm;
        cudaMalloc(&d_data, n * 4);
        cudaMalloc(&d_norm, 4);
        float h_data[${Math.max(n, 1)}];
        for (int i = 0; i < n; i++) h_data[i] = 2.0f; // Each item 2.0, sum of squares = 4*n, norm = 2*sqrt(n)
        cudaMemcpy(d_data, h_data, n * 4, cudaMemcpyHostToDevice);
        l2Norm<<<1, 256>>>(d_data, d_norm, n);
        cudaDeviceSynchronize();
        float h_norm[1];
        cudaMemcpy(h_norm, d_norm, 4, cudaMemcpyDeviceToHost);
        float expNorm = sqrtf(4.0f * n);
        float diff = fabsf(h_norm[0] - expNorm);
        if (diff > 0.01f) {
            printf("Norm mismatch: got %f expected %f\\n", h_norm[0], expNorm);
        }
        return 0;
    }
    `;
    const trans = window.CudaTranspiler.transpile(code);
    assert.strictEqual(trans.success, true);
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    new Function('__rt', trans.code)(rt);

    const diags = rt.getDiagnostics().filter(d => d.message.includes('mismatch'));
    assert.strictEqual(diags.length, 0);

    const c = rt.getCounters().toJSON();
    assert.strictEqual(c.sharedStores > 0, true, 'Shared stores must be > 0');
    assert.strictEqual(c.sharedLoads > 0, true, 'Shared loads must be > 0');
    assert.strictEqual(c.syncthreadsCount > 0, true, 'Barriers must be > 0');
  });
});

// ============================================================================
// SUITE 3: Matrix Multiplication (2D Shared Memory Tiling, Boundary Masking)
// ============================================================================
const matWidths = [4, 7, 15, 16, 23, 32];
matWidths.forEach(w => {
  runBenchmark('3. Matrix Multiplication', `Tiled Shared MatMul ${w}x${w}`, () => {
    const code = `
    #include <cuda_runtime.h>
    #define TILE 8
    __global__ void matmul(const float* A, const float* B, float* C, int width) {
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];
        int row = blockIdx.y * TILE + threadIdx.y;
        int col = blockIdx.x * TILE + threadIdx.x;
        float sum = 0.0f;
        int numTiles = (width + TILE - 1) / TILE;
        for (int t = 0; t < numTiles; t++) {
            int aCol = t * TILE + threadIdx.x;
            int bRow = t * TILE + threadIdx.y;
            As[threadIdx.y][threadIdx.x] = (row < width && aCol < width) ? A[row * width + aCol] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (bRow < width && col < width) ? B[bRow * width + col] : 0.0f;
            __syncthreads();
            for (int k = 0; k < TILE; k++) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            __syncthreads();
        }
        if (row < width && col < width) {
            C[row * width + col] = sum;
        }
    }
    int main() {
        int width = ${w};
        int sz = width * width;
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sz * 4);
        cudaMalloc(&d_B, sz * 4);
        cudaMalloc(&d_C, sz * 4);
        float h_A[${Math.max(w * w, 1)}];
        float h_B[${Math.max(w * w, 1)}];
        for (int i = 0; i < sz; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }
        cudaMemcpy(d_A, h_A, sz * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sz * 4, cudaMemcpyHostToDevice);
        dim3 block(8, 8);
        dim3 grid((width + 7) / 8, (width + 7) / 8);
        matmul<<<grid, block>>>(d_A, d_B, d_C, width);
        cudaDeviceSynchronize();
        float h_C[${Math.max(w * w, 1)}];
        cudaMemcpy(h_C, d_C, sz * 4, cudaMemcpyDeviceToHost);
        float expected = width * 2.0f;
        for (int i = 0; i < sz; i++) {
            if (fabsf(h_C[i] - expected) > 0.001f) {
                printf("Matmul mismatch at %d: got %f expected %f\\n", i, h_C[i], expected);
                break;
            }
        }
        return 0;
    }
    `;
    const trans = window.CudaTranspiler.transpile(code);
    assert.strictEqual(trans.success, true);
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    new Function('__rt', trans.code)(rt);

    const diags = rt.getDiagnostics().filter(d => d.message.includes('mismatch'));
    assert.strictEqual(diags.length, 0);

    const c = rt.getCounters().toJSON();
    assert.ok(c.sharedStores > 0);
    assert.ok(c.sharedLoads > 0);
    assert.ok(c.syncthreadsCount > 0);
  });
});

// ============================================================================
// SUITE 4: Distributed Softmax (Multi-GPU Sharded, Online Max/Sum, Normalized)
// ============================================================================
const clusterSizes = [2, 4, 8, 16];
clusterSizes.forEach(nGpus => {
  runBenchmark('4. Distributed Softmax', `Cluster size ${nGpus} GPUs`, () => {
    const shardSize = 4;
    const totalN = shardSize * nGpus;
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    const multi = rt.initMultiGPU(nGpus);

    // Each GPU has 4 logits: GPU r has values [r*4, r*4+1, r*4+2, r*4+3]
    const d_input = [];
    const d_output = [];
    const d_localMax = [];
    const d_localSum = [];

    for (let r = 0; r < nGpus; r++) {
      rt.setDevice(r);
      const inBuf = rt.cudaMalloc(shardSize * 4);
      const outBuf = rt.cudaMalloc(shardSize * 4);
      const mBuf = rt.cudaMalloc(4);
      const sBuf = rt.cudaMalloc(4);
      for (let i = 0; i < shardSize; i++) inBuf._rawSet(i, 1.0 + (r * shardSize + i) * 0.1);
      d_input.push(inBuf);
      d_output.push(outBuf);
      d_localMax.push(mBuf);
      d_localSum.push(sBuf);
    }

    // Step 1: Find local max on each GPU
    for (let r = 0; r < nGpus; r++) {
      let maxVal = -Infinity;
      for (let i = 0; i < shardSize; i++) maxVal = Math.max(maxVal, d_input[r]._rawGet(i));
      d_localMax[r]._rawSet(0, maxVal);
    }

    // Step 2: Native AllReduce MAX across GPUs
    const comm = { nranks: nGpus };
    const maxRecv = d_localMax.map((_, i) => { rt.setDevice(i); return rt.cudaMalloc(4); });
    multi.ncclAllReduce(d_localMax, maxRecv, 1, 'float32', 'max', comm);
    const globalMax = maxRecv[0]._rawGet(0);
    const expGlobalMax = 1.0 + (totalN - 1) * 0.1;
    assert.strictEqual(Math.abs(globalMax - expGlobalMax) < 1e-4, true);

    // Step 3: Compute local exp(logit - globalMax) sum on each GPU
    for (let r = 0; r < nGpus; r++) {
      let sumExp = 0.0;
      for (let i = 0; i < shardSize; i++) sumExp += Math.exp(d_input[r]._rawGet(i) - globalMax);
      d_localSum[r]._rawSet(0, sumExp);
    }

    // Step 4: Native AllReduce SUM of exponential sums
    const sumRecv = d_localSum.map((_, i) => { rt.setDevice(i); return rt.cudaMalloc(4); });
    multi.ncclAllReduce(d_localSum, sumRecv, 1, 'float32', 'sum', comm);
    const globalSum = sumRecv[0]._rawGet(0);

    // Step 5: Normalize locally
    let totalProb = 0.0;
    for (let r = 0; r < nGpus; r++) {
      for (let i = 0; i < shardSize; i++) {
        const p = Math.exp(d_input[r]._rawGet(i) - globalMax) / globalSum;
        d_output[r]._rawSet(i, p);
        totalProb += p;
      }
    }
    assert.strictEqual(Math.abs(totalProb - 1.0) < 1e-4, true, 'Probabilities must sum to 1.0');

    // Verify counter aggregation across all GPUs
    const c = rt.getCounters().toJSON();
    assert.strictEqual(c.ncclOps, 2 * nGpus, '2 AllReduce operations across all nodes');
    assert.ok(c.crossGpuBytes > 0, 'Cross GPU traffic recorded');
  });
});

// ============================================================================
// SUITE 5: Native NCCL AllReduce (SUM, MAX, MIN, PROD, AVG across 2,4,8,16 GPUs)
// ============================================================================
const allReduceOps = ['sum', 'max', 'min', 'prod', 'avg'];
clusterSizes.forEach(nGpus => {
  allReduceOps.forEach(op => {
    runBenchmark('5. AllReduce', `Op=${op.toUpperCase()} across ${nGpus} GPUs`, () => {
      const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
      const multi = rt.initMultiGPU(nGpus);
      const count = 16;
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(count * 4);
        const rc = rt.cudaMalloc(count * 4);
        for (let i = 0; i < count; i++) {
          s._rawSet(i, (r + 1) * 1.5);
        }
        sendbufs.push(s);
        recvbufs.push(rc);
      }
      multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', op, { nranks: nGpus });

      // Calculate expected IEEE-754 single precision float
      let exp;
      if (op === 'sum') {
        let s = 0.0; for (let r = 0; r < nGpus; r++) s += (r + 1) * 1.5; exp = Math.fround(s);
      } else if (op === 'max') {
        exp = Math.fround(nGpus * 1.5);
      } else if (op === 'min') {
        exp = Math.fround(1.5);
      } else if (op === 'prod') {
        let p = 1.0; for (let r = 0; r < nGpus; r++) p = Math.fround(p * Math.fround((r + 1) * 1.5)); exp = p;
      } else if (op === 'avg') {
        let s = 0.0; for (let r = 0; r < nGpus; r++) s += (r + 1) * 1.5; exp = Math.fround(s / nGpus);
      }

      // Check all devices received the exact same reduction
      for (let r = 0; r < nGpus; r++) {
        const val = recvbufs[r]._rawGet(0);
        const diff = Math.abs(val - exp);
        assert.ok(diff < 1e-3, `Device ${r} value mismatch: got ${val}, expected ${exp}`);
      }

      // Verify cross GPU metrics
      const c = rt.getCounters().toJSON();
      assert.strictEqual(c.ncclOps, nGpus);
      const expTraffic = count * 4 * (nGpus - 1) * nGpus;
      assert.strictEqual(c.crossGpuBytes, expTraffic);
    });
  });
});

// ============================================================================
// SUITE 6: Native NCCL AllGather (2, 4, 8, 16 GPUs, Uneven Shards)
// ============================================================================
const gatherCounts = [3, 7, 16];
clusterSizes.forEach(nGpus => {
  gatherCounts.forEach(sendcount => {
    runBenchmark('6. AllGather', `${nGpus} GPUs x ${sendcount} items/shard`, () => {
      const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
      const multi = rt.initMultiGPU(nGpus);
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(sendcount * 4);
        const rc = rt.cudaMalloc(sendcount * nGpus * 4);
        for (let i = 0; i < sendcount; i++) s._rawSet(i, (r * 100) + i);
        sendbufs.push(s);
        recvbufs.push(rc);
      }

      multi.ncclAllGather(sendbufs, recvbufs, sendcount, 'float32', { nranks: nGpus });

      // Verify every device gathered shards in rank order [shard0, shard1, ..., shardN-1]
      for (let r = 0; r < nGpus; r++) {
        for (let srcR = 0; srcR < nGpus; srcR++) {
          for (let i = 0; i < sendcount; i++) {
            const actual = recvbufs[r]._rawGet(srcR * sendcount + i);
            const expected = (srcR * 100) + i;
            assert.strictEqual(actual, expected, `Mismatch at device ${r}, shard ${srcR}, item ${i}`);
          }
        }
      }

      const c = rt.getCounters().toJSON();
      assert.strictEqual(c.ncclOps, nGpus);
      assert.strictEqual(c.crossGpuBytes, sendcount * 4 * (nGpus - 1) * nGpus);
    });
  });
});

// ============================================================================
// SUITE 7: Native NCCL ReduceScatter (2, 4, 8, 16 GPUs)
// ============================================================================
clusterSizes.forEach(nGpus => {
  runBenchmark('7. ReduceScatter', `SUM across ${nGpus} GPUs`, () => {
    const recvcount = 5; // non-power-of-two
    const totalCount = recvcount * nGpus;
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    const multi = rt.initMultiGPU(nGpus);
    const sendbufs = [];
    const recvbufs = [];
    for (let r = 0; r < nGpus; r++) {
      rt.setDevice(r);
      const s = rt.cudaMalloc(totalCount * 4);
      const rc = rt.cudaMalloc(recvcount * 4);
      for (let i = 0; i < totalCount; i++) s._rawSet(i, (r + 1) * 1.0);
      sendbufs.push(s);
      recvbufs.push(rc);
    }

    multi.ncclReduceScatter(sendbufs, recvbufs, recvcount, 'float32', 'sum', { nranks: nGpus });

    // Sum of (1 + 2 + ... + nGpus)
    let expectedSum = 0;
    for (let r = 0; r < nGpus; r++) expectedSum += (r + 1);

    for (let r = 0; r < nGpus; r++) {
      for (let i = 0; i < recvcount; i++) {
        const val = recvbufs[r]._rawGet(i);
        assert.strictEqual(val, expectedSum, `Device ${r} received chunk element ${i}`);
      }
    }

    const c = rt.getCounters().toJSON();
    assert.strictEqual(c.ncclOps, nGpus);
    assert.strictEqual(c.crossGpuBytes, recvcount * 4 * (nGpus - 1) * nGpus);
  });
});

// ============================================================================
// SUITE 8: Native NCCL Broadcast (2, 4, 8, 16 GPUs, Root 0 and Root N-1)
// ============================================================================
clusterSizes.forEach(nGpus => {
  [0, nGpus - 1].forEach(root => {
    runBenchmark('8. Broadcast', `Root=${root} across ${nGpus} GPUs`, () => {
      const count = 8;
      const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
      const multi = rt.initMultiGPU(nGpus);
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(count * 4);
        const rc = rt.cudaMalloc(count * 4);
        if (r === root) {
          for (let i = 0; i < count; i++) s._rawSet(i, 77.0 + i);
        }
        sendbufs.push(s);
        recvbufs.push(rc);
      }

      multi.ncclBroadcast(sendbufs, recvbufs, count, 'float32', root, { nranks: nGpus });

      for (let r = 0; r < nGpus; r++) {
        for (let i = 0; i < count; i++) {
          assert.strictEqual(recvbufs[r]._rawGet(i), 77.0 + i, `Rank ${r} element ${i}`);
        }
      }

      const c = rt.getCounters().toJSON();
      assert.strictEqual(c.ncclOps, nGpus);
      // Root sends to (nGpus - 1) ranks
      assert.strictEqual(c.crossGpuBytes, count * 4 * (nGpus - 1));
      // Root device crossGpuBytes should be 0; non-root should be count * 4
      assert.strictEqual(c.perDevice[root].crossGpuBytes, 0);
      for (let r = 0; r < nGpus; r++) {
        if (r !== root) {
          assert.strictEqual(c.perDevice[r].crossGpuBytes, count * 4);
        }
      }
    });
  });
});

// ============================================================================
// SUITE 9: Multi-GPU Isolation, SPMG Rank Loop & Runtime Reset
// ============================================================================
clusterSizes.forEach(nGpus => {
  runBenchmark('9. Multi-GPU Isolation', `Independent Memory & Isolation across ${nGpus} GPUs`, () => {
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    const multi = rt.initMultiGPU(nGpus);

    const ptrs = [];
    for (let r = 0; r < nGpus; r++) {
      rt.setDevice(r);
      const p = rt.cudaMalloc(16);
      p._rawSet(0, (r + 1) * 111.0);
      ptrs.push(p);
    }

    // Verify GPU memory isolation: altering GPU 0 does NOT touch GPU 1
    ptrs[0]._rawSet(0, 999.0);
    for (let r = 1; r < nGpus; r++) {
      assert.strictEqual(ptrs[r]._rawGet(0), (r + 1) * 111.0, `GPU ${r} memory was leaked!`);
    }

    // Verify SPMG rank-loop collective execution via CudaTranspiler
    const code = `
    #include <cuda_runtime.h>
    #include <nccl.h>
    int main() {
        int n = ${nGpus};
        ncclComm_t comms[${nGpus}];
        ncclCommInitAll(comms, n);
        float *d_s[${nGpus}];
        float *d_r[${nGpus}];
        for (int i = 0; i < n; i++) {
            cudaSetDevice(i);
            cudaMalloc(&d_s[i], 4);
            cudaMalloc(&d_r[i], 4);
            float h_val[1] = { 1.0f };
            cudaMemcpy(d_s[i], h_val, 4, cudaMemcpyHostToDevice);
        }
        for (int i = 0; i < n; i++) {
            cudaSetDevice(i);
            ncclAllReduce(d_s[i], d_r[i], 1, ncclFloat, ncclSum, comms[i], 0);
        }
        return 0;
    }
    `;
    const trans = window.CudaTranspiler.transpile(code);
    assert.strictEqual(trans.success, true);
    new Function('__rt', trans.code)(rt);

    const c1 = rt.getCounters().toJSON();
    assert.ok(c1.ncclOps > 0);

    // Reset runtime and verify total wipeout of counters
    rt.reset();
    const c2 = rt.getCounters().toJSON();
    assert.strictEqual(c2.ncclOps, 0, 'Counters must be 0 after reset');
    assert.strictEqual(c2.crossGpuBytes, 0, 'Cross GPU bytes must be 0 after reset');
    assert.strictEqual(c2.globalLoads, 0, 'Global loads must be 0 after reset');
    assert.strictEqual(c2.globalStores, 0, 'Global stores must be 0 after reset');
  });
});

// ============================================================================
// SUITE 10: Attention & FlashAttention-Style Kernels (vs CPU Reference)
// ============================================================================
function cpuAttention(Q, K, V, N, D) {
  const scores = new Float32Array(N * N);
  const out = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    let maxVal = -Infinity;
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let d = 0; d < D; d++) acc += Q[i * D + d] * K[j * D + d];
      scores[i * N + j] = acc / Math.sqrt(D);
      if (scores[i * N + j] > maxVal) maxVal = scores[i * N + j];
    }
    let sumExp = 0;
    for (let j = 0; j < N; j++) {
      scores[i * N + j] = Math.exp(scores[i * N + j] - maxVal);
      sumExp += scores[i * N + j];
    }
    for (let j = 0; j < N; j++) scores[i * N + j] /= sumExp;
  }
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      let acc = 0;
      for (let j = 0; j < N; j++) acc += scores[i * N + j] * V[j * D + d];
      out[i * D + d] = acc;
    }
  }
  return out;
}

const attnConfigs = [
  { N: 2, D: 2, label: 'Small 2x2' },
  { N: 7, D: 8, label: 'Non-multiple/Odd 7x8' },
  { N: 16, D: 16, label: 'Standard 16x16' }
];

attnConfigs.forEach(({ N, D, label }) => {
  runBenchmark('10. Attention Pipeline', `${label} vs CPU Ground Truth`, () => {
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

    __global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int d = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N && d < D) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) acc += attn[i * N + j] * V[j * D + d];
            output[i * D + d] = acc;
        }
    }

    int main() {
        int N = ${N};
        int D = ${D};
        float *d_Q, *d_K, *d_V, *d_scores, *d_output;
        cudaMalloc(&d_Q, N * D * 4);
        cudaMalloc(&d_K, N * D * 4);
        cudaMalloc(&d_V, N * D * 4);
        cudaMalloc(&d_scores, N * N * 4);
        cudaMalloc(&d_output, N * D * 4);

        float h_Q[${N * D}];
        float h_K[${N * D}];
        float h_V[${N * D}];
        for (int i = 0; i < N * D; i++) {
            h_Q[i] = (float)((i % 5) + 1);
            h_K[i] = (float)((i % 3) + 1);
            h_V[i] = (float)((i % 7) + 1);
        }
        cudaMemcpy(d_Q, h_Q, N * D * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, N * D * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, N * D * 4, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 gridScores((N + 15) / 16, (N + 15) / 16);
        scores_kernel<<<gridScores, block>>>(d_Q, d_K, d_scores, N, D);

        softmax_rows_kernel<<<N, 1>>>(d_scores, N);

        dim3 gridOut((D + 15) / 16, (N + 15) / 16);
        av_kernel<<<gridOut, block>>>(d_scores, d_V, d_output, N, D);
        cudaDeviceSynchronize();

        float h_output[${N * D}];
        cudaMemcpy(h_output, d_output, N * D * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N * D; i++) {
            printf("%.6f ", h_output[i]);
        }
        return 0;
    }
    `;

    const trans = window.CudaTranspiler.transpile(code);
    assert.strictEqual(trans.success, true);
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    new Function('__rt', trans.code)(rt);

    const h_Q = new Float32Array(N * D);
    const h_K = new Float32Array(N * D);
    const h_V = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) {
      h_Q[i] = (i % 5) + 1;
      h_K[i] = (i % 3) + 1;
      h_V[i] = (i % 7) + 1;
    }
    const expected = cpuAttention(h_Q, h_K, h_V, N, D);

    const diags = rt.getDiagnostics().filter(d => d.type === 'printf');
    const actualStr = diags.map(d => d.message.trim()).join(' ').split(/\s+/).filter(Boolean);
    const actual = actualStr.map(parseFloat);

    assert.strictEqual(actual.length, N * D);
    let maxDiff = 0;
    for (let i = 0; i < N * D; i++) {
      const diff = Math.abs(actual[i] - expected[i]);
      if (diff > maxDiff) maxDiff = diff;
    }
    assert.ok(maxDiff < 1e-4, `Max diff ${maxDiff} must be < 1e-4`);
  });
});

runBenchmark('10. Attention Pipeline', 'FlashAttention Online Softmax (isinf / -INFINITY)', () => {
  const code = `
  #include <cuda_runtime.h>
  #include <math.h>
  __global__ void flashOnlineMax(float* out, float* in, int n) {
      int tid = threadIdx.x;
      float m = -INFINITY;
      if (isinf(m)) {
          out[tid] = (tid < n) ? in[tid] + 10.0f : 0.0f;
      }
  }
  int main() {
      float *d_out, *d_in;
      cudaMalloc(&d_out, 4 * sizeof(float));
      cudaMalloc(&d_in, 4 * sizeof(float));
      float h_in[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
      cudaMemcpy(d_in, h_in, 4 * sizeof(float), cudaMemcpyHostToDevice);
      flashOnlineMax<<<1, 4>>>(d_out, d_in, 4);
      float h_out[4];
      cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < 4; i++) {
          if (h_out[i] != h_in[i] + 10.0f) {
              printf("FlashAttention online max failed at %d\\n", i);
              break;
          }
      }
      return 0;
  }
  `;
  const trans = window.CudaTranspiler.transpile(code);
  assert.strictEqual(trans.success, true);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  new Function('__rt', trans.code)(rt);
  const errs = rt.getDiagnostics().filter(d => d.message.includes('failed'));
  assert.strictEqual(errs.length, 0);
});

// ============================================================================
// SUITE 11: Multi-GPU Collective Edge Cases & Argument Validation
// ============================================================================
runBenchmark('11. Collective Edge Cases', 'Single-GPU Execution (N=1 Rank)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(1);
  const s = rt.cudaMalloc(16);
  const r = rt.cudaMalloc(16);
  s._rawSet(0, 99.0);
  multi.ncclAllReduce([s], [r], 4, 'float32', 'sum', { nranks: 1 });
  assert.strictEqual(r._rawGet(0), 99.0);
  const c = rt.getCounters().toJSON();
  assert.strictEqual(c.ncclOps, 1);
  assert.strictEqual(c.crossGpuBytes, 0, 'Single-GPU collective must have 0 cross-GPU bytes');
});

runBenchmark('11. Collective Edge Cases', 'Zero-Length Input (count=0)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(2);
  const s0 = rt.cudaMalloc(16);
  const r0 = rt.cudaMalloc(16);
  const s1 = rt.cudaMalloc(16);
  const r1 = rt.cudaMalloc(16);
  multi.ncclAllReduce([s0, s1], [r0, r1], 0, 'float32', 'sum', { nranks: 2 });
  const c = rt.getCounters().toJSON();
  assert.strictEqual(c.ncclOps, 2);
  assert.strictEqual(c.crossGpuBytes, 0);
});

runBenchmark('11. Collective Edge Cases', 'Invalid Root Rank Rejection', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(2);
  const s0 = rt.cudaMalloc(16);
  const r0 = rt.cudaMalloc(16);
  const s1 = rt.cudaMalloc(16);
  const r1 = rt.cudaMalloc(16);
  let threw = false;
  try {
    multi.ncclBroadcast([s0, s1], [r0, r1], 4, 'float32', 5, { nranks: 2 });
  } catch (err) {
    threw = true;
    assert.ok(err.message.includes('ncclInvalidArgument'), `Expected ncclInvalidArgument, got: ${err.message}`);
  }
  assert.strictEqual(threw, true, 'Should reject invalid root rank');
});

runBenchmark('11. Collective Edge Cases', 'Mismatched Buffer Array Length Rejection', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(2);
  const s0 = rt.cudaMalloc(16);
  const r0 = rt.cudaMalloc(16);
  let threw = false;
  try {
    multi.ncclAllReduce([s0], [r0], 4, 'float32', 'sum', { nranks: 2 });
  } catch (err) {
    threw = true;
    assert.ok(err.message.includes('ncclInvalidArgument'), `Expected ncclInvalidArgument, got: ${err.message}`);
  }
  assert.strictEqual(threw, true, 'Should reject mismatched buffer lengths');
});

// ============================================================================
// SUITE 12: Metric Instrumentation Verification (Accounting Rules)
// ============================================================================
runBenchmark('12. Metric Instrumentation', 'Exact Vector Add N=64 Architecture Model', () => {
  const code = `
  #include <cuda_runtime.h>
  #define N 64
  __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
          c[idx] = a[idx] + b[idx];
      }
  }
  int main() {
      float *d_a, *d_b, *d_c;
      cudaMalloc(&d_a, N * 4);
      cudaMalloc(&d_b, N * 4);
      cudaMalloc(&d_c, N * 4);
      vectorAdd<<<2, 32>>>(d_a, d_b, d_c, N);
      cudaDeviceSynchronize();
      return 0;
  }
  `;
  const trans = window.CudaTranspiler.transpile(code);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  new Function('__rt', trans.code)(rt);
  const c = rt.getCounters().toJSON();

  assert.strictEqual(c.globalLoads, 128, 'N=64 requires 128 loads');
  assert.strictEqual(c.globalStores, 64, 'N=64 requires 64 stores');
  assert.strictEqual(c.globalLoadBytes, 512, 'N=64 requires 512 load bytes');
  assert.strictEqual(c.globalStoreBytes, 256, 'N=64 requires 256 store bytes');
  assert.strictEqual(c.globalTransactions, 24, 'N=64 requires 24 32-byte coalesced transactions');
  assert.strictEqual(c.sharedLoads, 0);
  assert.strictEqual(c.sharedStores, 0);
  assert.strictEqual(c.bankConflicts, 0);
});

runBenchmark('12. Metric Instrumentation', 'Multi-GPU Telemetry Consistency (Totals == Sum of Parts)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);
  const s = [];
  const r = [];
  for (let i = 0; i < 4; i++) {
    rt.setDevice(i);
    s.push(rt.cudaMalloc(32));
    r.push(rt.cudaMalloc(32));
  }
  multi.ncclAllReduce(s, r, 8, 'float32', 'sum', { nranks: 4 });
  const c = rt.getCounters().toJSON();

  let sumOps = 0;
  let sumBytes = 0;
  for (let i = 0; i < 4; i++) {
    sumOps += c.perDevice[i].ncclOps;
    sumBytes += c.perDevice[i].crossGpuBytes;
  }
  assert.strictEqual(c.ncclOps, sumOps, 'Total ncclOps must match sum of perDevice');
  assert.strictEqual(c.crossGpuBytes, sumBytes, 'Total crossGpuBytes must match sum of perDevice');
});

console.log('\n================================================================');
console.log(`  BENCHMARK SUMMARY: ${passedTests}/${totalTests} TESTS PASSED (${failedTests} FAILED)`);
console.log('================================================================\n');

if (failedTests > 0) process.exit(1);
