const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

console.log('================================================================');
console.log('  GPU METRICS INSTRUMENTATION REGRESSION TEST SUITE');
console.log('================================================================\n');

let totalTests = 0;
let passedTests = 0;

function runTest(name, fn) {
  totalTests++;
  console.log(`[TEST ${totalTests}] ${name}...`);
  try {
    fn();
    passedTests++;
    console.log(`  ✓ PASSED\n`);
  } catch (err) {
    console.error(`  ✗ FAILED: ${err.message}\n`, err.stack);
    process.exit(1);
  }
}

// -----------------------------------------------------------------------------
// Test 1: Vector Addition (Global Memory Loads, Stores, Coalesced Transactions)
// -----------------------------------------------------------------------------
runTest('Vector Addition (Global Loads, Stores, Bytes, Transactions, No Shared)', () => {
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
  assert.strictEqual(trans.success, true, 'Transpilation should succeed');

  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const runner = new Function('__rt', trans.code);
  runner(rt);

  const c = rt.getCounters().toJSON();

  // 64 elements: a loaded (64), b loaded (64) = 128 loads
  assert.strictEqual(c.threadsLaunched, 64, 'Threads launched should be 64');
  assert.strictEqual(c.blocksLaunched, 2, 'Blocks launched should be 2');
  assert.strictEqual(c.globalLoads, 128, 'Global loads must be exactly 128');
  assert.strictEqual(c.globalStores, 64, 'Global stores must be exactly 64');
  assert.strictEqual(c.globalLoadBytes, 512, 'Global load bytes must be 512 (128 * 4)');
  assert.strictEqual(c.globalStoreBytes, 256, 'Global store bytes must be 256 (64 * 4)');
  assert.strictEqual(c.totalBytesRead, 512, 'totalBytesRead alias must be 512');
  assert.strictEqual(c.totalBytesWritten, 256, 'totalBytesWritten alias must be 256');

  // Transactions: 2 blocks, each with 32 threads.
  // Each block: 1 warp of 32 threads.
  // Warp reads 32 contiguous floats = 128 bytes = 4 sectors of 32B for a, 4 for b, 4 for c = 12 transactions per block.
  // Total across 2 blocks: 24 transactions.
  assert.ok(c.globalTransactions > 0, 'Global transactions must be non-zero');
  assert.strictEqual(c.globalTransactions, 24, 'Global transactions should be exactly 24');
  assert.strictEqual(c.coalescedTransactions, 24, 'coalescedTransactions alias should be 24');

  // Shared memory must be zero
  assert.strictEqual(c.sharedLoads, 0, 'Shared loads must be 0 for vectorAdd');
  assert.strictEqual(c.sharedStores, 0, 'Shared stores must be 0 for vectorAdd');
  assert.strictEqual(c.sharedLoadBytes, 0, 'Shared load bytes must be 0');
  assert.strictEqual(c.bankConflicts, 0, 'Bank conflicts must be 0');
  assert.strictEqual(c.crossGpuBytes, 0, 'Cross GPU bytes must be 0');
});

// -----------------------------------------------------------------------------
// Test 2: Matrix Multiplication with Shared Memory Tiling
// -----------------------------------------------------------------------------
runTest('Tiled Matrix Multiplication (Shared Memory, Bank Conflicts, Barriers)', () => {
  const code = `
  #include <cuda_runtime.h>
  #define TILE 4
  #define N 4
  __global__ void matmul(const float* A, const float* B, float* C) {
      __shared__ float As[TILE][TILE];
      __shared__ float Bs[TILE][TILE];
      int r = threadIdx.y;
      int c = threadIdx.x;
      As[r][c] = A[r * N + c];
      Bs[r][c] = B[r * N + c];
      __syncthreads();
      float sum = 0.0f;
      for (int k = 0; k < TILE; k++) {
          sum += As[r][k] * Bs[k][c];
      }
      __syncthreads();
      C[r * N + c] = sum;
  }
  int main() {
      float *d_A, *d_B, *d_C;
      cudaMalloc(&d_A, 16 * 4);
      cudaMalloc(&d_B, 16 * 4);
      cudaMalloc(&d_C, 16 * 4);
      dim3 block(4, 4);
      dim3 grid(1, 1);
      matmul<<<grid, block>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      return 0;
  }
  `;

  const trans = window.CudaTranspiler.transpile(code);
  assert.strictEqual(trans.success, true);

  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const runner = new Function('__rt', trans.code);
  runner(rt);

  const c = rt.getCounters().toJSON();

  assert.strictEqual(c.threadsLaunched, 16, 'Threads launched should be 16');
  assert.strictEqual(c.blocksLaunched, 1, 'Blocks launched should be 1');
  assert.strictEqual(c.globalLoads, 32, 'Global loads: 16 from A + 16 from B = 32');
  assert.strictEqual(c.globalStores, 16, 'Global stores: 16 to C');
  assert.ok(c.globalTransactions > 0, 'Global transactions must be non-zero');

  // Shared memory checks
  assert.ok(c.sharedStores > 0, 'Shared memory stores must be recorded');
  assert.strictEqual(c.sharedStores, 32, 'Shared stores: 16 to As + 16 to Bs = 32');
  assert.ok(c.sharedLoads > 0, 'Shared memory loads must be recorded');
  // 16 threads each loop 4 times = 64 reads from As + 64 reads from Bs = 128
  assert.strictEqual(c.sharedLoads, 128, 'Shared loads should be 128');
  assert.strictEqual(c.sharedStoreBytes, 128, 'Shared store bytes should be 128');
  assert.strictEqual(c.sharedLoadBytes, 512, 'Shared load bytes should be 512');
  assert.strictEqual(c.syncthreadsCount, 32, 'syncthreadsCount should be 32 (16 threads * 2 barriers)');
});

// -----------------------------------------------------------------------------
// Test 3: Softmax (Exp, Math FLOPs, Reduction, Normalization)
// -----------------------------------------------------------------------------
runTest('Softmax (Math FLOPs, Global Loads/Stores, Reductions)', () => {
  const code = `
  #include <cuda_runtime.h>
  #include <math.h>
  #define N 8
  __global__ void softmax(const float* in, float* out) {
      __shared__ float s[N];
      int tid = threadIdx.x;
      s[tid] = expf(in[tid]);
      __syncthreads();
      if (tid == 0) {
          float sum = 0.0f;
          for (int i = 0; i < N; i++) sum += s[i];
          s[0] = sum;
      }
      __syncthreads();
      out[tid] = s[tid] / s[0];
  }
  int main() {
      float *d_in, *d_out;
      cudaMalloc(&d_in, N * 4);
      cudaMalloc(&d_out, N * 4);
      softmax<<<1, N>>>(d_in, d_out);
      cudaDeviceSynchronize();
      return 0;
  }
  `;

  const trans = window.CudaTranspiler.transpile(code);
  assert.strictEqual(trans.success, true);

  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const runner = new Function('__rt', trans.code);
  runner(rt);

  const c = rt.getCounters().toJSON();

  assert.strictEqual(c.threadsLaunched, 8);
  assert.strictEqual(c.globalLoads, 8, '8 global reads from in');
  assert.strictEqual(c.globalStores, 8, '8 global writes to out');
  assert.ok(c.globalTransactions > 0, 'Transactions must be recorded');
  assert.ok(c.sharedLoads > 0, 'Shared reads must be recorded');
  assert.ok(c.sharedStores > 0, 'Shared writes must be recorded');
  assert.strictEqual(c.syncthreadsCount, 16, 'Two syncthreads for 8 threads = 16');
});

// -----------------------------------------------------------------------------
// Test 4: NCCL AllGather (4 GPUs, Inter-GPU Traffic, No Kernel Memory Access)
// -----------------------------------------------------------------------------
runTest('NCCL AllGather across 4 GPUs (crossGpuBytes, ncclOps, Clean Separation)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);

  const count = 32;
  const sendbufs = [];
  const recvbufs = [];

  for (let r = 0; r < 4; r++) {
    rt.setDevice(r);
    const s = rt.cudaMalloc(count * 4);
    const rc = rt.cudaMalloc(count * 4 * 4); // each rank receives 4 * count
    sendbufs.push(s);
    recvbufs.push(rc);
  }

  const comm = { nranks: 4 };
  multi.ncclAllGather(sendbufs, recvbufs, count, 'float32', comm);

  const c = rt.getCounters().toJSON();

  assert.strictEqual(c.ncclOps, 4, 'ncclOps should be 4 (1 per device)');
  // Each device receives count * 4 bytes from (4 - 1) peers = 32 * 4 * 3 = 384 bytes
  // Across 4 devices: 384 * 4 = 1536 bytes
  assert.strictEqual(c.crossGpuBytes, 1536, 'crossGpuBytes should be 1536');
  // Kernel memory must remain 0 because this was pure collective communication!
  assert.strictEqual(c.globalLoads, 0, 'Global loads must be 0 for pure collective');
  assert.strictEqual(c.globalStores, 0, 'Global stores must be 0 for pure collective');
  assert.strictEqual(c.globalTransactions, 0, 'Global transactions must be 0');

  // Verify per-device breakdown
  assert.strictEqual(c.perDevice.length, 4, 'Should have 4 per-device records');
  for (let r = 0; r < 4; r++) {
    assert.strictEqual(c.perDevice[r].crossGpuBytes, 384, `Device ${r} crossGpuBytes should be 384`);
    assert.strictEqual(c.perDevice[r].ncclOps, 1, `Device ${r} ncclOps should be 1`);
  }
});

// -----------------------------------------------------------------------------
// Test 5: NCCL ReduceScatter across 4 GPUs (crossGpuBytes, ncclOps)
// -----------------------------------------------------------------------------
runTest('NCCL ReduceScatter across 4 GPUs (Metrics verification)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);

  const recvcount = 16;
  const totalCount = recvcount * 4;
  const sendbufs = [];
  const recvbufs = [];

  for (let r = 0; r < 4; r++) {
    rt.setDevice(r);
    sendbufs.push(rt.cudaMalloc(totalCount * 4));
    recvbufs.push(rt.cudaMalloc(recvcount * 4));
  }

  const comm = { nranks: 4 };
  multi.ncclReduceScatter(sendbufs, recvbufs, recvcount, 'float32', 'sum', comm);

  const c = rt.getCounters().toJSON();

  assert.strictEqual(c.ncclOps, 4, 'ncclOps should be 4');
  // Each rank receives recvcount * 4 bytes from (4 - 1) peers = 16 * 4 * 3 = 192 bytes
  // Total across 4 devices = 192 * 4 = 768 bytes
  assert.strictEqual(c.crossGpuBytes, 768, 'crossGpuBytes should be 768');
  assert.strictEqual(c.globalLoads, 0, 'Kernel global loads must be 0');
  assert.strictEqual(c.globalStores, 0, 'Kernel global stores must be 0');
});

// -----------------------------------------------------------------------------
// Test 6: NCCL Broadcast across 4 GPUs (crossGpuBytes, root vs non-root)
// -----------------------------------------------------------------------------
runTest('NCCL Broadcast across 4 GPUs (Root vs non-root traffic)', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(4);

  const count = 64;
  const sendbufs = [];
  const recvbufs = [];

  for (let r = 0; r < 4; r++) {
    rt.setDevice(r);
    sendbufs.push(rt.cudaMalloc(count * 4));
    recvbufs.push(rt.cudaMalloc(count * 4));
  }

  const comm = { nranks: 4 };
  multi.ncclBroadcast(sendbufs, recvbufs, count, 'float32', 0, comm); // root is 0

  const c = rt.getCounters().toJSON();

  assert.strictEqual(c.ncclOps, 4, 'ncclOps should be 4');
  // 3 non-root ranks receive 64 * 4 = 256 bytes each -> 768 bytes total
  assert.strictEqual(c.crossGpuBytes, 768, 'crossGpuBytes should be 768');
  assert.strictEqual(c.perDevice[0].crossGpuBytes, 0, 'Root rank 0 crossGpuBytes should be 0');
  assert.strictEqual(c.perDevice[1].crossGpuBytes, 256, 'Non-root rank 1 crossGpuBytes should be 256');
});

// -----------------------------------------------------------------------------
// Test 7: Multi-GPU Distributed Execution (Kernels on all 4 GPUs)
// -----------------------------------------------------------------------------
runTest('Multi-GPU Distributed Kernel Execution across 4 GPUs', () => {
  const code = `
  #include <cuda_runtime.h>
  #define CHUNK 16
  __global__ void rankKernel(const float* in, float* out, int rank) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < CHUNK) {
          out[tid] = in[tid] + rank;
      }
  }
  int main() {
      int nGpus = 4;
      float *d_in[4];
      float *d_out[4];
      for (int g = 0; g < nGpus; g++) {
          cudaSetDevice(g);
          cudaMalloc(&d_in[g], CHUNK * sizeof(float));
          cudaMalloc(&d_out[g], CHUNK * sizeof(float));
          rankKernel<<<1, CHUNK>>>(d_in[g], d_out[g], g);
      }
      cudaDeviceSynchronize();
      return 0;
  }
  `;

  const trans = window.CudaTranspiler.transpile(code);
  assert.strictEqual(trans.success, true);

  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  rt.initMultiGPU(4);
  const runner = new Function('__rt', trans.code);
  runner(rt);

  const agg = rt.getCounters().toJSON();

  // Each GPU launched 16 threads, 1 block, 16 loads, 16 stores
  // Total across 4 GPUs:
  assert.strictEqual(agg.threadsLaunched, 64, 'Aggregated threads launched should be 64');
  assert.strictEqual(agg.blocksLaunched, 4, 'Aggregated blocks launched should be 4');
  assert.strictEqual(agg.globalLoads, 64, 'Aggregated global loads should be 64');
  assert.strictEqual(agg.globalStores, 64, 'Aggregated global stores should be 64');
  assert.strictEqual(agg.globalLoadBytes, 256, 'Aggregated global load bytes should be 256');
  assert.strictEqual(agg.globalStoreBytes, 256, 'Aggregated global store bytes should be 256');
  assert.ok(agg.globalTransactions > 0, 'Aggregated transactions must be non-zero');

  // Verify that EVERY node has identical individual non-zero counts
  assert.strictEqual(agg.perDevice.length, 4);
  for (let g = 0; g < 4; g++) {
    const dev = rt.getCounters(g).toJSON();
    assert.strictEqual(dev.threadsLaunched, 16, `Device ${g} threads should be 16`);
    assert.strictEqual(dev.blocksLaunched, 1, `Device ${g} blocks should be 1`);
    assert.strictEqual(dev.globalLoads, 16, `Device ${g} loads should be 16`);
    assert.strictEqual(dev.globalStores, 16, `Device ${g} stores should be 16`);
    assert.strictEqual(dev.globalLoadBytes, 64, `Device ${g} load bytes should be 64`);
    assert.strictEqual(dev.globalStoreBytes, 64, `Device ${g} store bytes should be 64`);
    assert.ok(dev.globalTransactions > 0, `Device ${g} transactions must be non-zero`);
  }
});

// -----------------------------------------------------------------------------
// Test 8: Runtime Reset Between Runs
// -----------------------------------------------------------------------------
runTest('Runtime Reset between consecutive runs', () => {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  rt.initMultiGPU(2);

  // Perform some allocations and kernel launches
  rt.setDevice(0);
  const p0 = rt.cudaMalloc(32);
  p0.set(0, 1.0); // 1 store
  p0.get(0);      // 1 load

  rt.setDevice(1);
  const p1 = rt.cudaMalloc(32);
  p1.set(0, 2.0); // 1 store

  let c = rt.getCounters().toJSON();
  assert.ok(c.globalLoads > 0, 'Loads should be > 0 before reset');
  assert.ok(c.globalStores > 0, 'Stores should be > 0 before reset');

  // Now perform reset
  rt.reset();
  c = rt.getCounters().toJSON();

  assert.strictEqual(c.threadsLaunched, 0, 'threadsLaunched must be 0 after reset');
  assert.strictEqual(c.blocksLaunched, 0, 'blocksLaunched must be 0 after reset');
  assert.strictEqual(c.globalLoads, 0, 'globalLoads must be 0 after reset');
  assert.strictEqual(c.globalStores, 0, 'globalStores must be 0 after reset');
  assert.strictEqual(c.globalLoadBytes, 0, 'globalLoadBytes must be 0 after reset');
  assert.strictEqual(c.globalStoreBytes, 0, 'globalStoreBytes must be 0 after reset');
  assert.strictEqual(c.globalTransactions, 0, 'globalTransactions must be 0 after reset');
  assert.strictEqual(c.sharedLoads, 0, 'sharedLoads must be 0 after reset');
  assert.strictEqual(c.sharedStores, 0, 'sharedStores must be 0 after reset');
  assert.strictEqual(c.crossGpuBytes, 0, 'crossGpuBytes must be 0 after reset');
  assert.strictEqual(c.ncclOps, 0, 'ncclOps must be 0 after reset');

  // Verify per-device counters are also 0
  for (let g = 0; g < 2; g++) {
    const devC = rt.getCounters(g).toJSON();
    assert.strictEqual(devC.globalLoads, 0, `Device ${g} loads must be 0 after reset`);
    assert.strictEqual(devC.globalStores, 0, `Device ${g} stores must be 0 after reset`);
  }
});

console.log('================================================================');
console.log(`  ALL ${passedTests}/${totalTests} REGRESSION TESTS PASSED SUCCESSFULLY!`);
console.log('================================================================');
