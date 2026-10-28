const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

console.log('--- AUDITING PHASE 2: ALGORITHMS 2 to 9 ---');

// 2. Reduction & Normalization (L2 norm or sum reduce on odd sizes)
function auditReductionNorm() {
  console.log('\n[2] Reduction & Normalization Audit...');
  const sizes = [1, 5, 17, 33, 64, 129];
  for (const n of sizes) {
    const code = `
    #include <cuda_runtime.h>
    __global__ void reduceSum(const float* in, float* out, int n) {
        __shared__ float s[128];
        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        s[tid] = (gid < n) ? in[gid] : 0.0f;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    int main() {
        int n = ${n};
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * 4);
        cudaMalloc(&d_out, 4);
        float h_in[${Math.max(n, 1)}];
        float expected = 0.0f;
        for (int i = 0; i < n; i++) { h_in[i] = 1.0f; expected += 1.0f; }
        cudaMemcpy(d_in, h_in, n * 4, cudaMemcpyHostToDevice);
        reduceSum<<<1, 128>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
        float h_out[1];
        cudaMemcpy(h_out, d_out, 4, cudaMemcpyDeviceToHost);
        if (h_out[0] != expected) {
            printf("Mismatch for size %d: got %f expected %f\\n", n, h_out[0], expected);
        }
        return 0;
    }
    `;
    const trans = window.CudaTranspiler.transpile(code);
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    try {
      new Function('__rt', trans.code)(rt);
      const diags = rt.getDiagnostics().filter(d => d.message.includes('Mismatch'));
      if (diags.length > 0) console.log(`  Size ${n}: FAIL: ${diags[0].message}`);
      else console.log(`  Size ${n}: PASS`);
    } catch (e) {
      console.log(`  Size ${n}: Runtime ERROR: ${e.message}`);
    }
  }
}

// 3. Matrix Multiplication (Shared Memory Tiling with non-multiples of TILE)
function auditTiledMatMul() {
  console.log('\n[3] Tiled Matrix Multiplication Audit...');
  const widths = [4, 7, 16, 25, 32];
  for (const w of widths) {
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
            if (h_C[i] != expected) {
                printf("Mismatch at %d: got %f expected %f\\n", i, h_C[i], expected);
                break;
            }
        }
        return 0;
    }
    `;
    const trans = window.CudaTranspiler.transpile(code);
    if (!trans.success) {
      console.log(`  Width ${w}: Transpile FAIL: ${trans.error}`);
      continue;
    }
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    try {
      new Function('__rt', trans.code)(rt);
      const diags = rt.getDiagnostics().filter(d => d.message.includes('Mismatch'));
      if (diags.length > 0) console.log(`  Width ${w}: FAIL: ${diags[0].message}`);
      else console.log(`  Width ${w}: PASS`);
    } catch (e) {
      console.log(`  Width ${w}: Runtime ERROR: ${e.message}`);
    }
  }
}

// 5 to 8. Collectives across 2, 4, 8, 16 GPUs
function auditCollectives() {
  console.log('\n[5-8] Native Collectives Audit (2, 4, 8, 16 GPUs)...');
  const clusterSizes = [2, 4, 8, 16];
  const ops = ['sum', 'max', 'min', 'prod'];

  for (const nGpus of clusterSizes) {
    console.log(`\n  Testing Cluster Size: ${nGpus} GPUs:`);
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    const multi = rt.initMultiGPU(nGpus);

    // Test AllReduce for each op
    for (const op of ops) {
      const count = 16;
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(count * 4);
        const rc = rt.cudaMalloc(count * 4);
        for (let i = 0; i < count; i++) {
          s._rawSet(i, (r + 1) * 2.0); // Rank 0 has 2, Rank 1 has 4, etc.
        }
        sendbufs.push(s);
        recvbufs.push(rc);
      }
      const comm = { nranks: nGpus };
      multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', op, comm);

      // Verify rank 0 output
      const r0 = recvbufs[0];
      let expected;
      if (op === 'sum') {
        let s = 0; for (let r = 0; r < nGpus; r++) s += (r + 1) * 2.0; expected = s;
      } else if (op === 'max') {
        expected = nGpus * 2.0;
      } else if (op === 'min') {
        expected = 2.0;
      } else if (op === 'prod') {
        let p = 1.0; for (let r = 0; r < nGpus; r++) p *= (r + 1) * 2.0; expected = p;
      }
      const actual = r0._rawGet(0);
      assert.strictEqual(actual, expected, `AllReduce ${op} on ${nGpus} GPUs failed: got ${actual}, expected ${expected}`);
    }
    console.log(`    ✓ AllReduce SUM, MAX, MIN, PROD PASSED`);

    // Test AllGather
    {
      const sendcount = 4;
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(sendcount * 4);
        const rc = rt.cudaMalloc(sendcount * nGpus * 4);
        for (let i = 0; i < sendcount; i++) s._rawSet(i, r * 10 + i);
        sendbufs.push(s);
        recvbufs.push(rc);
      }
      multi.ncclAllGather(sendbufs, recvbufs, sendcount, 'float32', { nranks: nGpus });
      // Validate every rank has full sequence
      for (let r = 0; r < nGpus; r++) {
        const buf = recvbufs[r];
        for (let srcR = 0; srcR < nGpus; srcR++) {
          for (let i = 0; i < sendcount; i++) {
            assert.strictEqual(buf._rawGet(srcR * sendcount + i), srcR * 10 + i);
          }
        }
      }
      console.log(`    ✓ AllGather PASSED`);
    }

    // Test ReduceScatter
    {
      const recvcount = 4;
      const totalCount = recvcount * nGpus;
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(totalCount * 4);
        const rc = rt.cudaMalloc(recvcount * 4);
        for (let i = 0; i < totalCount; i++) s._rawSet(i, 1.0);
        sendbufs.push(s);
        recvbufs.push(rc);
      }
      multi.ncclReduceScatter(sendbufs, recvbufs, recvcount, 'float32', 'sum', { nranks: nGpus });
      for (let r = 0; r < nGpus; r++) {
        for (let i = 0; i < recvcount; i++) {
          assert.strictEqual(recvbufs[r]._rawGet(i), nGpus * 1.0);
        }
      }
      console.log(`    ✓ ReduceScatter PASSED`);
    }

    // Test Broadcast
    {
      const count = 8;
      const sendbufs = [];
      const recvbufs = [];
      for (let r = 0; r < nGpus; r++) {
        rt.setDevice(r);
        const s = rt.cudaMalloc(count * 4);
        const rc = rt.cudaMalloc(count * 4);
        if (r === 0) {
          for (let i = 0; i < count; i++) s._rawSet(i, 42.0 + i);
        }
        sendbufs.push(s);
        recvbufs.push(rc);
      }
      multi.ncclBroadcast(sendbufs, recvbufs, count, 'float32', 0, { nranks: nGpus });
      for (let r = 0; r < nGpus; r++) {
        for (let i = 0; i < count; i++) {
          assert.strictEqual(recvbufs[r]._rawGet(i), 42.0 + i);
        }
      }
      console.log(`    ✓ Broadcast PASSED`);
    }
  }
}

auditReductionNorm();
auditTiledMatMul();
auditCollectives();
