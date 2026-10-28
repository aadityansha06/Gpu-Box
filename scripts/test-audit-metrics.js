global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cuda_runtime.h>
#define N 16
__global__ void vecAdd(const float* A, const float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * 4);
    cudaMalloc(&d_B, N * 4);
    cudaMalloc(&d_C, N * 4);
    vecAdd<<<1, 16>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    return 0;
}
`;

const tRes = window.CudaTranspiler.transpile(code);
console.log('Transpiled code:\n', tRes.code);

const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
const fn = new Function('__rt', tRes.code);
fn(rt);

console.log('--- Single GPU Counters ---');
console.log(JSON.stringify(rt.getCounters().toJSON(), null, 2));

console.log('\n--- Multi-GPU Test (2 Nodes) ---');
const rtMulti = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rtMulti.initMultiGPU(2);
const fnMulti = new Function('__rt', tRes.code);
fnMulti(rtMulti);

console.log('rtMulti.getCounters():');
console.log(JSON.stringify(rtMulti.getCounters().toJSON(), null, 2));
for (let d = 0; d < 2; d++) {
  console.log(`Device ${d} counters:`);
  console.log(JSON.stringify(rtMulti._multiGPU.getDevice(d).counters.toJSON(), null, 2));
}
