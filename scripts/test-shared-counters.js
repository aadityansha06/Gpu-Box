global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cuda_runtime.h>
__global__ void testShared(const float* in, float* out) {
    __shared__ float s[64];
    int tid = threadIdx.x;
    s[tid] = in[tid];
    __syncthreads();
    out[tid] = s[tid];
}
int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in, 64 * 4);
    cudaMalloc(&d_out, 64 * 4);
    testShared<<<1, 64>>>(d_in, d_out);
    cudaDeviceSynchronize();
    return 0;
}
`;

const tRes = window.CudaTranspiler.transpile(code);
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
const fn = new Function('__rt', tRes.code);
fn(rt);

console.log('--- Counters for testShared ---');
console.log(JSON.stringify(rt.getCounters().toJSON(), null, 2));
