global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    __shared__ float s[256];
    s[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    for (int s_offset = 128; s_offset > 0; s_offset >>= 1) {
        if (tid < s_offset) s[tid] += s[tid + s_offset];
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}

int main() {
    int nGpus = 2;
    cudaGetDeviceCount(&nGpus);
    int N = 16;
    int chunk = N / nGpus;
    float h_data[16];
    for (int i = 0; i < N; i++) h_data[i] = i + 1;

    float* d_in[2];
    float* d_out[2];
    float sum[2];

    for (int i = 0; i < nGpus; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_in[i], chunk * sizeof(float));
        cudaMalloc(&d_out[i], sizeof(float));
        cudaMemcpy(d_in[i], h_data + i * chunk, chunk * sizeof(float), cudaMemcpyHostToDevice);
        reduce<<<1, 256>>>(d_in[i], d_out[i], chunk);
    }
    cudaDeviceSynchronize();

    float globalSum = 0;
    for (int i = 0; i < nGpus; i++) {
        cudaSetDevice(i);
        cudaMemcpy(&sum[i], d_out[i], sizeof(float), cudaMemcpyDeviceToHost);
        printf("GPU Node %d: elements [%d, %d), sum = %.0f\\n", i, i * chunk, (i + 1) * chunk, sum[i]);
        globalSum += sum[i];
    }
    printf("Global sum = %.0f\\n", globalSum);
    return 0;
}
`;

const tRes = window.CudaTranspiler.transpile(code);
console.log('--- Transpiled Code ---');
console.log(tRes.code);
console.log('--- Executing 2 GPUs ---');
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rt.initMultiGPU(2);
const fn = new Function('__rt', tRes.code);
fn(rt);

console.log('\n--- Executing 4 GPUs ---');
const rt4 = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rt4.initMultiGPU(4);
const fn4 = new Function('__rt', tRes.code);
fn4(rt4);

console.log('\n--- Executing 8 GPUs ---');
const rt8 = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rt8.initMultiGPU(8);
const fn8 = new Function('__rt', tRes.code);
fn8(rt8);

console.log('\n--- Executing 16 GPUs ---');
const rt16 = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rt16.initMultiGPU(16);
const fn16 = new Function('__rt', tRes.code);
fn16(rt16);
