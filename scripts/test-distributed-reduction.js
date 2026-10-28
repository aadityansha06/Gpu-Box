// Test script for Two-GPU and N-GPU Distributed Vector Reduction
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');
const { transpile } = window.CudaTranspiler;

const cudaTwoGpuReductionCode = `
#include <iostream>
#include <cuda_runtime.h>

__global__ void blockReduce(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

int main() {
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    std::cout << "Available GPU devices: " << numDevices << std::endl;

    const int N = 1024;
    const int nPerGpu = N / numDevices;
    const size_t bytesPerGpu = nPerGpu * sizeof(float);

    float* d_in[16];
    float* d_out[16];
    float h_in[1024];

    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }

    // Allocate & initialize per device
    for (int d = 0; d < numDevices; d++) {
        cudaSetDevice(d);
        cudaMalloc(&d_in[d], bytesPerGpu);
        cudaMalloc(&d_out[d], sizeof(float));
        cudaMemset(d_out[d], 0, sizeof(float));
        
        cudaMemcpy(d_in[d], &h_in[d * nPerGpu], bytesPerGpu, cudaMemcpyHostToDevice);
        
        // Launch partial reduction
        blockReduce<<<2, 256>>>(d_in[d], d_out[d], nPerGpu);
    }

    cudaDeviceSynchronize();

    // Reduce results from all GPUs to GPU 0 using Peer-to-Peer access
    for (int d = 1; d < numDevices; d++) {
        cudaDeviceEnablePeerAccess(d, 0);
        // Copy d_out[d] to host or GPU 0
    }

    float totalSum = 0.0f;
    for (int d = 0; d < numDevices; d++) {
        cudaSetDevice(d);
        float partial = 0.0f;
        cudaMemcpy(&partial, d_out[d], sizeof(float), cudaMemcpyDeviceToHost);
        totalSum += partial;
    }

    std::cout << "Total Distributed Reduced Sum = " << totalSum << std::endl;

    for (int d = 0; d < numDevices; d++) {
        cudaSetDevice(d);
        cudaFree(d_in[d]);
        cudaFree(d_out[d]);
    }

    return 0;
}
`;

console.log('--- Testing Distributed Vector Reduction on 2 GPUs ---');
const prof = window.GPU_PROFILES.getProfile('A100');
const runtime2 = new window.CudaRuntime(prof);
runtime2.initMultiGPU(2);

const tRes2 = transpile(cudaTwoGpuReductionCode);
console.log('Transpilation success:', tRes2.success);
if (!tRes2.success) {
  console.error('Errors:', tRes2.errors);
  process.exit(1);
}

try {
  const fn2 = new Function('__rt', tRes2.code);
  fn2(runtime2);
  console.log('✓ 2-GPU reduction executed without syntax errors!');
} catch (e) {
  console.error('Execution error:', e);
  console.log('Transpiled code:\n', tRes2.code);
  process.exit(1);
}

console.log('\n--- Testing Distributed Vector Reduction on 8 GPUs ---');
const runtime8 = new window.CudaRuntime(prof);
runtime8.initMultiGPU(8);

try {
  const fn8 = new Function('__rt', tRes2.code);
  fn8(runtime8);
  console.log('✓ 8-GPU reduction executed without syntax errors!');
} catch (e) {
  console.error('Execution error:', e);
  process.exit(1);
}

console.log('\n--- Testing Distributed Vector Reduction on 16 GPUs ---');
const runtime16 = new window.CudaRuntime(prof);
runtime16.initMultiGPU(16);

try {
  const fn16 = new Function('__rt', tRes2.code);
  fn16(runtime16);
  console.log('✓ 16-GPU reduction executed without syntax errors!');
} catch (e) {
  console.error('Execution error:', e);
  process.exit(1);
}
