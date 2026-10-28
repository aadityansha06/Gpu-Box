const fs = require('fs');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {                                      \\
    cudaError_t err = (call);                                      \\
    if (err != cudaSuccess) {                                      \\
        printf("CUDA error: %s\\n", cudaGetErrorString(err));       \\
        exit(1);                                                   \\
    }                                                              \\
} while (0)

const int SHARD_SIZE = 2;
const int NUM_GPUS = 2;

// Stage 1: Find the maximum logit on each GPU.
__global__ void localMaxKernel(
    const float* input, float* localMax, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float m = input[0];
        for (int i = 1; i < n; i++) {
            if (input[i] > m)
                m = input[i];
        }
        *localMax = m;
    }
}

// Stage 2: Sum exp(logit - globalMax) on each GPU.
__global__ void localExpSumKernel(
    const float* input, float* localSum,
    float globalMax, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += expf(input[i] - globalMax);

        *localSum = sum;
    }
}

// Stage 3: Normalize using the global sum.
__global__ void normalizeKernel(
    const float* input, float* output,
    float globalMax, float globalSum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] =
            expf(input[i] - globalMax) / globalSum;
    }
}

int main()
{
    // Hardcoded distributed input:
    // GPU 0: [1, 2]
    // GPU 1: [3, 4]
    float hostInput[NUM_GPUS][SHARD_SIZE] = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    float hostOutput[NUM_GPUS][SHARD_SIZE];
    float hostLocalMax[NUM_GPUS];
    float hostLocalSum[NUM_GPUS];

    float *d_input[NUM_GPUS];
    float *d_output[NUM_GPUS];
    float *d_localMax[NUM_GPUS];
    float *d_localSum[NUM_GPUS];

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < NUM_GPUS) {
        printf("Need 2 CUDA devices; found %d\\n",
               deviceCount);
        return 1;
    }

    // Allocate memory and copy each shard to its GPU.
    for (int g = 0; g < NUM_GPUS; g++) {
        CUDA_CHECK(cudaSetDevice(g));

        CUDA_CHECK(cudaMalloc(&d_input[g],
            SHARD_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output[g],
            SHARD_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_localMax[g],
            sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_localSum[g],
            sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input[g], hostInput[g],
            SHARD_SIZE * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    // Phase 1: Compute local maxima.
    for (int g = 0; g < NUM_GPUS; g++) {
        CUDA_CHECK(cudaSetDevice(g));

        localMaxKernel<<<1, 1>>>(
            d_input[g], d_localMax[g], SHARD_SIZE);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&hostLocalMax[g],
            d_localMax[g], sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    // Host combines local maxima.
    float globalMax = fmaxf(
        hostLocalMax[0], hostLocalMax[1]);

    // Phase 2: Compute local exponential sums,
    // both shifted by the same global maximum.
    for (int g = 0; g < NUM_GPUS; g++) {
        CUDA_CHECK(cudaSetDevice(g));

        localExpSumKernel<<<1, 1>>>(
            d_input[g], d_localSum[g],
            globalMax, SHARD_SIZE);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&hostLocalSum[g],
            d_localSum[g], sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    // Host combines local sums into the global sum.
    float globalSum = hostLocalSum[0] + hostLocalSum[1];

    // Phase 3: Normalize each shard using global values.
    for (int g = 0; g < NUM_GPUS; g++) {
        CUDA_CHECK(cudaSetDevice(g));

        normalizeKernel<<<1, 32>>>(
            d_input[g], d_output[g],
            globalMax, globalSum, SHARD_SIZE);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hostOutput[g], d_output[g],
            SHARD_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    printf("Global max: %.6f\\n", globalMax);
    printf("Global exponential sum: %.6f\\n", globalSum);

    printf("GPU 0 softmax: %.6f %.6f\\n",
        hostOutput[0][0], hostOutput[0][1]);

    printf("GPU 1 softmax: %.6f %.6f\\n",
        hostOutput[1][0], hostOutput[1][1]);

    float probabilitySum = 0.0f;
    for (int g = 0; g < NUM_GPUS; g++)
        for (int i = 0; i < SHARD_SIZE; i++)
            probabilitySum += hostOutput[g][i];

    printf("Global probability sum: %.6f\\n",
        probabilitySum);

    // Cleanup.
    for (int g = 0; g < NUM_GPUS; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        cudaFree(d_input[g]);
        cudaFree(d_output[g]);
        cudaFree(d_localMax[g]);
        cudaFree(d_localSum[g]);
    }

    return 0;
}
`;

const res = window.CudaTranspiler.transpile(code);
console.log('Success:', res.success);
console.log('Error:', res.error);
console.log('--- JS Code ---');
console.log(res.code);
console.log('\n--- Executing Simulation on 2 GPUs ---');
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rt.initMultiGPU(2);
const fn = new Function('__rt', res.code);
fn(rt);

console.log('--- Aggregated Counters ---');
console.log(JSON.stringify(rt.getCounters().toJSON(), null, 2));
