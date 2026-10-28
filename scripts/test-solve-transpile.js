global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const fs = require('fs');

// We will test transpile on the attention code
const code = `
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void scores_kernel(
    const float* Q, const float* K, float* scores, int N, int D)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++)
            acc += Q[i * D + d] * K[j * D + d];

        scores[i * N + j] = acc / sqrtf((float)D);
    }
}

__global__ void softmax_rows_kernel(float* scores, int N)
{
    int row = blockIdx.x;
    if (row >= N) return;

    float* r = scores + row * N;
    int tid = threadIdx.x;

    __shared__ float buf[256];
    __shared__ float s_max;
    __shared__ float s_sum;

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x)
        local_max = fmaxf(local_max, r[j]);

    buf[tid] = local_max;
    __syncthreads();

    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            buf[tid] = fmaxf(buf[tid], buf[tid + st]);
        __syncthreads();
    }

    if (tid == 0) s_max = buf[0];
    __syncthreads();

    float m = s_max;
    float local_sum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(r[j] - m);
        r[j] = e;
        local_sum += e;
    }

    buf[tid] = local_sum;
    __syncthreads();

    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            buf[tid] += buf[tid + st];
        __syncthreads();
    }

    if (tid == 0) s_sum = buf[0];
    __syncthreads();

    float inv = 1.0f / s_sum;
    for (int j = tid; j < N; j += blockDim.x)
        r[j] *= inv;
}

__global__ void av_kernel(
    const float* attn, const float* V,
    float* output, int N, int D)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && d < D) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += attn[i * N + j] * V[j * D + d];

        output[i * D + d] = acc;
    }
}

extern "C" void solve(
    const float* Q, const float* K, const float* V,
    float* output, int N, int D)
{
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);

    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);
    softmax_rows_kernel<<<N, 256>>>(scores, N);

    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);

    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    cudaDeviceSynchronize();
    cudaFree(scores);
}

int main()
{
    const int N = 2;
    const int D = 2;

    float h_Q[N * D] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    float h_K[N * D] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    float h_V[N * D] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float *d_Q, *d_K, *d_V, *d_output;

    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_output, N * D * sizeof(float));

    cudaMemcpy(d_Q, h_Q, sizeof(h_Q), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sizeof(h_K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(h_V), cudaMemcpyHostToDevice);

    solve(d_Q, d_K, d_V, d_output, N, D);

    float h_output[N * D];
    cudaMemcpy(h_output, d_output, sizeof(h_output),
               cudaMemcpyDeviceToHost);

    printf("Attention Output:\\n");
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++)
            printf("%.6f ", h_output[i * D + d]);
        printf("\\n");
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}
`;

console.log('Testing Attention transpilation...');
