// Data-parallel training step on 8 virtual GPUs: every rank computes a local gradient,
// then ncclAllReduce averages it. Check "Cluster & NCCL" for per-rank traffic and time.
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>

__global__ void localGradient(float* grad, int n, float rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = rank + 1.0f;                    // pretend gradient = rank + 1
}

int main() {
    const int gpus = 8;
    const int n = 1024;
    ncclComm_t comms[8];
    ncclCommInitAll(comms, gpus);

    float *grad[8], *avg[8];
    for (int r = 0; r < gpus; r++) {
        cudaSetDevice(r);
        cudaMalloc(&grad[r], n * sizeof(float));
        cudaMalloc(&avg[r], n * sizeof(float));
        localGradient<<<(n + 255) / 256, 256>>>(grad[r], n, (float)r);
    }

    for (int r = 0; r < gpus; r++) {
        cudaSetDevice(r);
        ncclAllReduce(grad[r], avg[r], n, ncclFloat, ncclAvg, comms[r], 0);
    }

    for (int r = 0; r < gpus; r += 7) {
        cudaSetDevice(r);
        float h[1024];
        cudaMemcpy(h, avg[r], n * sizeof(float), cudaMemcpyDeviceToHost);
        printf("GPU %d averaged gradient[0] = %f (expected 4.500000)\n", r, h[0]);
    }
    return 0;
}
