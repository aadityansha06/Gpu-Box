// BUG DEMO: divergent __syncthreads() (a barrier deadlock on real hardware).
// Only half of the threads reach the barrier. Gpubox should flag it.
#include <cstdio>

__global__ void buggyBarrier(float* data) {
    __shared__ float tile[64];
    int t = threadIdx.x;
    tile[t] = (float)t;
    if (t < 32) {
        __syncthreads();   // never reached by threads 32..63
    }
    data[t] = tile[t];
}

int main() {
    float* d_data;
    cudaMalloc(&d_data, 64 * sizeof(float));

    buggyBarrier<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();

    printf("If you can read this line, the bug was NOT caught.\n");
    return 0;
}
