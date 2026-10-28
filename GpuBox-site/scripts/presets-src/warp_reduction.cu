// Parallel Reduction using Warp Shuffle Primitives (__shfl_down_sync)
#include <cstdio>

__global__ void blockReduce(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (gid < n) ? in[gid] : 0.0f;

    // Inline warp reduction: each lane reads from lane+offset.
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        out[blockIdx.x] = sum;
    }
}

int main() {
    int n = 32;
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));

    float h_in[32];
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    blockReduce<<<1, 32>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    float h_out[1];
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Warp reduction result = %f (Expected: 32.0)\n", h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
