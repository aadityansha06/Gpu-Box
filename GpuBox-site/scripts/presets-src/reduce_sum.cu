// Parallel reduction: shared-memory tree + one atomicAdd per block
#include <cstdio>

__global__ void reduceSum(const float* in, float* out, int n) {
    __shared__ float s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) atomicAdd(out, s[0]);
}

int main() {
    const int n = 4096;
    float h_in[4096];
    for (int i = 0; i < n; i++) h_in[i] = 0.5f;          // expected sum = 2048

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    reduceSum<<<(n + 255) / 256, 256>>>(d_in, d_out, n);

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum = %f (expected 2048.000000)\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
