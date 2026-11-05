// BUG: a shared-memory read-after-write race. Thread t reads its neighbour's value
// but there is no __syncthreads() between the write and the read.
// Gpubox flags the hazard (RAW) before it silently corrupts results on real hardware.
// Fix: add __syncthreads(); between the two statements.
#include <cstdio>

__global__ void neighbourSum(const float* in, float* out) {
    __shared__ float s[64];
    int t = threadIdx.x;
    s[t] = in[t];
    // missing __syncthreads();
    out[t] = s[t] + s[(t + 1) % 64];
}

int main() {
    float h[64];
    for (int i = 0; i < 64; i++) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, 64 * sizeof(float));
    cudaMalloc(&d_out, 64 * sizeof(float));
    cudaMemcpy(d_in, h, 64 * sizeof(float), cudaMemcpyHostToDevice);
    neighbourSum<<<1, 64>>>(d_in, d_out);
    cudaMemcpy(h, d_out, 64 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0] = %f\n", h[0]);
    return 0;
}
