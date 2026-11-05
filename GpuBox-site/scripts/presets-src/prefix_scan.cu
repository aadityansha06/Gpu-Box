// Inclusive prefix sum (Hillis-Steele) using ping-pong shared buffers
#include <cstdio>

__global__ void scan(const float* in, float* out, int n) {
    __shared__ float bufA[64];
    __shared__ float bufB[64];
    int t = threadIdx.x;
    bufA[t] = in[t];
    __syncthreads();

    float* src = bufA;
    float* dst = bufB;
    for (int offset = 1; offset < n; offset <<= 1) {
        if (t >= offset) dst[t] = src[t] + src[t - offset];
        else             dst[t] = src[t];
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }
    out[t] = src[t];
}

int main() {
    const int n = 64;
    float h[64];
    for (int i = 0; i < n; i++) h[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h, n * sizeof(float), cudaMemcpyHostToDevice);
    scan<<<1, 64>>>(d_in, d_out, n);
    cudaMemcpy(h, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("scan[0] = %f, scan[31] = %f, scan[63] = %f\n", h[0], h[31], h[63]);
    return 0;
}
