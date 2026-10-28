// Vector Addition in CUDA C++
#include <cstdio>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    float h_a[1024];
    float h_b[1024];
    float h_c[1024];

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch with 4 blocks of 256 threads
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Vector Add Verification:\n");
    printf("  h_c[0]   = %f (Expected: 0.0)\n", h_c[0]);
    printf("  h_c[10]  = %f (Expected: 30.0)\n", h_c[10]);
    printf("  h_c[1023] = %f (Expected: 3069.0)\n", h_c[1023]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
