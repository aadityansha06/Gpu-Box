// SAXPY: y = a*x + y  (NVIDIA cuda-samples pattern)
#include <cstdio>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 512;
    size_t bytes = n * sizeof(float);

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    float h_x[512];
    float h_y[512];
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    saxpy<<<2, 256>>>(n, 3.0f, d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    printf("SAXPY: y[0] = %f (Expected: 5.0)\n", h_y[0]);
    printf("SAXPY: y[511] = %f (Expected: 5.0)\n", h_y[511]);

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
