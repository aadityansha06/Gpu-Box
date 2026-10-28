// Atomic Operations and Contention Profiling
// Uses float device pointers so DevicePointer.get() (a float32 view) returns
// the values written by the host correctly.
#include <cstdio>

__global__ void histogram(const float* data, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = data[i] % 8; // 8 bins
        atomicAdd(&bins[bin], 1);
    }
}

int main() {
    int n = 256;
    float *d_data;
    int *d_bins;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_bins, 8 * sizeof(int));
    cudaMemset(d_bins, 0, 8 * sizeof(int));

    float h_data[256];
    for (int i = 0; i < n; i++) h_data[i] = (float)i;

    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    histogram<<<4, 64>>>(d_data, d_bins, n);
    cudaDeviceSynchronize();

    float h_bins[8];
    cudaMemcpy(h_bins, d_bins, 8 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Atomic Histogram Results (256 items into 8 bins):\n");
    for (int b = 0; b < 8; b++) {
        printf("  Bin %d: count = %d\n", b, h_bins[b]);
    }

    cudaFree(d_data);
    cudaFree(d_bins);
    return 0;
}
