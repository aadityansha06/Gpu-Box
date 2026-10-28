// BUG DEMO: off-by-one boundary error. The "if (i < n)" guard is missing.
// Gpubox should report an out-of-bounds access with kernel / block / thread.
#include <cstdio>

__global__ void buggyCopy(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[i];   // threads 100..127 access past the end of the buffer
}

int main() {
    int n = 100;
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    buggyCopy<<<1, 128>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    printf("If you can read this line, the bug was NOT caught.\n");
    return 0;
}
