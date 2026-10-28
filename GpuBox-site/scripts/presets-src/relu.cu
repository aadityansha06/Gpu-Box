// ReLU activation: out = max(0, x)  (deep learning primitive)
#include <cstdio>

__global__ void relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

int main() {
    int n = 64;
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    float h_in[64];
    float h_out[64];
    for (int i = 0; i < n; i++) {
        h_in[i] = (float)(i - 32);
    }

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    relu<<<2, 32>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("ReLU: in[0] = %f -> out[0] = %f (Expected: 0.0)\n", h_in[0], h_out[0]);
    printf("ReLU: in[63] = %f -> out[63] = %f (Expected: 31.0)\n", h_in[63], h_out[63]);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
