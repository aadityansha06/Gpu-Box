// LayerNorm forward: mean/variance reduction + rsqrtf, as used in every transformer block
#include <cstdio>
#include <math.h>

__global__ void layerNorm(const float* in, float* out, int cols) {
    __shared__ float sh[32];
    int row = blockIdx.x, t = threadIdx.x;

    float s = 0.0f;
    for (int c = t; c < cols; c += blockDim.x) s += in[row * cols + c];
    sh[t] = s;
    __syncthreads();
    for (int k = 16; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; __syncthreads(); }
    float mean = sh[0] / cols;
    __syncthreads();

    float v = 0.0f;
    for (int c = t; c < cols; c += blockDim.x) { float d = in[row * cols + c] - mean; v += d * d; }
    sh[t] = v;
    __syncthreads();
    for (int k = 16; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; __syncthreads(); }
    float inv = rsqrtf(sh[0] / cols + 1e-5f);

    for (int c = t; c < cols; c += blockDim.x) out[row * cols + c] = (in[row * cols + c] - mean) * inv;
}

int main() {
    const int rows = 4, cols = 96;
    float h[384];
    for (int i = 0; i < rows * cols; i++) h[i] = (float)((i * 5) % 17) * 0.5f + (float)(i / cols);

    float *d_in, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));
    cudaMemcpy(d_in, h, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    layerNorm<<<rows, 32>>>(d_in, d_out, cols);
    cudaMemcpy(h, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    float mean = 0.0f, var = 0.0f;
    for (int c = 0; c < cols; c++) mean += h[c];
    mean /= cols;
    for (int c = 0; c < cols; c++) var += (h[c] - mean) * (h[c] - mean);
    var /= cols;
    printf("row 0 after LayerNorm: mean = %.4f, variance = %.4f\n", mean, var);
    return 0;
}
