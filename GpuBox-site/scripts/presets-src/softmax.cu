// Row-wise softmax: the core of every attention layer
#include <cstdio>
#include <math.h>

__global__ void softmaxRows(const float* in, float* out, int cols) {
    __shared__ float red[64];
    int row = blockIdx.x, t = threadIdx.x;

    float m = -1e30f;                                    // 1) row max (numerical stability)
    for (int c = t; c < cols; c += blockDim.x) m = fmaxf(m, in[row * cols + c]);
    red[t] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float sum = 0.0f;                                    // 2) sum of exp
    for (int c = t; c < cols; c += blockDim.x) sum += expf(in[row * cols + c] - m);
    red[t] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    sum = red[0];

    for (int c = t; c < cols; c += blockDim.x)           // 3) normalise
        out[row * cols + c] = expf(in[row * cols + c] - m) / sum;
}

int main() {
    const int rows = 4, cols = 100;
    float h[400];
    for (int i = 0; i < rows * cols; i++) h[i] = (float)((i * 7) % 13) - 6.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));
    cudaMemcpy(d_in, h, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    softmaxRows<<<rows, 64>>>(d_in, d_out, cols);
    cudaMemcpy(h, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; r++) {
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += h[r * cols + c];
        printf("row %d probabilities sum to %f\n", r, s);
    }
    return 0;
}
