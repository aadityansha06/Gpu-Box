// Matrix transpose through shared memory. Run it, then open the "32-Bank Memory" tab:
// the [32][32] tile makes every column read hit the same bank (32-way conflict).
// Change the tile to [32][33] (one padding column) and run again: conflicts drop to zero.
#include <cstdio>

__global__ void transpose(const float* in, float* out, int n) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int n = 64;
    float h_in[4096], h_out[4096];
    for (int i = 0; i < n * n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * n * sizeof(float));
    cudaMalloc(&d_out, n * n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(n / 32, n / 32);
    transpose<<<grid, block>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            if (h_out[c * n + r] != h_in[r * n + c]) bad++;
    printf("transpose verified: %d mismatches\n", bad);
    return 0;
}
