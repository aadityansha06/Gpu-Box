// 2D Matrix Multiplication using Shared Memory Tiling
#include <cstdio>

#define TILE_SIZE 16

__global__ void matmulTiled(const float* A, const float* B, float* C, int width) {
    __shared__ float s_a[TILE_SIZE * TILE_SIZE];
    __shared__ float s_b[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < width / TILE_SIZE; ++t) {
        s_a[threadIdx.y * TILE_SIZE + threadIdx.x] = A[row * width + (t * TILE_SIZE + threadIdx.x)];
        s_b[threadIdx.y * TILE_SIZE + threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_a[threadIdx.y * TILE_SIZE + k] * s_b[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 32;
    int size = width * width;
    size_t bytes = size * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    float h_A[1024];
    float h_B[1024];
    float h_C[1024];

    for (int i = 0; i < size; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(width / TILE_SIZE, width / TILE_SIZE);
    matmulTiled<<<grid, block>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Tiled MatMul complete. C[0,0] = %f (Expected: 64.0)\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
