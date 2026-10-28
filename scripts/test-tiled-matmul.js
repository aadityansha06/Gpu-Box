global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <stdio.h>
#include <cuda_runtime.h>

#define N 8
#define TILE 4

// Optimized tiled matrix multiplication
__global__ void tiledMatMul(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Process matrices one tile at a time
    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Load tiles into shared memory
        if (row < N && aCol < N)
            As[threadIdx.y][threadIdx.x] =
                A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait until all tile elements are loaded
        __syncthreads();

        // Multiply tiles and accumulate
        #pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            sum += As[threadIdx.y][k] *
                   Bs[k][threadIdx.x];
        }

        // Ensure all threads finish using the tiles
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main()
{
    const int size = N * N * sizeof(float);

    float h_A[N * N];
    float h_B[N * N];
    float h_C[N * N];

    // Hardcoded deterministic input values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = i * N + j + 1;
            h_B[i * N + j] = i * N + j + 1;
        }
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA execution error: %s\\n",
               cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("First 5 values of C = A * B:\\n");

    for (int i = 0; i < 5; i++)
        printf("%.0f ", h_C[i]);

    printf("\\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
`;

const res = window.CudaTranspiler.transpile(code);
console.log('Success:', res.success);
console.log('Transpiled code:\n', res.code);
try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const fn = new Function('__rt', res.code);
  fn(rt);
  console.log('Execution completed! Diagnostics count:', rt.getDiagnostics().length);
  console.log('--- OUTPUT ---');
  for (const d of rt.getDiagnostics()) {
    process.stdout.write(d.message);
  }
  console.log('--- END OUTPUT ---');
  console.log('--- ERRORS ---');
  for (const d of rt.getDiagnostics()) {
    if (d.type === 'error') console.log(d.message);
  }
  console.log('--- END ERRORS ---');
} catch (e) {
  console.log('Execution error:', e.message, e.stack);
}
