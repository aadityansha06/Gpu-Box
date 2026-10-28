global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void vectorAdd(const float *A, const float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    A = new float[N];
    B = new float[N];
    C = new float[N];

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate GPU memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy input vectors to GPU
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel: 1 block with 1024 threads
    vectorAdd<<<1, 1024>>>(d_A, d_B, d_C);

    // Wait for GPU execution
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10; i++)
    {
        std::cout << A[i] << " + " << B[i]
                  << " = " << C[i] << '\\n';
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}`;

const res = window.CudaTranspiler.transpile(code);
console.log('Success:', res.success);
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
const fn = new Function('__rt', res.code);
fn(rt);
console.log('Diagnostics count:', rt.getDiagnostics().length);
console.log('Output from execution:');
for (const d of rt.getDiagnostics()) {
  process.stdout.write(d.message);
}
console.log('Counters:', rt.getCounters().toJSON());
