global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const prodCode = `
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#define TILE_DIM 4

#define CUDA_CHECK(call) \\
do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        printf("CUDA error\\n"); \\
    } \\
} while(0)

struct Point {
    float x;
    float y;
};

template <typename T>
__global__ void testKernel(T* data, int n) {
    extern __shared__ float s_dynamic[];
    __shared__ float s_tile[TILE_DIM][TILE_DIM];

    int tid = threadIdx.x;
    if (tid < n) {
        s_dynamic[tid] = data[tid];
        s_tile[0][tid % TILE_DIM] = data[tid];
    }
    __syncthreads();

    if (tid < n) {
        float val = s_dynamic[tid];
        val = std::max(val, 0.0f);
        data[tid] = val * 2.0f;
    }
}

int main() {
    const int N = 8;
    const int bytes = N * sizeof(float);

    Point pt;
    pt.x = 10.0f;
    pt.y = 20.0f;
    Point* pPtr = &pt;
    printf("Point coords: %.1f, %.1f\\n", pPtr->x, pPtr->y);

    float* h_A = new float[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i - 3);
    }

    float* d_A;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    testKernel<float><<<1, N, N * sizeof(float)>>>(d_A, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel elapsed time: %.3f ms\\n", ms);

    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    printf("Results:\\n");
    for (int i = 0; i < N; i++) {
        printf("%.1f ", h_A[i]);
    }
    printf("\\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    delete[] h_A;

    return 0;
}
`;

const res = window.CudaTranspiler.transpile(prodCode);
console.log('Success:', res.success);
console.log('Transpiled code:\n', res.code);

try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const fn = new Function('__rt', res.code);
  fn(rt);
  console.log('Execution completed! Output:');
  for (const d of rt.getDiagnostics()) {
    process.stdout.write(d.message);
  }
} catch (e) {
  console.log('Execution error:', e.message, e.stack);
}
