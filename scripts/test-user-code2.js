global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#define N 16
#define THREADS 128

// Phase 1: Compute the L1 norm using float4 loads
// and warp-shuffle reduction.
__global__ void computeL1Norm(const float *x, float *norm)
{
    __shared__ float warpSums[32];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    float sum = 0.0f;

    // Each thread processes float4 elements.
    for (int i = tid; i < N / 4; i += blockDim.x)
    {
        float4 v = reinterpret_cast<const float4 *>(x)[i];

        sum += fabsf(v.x) + fabsf(v.y)
             + fabsf(v.z) + fabsf(v.w);
    }

    // Reduce within each warp using shuffle instructions.
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // One thread per warp writes its partial sum.
    if (lane == 0)
        warpSums[warp] = sum;

    __syncthreads();

    // First warp reduces the warp-level partial sums.
    if (warp == 0)
    {
        sum = (lane < (blockDim.x + 31) / 32)
                  ? warpSums[lane]
                  : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(
                0xffffffff, sum, offset);
        }

        if (lane == 0)
            *norm = sum;
    }
}

// Phase 2: Normalize using the computed norm.
__global__ void normalizeL1(
    const float *x,
    float *y,
    const float *norm)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized float4 loads and stores.
    for (int i = tid; i < N / 4;
         i += gridDim.x * blockDim.x)
    {
        float4 v = reinterpret_cast<const float4 *>(x)[i];

        float denominator = *norm;

        v.x /= denominator;
        v.y /= denominator;
        v.z /= denominator;
        v.w /= denominator;

        reinterpret_cast<float4 *>(y)[i] = v;
    }
}

int main()
{
    // Hardcoded input
    float h_x[N] = {
         1, -2,  3, -4,
         5, -6,  7, -8,
         9, -10, 11, -12,
        13, -14, 15, -16
    };

    float h_y[N];
    float h_norm;

    float *d_x, *d_y, *d_norm;

    size_t bytes = N * sizeof(float);

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_norm, sizeof(float));

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    // Phase 1: One block computes the norm.
    computeL1Norm<<<1, THREADS>>>(d_x, d_norm);

    // Phase 2: Normalize all elements.
    int blocks = ((N / 4) + THREADS - 1) / THREADS;
    normalizeL1<<<blocks, THREADS>>>(d_x, d_y, d_norm);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_norm, d_norm, sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "L1 Norm = " << h_norm << "\\n\\n";
    std::cout << "Normalized vector:\\n";

    for (int i = 0; i < N; i++)
    {
        std::cout << h_y[i] << " ";
        if ((i + 1) % 4 == 0)
            std::cout << "\\n";
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_norm);

    return 0;
}
`;

const res = window.CudaTranspiler.transpile(code);
console.log('Success:', res.success);
console.log('Transpiled Code:\n', res.code);
try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const fn = new Function('__rt', res.code);
  fn(rt);
  console.log('Execution completed! Diagnostics count:', rt.getDiagnostics().length);
  console.log('--- ERRORS ---');
  for (const d of rt.getDiagnostics()) {
    if (d.type === 'error') console.log(d.message);
  }
  console.log('--- END ERRORS ---');
  console.log('--- OUTPUT ---');
  for (const d of rt.getDiagnostics()) {
    process.stdout.write(d.message);
  }
  console.log('--- END OUTPUT ---');
  console.log('Counters:', rt.getCounters().toJSON());
} catch (e) {
  console.log('Execution error:', e.message, e.stack);
}
