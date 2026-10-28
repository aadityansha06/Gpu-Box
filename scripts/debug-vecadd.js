global.window = global;
require('../demo/cuda-runtime.js');
const CudaTranspiler = require('../demo/cuda-transpiler.js');
const prof = require('../demo/gpu-profiles.js').getProfile('A100');

const n = 1;
const code = `
    #include <cuda_runtime.h>
    __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    int main() {
        int n = ${n};
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n * 4);
        cudaMalloc(&d_b, n * 4);
        cudaMalloc(&d_c, n * 4);
        float h_a[1]; float h_b[1];
        h_a[0] = 1.5f; h_b[0] = 2.5f;
        cudaMemcpy(d_a, h_a, n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, n * 4, cudaMemcpyHostToDevice);
        vecAdd<<<1, 32>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        float h_c[1];
        cudaMemcpy(h_c, d_c, n * 4, cudaMemcpyDeviceToHost);
        printf("c[0] = %f\\n", h_c[0]);
        return 0;
    }
`;

const trans = CudaTranspiler.transpile(code);
const rt = new window.CudaRuntime(prof);
try {
  new Function('__rt', trans.code)(rt);
  console.log('Diagnostics:', rt.getDiagnostics());
  console.log('Counters:', rt.getCounters().toJSON());
} catch (e) {
  console.error('Error:', e);
}
