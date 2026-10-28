global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void flashAttention(float* out, float* q, int n) {
    int tid = threadIdx.x;
    float m = -INFINITY;
    if (isinf(m)) {
        out[tid] = 42.0f;
    } else {
        out[tid] = 0.0f;
    }
}

int main() {
    float *d_out, *d_q;
    cudaMalloc(&d_out, 4 * sizeof(float));
    cudaMalloc(&d_q, 4 * sizeof(float));
    flashAttention<<<1, 4>>>(d_out, d_q, 4);
    float h_out[4];
    cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++) printf("%.0f ", h_out[i]);
    printf("\\n");
    cudaFree(d_out);
    cudaFree(d_q);
    return 0;
}
`;

const res = window.CudaTranspiler.transpile(code);
console.log('Transpiled code:\n', res.code);

const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
const fn = new Function('__rt', res.code);
fn(rt);

console.log('Diagnostics count:', rt.getDiagnostics().length);
for (const d of rt.getDiagnostics()) {
  console.log(d.type, ':', d.message);
}
