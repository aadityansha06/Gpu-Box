global.window = global;
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');
require('../demo/gpu-profiles.js');

const n = 4;
const code = `
    #include <cuda_runtime.h>
    #include <math.h>
    __global__ void l2Norm(float* data, float* normOut, int n) {
        __shared__ float s[256];
        int tid = threadIdx.x;
        float val = (tid < n) ? data[tid] : 0.0f;
        s[tid] = val * val;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            normOut[0] = sqrtf(s[0]);
        }
        __syncthreads();
        if (tid < n && normOut[0] > 0.0f) {
            data[tid] = data[tid] / normOut[0];
        }
    }
    int main() {
        int n = ${n};
        float *d_data, *d_norm;
        cudaMalloc(&d_data, n * 4);
        cudaMalloc(&d_norm, 4);
        float h_data[${Math.max(n, 1)}];
        for (int i = 0; i < n; i++) h_data[i] = 2.0f;
        cudaMemcpy(d_data, h_data, n * 4, cudaMemcpyHostToDevice);
        l2Norm<<<1, 256>>>(d_data, d_norm, n);
        cudaDeviceSynchronize();
        float h_norm[1];
        cudaMemcpy(h_norm, d_norm, 4, cudaMemcpyDeviceToHost);
        printf("norm: %f\\n", h_norm[0]);
        return 0;
    }
`;

const trans = window.CudaTranspiler.transpile(code);
console.log('Transpiled code:\n', trans.code);
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
try {
  new Function('__rt', trans.code)(rt);
  console.log('Diagnostics:', rt.getDiagnostics());
} catch (e) {
  console.error('Runtime error:', e);
}
