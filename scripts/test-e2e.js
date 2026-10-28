global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const cudaCode = `
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 64;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    float h_a[64];
    float h_b[64];
    float h_c[64];
    for (int i = 0; i < 64; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    vec_add<<<2, 32>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result[0] = %f, Result[63] = %f\\n", h_c[0], h_c[63]);
    return 0;
}
`;

const result = window.CudaTranspiler.transpile(cudaCode);
console.log('Transpilation success:', result.success);
if (!result.success) {
    console.error('Errors:', result.errors);
} else {
    console.log('Transpiled code:\n----------------\n' + result.code + '\n----------------');
    const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
    const fn = new Function('__rt', result.code);
    fn(rt);
    console.log('Diagnostics count:', rt.getDiagnostics().length);
    console.log('Diagnostics:', rt.getDiagnostics());
    console.log('Counters:', rt.getCounters().toJSON());
}
