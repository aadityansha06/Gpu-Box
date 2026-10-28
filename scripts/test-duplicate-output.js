const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int numGpus = 2;
    cudaGetDeviceCount(&numGpus);
    cudaSetDevice(0);
    printf("GPU 0 AllReduce MAX: 8 12 15 20\\n");
    cudaSetDevice(1);
    printf("GPU 1 AllReduce MAX: 8 12 15 20\\n");
    printf("AllReduce MAX validation: PASS\\n");
    return 0;
}
`;

const trans = window.CudaTranspiler.transpile(code);
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
rt.initMultiGPU(2);
const runner = new Function('__rt', trans.code);
runner(rt);

const diag = rt.getDiagnostics();
console.log('--- Diagnostics Count ---', diag.length);
diag.forEach((d, i) => {
  console.log(`[${i}] ${d.message}`);
});

assert.strictEqual(diag.length, 3, 'Should have exactly 3 diagnostics, not 6');
assert.strictEqual(diag[0].message, 'GPU 0 AllReduce MAX: 8 12 15 20\n');
assert.strictEqual(diag[1].message, 'GPU 1 AllReduce MAX: 8 12 15 20\n');
assert.strictEqual(diag[2].message, 'AllReduce MAX validation: PASS\n');
console.log('✓ No duplicate output test passed!');
