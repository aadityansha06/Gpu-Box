const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cuda_runtime.h>
#include <nccl.h>

int main() {
    int nGpus = 2;
    int count = 4;
    ncclComm_t comms[2];
    ncclCommInitAll(comms, 2);

    float *d_send[2];
    float *d_recv[2];
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_send[i], count * sizeof(float));
        cudaMalloc(&d_recv[i], count * sizeof(float));
        float h_data[4];
        for (int k = 0; k < 4; k++) h_data[k] = (float)(i + 1); // GPU 0: 1.0, GPU 1: 2.0
        cudaMemcpy(d_send[i], h_data, count * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Call ncclAllReduce per rank:
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        ncclAllReduce(d_send[i], d_recv[i], count, ncclFloat, ncclSum, comms[i], 0);
    }

    // Copy back and verify
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        float h_res[4];
        cudaMemcpy(h_res, d_recv[i], count * sizeof(float), cudaMemcpyDeviceToHost);
        for (int k = 0; k < 4; k++) {
            if (h_res[k] != 3.0f) {
                printf("Error on GPU %d at %d: got %f expected 3.0\\n", i, k, h_res[k]);
            }
        }
        printf("GPU %d reduced element 0: %f\\n", i, h_res[0]);
    }
    return 0;
}
`;

const trans = window.CudaTranspiler.transpile(code);
assert.strictEqual(trans.success, true);

const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
const runner = new Function('__rt', trans.code);
runner(rt);

const diags = rt.getDiagnostics();
console.log('Diagnostics:', diags.map(d => d.message));

const c = rt.getCounters().toJSON();
console.log('Aggregated Counters:');
console.log('  ncclOps:', c.ncclOps);
console.log('  crossGpuBytes:', c.crossGpuBytes);
console.log('  perDevice:', c.perDevice.map(d => ({ deviceId: d.deviceId, ncclOps: d.ncclOps, crossGpuBytes: d.crossGpuBytes })));

assert.strictEqual(c.ncclOps, 2, 'Should be 2 ncclOps');
assert.strictEqual(c.crossGpuBytes, 32, 'crossGpuBytes should be 32 (4 floats * 4 bytes * 1 peer * 2 gpus)');
console.log('ALL CHECKS PASSED!');
