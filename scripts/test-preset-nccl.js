const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>

int main() {
    int numGpus = 4;
    int count = 64;
    size_t bytes = count * sizeof(float);

    printf("Initializing Multi-GPU cluster with %d devices...\\n", numGpus);

    ncclComm_t comms[4];
    ncclCommInitAll(comms, numGpus);

    float *d_send[4];
    float *d_recv[4];
    float h_send[64];
    for (int i = 0; i < count; i++) h_send[i] = 1.0f;

    // Allocate memory on each virtual GPU node and copy input shard
    for (int r = 0; r < numGpus; r++) {
        cudaSetDevice(r);
        cudaMalloc(&d_send[r], bytes);
        cudaMalloc(&d_recv[r], bytes);
        cudaMemcpy(d_send[r], h_send, bytes, cudaMemcpyHostToDevice);
    }

    // Execute Native Ring AllReduce across all ranks
    printf("Invoking ncclAllReduce(sum) across %d GPUs...\\n", numGpus);
    for (int r = 0; r < numGpus; r++) {
        cudaSetDevice(r);
        ncclAllReduce(d_send[r], d_recv[r], count, ncclFloat, ncclSum, comms[r], 0);
    }

    // Verify reduction output on each device
    for (int r = 0; r < numGpus; r++) {
        cudaSetDevice(r);
        float h_recv[64];
        cudaMemcpy(h_recv, d_recv[r], bytes, cudaMemcpyDeviceToHost);
        printf("GPU %d reduced element 0: %f (Expected: %f)\\n", r, h_recv[0], (float)numGpus);
    }

    return 0;
}
`;

const trans = window.CudaTranspiler.transpile(code);
assert.strictEqual(trans.success, true);
const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
new Function('__rt', trans.code)(rt);
console.log('✓ Preset test passed!');
