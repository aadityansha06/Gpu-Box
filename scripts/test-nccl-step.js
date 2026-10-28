const fs = require('fs');
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
    }
    return 0;
}
`;

const res = window.CudaTranspiler.transpile(code);
console.log('Result:\n', res.code);
