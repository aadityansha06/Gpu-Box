const fs = require('fs');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const code = `
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

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        ncclAllReduce(d_send[i], d_recv[i], count, ncclFloat, ncclSum, comms[i], 0);
    }
`;

console.log('Result:', window.CudaTranspiler.transpileHostCode(code, []));
