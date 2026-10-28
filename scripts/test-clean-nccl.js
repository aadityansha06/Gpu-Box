const fs = require('fs');

let code = `
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

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        ncclAllReduce(d_send[i], d_recv[i], count, ncclFloat, ncclSum, comms[i], 0);
    }
    return 0;
}
`;

// Test simulated clean regex
code = code.replace(/ncclCommInitAll\s*\(\s*&?\s*([\w.\[\]]+)\s*,\s*([^,)]+)(?:,[^)]*)?\)/g, '__rt.ncclCommInitAll($1, $2)');
code = code.replace(/ncclAllReduce\s*\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,)]+)(?:,[^)]*)?\)/g, '__rt.ncclAllReduce($1, $2, $3, $4, $5, $6)');

console.log('Transpiled clean snippet:\n', code);
