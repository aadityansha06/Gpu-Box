const fs = require('fs');
let code = `
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

// Let's run transpiler and see what happens at each regex
const transpilerSrc = fs.readFileSync('demo/cuda-transpiler.js', 'utf8');

// Let's inspect where pointer arrays and assignments happen in cuda-transpiler.js
console.log('Testing step by step...');
