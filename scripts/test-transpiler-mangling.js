const fs = require('fs');

const src = fs.readFileSync('demo/cuda-transpiler.js', 'utf8');

// Let's create an instrumented version of cuda-transpiler.js in scratch/ or run it
const testInput = `
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
`;

// Let's see what happens to `ncclCommInitAll`
// Line 1465 in cuda-transpiler.js:
// code = code.replace(/ncclCommInitAll\s*\(\s*&?\s*([\w.\[\]]+)\s*,\s*([^,]+)[^)]*\)/g, ...)
// Look at what regex matched in testInput!
console.log('ncclCommInitAll match:');
const m = testInput.match(/ncclCommInitAll\s*\(\s*&?\s*([\w.\[\]]+)\s*,\s*([^,]+)[^)]*\)/);
console.log(m);
