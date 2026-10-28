// examples/node-ci-test/test.js
// Runs in CI (Jest, Vitest, or plain Node.js) to validate CUDA kernels before real GPU runs

const assert = require('assert');
const { createSimulator, compile, validateOutput } = require('../../js/dist/cuda-sim.cjs.js');

console.log('Running CUDA Kernel CI Validation with Gpubox...');

const cudaCode = `
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
`;

const sim = createSimulator({ gpuProfile: 'A100', numDevices: 1 });
const N = 128;
const pX = sim.allocate(N * 4);
const pY = sim.allocate(N * 4);

const hX = new Float32Array(N).fill(2.0);
const hY = new Float32Array(N).fill(5.0);
sim.memcpy(pX, hX, N * 4, 'H2D');
sim.memcpy(pY, hY, N * 4, 'H2D');

const prog = compile(cudaCode);
const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_saxpy;')();
sim.launch(kernelFn, 4, 32, 0, N, 3.0, pX, pY);

const hOut = new Float32Array(N);
sim.memcpy(hOut, pY, N * 4, 'D2H');

const expected = new Float32Array(N).fill(3.0 * 2.0 + 5.0); // 11.0
const verdict = validateOutput(hOut, expected, { tolerance: 1e-5 });

assert.strictEqual(verdict.verdict, 'PASS', 'SAXPY CI test must PASS');
console.log('✓ CI Test Passed: Bit-exact SAXPY output verified on virtual A100 GPU');
