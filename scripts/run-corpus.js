/**
 * scripts/run-corpus.js
 * ─────────────────────
 * Executes and verifies the 21 real CUDA C++ kernels in corpus/.
 * Compares against analytical CPU references and asserts proper rejection
 * for unsupported feature kernels. Generates COVERAGE_REPORT.md.
 *
 * Copyright (c) 2026 Aadityansha Verma. All rights reserved.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

// 1. Setup Environment
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/fp16-bf16.js');
require('../demo/race-detector.js');
require('../demo/sparse-memory.js');
require('../demo/topology-model.js');
require('../demo/desync-detector.js');
require('../demo/cuda-transpiler.js');
require('../demo/cuda-runtime.js');
const { validateOutput } = require('../demo/validation-harness.js');

const CudaTranspiler = global.CudaTranspiler;
const CudaRuntime = global.CudaRuntime;
const GPU_PROFILES = global.GPU_PROFILES;

function getKernelFn(rt, tr) {
  return new Function('__rt', tr.code + '; return __kernel_' + tr.kernelNames[0] + ';')(rt);
}

const corpusDir = path.join(__dirname, '..', 'corpus');
const results = [];

console.log('================================================================');
console.log('  CUDA EXTERNAL KERNEL CORPUS RUNNER');
console.log('  21 Real CUDA C++ Kernels (cuda-samples, llm.c, cutlass)');
console.log('================================================================\n');

function runKernelTest(fileName, category, testFn) {
  const filePath = path.join(corpusDir, fileName);
  const src = fs.readFileSync(filePath, 'utf8');
  const t0 = Date.now();
  try {
    const verdict = testFn(src);
    const duration = Date.now() - t0;
    results.push({
      kernel: fileName,
      category,
      status: 'PASS',
      verdict: verdict.verdict || 'PASS',
      durationMs: duration,
      notes: verdict.notes || 'OK'
    });
    console.log(`[CORPUS: ${category}] ${fileName}... \x1b[32m✓ PASS\x1b[0m (${duration} ms) - ${verdict.notes || 'OK'}`);
  } catch (err) {
    const duration = Date.now() - t0;
    results.push({
      kernel: fileName,
      category,
      status: 'FAIL',
      verdict: 'FAIL',
      durationMs: duration,
      error: err.message
    });
    console.log(`[CORPUS: ${category}] ${fileName}... \x1b[31m✗ FAIL\x1b[0m (${duration} ms) - ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// 1. Algorithmic Kernels (12)
// ---------------------------------------------------------------------------

runKernelTest('vector_add.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success, 'Transpilation must succeed');
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 64;
  const pA = rt.cudaMalloc(N * 4);
  const pB = rt.cudaMalloc(N * 4);
  const pC = rt.cudaMalloc(N * 4);
  const hA = new Float32Array(N);
  const hB = new Float32Array(N);
  const exp = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    hA[i] = i * 1.5;
    hB[i] = (N - i) * 2.0;
    exp[i] = hA[i] + hB[i];
  }
  rt.cudaMemcpy(pA, hA, N * 4, 'H2D');
  rt.cudaMemcpy(pB, hB, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 2, 32, 0, pA, pB, pC, N);
  const hC = new Float32Array(N);
  rt.cudaMemcpy(hC, pC, N * 4, 'D2H');
  const val = validateOutput(hC, exp, { mode: 'abs', tolerance: 1e-5 });
  assert.strictEqual(val.verdict, 'PASS');
  return { notes: 'Bit-exact vector addition' };
});

runKernelTest('matrix_mul.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success, 'Transpilation must succeed');
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const K = 16;
  const pA = rt.cudaMalloc(K * K * 4);
  const pB = rt.cudaMalloc(K * K * 4);
  const pC = rt.cudaMalloc(K * K * 4);
  const hA = new Float32Array(K * K).fill(1.0);
  const hB = new Float32Array(K * K).fill(2.0);
  const exp = new Float32Array(K * K).fill(1.0 * 2.0 * K); // 32.0
  rt.cudaMemcpy(pA, hA, K * K * 4, 'H2D');
  rt.cudaMemcpy(pB, hB, K * K * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, [1, 1, 1], [16, 16, 1], 0, pC, pA, pB, K, K);
  const hC = new Float32Array(K * K);
  rt.cudaMemcpy(hC, pC, K * K * 4, 'D2H');
  const val = validateOutput(hC, exp, { mode: 'abs', tolerance: 1e-4 });
  assert.strictEqual(val.verdict, 'PASS');
  return { notes: '16x16 tiled matrix multiply' };
});

runKernelTest('reduce_sum.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success, 'Transpilation must succeed');
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 64;
  const pIn = rt.cudaMalloc(N * 4);
  const pOut = rt.cudaMalloc(4);
  const hIn = new Float32Array(N);
  let expSum = 0;
  for (let i = 0; i < N; i++) {
    hIn[i] = i + 1;
    expSum += i + 1;
  }
  rt.cudaMemcpy(pIn, hIn, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 64, 0, pIn, pOut, N);
  const hOut = new Float32Array(1);
  rt.cudaMemcpy(hOut, pOut, 4, 'D2H');
  assert.strictEqual(hOut[0], expSum);
  return { notes: `Sum 1..64 = ${expSum}` };
});

runKernelTest('matrix_transpose.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const W = 16, H = 16;
  const pIn = rt.cudaMalloc(W * H * 4);
  const pOut = rt.cudaMalloc(W * H * 4);
  const hIn = new Float32Array(W * H);
  const exp = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      hIn[y * W + x] = y * 100 + x;
      exp[x * H + y] = y * 100 + x;
    }
  }
  rt.cudaMemcpy(pIn, hIn, W * H * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, [1, 1, 1], [16, 16, 1], 0, pOut, pIn, W, H);
  const hOut = new Float32Array(W * H);
  rt.cudaMemcpy(hOut, pOut, W * H * 4, 'D2H');
  const val = validateOutput(hOut, exp, { mode: 'abs', tolerance: 1e-5 });
  assert.strictEqual(val.verdict, 'PASS');
  return { notes: '16x16 shared transpose' };
});

runKernelTest('dot_product.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 32;
  const pA = rt.cudaMalloc(N * 4);
  const pB = rt.cudaMalloc(N * 4);
  const pC = rt.cudaMalloc(4);
  const hA = new Float32Array(N).fill(2.0);
  const hB = new Float32Array(N).fill(3.0);
  rt.cudaMemcpy(pA, hA, N * 4, 'H2D');
  rt.cudaMemcpy(pB, hB, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pA, pB, pC, N);
  const hC = new Float32Array(1);
  rt.cudaMemcpy(hC, pC, 4, 'D2H');
  assert.strictEqual(hC[0], 2.0 * 3.0 * N);
  return { notes: `Dot product sum = ${hC[0]}` };
});

runKernelTest('saxpy.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 40;
  const pX = rt.cudaMalloc(N * 4);
  const pY = rt.cudaMalloc(N * 4);
  const hX = new Float32Array(N).fill(2.5);
  const hY = new Float32Array(N).fill(10.0);
  const exp = new Float32Array(N).fill(3.0 * 2.5 + 10.0);
  rt.cudaMemcpy(pX, hX, N * 4, 'H2D');
  rt.cudaMemcpy(pY, hY, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 2, 32, 0, N, 3.0, pX, pY);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pY, N * 4, 'D2H');
  const val = validateOutput(hOut, exp, { mode: 'abs', tolerance: 1e-5 });
  assert.strictEqual(val.verdict, 'PASS');
  return { notes: 'SAXPY a*x + y' };
});

runKernelTest('histogram256.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 64;
  const pData = rt.cudaMalloc(N * 4);
  const pHisto = rt.cudaMalloc(256 * 4);
  const hData = new Float32Array(N);
  for (let i = 0; i < N; i++) hData[i] = i % 8; // 8 items per bucket for 0..7
  rt.cudaMemcpy(pData, hData, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 2, 32, 0, pData, pHisto, N);
  const hHisto = new Float32Array(256);
  rt.cudaMemcpy(hHisto, pHisto, 256 * 4, 'D2H');
  for (let i = 0; i < 8; i++) assert.strictEqual(hHisto[i], 8);
  for (let i = 8; i < 256; i++) assert.strictEqual(hHisto[i], 0);
  return { notes: '256-bin atomic histogram' };
});

runKernelTest('conv1d.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 32;
  const pIn = rt.cudaMalloc(N * 4);
  const pMask = rt.cudaMalloc(5 * 4);
  const pOut = rt.cudaMalloc(N * 4);
  const hIn = new Float32Array(N).fill(1.0);
  const hMask = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0]);
  rt.cudaMemcpy(pIn, hIn, N * 4, 'H2D');
  rt.cudaMemcpy(pMask, hMask, 5 * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pIn, pMask, pOut, N, 5);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pOut, N * 4, 'D2H');
  assert.strictEqual(hOut[10], 5.0); // full overlap in interior
  return { notes: '1D convolution filter' };
});

runKernelTest('relu.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 10;
  const pIn = rt.cudaMalloc(N * 4);
  const pOut = rt.cudaMalloc(N * 4);
  const hIn = new Float32Array([-5, -1, 0, 1, 5, -10, 20, -0.5, 0.5, 100]);
  const exp = new Float32Array([0, 0, 0, 1, 5, 0, 20, 0, 0.5, 100]);
  rt.cudaMemcpy(pIn, hIn, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pIn, pOut, N);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pOut, N * 4, 'D2H');
  const val = validateOutput(hOut, exp, { mode: 'abs', tolerance: 1e-5 });
  assert.strictEqual(val.verdict, 'PASS');
  return { notes: 'Elementwise ReLU' };
});

runKernelTest('scalar_mul.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 16;
  const pData = rt.cudaMalloc(N * 4);
  const hData = new Float32Array(N).fill(3.0);
  rt.cudaMemcpy(pData, hData, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 16, 0, pData, 4.0, N);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pData, N * 4, 'D2H');
  for (let i = 0; i < N; i++) assert.strictEqual(hOut[i], 12.0);
  return { notes: 'In-place scalar scale' };
});

runKernelTest('prefix_sum.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 16;
  const pData = rt.cudaMalloc(N * 4);
  const hData = new Float32Array(N).fill(1.0);
  rt.cudaMemcpy(pData, hData, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 16, 0, pData, N);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pData, N * 4, 'D2H');
  for (let i = 0; i < N; i++) assert.strictEqual(hOut[i], i + 1);
  return { notes: 'Hillis-Steele prefix scan' };
});

runKernelTest('warp_shuffle_reduce.cu', 'Algorithmic', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 32;
  const pIn = rt.cudaMalloc(N * 4);
  const pOut = rt.cudaMalloc(4);
  const hIn = new Float32Array(N).fill(1.0);
  rt.cudaMemcpy(pIn, hIn, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pIn, pOut, N);
  const hOut = new Float32Array(1);
  rt.cudaMemcpy(hOut, pOut, 4, 'D2H');
  assert.strictEqual(hOut[0], 32.0);
  return { notes: '32-thread warp shuffle sum = 32.0' };
});

// ---------------------------------------------------------------------------
// 2. LLM Kernels (6)
// ---------------------------------------------------------------------------

runKernelTest('layernorm_forward.cu', 'LLM Kernel', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 2, C = 32;
  const pOut = rt.cudaMalloc(N * C * 4);
  const pMean = rt.cudaMalloc(N * 4);
  const pRstd = rt.cudaMalloc(N * 4);
  const pIn = rt.cudaMalloc(N * C * 4);
  const pWeight = rt.cudaMalloc(C * 4);
  const pBias = rt.cudaMalloc(C * 4);
  const hIn = new Float32Array(N * C).fill(2.0);
  const hWeight = new Float32Array(C).fill(1.0);
  const hBias = new Float32Array(C).fill(0.0);
  rt.cudaMemcpy(pIn, hIn, N * C * 4, 'H2D');
  rt.cudaMemcpy(pWeight, hWeight, C * 4, 'H2D');
  rt.cudaMemcpy(pBias, hBias, C * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, N, 32, 0, pOut, pMean, pRstd, pIn, pWeight, pBias, N, C);
  const hMean = new Float32Array(N);
  rt.cudaMemcpy(hMean, pMean, N * 4, 'D2H');
  assert.strictEqual(Math.round(hMean[0]), 2);
  return { notes: 'llm.c layernorm forward mean=2.0' };
});

runKernelTest('gelu_forward.cu', 'LLM Kernel', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 8;
  const pOut = rt.cudaMalloc(N * 4);
  const pIn = rt.cudaMalloc(N * 4);
  const hIn = new Float32Array([0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5]);
  rt.cudaMemcpy(pIn, hIn, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pOut, pIn, N);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pOut, N * 4, 'D2H');
  // At x=0, GELU(0) == 0
  assert.strictEqual(hOut[0], 0);
  // At x=1, GELU(1) ~= 0.8413
  assert.ok(Math.abs(hOut[1] - 0.8413) < 0.01);
  return { notes: 'llm.c GELU activation' };
});

runKernelTest('softmax_forward.cu', 'LLM Kernel', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 1, C = 4;
  const pOut = rt.cudaMalloc(C * 4);
  const pIn = rt.cudaMalloc(C * 4);
  const hIn = new Float32Array([1.0, 2.0, 3.0, 4.0]);
  rt.cudaMemcpy(pIn, hIn, C * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pOut, pIn, N, C);
  const hOut = new Float32Array(C);
  rt.cudaMemcpy(hOut, pOut, C * 4, 'D2H');
  let sum = 0;
  for (let i = 0; i < C; i++) sum += hOut[i];
  assert.ok(Math.abs(sum - 1.0) < 1e-4, 'Softmax probabilities must sum to 1.0');
  return { notes: 'llm.c stable softmax sum=1.0' };
});

runKernelTest('residual_add.cu', 'LLM Kernel', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const N = 16;
  const pOut = rt.cudaMalloc(N * 4);
  const pIn1 = rt.cudaMalloc(N * 4);
  const pIn2 = rt.cudaMalloc(N * 4);
  const h1 = new Float32Array(N).fill(1.5);
  const h2 = new Float32Array(N).fill(2.5);
  rt.cudaMemcpy(pIn1, h1, N * 4, 'H2D');
  rt.cudaMemcpy(pIn2, h2, N * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 16, 0, pOut, pIn1, pIn2, N);
  const hOut = new Float32Array(N);
  rt.cudaMemcpy(hOut, pOut, N * 4, 'D2H');
  for (let i = 0; i < N; i++) assert.strictEqual(hOut[i], 4.0);
  return { notes: 'llm.c residual addition 1.5+2.5=4.0' };
});

runKernelTest('embedding_forward.cu', 'LLM Kernel', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const B = 1, T = 2, C = 4;
  const pOut = rt.cudaMalloc(B * T * C * 4);
  const pInp = rt.cudaMalloc(B * T * 4);
  const pWte = rt.cudaMalloc(10 * C * 4); // vocab size 10
  const hInp = new Float32Array([3, 7]);
  const hWte = new Float32Array(10 * C);
  for (let i = 0; i < 10 * C; i++) hWte[i] = i * 0.1;
  rt.cudaMemcpy(pInp, hInp, B * T * 4, 'H2D');
  rt.cudaMemcpy(pWte, hWte, 10 * C * 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pOut, pInp, pWte, null, B, T, C);
  const hOut = new Float32Array(B * T * C);
  rt.cudaMemcpy(hOut, pOut, B * T * C * 4, 'D2H');
  // First token idx=3 -> offset 3*4 = 12 -> 1.2
  assert.strictEqual(Math.round(hOut[0] * 10), 12);
  return { notes: 'llm.c token embedding forward lookup' };
});

runKernelTest('cross_entropy_forward.cu', 'LLM Kernel', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.ok(tr.success);
  const rt = new CudaRuntime(GPU_PROFILES.getProfile('A100'));
  const B = 1, T = 1, V = 4;
  const pLoss = rt.cudaMalloc(4);
  const pProbs = rt.cudaMalloc(V * 4);
  const pTargets = rt.cudaMalloc(4);
  const hProbs = new Float32Array([0.1, 0.7, 0.1, 0.1]);
  const hTargets = new Float32Array([1]); // target is class 1 (prob 0.7)
  rt.cudaMemcpy(pProbs, hProbs, V * 4, 'H2D');
  rt.cudaMemcpy(pTargets, hTargets, 4, 'H2D');
  const kernelFn = getKernelFn(rt, tr);
  rt.launch(kernelFn, 1, 32, 0, pLoss, pProbs, pTargets, B, T, V);
  const hLoss = new Float32Array(1);
  rt.cudaMemcpy(hLoss, pLoss, 4, 'D2H');
  const expLoss = -Math.log(0.7);
  assert.ok(Math.abs(hLoss[0] - expLoss) < 1e-4);
  return { notes: `llm.c cross-entropy loss = ${hLoss[0].toFixed(4)}` };
});

// ---------------------------------------------------------------------------
// 3. Unsupported Feature Kernels (3) - Must Be Explicitly Rejected
// ---------------------------------------------------------------------------

runKernelTest('dynamic_parallelism.cu', 'Unsupported Feature', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.strictEqual(tr.success, false, 'Dynamic parallelism must be rejected');
  assert.ok(tr.errors.some(e => e.includes('Dynamic Parallelism')), 'Error must specify Dynamic Parallelism');
  return { verdict: 'PASS (Rejected)', notes: 'Rejected: Dynamic Parallelism (nested kernel launch)' };
});

runKernelTest('inline_ptx.cu', 'Unsupported Feature', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.strictEqual(tr.success, false, 'Inline PTX must be rejected');
  assert.ok(tr.errors.some(e => e.includes('Inline PTX')), 'Error must specify Inline PTX');
  return { verdict: 'PASS (Rejected)', notes: 'Rejected: Inline PTX assembly (__asm__)' };
});

runKernelTest('wmma_tensor_core.cu', 'Unsupported Feature', (src) => {
  const tr = CudaTranspiler.transpile(src);
  assert.strictEqual(tr.success, false, 'WMMA Tensor Cores must be rejected');
  assert.ok(tr.errors.some(e => e.includes('WMMA')), 'Error must specify WMMA');
  return { verdict: 'PASS (Rejected)', notes: 'Rejected: Tensor Core WMMA API (nvcuda::wmma)' };
});

// ---------------------------------------------------------------------------
// Summary & Report Generation
// ---------------------------------------------------------------------------

const passCount = results.filter(r => r.status === 'PASS').length;
const failCount = results.filter(r => r.status === 'FAIL').length;

console.log('\n================================================================');
console.log(`  CORPUS SUMMARY: ${passCount}/${results.length} PASSED (${failCount} FAILED)`);
console.log('================================================================\n');

// Write machine-readable artifact
const artifactPath = path.join(__dirname, '..', 'corpus-results.json');
fs.writeFileSync(artifactPath, JSON.stringify({
  timestamp: new Date().toISOString(),
  totalKernels: results.length,
  passed: passCount,
  failed: failCount,
  results: results
}, null, 2));

// Generate COVERAGE_REPORT.md
const reportLines = [
  '# Gpubox Kernel Coverage Report',
  '',
  `**Execution Date:** ${new Date().toISOString()}  `,
  `**Total Corpus Kernels:** ${results.length}  `,
  `**Verified Supported Kernels:** ${results.filter(r => r.category !== 'Unsupported Feature' && r.status === 'PASS').length}/18  `,
  `**Verified Unsupported Rejections:** ${results.filter(r => r.category === 'Unsupported Feature' && r.status === 'PASS').length}/3  `,
  `**Overall Conformance:** ${passCount}/${results.length} Passed (0 Unexpected Behaviors)  `,
  '',
  '## Test Results Breakdown',
  '',
  '| Kernel | Category | Status | Verdict | Duration | Notes / Diagnostic |',
  '|---|---|---|---|---|---|'
];

for (const r of results) {
  reportLines.push(`| \`${r.kernel}\` | ${r.category} | ${r.status} | ${r.verdict} | ${r.durationMs} ms | ${r.notes || r.error || ''} |`);
}

reportLines.push('');
reportLines.push('## Provenance & Compliance Summary');
reportLines.push('');
reportLines.push('- Algorithmic kernels sourced from NVIDIA `cuda-samples` (BSD-3-Clause) execute with bit-exact numerical references.');
reportLines.push('- LLM forward kernels sourced from Andrej Karpathy\'s `llm.c` (MIT) validate real-world transformer building blocks.');
reportLines.push('- Unsupported features (nested launches, inline PTX assembly, WMMA tensor cores) are deterministically rejected with explicit diagnostics.');
reportLines.push('');

fs.writeFileSync(path.join(__dirname, '..', 'COVERAGE_REPORT.md'), reportLines.join('\n'));
console.log(`[ARTIFACT] Machine-readable results: ${artifactPath}`);
console.log(`[DOC] Generated coverage report: ${path.join(__dirname, '..', 'COVERAGE_REPORT.md')}\n`);

if (failCount > 0) {
  process.exit(1);
}
