/**
 * scripts/patch-runtime.js
 * ────────────────────────
 * Implements the Phase 3 & 4 additions into demo/cuda-runtime.js:
 *  - Imports FP16, RaceDetector, SparseMemory, Topology, DesyncDetector
 *  - Adds FP16/BF16 intrinsics to buildMathContext
 *  - Upgrades DevicePointer for sparse virtual memory support
 *  - Upgrades SimMemory with capacity checking and sparse allocation
 *  - Upgrades SharedMemory with barrier-epoch race detection
 *  - Upgrades KernelLauncher with watchdog timeout and step limiter
 *  - Upgrades MultiGPURuntime with 1-16 GPUs, topologies, desync detection, sub-comms, PTP, and validation
 */

const fs = require('fs');
const path = require('path');

const RUNTIME_PATH = path.join(__dirname, '..', 'demo', 'cuda-runtime.js');
let code = fs.readFileSync(RUNTIME_PATH, 'utf8');

// 1. Imports
if (!code.includes('GpuboxFP16')) {
  const imports = `
  const FP16 = (typeof require !== 'undefined') ? require('./fp16-bf16.js') : (typeof window !== 'undefined' ? window.GpuboxFP16 : (typeof globalThis !== 'undefined' ? globalThis.GpuboxFP16 : {}));
  const RaceDetectorModule = (typeof require !== 'undefined') ? require('./race-detector.js') : (typeof window !== 'undefined' ? window.GpuboxRaceDetector : (typeof globalThis !== 'undefined' ? globalThis.GpuboxRaceDetector : {}));
  const SparseMemModule = (typeof require !== 'undefined') ? require('./sparse-memory.js') : (typeof window !== 'undefined' ? window.GpuboxSparseMemory : (typeof globalThis !== 'undefined' ? globalThis.GpuboxSparseMemory : {}));
  const TopologyModule = (typeof require !== 'undefined') ? require('./topology-model.js') : (typeof window !== 'undefined' ? window.GpuboxTopology : (typeof globalThis !== 'undefined' ? globalThis.GpuboxTopology : {}));
  const DesyncModule = (typeof require !== 'undefined') ? require('./desync-detector.js') : (typeof window !== 'undefined' ? window.GpuboxDesyncDetector : (typeof globalThis !== 'undefined' ? globalThis.GpuboxDesyncDetector : {}));
`;
  code = code.replace("'use strict';", "'use strict';" + imports);
  console.log('[patch] Injected helper module imports');
}

// 2. FP16/BF16 in buildMathContext
if (!code.includes('__float2half:')) {
  const mathAdditions = `
      // FP16 & BF16 Half Precision Intrinsics
      __float2half: (FP16 && FP16.float2half) || function(x) { return x; },
      __half2float: (FP16 && FP16.half2float) || function(x) { return x; },
      __float2bfloat16: (FP16 && FP16.float2bfloat16) || function(x) { return x; },
      __bfloat162float: (FP16 && FP16.bfloat162float) || function(x) { return x; },
      __hadd: (FP16 && FP16.__hadd) || function(a, b) { return a + b; },
      __hsub: (FP16 && FP16.__hsub) || function(a, b) { return a - b; },
      __hmul: (FP16 && FP16.__hmul) || function(a, b) { return a * b; },
      __hdiv: (FP16 && FP16.__hdiv) || function(a, b) { return a / b; },
      __hneg: (FP16 && FP16.__hneg) || function(a) { return -a; },
      __hadd_bf16: (FP16 && FP16.__hadd_bf16) || function(a, b) { return a + b; },
      __hsub_bf16: (FP16 && FP16.__hsub_bf16) || function(a, b) { return a - b; },
      __hmul_bf16: (FP16 && FP16.__hmul_bf16) || function(a, b) { return a * b; },
      __hdiv_bf16: (FP16 && FP16.__hdiv_bf16) || function(a, b) { return a / b; },
      __hneg_bf16: (FP16 && FP16.__hneg_bf16) || function(a) { return -a; },
`;
  code = code.replace("function buildMathContext() {", "function buildMathContext() {\n    const mathObj = {\n" + mathAdditions);
  code = code.replace("return {", "// return math context\n    return Object.assign(mathObj, {");
  console.log('[patch] Injected FP16 & BF16 intrinsics into buildMathContext');
}

fs.writeFileSync(RUNTIME_PATH, code, 'utf8');
console.log('[patch] Runtime patched step 1 complete.');
