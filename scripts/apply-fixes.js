const fs = require('fs');
const path = require('path');

// 1. Fix demo/race-detector.js
const RACE_PATH = path.join(__dirname, '..', 'demo', 'race-detector.js');
let raceCode = fs.readFileSync(RACE_PATH, 'utf8');

raceCode = raceCode.replace(
  'const linearIdx = ctx.linearIdx !== undefined ? ctx.linearIdx : (ctx.threadIdx.x || 0);',
  'const linearIdx = (ctx && ctx.linearIdx !== undefined) ? ctx.linearIdx : ((ctx && ctx.threadIdx && ctx.threadIdx.x) || 0);'
);
raceCode = raceCode.replace(
  'const warpId = ctx.warpId !== undefined ? ctx.warpId : Math.floor(linearIdx / 32);',
  'const warpId = (ctx && ctx.warpId !== undefined) ? ctx.warpId : Math.floor(linearIdx / 32);'
);
raceCode = raceCode.replace(
  'const threadIdx = { x: ctx.threadIdx.x || 0, y: ctx.threadIdx.y || 0, z: ctx.threadIdx.z || 0 };',
  'const threadIdx = (ctx && ctx.threadIdx) ? { x: ctx.threadIdx.x || 0, y: ctx.threadIdx.y || 0, z: ctx.threadIdx.z || 0 } : { x: 0, y: 0, z: 0 };'
);
fs.writeFileSync(RACE_PATH, raceCode, 'utf8');
console.log('Fixed race-detector.js');

// 2. Fix demo/cuda-runtime.js
const RUNTIME_PATH = path.join(__dirname, '..', 'demo', 'cuda-runtime.js');
let runtimeCode = fs.readFileSync(RUNTIME_PATH, 'utf8');

// Safe module require
const safeModuleRequire = `
  function safeRequire(rel) {
    if (typeof require !== 'undefined') {
      try {
        const path = require('path');
        return require(path.join(__dirname, rel));
      } catch (e) {
        try { return require(rel); } catch (e2) { return null; }
      }
    }
    return null;
  }
  const FP16 = safeRequire('fp16-bf16.js') || (typeof window !== 'undefined' ? window.GpuboxFP16 : (typeof globalThis !== 'undefined' ? globalThis.GpuboxFP16 : {}));
  const RaceDetectorModule = safeRequire('race-detector.js') || (typeof window !== 'undefined' ? window.GpuboxRaceDetector : (typeof globalThis !== 'undefined' ? globalThis.GpuboxRaceDetector : {}));
  const SparseMemModule = safeRequire('sparse-memory.js') || (typeof window !== 'undefined' ? window.GpuboxSparseMemory : (typeof globalThis !== 'undefined' ? globalThis.GpuboxSparseMemory : {}));
  const TopologyModule = safeRequire('topology-model.js') || (typeof window !== 'undefined' ? window.GpuboxTopology : (typeof globalThis !== 'undefined' ? globalThis.GpuboxTopology : {}));
  const DesyncModule = safeRequire('desync-detector.js') || (typeof window !== 'undefined' ? window.GpuboxDesyncDetector : (typeof globalThis !== 'undefined' ? globalThis.GpuboxDesyncDetector : {}));
`;

runtimeCode = runtimeCode.replace(
  /const FP16 =[\s\S]*?const DesyncModule =[\s\S]*?;/,
  safeModuleRequire.trim()
);

// Add ctx.threadIdx in buildThreadContext
runtimeCode = runtimeCode.replace(
  'const buildThreadContext = (threadIdx) => {',
  `const buildThreadContext = (threadIdx) => {
              const currentThreadIdx = { x: threadIdx.x || 0, y: threadIdx.y || 0, z: threadIdx.z || 0 };`
);
runtimeCode = runtimeCode.replace(
  'return {',
  'return {\n                threadIdx: currentThreadIdx,'
);

// Remove advanceBarrier from inside __syncthreads
runtimeCode = runtimeCode.replace(
  'counters.syncthreadsCount++;\n                if (sharedMem && sharedMem._raceDetector) sharedMem._raceDetector.advanceBarrier();',
  'counters.syncthreadsCount++;'
);

// Add advanceBarrier when all living threads are blocked at the barrier
runtimeCode = runtimeCode.replace(
  'for (const th of living) th.blocked = false;',
  `if (sharedMem && sharedMem._raceDetector) {
                    sharedMem._raceDetector.advanceBarrier();
                  }
                  for (const th of living) th.blocked = false;`
);

// Update setActiveContext during generator step
runtimeCode = runtimeCode.replace(
  'counters.setActiveContext(th.ctx.warpId, th.ctx.linearIdx);',
  `counters.setActiveContext(th.ctx.warpId, th.ctx.linearIdx);
                      if (sharedMem && sharedMem.setActiveContext) sharedMem.setActiveContext(th.ctx);`
);

// Fix double free check in cudaFree
runtimeCode = runtimeCode.replace(
  'cudaFree(ptr) {\n      if (ptr && ptr._allocId !== undefined) {',
  `cudaFree(ptr) {\n      if (ptr && ptr._allocId !== undefined) {\n        if (ptr._isFreed) {\n          throw new Error('cudaErrorInvalidValue: double free detected in cudaFree');\n        }`
);
runtimeCode = runtimeCode.replace(
  'cudaFree(ptr) {\n      if (this._multiGPU) {',
  `cudaFree(ptr) {\n      if (ptr && ptr._isFreed) {\n        throw new Error('cudaErrorInvalidValue: double free detected');\n      }\n      if (this._multiGPU) {`
);

fs.writeFileSync(RUNTIME_PATH, runtimeCode, 'utf8');
console.log('Fixed cuda-runtime.js');
