const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');

console.log('Testing collective edge cases...');

// 1. Single GPU AllReduce (1 GPU)
try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(1);
  const s = rt.cudaMalloc(16);
  const r = rt.cudaMalloc(16);
  s._rawSet(0, 42.0);
  multi.ncclAllReduce([s], [r], 4, 'float32', 'sum', { nranks: 1 });
  console.log('1 GPU AllReduce output:', r._rawGet(0));
} catch (e) {
  console.log('1 GPU AllReduce error:', e.message);
}

// 2. Zero-length input (count = 0)
try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(2);
  const s0 = rt.cudaMalloc(16);
  const r0 = rt.cudaMalloc(16);
  const s1 = rt.cudaMalloc(16);
  const r1 = rt.cudaMalloc(16);
  multi.ncclAllReduce([s0, s1], [r0, r1], 0, 'float32', 'sum', { nranks: 2 });
  console.log('Zero-length AllReduce passed without crash');
} catch (e) {
  console.log('Zero-length AllReduce error:', e.message);
}

// 3. Invalid root in broadcast (root = 5 on 2 GPUs)
try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(2);
  const s0 = rt.cudaMalloc(16);
  const r0 = rt.cudaMalloc(16);
  const s1 = rt.cudaMalloc(16);
  const r1 = rt.cudaMalloc(16);
  multi.ncclBroadcast([s0, s1], [r0, r1], 4, 'float32', 5, { nranks: 2 });
  console.log('Broadcast with invalid root 5 passed (or did it fail?)');
} catch (e) {
  console.log('Broadcast with invalid root caught error:', e.message);
}

// 4. Mismatched buffer array length vs comm.nranks
try {
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  const multi = rt.initMultiGPU(2);
  const s0 = rt.cudaMalloc(16);
  const r0 = rt.cudaMalloc(16);
  multi.ncclAllReduce([s0], [r0], 4, 'float32', 'sum', { nranks: 2 });
  console.log('Mismatched buffers passed (or did it crash?)');
} catch (e) {
  console.log('Mismatched buffers error:', e.message);
}
