# Gpubox Integration Guide

Gpubox is a headless, client-side CUDA and multi-GPU simulation and pre-validation engine. It allows developers, AI startups, enterprises, and educational platforms to validate CUDA algorithms, test distributed NCCL collectives across 1 to 16 virtual GPUs, and detect logic errors (out-of-bounds, race conditions, collective desync) before spending compute budget on physical GPU infrastructure.

## Installation

```bash
npm install gpubox
```

## Quick Start Environments

### 1. Node.js CommonJS & CI Runner

```javascript
const { createSimulator, compile, validateOutput } = require('gpubox');

// 1. Initialize virtual GPU cluster (1 to 16 GPUs)
const sim = createSimulator({ gpuProfile: 'A100', numDevices: 1 });

// 2. Allocate device memory
const N = 128;
const d_a = sim.allocate(N * 4);
const d_b = sim.allocate(N * 4);
const d_c = sim.allocate(N * 4);

// 3. Copy host memory to virtual GPU
sim.memcpy(d_a, new Float32Array(N).fill(1.5), N * 4, 'H2D');
sim.memcpy(d_b, new Float32Array(N).fill(2.5), N * 4, 'H2D');

// 4. Compile and launch CUDA kernel
const prog = compile(`
  __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) c[i] = a[i] + b[i];
  }
`);

const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_vecAdd;')();
sim.launch(kernelFn, 4, 32, 0, d_a, d_b, d_c, N);

// 5. Read back results and validate against expected CPU reference
const h_c = new Float32Array(N);
sim.memcpy(h_c, d_c, N * 4, 'D2H');
const verdict = validateOutput(h_c, new Float32Array(N).fill(4.0), { tolerance: 1e-5 });
console.log('Simulation Verdict:', verdict.verdict); // PASS
```

### 2. Browser ESM (Vite, Webpack, Next.js)

```javascript
import { createSimulator, compile } from 'gpubox';

const sim = createSimulator({ gpuProfile: 'H100', numDevices: 4 });
// Use identical headless API in modern web frameworks
```

### 3. CDN Script Tag (UMD)

```html
<script src="https://cdn.jsdelivr.net/npm/gpubox/dist/cuda-sim.min.js"></script>
<script>
  const sim = CudaSimLib.createSimulator({ gpuProfile: 'RTX4090', numDevices: 2 });
</script>
```

### 4. Web Worker (Background Thread)

```javascript
import { createWorkerSimulator } from 'gpubox';

const workerSim = createWorkerSimulator({
  limits: { maxSteps: 100000, maxTimeMs: 5000 }
});
```

---

## Core Concepts

- **Virtual Cluster:** Simulated cluster of 1 to 16 GPUs with per-device VRAM, hardware latency profiles, and peer-to-peer memory access.
- **Topologies:** Network models including `nvswitch_mesh`, `pcie_tree`, and `multi_node` measuring inter-GPU bandwidth, latency, and pairwise traffic.
- **Collectives:** Ring and tree models for `ncclAllReduce`, `ncclBroadcast`, `ncclReduceScatter`, `ncclAllGather`, `ncclReduce`, and `ncclAllToAll`.
- **Desync & Deadlock Detection:** Real-time tracking of rank sequence numbers catching missing ranks, count/type mismatches, and collective ordering bugs without hanging.
- **Sparse Virtual VRAM:** Virtual memory allocator supporting up to 80 GiB per device (1.28 TiB cluster total) with sub-256 MB real host heap memory.

---

## Error Handling & SimError

All runtime and compilation errors are typed:

```javascript
const { SimError } = require('gpubox');

try {
  sim.allocate(-100);
} catch (err) {
  if (err instanceof SimError) {
    console.error(`[${err.code}] ${err.message}`);
    console.error(`Documentation: ${err.docsUrl}`);
  }
}
```
