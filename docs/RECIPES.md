# Gpubox Recipes & Integrator Patterns

Practical, executable patterns for integrating Gpubox into test suites, documentation platforms, cloud runtimes, and distributed AI engineering workflows.

---

## Recipe 1: Assert Kernel Outputs in CI (Jest / Vitest)

```javascript
const assert = require('assert');
const { createSimulator, compile, validateOutput } = require('gpubox');

// Automated test verifying vector addition kernel
const sim = createSimulator({ gpuProfile: 'A100' });
const N = 64;
const pA = sim.allocate(N * 4);
const pB = sim.allocate(N * 4);
const pC = sim.allocate(N * 4);

sim.memcpy(pA, new Float32Array(N).fill(2.0), N * 4, 'H2D');
sim.memcpy(pB, new Float32Array(N).fill(3.0), N * 4, 'H2D');

const prog = compile(`
  __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) c[i] = a[i] + b[i];
  }
`);
const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_vecAdd;')();
sim.launch(kernelFn, 2, 32, 0, pA, pB, pC, N);

const hC = new Float32Array(N);
sim.memcpy(hC, pC, N * 4, 'D2H');
const verdict = validateOutput(hC, new Float32Array(N).fill(5.0), { tolerance: 1e-5 });
assert.strictEqual(verdict.verdict, 'PASS');
```

---

## Recipe 2: Validate 16-GPU Collectives & Catch Desync

```javascript
const { createSimulator } = require('gpubox');

const sim = createSimulator({ numDevices: 16, gpuProfile: 'A100' });
const comm = sim.ncclCommInitRank(16, 0, { algorithm: 'ring' });

const count = 1024;
const sBufs = [], rBufs = [];
for (let r = 0; r < 16; r++) {
  const s = sim.allocate(count * 4, r);
  const rec = sim.allocate(count * 4, r);
  sim.memcpy(s, new Float32Array(count).fill(r + 1), count * 4, 'H2D');
  sBufs.push(s);
  rBufs.push(rec);
}

sim.allReduce(sBufs, rBufs, count, 'float32', 'sum', comm);
const agg = sim.getAggregatedCounters().toJSON();
console.log('Total 16-rank ring traffic (bytes):', agg.crossGpuBytes); // 122880
```

---

## Recipe 3: Build a Custom Dashboard from the Event Stream

```javascript
const { createSimulator } = require('gpubox');

const sim = createSimulator({ traceLevel: 'events', numDevices: 4 });

sim.on('collective', (ev) => {
  console.log(`[Dashboard] Collective posted: op=${ev.op}, count=${ev.count}`);
});

sim.on('kernel_start', (ev) => {
  console.log(`[Dashboard] Kernel launched with grid:`, ev.gridDim);
});
```

---

## Recipe 4: Define a Custom GPU Profile & Topology

```javascript
const { registerProfile, registerTopology, createSimulator } = require('gpubox');

// Register custom accelerator profile
registerProfile('CustomNPU', {
  name: 'Custom Enterprise NPU',
  architecture: 'CustomTile',
  computeCapability: '9.0',
  clockSpeedMHz: 1800,
  peakTFLOPS: 50.0,
  peakGFLOPS: 50000,
  memBandwidthGBs: 3200,
  smCount: 144,
  maxThreadsPerSM: 2048,
  globalMemLatencyCycles: 250,
  sharedMemLatencyCycles: 15,
  atomicLatencyCycles: 30,
  syncLatencyCycles: 10,
  flopLatencyCycles: 1,
  maxSharedMemPerBlock: 196608,
  maxRegistersPerBlock: 65536
});

// Register custom torus cluster topology
registerTopology('torus_2d', {
  name: '2D Torus Interconnect',
  type: 'custom',
  nodeCount: 4,
  gpusPerNode: 4,
  interGpuBandwidthGBs: 400,
  interNodeBandwidthGBs: 200,
  latencyUs: 0.8
});

const sim = createSimulator({ gpuProfile: 'CustomNPU', topology: 'torus_2d', numDevices: 8 });
```

---

## Recipe 5: Embed in Documentation Site (Docusaurus / VitePress)

```html
<script src="https://cdn.jsdelivr.net/npm/gpubox/dist/cuda-sim.min.js"></script>
<script>
  // Interactive client-side CUDA runner without physical GPU backend
  function runSample() {
    const sim = CudaSimLib.createSimulator({ gpuProfile: 'A100' });
    const prog = CudaSimLib.compile(`
      __global__ void scale(float* x, float s, int n) {
        int i = threadIdx.x;
        if (i < n) x[i] *= s;
      }
    `);
    console.log('Compiled kernels:', prog.kernelNames);
  }
</script>
```

---

## Recipe 6: Run Headless as a CLI

```bash
node -e "
  const { createSimulator, compile } = require('gpubox');
  const sim = createSimulator({ gpuProfile: 'A100', numDevices: 2 });
  console.log('Virtual cluster initialized with 2 devices.');
"
```

---

## Recipe 7: Save, Diff, and Compare Run Reports Across Code Versions

```javascript
const fs = require('fs');
const { createSimulator, compile } = require('gpubox');

const sim = createSimulator({ gpuProfile: 'A100' });
const res = sim.run(`__global__ void k(float* x) { x[threadIdx.x] = 1.0f; }`);

fs.writeFileSync('report-v1.json', JSON.stringify(res, null, 2));
```

---

## Recipe 8: Run Untrusted Code Safely in a Worker with Limits

```javascript
const { createWorkerSimulator } = require('gpubox');

const workerSim = createWorkerSimulator({
  limits: { maxSteps: 50000, maxTimeMs: 2000 }
});

async function runUntrusted(userCode) {
  const result = await workerSim.run(userCode);
  if (result.status === 'ERROR') {
    console.warn('Execution aborted by watchdog limit:', result.diagnostics[0].message);
  }
  return result;
}
```
