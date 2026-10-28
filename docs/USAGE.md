# Gpubox User Guide & Usage Manual

> **Gpubox** is a browser-based CUDA and multi-GPU simulation and pre-validation library. It allows AI engineers, startups, and enterprises to catch CUDA logic bugs, memory hazards, and collective communication desynchronizations before spending money on physical GPU hardware.

---

## 1. Installation

### Node.js / Bundlers (npm / yarn / pnpm)
```bash
npm install gpubox
```

### Browser Script Tag (CDN)
```html
<script src="https://cdn.jsdelivr.net/npm/gpubox@latest/dist/cuda-sim.min.js"></script>
<script>
  const { createSimulator, compile, validateOutput } = window.Gpubox;
</script>
```

### ES Modules (Vite, Webpack, Next.js)
```javascript
import { createSimulator, compile, analyze, run, validateOutput } from 'gpubox';
```

---

## 2. Quick Start: Single-Kernel Execution

Here is a minimal 10-line example that compiles a vector addition kernel, allocates virtual device memory, executes the kernel on a simulated NVIDIA A100 GPU, copies results back, and validates accuracy:

```javascript
const { createSimulator, compile, validateOutput } = require('gpubox');

// 1. Initialize simulator with target GPU profile
const sim = createSimulator({ gpuProfile: 'A100', numDevices: 1 });

// 2. Transpile CUDA C++ kernel to client-side SIMT execution
const prog = compile(`
  __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
          c[idx] = a[idx] + b[idx];
      }
  }
`);

// 3. Allocate device memory & upload host buffers
const N = 1024;
const d_a = sim.allocate(N * 4);
const d_b = sim.allocate(N * 4);
const d_c = sim.allocate(N * 4);

sim.memcpy(d_a, new Float32Array(N).fill(1.5), N * 4, 'H2D');
sim.memcpy(d_b, new Float32Array(N).fill(2.5), N * 4, 'H2D');

// 4. Launch kernel (gridDim=4, blockDim=256, sharedMem=0)
const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_vecAdd;')();
sim.launch(kernelFn, 4, 256, 0, d_a, d_b, d_c, N);

// 5. Download result & validate
const h_c = new Float32Array(N);
sim.memcpy(h_c, d_c, N * 4, 'D2H');

const verdict = validateOutput(h_c, new Float32Array(N).fill(4.0), { tolerance: 1e-5 });
console.log('Verdict:', verdict.verdict); // 'PASS'

// 6. Inspect modeled telemetry
const counters = sim.getCounters().toJSON();
console.log(`Executed: ${counters.instructionsExecuted} instructions, ${counters.globalLoads} loads, ${counters.globalStores} stores.`);
```

---

## 3. High-Level One-Shot Runner

If you prefer a one-line execution that handles compilation, execution, metric aggregation, and structured reporting automatically:

```javascript
const { run } = require('gpubox');

const source = `
  __global__ void square(float* d_out, const float* d_in, int n) {
      int i = threadIdx.x;
      if (i < n) d_out[i] = d_in[i] * d_in[i];
  }
  int main() {
      float *d_in, *d_out;
      cudaMalloc(&d_in, 64 * sizeof(float));
      cudaMalloc(&d_out, 64 * sizeof(float));
      square<<<1, 64>>>(d_out, d_in, 64);
      cudaFree(d_in);
      cudaFree(d_out);
      return 0;
  }
`;

const result = run(source, { gpuProfile: 'A100', numDevices: 1 });
console.log('Status:', result.status); // 'PASS'
console.log('Estimated Time:', result.estimatedTimeMs, 'ms');
console.log('Leak Report:', result.leakReport); // { totalAllocations: 0, totalLeakedBytes: 0, leaks: [] }
```

---

## 4. Multi-GPU Cluster Simulation (1–16 Virtual GPUs)

Gpubox simulates distributed clusters with 1 to 16 virtual GPU nodes, rank-isolated memory spaces, and ring-based NCCL collectives.

### 16-GPU AllReduce Example
```javascript
const { createSimulator, validateOutput } = require('gpubox');

// Initialize a 16-GPU cluster with NVSwitch mesh topology
const sim = createSimulator({
  gpuProfile: 'A100',
  numDevices: 16,
  topology: 'nvswitch_mesh'
});

const count = 1024;
const sendBuffers = [];
const recvBuffers = [];

// Allocate memory per virtual rank
for (let rank = 0; rank < 16; rank++) {
  const s = sim.allocate(count * 4, rank);
  const r = sim.allocate(count * 4, rank);
  // Fill each rank with rank+1 (1.0 on rank 0, 2.0 on rank 1, ..., 16.0 on rank 15)
  sim.memcpy(s, new Float32Array(count).fill(rank + 1), count * 4, 'H2D');
  sendBuffers.push(s);
  recvBuffers.push(r);
}

// Execute 16-rank ring AllReduce (SUM)
sim.allReduce(sendBuffers, recvBuffers, count, 'float32', 'sum');

// Verify output on rank 0 (Sum 1..16 = 136.0)
const hostOut = new Float32Array(count);
sim.memcpy(hostOut, recvBuffers[0], count * 4, 'D2H');
const verdict = validateOutput(hostOut, new Float32Array(count).fill(136.0), { tolerance: 1e-4 });
console.log('Cluster AllReduce Verdict:', verdict.verdict); // 'PASS'

// Inspect 16x16 ring communication traffic matrix
const counters = sim.getCounters(0).toJSON();
console.log(`Rank 0 cross-GPU ring traffic: ${counters.crossGpuBytes.toLocaleString()} bytes`);
```

### Supported Collectives
- `sim.allReduce(sendbufs, recvbufs, count, datatype, op, comm)` (SUM, MAX, MIN, PROD, AVG)
- `sim.broadcast(sendbufs, recvbufs, count, datatype, root, comm)`
- `sim.allGather(sendbufs, recvbufs, sendcount, datatype, comm)`
- `sim.reduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm)`
- `sim.allToAll(sendbufs, recvbufs, countPerRank, datatype, comm)`
- `sim.send(sendbuf, count, datatype, peer, comm)` & `sim.recv(recvbuf, count, datatype, peer, comm)`
- `sim.groupStart()` & `sim.groupEnd()`

---

## 5. Diagnostic Tools & Bug Detection

Gpubox includes deep instrumentation that triggers automated diagnostics for common CUDA pitfalls:

### 1. Shared Memory Race Detector
Detects Read-After-Write (RAW), Write-After-Read (WAR), and Write-After-Write (WAW) hazards between threads in the same block across barrier epochs (`__syncthreads()`):
```javascript
const { createSimulator, compile } = require('gpubox');
const sim = createSimulator({ raceDetection: true });

// Deliberate shared memory race: thread 0 writes while thread 1 reads without __syncthreads()
const code = `
  __global__ void buggyRace(float* out) {
      __shared__ float s[32];
      int tid = threadIdx.x;
      s[tid] = tid * 2.0f;
      // BUG: missing __syncthreads() before reading neighbor's shared slot
      out[tid] = s[(tid + 1) % 32];
  }
`;
const prog = compile(code);
const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_buggyRace;')();
const d_out = sim.allocate(32 * 4);
sim.launch(kernelFn, 1, 32, 0, d_out);

const diags = sim.getDiagnostics();
console.log('Detected Hazards:', diags.filter(d => d.category === 'RACE_CONDITION'));
```

### 2. Multi-GPU Desync & Deadlock Detector
Identifies communicator mismatches before a physical cluster hangs:
- Missing ranks in collective calls
- Count mismatches between ranks
- Root rank divergence
- Collective operation order divergence

### 3. Runaway Loop Watchdog
Guards against infinite loops (`while(true) {}`) or runaway thread code by automatically aborting execution after step or execution time limits, pinpointing the kernel name, block, and thread index.

### 4. Virtual Sparse VRAM & Leak Reporting
Simulates up to 80 GiB virtual VRAM per device without consuming host RAM using on-demand 64 KB paging. `sim.getLeakReport()` tracks unfreed pointers and reports byte counts and allocations.

---

## 6. Pre-Execution Static Analysis

Before executing, use `analyze(code)` to verify compatibility with supported CUDA features:

```javascript
const { analyze } = require('gpubox');

const res = analyze(`
  __global__ void unsupportedKernel() {
      asm("nop;"); // Inline PTX
  }
`);

if (!res.success) {
  console.error('Compatibility failure:', res.diagnostics[0].message);
  console.error('Docs:', res.diagnostics[0].docsUrl);
}
```

---

## 7. Web Worker Isolation (Non-Blocking Simulation)

To run simulations in background browser threads without freezing the UI:

```javascript
import { createWorkerSimulator } from 'gpubox';

const sim = createWorkerSimulator({ gpuProfile: 'A100', numDevices: 4 });

const result = await sim.run(`
  __global__ void compute(float* x) {
      x[threadIdx.x] = 42.0f;
  }
`);

console.log('Worker Result Status:', result.status);
```

---

## 8. Hardware Profiles & Cluster Topologies

### Available GPU Profiles (`listProfiles()`)
- `A100`: NVIDIA A100 SXM4 80GB (Ampere, 108 SMs, 2039 GB/s HBM2e)
- `H100`: NVIDIA H100 SXM5 80GB (Hopper, 132 SMs, 3350 GB/s HBM3)
- `RTX 4090`: NVIDIA GeForce RTX 4090 (Ada Lovelace, 128 SMs, 1008 GB/s GDDR6X)
- `RTX 4050 Laptop`: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, 20 SMs, 192 GB/s GDDR6) — *Physically verified on hardware*
- `RTX 5060 Ti`: Modeled Next-Gen Architecture (`sm_120 [TO VERIFY]`, 36 SMs, 288 GB/s)
- `RTX 3090`: NVIDIA GeForce RTX 3090 (Ampere, 82 SMs, 936 GB/s)
- `V100`: NVIDIA Tesla V100 SXM2 16GB (Volta, 80 SMs, 900 GB/s HBM2)

### Available Topologies (`listTopologies()`)
- `nvswitch_mesh`: All-to-all crossbar interconnect (full bidirectional NVLink bandwidth)
- `ring`: Unidirectional / bidirectional ring interconnect
- `torus_2d`: 2D Torus grid mesh for large virtual clusters
- `pcie_tree`: Host-mediated PCIe Gen 4/5 tree with root complex bottleneck

---

## 9. Commercial Packaging & Engine-as-a-Service (EaaS) Delivery

Gpubox is commercially licensed under a client-side delivery model:

- **Zero Provider Hosting Costs**: All AST transpilation, SIMT execution, and collective communication runs on the client machine or user browser. The provider incurs $0 in cloud server infrastructure bills.
- **Client Privacy & Intellectual Property**: Customer CUDA code remains strictly local and is never transmitted over external networks.
- **Protected Binaries**: The numerical simulation core is compiled to binary WebAssembly (`.wasm`), and client runtimes are minified to safeguard intellectual property.
- **Private Distribution**: Deliverable via private tarball packages (`gpubox-1.0.0.tgz`), private scoped npm registries, or direct customer CDN assets.

---

## 10. Legal & Trademark Disclaimer

> **Disclaimer**: Gpubox is not affiliated with, sponsored by, or endorsed by NVIDIA Corporation. CUDA, NVIDIA, NVLink, and NVSwitch are registered trademarks or trademarks of NVIDIA Corporation. All trademarks are used strictly for reference purposes to identify software compatibility and architectural profiles.
