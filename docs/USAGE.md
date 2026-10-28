# Gpubox User Guide & Client Manual

Welcome to the official developer guide for **Gpubox** (v1.0.0).

Gpubox is a browser-based CUDA and multi-GPU simulation and pre-validation library. It allows AI engineers, startups, enterprise teams, and educational platforms to validate CUDA algorithms, verify numerical correctness against references, experiment with virtual 1–16 GPU clusters, and detect memory hazards before spending budget on physical GPU hardware.

---

## 1. Installation & Setup

### Option A: npm / yarn / pnpm (Node.js & Bundlers)
```bash
npm install gpubox
```

### Option B: Direct Browser Script Tag (CDN / Local Hosting)
```html
<script src="https://cdn.jsdelivr.net/npm/gpubox@latest/dist/cuda-sim.min.js"></script>
<script>
  // Available globally under window.Gpubox
  const { createSimulator, compile, run, validateOutput } = window.Gpubox;
</script>
```

### Option C: ES Modules (Vite, Webpack, Next.js, Rollup)
```javascript
import { createSimulator, compile, analyze, run, validateOutput } from 'gpubox';
```

---

## 2. Quick Start: Vector Addition in 60 Seconds

The following self-contained example initializes a virtual NVIDIA A100 GPU, compiles a CUDA C++ kernel, uploads host memory, executes across 4 blocks of 256 threads, reads back the output, and validates numerical correctness:

```javascript
const { createSimulator, compile, validateOutput } = require('gpubox');

// 1. Initialize simulator with target GPU hardware profile
const sim = createSimulator({ gpuProfile: 'A100', numDevices: 1 });

// 2. Compile CUDA C++ kernel source
const prog = compile(`
  __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
          c[idx] = a[idx] + b[idx];
      }
  }
`);

// 3. Allocate virtual device memory & copy data from host (H2D)
const N = 1024;
const d_a = sim.allocate(N * 4);
const d_b = sim.allocate(N * 4);
const d_c = sim.allocate(N * 4);

sim.memcpy(d_a, new Float32Array(N).fill(1.5), N * 4, 'H2D');
sim.memcpy(d_b, new Float32Array(N).fill(2.5), N * 4, 'H2D');

// 4. Launch kernel across grid (gridDim=4 blocks, blockDim=256 threads, sharedMem=0 bytes)
const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_vecAdd;')();
sim.launch(kernelFn, 4, 256, 0, d_a, d_b, d_c, N);

// 5. Copy result from device back to host (D2H) and validate
const h_c = new Float32Array(N);
sim.memcpy(h_c, d_c, N * 4, 'D2H');

const verdict = validateOutput(h_c, new Float32Array(N).fill(4.0), { tolerance: 1e-5 });
console.log('Validation Verdict:', verdict.verdict); // 'PASS'

// 6. Read hardware metrics & counters
const counters = sim.getCounters().toJSON();
console.log(`Instructions: ${counters.instructionsExecuted}`);
console.log(`Global Loads: ${counters.globalLoads} (${counters.globalLoadBytes} bytes)`);
console.log(`Global Stores: ${counters.globalStores} (${counters.globalStoreBytes} bytes)`);
```

---

## 3. How Customers Access Functions: The Complete Catalog

Gpubox gives developers multiple ways to interact with the engine depending on their workflow.

### 3.1. Method 1: The One-Shot Runner (`run`)
When you want to compile and execute a complete CUDA program (with host `main()`) in one step without manual memory management:

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
console.log('Run Status:', result.status); // 'PASS'
console.log('Estimated Wall Time:', result.estimatedTimeMs, 'ms');
console.log('Memory Leaks:', result.leakReport.totalLeakedBytes); // 0 bytes
```

### 3.2. Method 2: Standard CUDA Driver API Aliases
If porting existing CUDA C host code, Gpubox provides 1:1 aliases matching standard CUDA Driver and Runtime APIs:

```javascript
const sim = createSimulator({ gpuProfile: 'A100', numDevices: 2 });

// CUDA Driver API calls
sim.cudaSetDevice(0); // Switch to virtual GPU 0
const d_x = sim.cudaMalloc(1024 * 4);
sim.cudaMemcpy(d_x, hostData, 1024 * 4, 'H2D');

// Peer-to-peer memory copy between GPU 0 and GPU 1
sim.cudaDeviceEnablePeerAccess(1);
sim.cudaMemcpyPeer(d_y, 1, d_x, 0, 1024 * 4);

// Free memory
sim.cudaFree(d_x);
```

### 3.3. Method 3: Static Pre-Execution Analysis (`analyze`)
To check whether user code contains unsupported CUDA features before running:

```javascript
const { analyze } = require('gpubox');

const analysis = analyze(`
  __global__ void unsupported() {
      asm("nop;"); // Inline PTX assembly
  }
`);

if (!analysis.success) {
  console.log('Feature rejected:', analysis.unsupportedFeatures[0].feature);
  console.log('Reason:', analysis.unsupportedFeatures[0].reason);
}
```

### 3.4. Method 4: Automated Output Validation (`validateOutput`)
Supports three tolerance modes: absolute, relative, and float32 ULP:

```javascript
const { validateOutput } = require('gpubox');

// Mode 1: Absolute difference (default)
const res1 = validateOutput(actualArray, expectedArray, { tolerance: 1e-4, mode: 'abs' });

// Mode 2: Relative difference
const res2 = validateOutput(actualArray, expectedArray, { tolerance: 1e-3, mode: 'relative' });

// Mode 3: Float32 ULP (Unit in the Last Place)
const res3 = validateOutput(actualArray, expectedArray, { tolerance: 4, mode: 'ulp' });
```

---

## 4. Multi-GPU Cluster Simulation (1–16 Virtual GPUs)

Gpubox simulates distributed clusters with 1 to 16 virtual GPU nodes, isolated device memory spaces, and ring-based NCCL collectives.

### 16-GPU AllReduce Example:
```javascript
const { createSimulator, validateOutput } = require('gpubox');

// 1. Create a 16-GPU cluster with NVSwitch mesh interconnect
const sim = createSimulator({
  gpuProfile: 'A100',
  numDevices: 16,
  topology: 'nvswitch_mesh'
});

const count = 1024;
const sendBufs = [];
const recvBufs = [];

// 2. Allocate memory and initialize inputs per rank
for (let rank = 0; rank < 16; rank++) {
  const s = sim.allocate(count * 4, rank);
  const r = sim.allocate(count * 4, rank);
  // Rank 0 has 1.0, Rank 1 has 2.0, ..., Rank 15 has 16.0
  sim.memcpy(s, new Float32Array(count).fill(rank + 1), count * 4, 'H2D');
  sendBufs.push(s);
  recvBufs.push(r);
}

// 3. Execute 16-rank ring AllReduce (SUM)
sim.allReduce(sendBufs, recvBufs, count, 'float32', 'sum');

// 4. Verify output on rank 0 (Sum 1..16 = 136.0)
const h_out = new Float32Array(count);
sim.memcpy(h_out, recvBufs[0], count * 4, 'D2H');
const verdict = validateOutput(h_out, new Float32Array(count).fill(136.0));
console.log('AllReduce Status:', verdict.verdict); // 'PASS'

// 5. Inspect 16x16 pairwise inter-GPU traffic
const rank0Counters = sim.getCounters(0).toJSON();
console.log(`Rank 0 Cross-GPU Ring Traffic: ${rank0Counters.crossGpuBytes.toLocaleString()} bytes`);
```

### Supported Collective Operations:
- `sim.allReduce(sendbufs, recvbufs, count, datatype, op, comm)` (SUM, MAX, MIN, PROD, AVG)
- `sim.broadcast(sendbufs, recvbufs, count, datatype, root, comm)`
- `sim.allGather(sendbufs, recvbufs, sendcount, datatype, comm)`
- `sim.reduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm)`
- `sim.allToAll(sendbufs, recvbufs, countPerRank, datatype, comm)`
- `sim.send(sendbuf, count, datatype, peer, comm)` & `sim.recv(recvbuf, count, datatype, peer, comm)`
- `sim.groupStart()` & `sim.groupEnd()`

---

## 5. Diagnostic Tools & Hazard Detection

Gpubox includes automated safety and diagnostic detectors:

### 1. Shared Memory Race Hazard Detector
Flags Read-After-Write (RAW), Write-After-Read (WAR), and Write-After-Write (WAW) hazards between threads across barrier epochs when `__syncthreads()` is omitted:
```javascript
const sim = createSimulator({ raceDetection: true });
// If threads in the same block access shared memory slots without __syncthreads():
const diags = sim.getDiagnostics();
const races = diags.filter(d => d.category === 'RACE_CONDITION');
```

### 2. Multi-GPU Desync & Deadlock Detector
Catches rank sequence mismatches, count mismatches, and collective ordering bugs before they hang physical GPU hardware clusters.

### 3. Runaway Loop Watchdog
Automatically aborts infinite loops (`while(true) {}`) and names the kernel, block, and thread responsible.

### 4. Virtual Sparse VRAM & Memory Leak Tracking
Models up to 80 GiB virtual VRAM per device using 64 KB sparse paging without consuming host RAM. `sim.getLeakReport()` tracks any unfreed allocations.

---

## 6. Telemetry & Event Hooks

You can monitor simulation execution in real time:

```javascript
const sim = createSimulator({ traceLevel: 'events' });

sim.on('alloc', (ev) => console.log(`Allocated ${ev.sizeBytes} bytes on GPU ${ev.deviceId}`));
sim.on('memcpy', (ev) => console.log(`Memcpy ${ev.kind}: ${ev.sizeBytes} bytes`));
sim.on('kernel_start', (ev) => console.log(`Kernel started on grid`, ev.gridDim));
sim.on('collective', (ev) => console.log(`Collective ${ev.op} count: ${ev.count}`));
```

---

## 7. Web Worker Isolation (Background Execution)

To run simulations in background browser threads without blocking the UI:

```javascript
import { createWorkerSimulator } from 'gpubox';

const sim = createWorkerSimulator({ gpuProfile: 'A100', numDevices: 4 });

const result = await sim.run(`
  __global__ void compute(float* x) {
      x[threadIdx.x] = 42.0f;
  }
`);

console.log('Worker Result:', result.status);
```

---

## 8. Hardware Profiles & Cluster Topologies

### GPU Profiles (`listProfiles()`)
- `A100`: NVIDIA A100 SXM4 80GB (Ampere, 108 SMs, 2039 GB/s HBM2e)
- `H100`: NVIDIA H100 SXM5 80GB (Hopper, 132 SMs, 3350 GB/s HBM3)
- `RTX 4090`: NVIDIA GeForce RTX 4090 (Ada Lovelace, 128 SMs, 1008 GB/s GDDR6X)
- `RTX 4050 Laptop`: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, 20 SMs, 192 GB/s GDDR6) — *Physically verified*
- `RTX 5060 Ti`: Modeled Architecture (`sm_120 [TO VERIFY]`, 36 SMs, 288 GB/s)
- `RTX 3090`: NVIDIA GeForce RTX 3090 (Ampere, 82 SMs, 936 GB/s)
- `V100`: NVIDIA Tesla V100 SXM2 16GB (Volta, 80 SMs, 900 GB/s HBM2)

### Cluster Topologies (`listTopologies()`)
- `nvswitch_mesh`: All-to-all crossbar interconnect with bidirectional NVLink bandwidth.
- `ring`: Unidirectional / bidirectional ring interconnect.
- `torus_2d`: 2D Torus grid mesh for virtual multi-node clusters.
- `pcie_tree`: Host-mediated PCIe Gen 4/5 tree with root complex bottleneck.

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

Copyright (c) 2026 Aadityansha Verma. All rights reserved. Commercial License.
