# Gpubox API Reference

Complete reference for the public TypeScript and JavaScript API of `gpubox` (v1.0.0).

---

## Table of Contents
- Top-Level Exports
  - `createSimulator(config)`
  - `run(programOrSource, options)`
  - `compile(source)`
  - `analyze(source)`
  - `validateOutput(actual, expected, options)`
  - `createWorkerSimulator(config)`
  - `getCapabilities(engine)`
  - `listProfiles()` / `getProfile(name)` / `registerProfile(name, spec)`
  - `listTopologies()` / `getTopology(name)` / `registerTopology(name, spec)`
  - `SimError`
- GpuboxSimulator Instance Methods
  - Memory & Pointers
  - CUDA Driver Aliases
  - Kernel Execution
  - Multi-GPU NCCL Collectives
  - Telemetry & Diagnostics
  - Event Subscriptions
- TypeScript Types & Schema Reference

---

## Top-Level Exports

### `createSimulator(config?: SimConfig): GpuboxSimulator`
Instantiates a headless GPU simulator and virtual cluster.

**Parameters:**
- `config.gpuProfile`: string (`'A100'`, `'H100'`, `'RTX 4090'`, `'RTX 4050 Laptop'`, `'RTX 5060 Ti'`, `'RTX 3090'`, `'V100'`) or custom `GpuProfileSpec` object. Default: `'A100'`.
- `config.numDevices`: integer between 1 and 16. Default: 1.
- `config.topology`: `'nvswitch_mesh'`, `'ring'`, `'torus_2d'`, `'pcie_tree'`. Default: `'nvswitch_mesh'`.
- `config.limits`: `{ maxSteps?: number, maxTimeMs?: number, maxHeapBytes?: number }`.
- `config.raceDetection`: boolean or `{ enabled: boolean, strict: boolean }`. Default: `true`.
- `config.traceLevel`: `'off'`, `'summary'`, `'events'`, `'full'`. Default: `'off'`.
- `config.seed`: integer PRNG seed. Default: `1337`.
- `config.engine`: `'js'` | `'wasm'`. Default: `'js'`.

**Example:**
```javascript
const { createSimulator } = require('gpubox');

const sim = createSimulator({
  gpuProfile: 'A100',
  numDevices: 8,
  topology: 'nvswitch_mesh',
  traceLevel: 'events'
});
```

---

### `run(programOrSource: Program | string, options?: RunOptions): RunResult`
High-level, one-shot runner. Handles compilation, execution, metric aggregation, and structured reporting in a single synchronous call.

**Parameters:**
- `programOrSource`: CUDA C++ source code string or precompiled `Program` object.
- `options.gpuProfile`: Hardware profile. Default: `'A100'`.
- `options.numDevices`: Number of virtual devices (1 to 16).
- `options.validation`: `{ expected: ArrayLike<number>, tolerance?: number, mode?: 'abs' | 'relative' | 'ulp' }`.

**Returns `RunResult`:**
```typescript
interface RunResult {
  schemaVersion: '1.0.0';
  engineVersion: '1.0.0';
  status: 'PASS' | 'FAIL' | 'UNSUPPORTED' | 'ERROR';
  stdout: string[];
  outputs: Record<string, any>;
  counters: SimCountersReport;
  diagnostics: SimDiagnostic[];
  collectives: any[];
  trafficMatrix: number[][];
  estimatedTimeMs: number;
  leakReport: LeakReport;
  validation?: ValidationResult;
  trace?: TraceEvent[];
}
```

**Example:**
```javascript
const { run } = require('gpubox');

const result = run(`
  __global__ void square(float* d_out, const float* d_in, int n) {
      int i = threadIdx.x;
      if (i < n) d_out[i] = d_in[i] * d_in[i];
  }
  int main() {
      float *d_in, *d_out;
      cudaMalloc(&d_in, 32 * sizeof(float));
      cudaMalloc(&d_out, 32 * sizeof(float));
      square<<<1, 32>>>(d_out, d_in, 32);
      cudaFree(d_in);
      cudaFree(d_out);
      return 0;
  }
`, { gpuProfile: 'A100' });

console.log('Status:', result.status); // 'PASS'
console.log('Estimated Time:', result.estimatedTimeMs, 'ms');
```

---

### `compile(source: string): Program`
Transpiles CUDA C++ source code into client-side SIMT generator functions and AST metadata.

**Returns `Program`:**
- `source`: Original source string.
- `transpiledCode`: Transpiled JavaScript generator code.
- `kernels`: String array of discovered `__global__` kernel names.
- `analysis`: `ProgramAnalysis` object.

---

### `analyze(source: string): ProgramAnalysis`
Performs static inspection on CUDA C++ source code without executing. Detects kernels, memory qualifiers, and flags unsupported features.

**Returns `ProgramAnalysis`:**
- `success`: boolean (`false` if unsupported CUDA features or syntax errors are found).
- `kernels`: Discovered kernel descriptors (`name`, `sharedMemoryBytes`, `hasBarriers`).
- `diagnostics`: Array of `SimDiagnostic` objects.
- `unsupportedFeatures`: Array of detected unsupported features (`'dynamic_parallelism'`, `'inline_ptx'`, `'wmma_tensor_core'`).

---

### `validateOutput(actual, expected, options?): ValidationResult`
Compares an actual output buffer against an expected reference buffer with configurable numeric tolerances.

**Parameters:**
- `actual`: `ArrayLike<number>` or `Float32Array`.
- `expected`: `ArrayLike<number>` or `Float32Array`.
- `options.tolerance`: Tolerance threshold (default: `1e-5`).
- `options.mode`:
  - `'abs'`: Absolute difference `|actual[i] - expected[i]| <= tolerance`.
  - `'relative'`: Relative difference `|actual[i] - expected[i]| / max(|expected[i]|, 1e-12) <= tolerance`.
  - `'ulp'`: Unit in the Last Place float32 comparison.

**Returns `ValidationResult`:**
- `verdict`: `'PASS'` | `'FAIL'`.
- `maxAbsDiff`: Maximum absolute difference observed.
- `maxRelDiff`: Maximum relative difference observed.
- `mismatchCount`: Count of elements exceeding tolerance.
- `firstMismatch`: `{ index, actual, expected, diff }` (if any).

---

### `createWorkerSimulator(config?: SimConfig): WorkerSimulator`
Creates a Web Worker-isolated simulator for non-blocking browser execution.

**Methods:**
- `run(programOrSource, options): Promise<RunResult>`
- `compile(source): Promise<Program>`
- `analyze(source): Promise<ProgramAnalysis>`
- `reset(): Promise<void>`
- `terminate(): void`

---

### `getCapabilities(engine?: 'js' | 'wasm'): EngineCapabilities`
Returns the machine-readable capability matrix, specifying supported CUDA features, memory tiers, and unsupported boundaries.

---

### Profile Registry
- `listProfiles(): string[]` — Returns `['A100', 'H100', 'RTX 4090', 'RTX 4050 Laptop', 'RTX 5060 Ti', 'RTX 3090', 'V100']`.
- `getProfile(name: string): GpuProfileSpec` — Returns hardware specs (SM count, clock, peak memory bandwidth, NVLink bandwidth).
- `registerProfile(name: string, spec: GpuProfileSpec): void` — Registers a custom GPU profile.

### Topology Registry
- `listTopologies(): string[]` — Returns `['nvswitch_mesh', 'ring', 'torus_2d', 'pcie_tree']`.
- `getTopology(name: string): TopologySpec` — Returns latency base and bandwidth multipliers.
- `registerTopology(name: string, spec: TopologySpec): void` — Registers a custom cluster topology.

---

### `SimError`
Typed error class thrown for all runtime, driver, and compilation failures.

**Properties:**
- `code`: Machine-readable error code (`'OUT_OF_BOUNDS'`, `'RACE_CONDITION'`, `'COLLECTIVE_DESYNC'`, `'UNSUPPORTED_FEATURE'`).
- `category`: Error category (`'MEMORY_ERROR'`, `'RACE_CONDITION'`, `'CLUSTER_ERROR'`).
- `message`: Diagnostic message.
- `location`: `{ kernel?: string, block?: any, thread?: any, rank?: number }`.
- `docsUrl`: Reference URL explaining the error and fix.

---

## GpuboxSimulator Instance Methods

### Memory Management
```javascript
const sim = createSimulator({ gpuProfile: 'A100', numDevices: 2 });
```

#### `sim.allocate(sizeBytes: number, deviceId: number = 0): DevicePointer`
Allocates virtual device memory on the specified GPU rank using 64 KB sparse paging (supports up to 80 GiB per device).

#### `sim.free(ptr: DevicePointer, deviceId: number = 0): number`
Frees an allocated device pointer. Detects and reports double-free operations.

#### `sim.memcpy(dst, src, sizeBytes: number, kind: 'H2D' | 'D2H' | 'D2D' = 'H2D'): number`
Copies memory between host and device, or between device allocations.

#### `sim.memcpyPeer(dst, dstDev: number, src, srcDev: number, sizeBytes: number): number`
Direct peer-to-peer memory copy between two virtual GPU ranks.

---

### CUDA Driver Aliases
For drop-in compatibility with standard CUDA C Driver API host code:
- `sim.cudaMalloc(sizeBytes, deviceId)` — Alias for `allocate()`.
- `sim.cudaFree(ptr, deviceId)` — Alias for `free()`.
- `sim.cudaMemcpy(dst, src, sizeBytes, kind)` — Alias for `memcpy()`.
- `sim.cudaMemcpyPeer(dst, dstDev, src, srcDev, sizeBytes)` — Alias for `memcpyPeer()`.
- `sim.cudaSetDevice(deviceId: number): number` — Sets active virtual GPU rank (0 to 15).
- `sim.cudaGetDevice(): number` — Returns active virtual GPU rank.
- `sim.cudaDeviceEnablePeerAccess(peerId: number): number` — Enables P2P access between ranks.
- `sim.cudaDeviceDisablePeerAccess(peerId: number): number` — Disables P2P access between ranks.

---

### Kernel Execution
#### `sim.launch(kernelFn, gridDim, blockDim, sharedMemBytes, ...args): void`
Launches a compiled kernel function across the virtual SIMT grid.
- `kernelFn`: Kernel function compiled via `compile()`.
- `gridDim`: integer or `[x, y, z]` block dimensions.
- `blockDim`: integer or `[x, y, z]` thread dimensions.
- `sharedMemBytes`: Dynamic shared memory allocation in bytes.
- `...args`: Device pointers and scalar arguments.

#### `sim.reset(): void`
Resets all device memories, zeroing telemetry counters, diagnostics, and traces across all virtual ranks.

---

### Multi-GPU NCCL Collectives (1 to 16 Virtual GPUs)

#### `sim.allReduce(sendbufs, recvbufs, count, datatype, op, comm)`
Executes an AllReduce across all participating virtual GPU ranks.
- `sendbufs`: Array of device pointers (one per rank).
- `recvbufs`: Array of device pointers (one per rank).
- `count`: Number of elements per rank.
- `datatype`: `'float32'` | `'float16'` | `'int32'`.
- `op`: `'sum'` | `'max'` | `'min'` | `'prod'` | `'avg'`.

#### `sim.broadcast(sendbufs, recvbufs, count, datatype, root, comm)`
Broadcasts data from `root` rank to all ranks.

#### `sim.reduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm)`
Reduces data across ranks and scatters equal shards to each rank.

#### `sim.allGather(sendbufs, recvbufs, sendcount, datatype, comm)`
Gathers equal shards from each rank to all ranks.

#### `sim.allToAll(sendbufs, recvbufs, countPerRank, datatype, comm)`
Each rank sends distinct data shards to every other rank.

#### `sim.send(sendbuf, count, datatype, peer, comm)` / `sim.recv(recvbuf, count, datatype, peer, comm)`
Point-to-point ring shift transmission between ranks.

#### `sim.groupStart()` / `sim.groupEnd()`
Enables batched non-blocking collective schedules.

*Note: All collectives also have `nccl*` aliases (`sim.ncclAllReduce`, `sim.ncclBroadcast`, etc.).*

---

### Telemetry, Diagnostics & Event Hooks

#### `sim.getCounters(deviceId?: number): SimCounters`
Returns execution metrics for the specified device (or device 0).
- `instructionsExecuted`: Total SIMT instructions.
- `globalLoads` / `globalStores`: Count of global memory memory operations.
- `globalLoadBytes` / `globalStoreBytes`: Exact byte counts.
- `coalescedLoads` / `coalescedStores`: Modeled 32-byte transaction sector metrics.
- `sharedLoads` / `sharedStores`: Shared memory operations.
- `bankConflicts`: Shared memory 32-bank conflict count.
- `branchDivergence`: Warp divergence events.
- `ncclOps`: Total collective operations executed.
- `crossGpuBytes`: Total inter-device communication traffic.

#### `sim.getDiagnostics(): SimDiagnostic[]`
Returns list of structured diagnostics detected during execution (memory bounds violations, shared memory race hazards, collective desync).

#### `sim.getLeakReport(): LeakReport`
Returns unfreed memory allocations across all devices:
`{ totalAllocations: number, totalLeakedBytes: number, leaks: Array<{ deviceId, allocId, sizeBytes, offset }> }`.

#### `sim.on(eventName, callback): this`
Subscribes to telemetry events:
- `'alloc'`: Fired on `allocate()` (`{ deviceId, sizeBytes, ptr }`).
- `'free'`: Fired on `free()` (`{ deviceId, ptr }`).
- `'memcpy'`: Fired on host-device copies (`{ kind, sizeBytes }`).
- `'p2p'`: Fired on peer-to-peer copies (`{ srcDev, dstDev, sizeBytes }`).
- `'kernel_start'` / `'kernel_end'`: Fired on kernel execution (`{ gridDim, blockDim, sharedMem }`).
- `'collective'`: Fired on collective execution (`{ op, count, root?, reduceOp? }`).

---

Copyright (c) 2026 Aadityansha Verma. All rights reserved. Commercial License.
