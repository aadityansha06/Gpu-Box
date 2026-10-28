# Gpubox API Reference

Complete reference for the public TypeScript and JavaScript API of `gpubox`.

## Top-Level Functions

### `createSimulator(config?: SimConfig): GpuboxSimulator`
Instantiates a headless GPU simulator and virtual cluster.

- **Parameters:**
  - `config.gpuProfile`: string (`'A100'`, `'H100'`, `'RTX4090'`, `'RTX4050_Laptop'`, `'RTX3090'`, `'V100'`) or custom profile object.
  - `config.numDevices`: integer between 1 and 16 (default: 1).
  - `config.topology`: `'nvswitch_mesh'`, `'pcie_tree'`, `'multi_node'`, `'custom'`.
  - `config.limits`: `{ maxSteps?: number, maxTimeMs?: number, maxHeapBytes?: number }`.
  - `config.raceDetection`: boolean or `{ enabled: boolean, strict: boolean }`.
  - `config.traceLevel`: `'off'`, `'summary'`, `'events'`, `'full'`.
  - `config.seed`: integer (default: 1337).

### `compile(source: string): Program`
Transpiles CUDA C++ source code into JavaScript generator functions.

### `analyze(source: string): ProgramAnalysis`
Performs static inspection on CUDA C++ source WITHOUT running. Reports supported kernels, syntax diagnostics, and any unsupported features (such as dynamic parallelism, inline PTX, or WMMA).

### `validateOutput(actual, expected, options?): ValidationResult`
Compares an actual output array against an expected CPU reference array using absolute (`'abs'`), relative (`'relative'`), or ULP (`'ulp'`) tolerance.

### `listProfiles(): string[]`
Returns list of registered GPU hardware profile identifiers.

### `getProfile(name: string): GpuProfileSpec`
Returns hardware specifications (SM count, clock speed, memory bandwidth, latency cycles) for a given profile.

### `listTopologies(): string[]`
Returns list of registered cluster topologies.

### `getTopology(name: string): TopologySpec`
Returns network latency and bandwidth specifications for a topology preset.

### `getCapabilities(engine?: 'js' | 'wasm'): EngineCapabilities`
Returns the machine-readable capability matrix from `compat.json`.

---

## GpuboxSimulator Class

### Memory & Execution
- `allocate(sizeBytes: number, deviceId?: number): DevicePointer`
- `free(ptr: DevicePointer, deviceId?: number): number`
- `memcpy(dst, src, sizeBytes, kind: 'H2D'|'D2H'|'D2D'): number`
- `memcpyPeer(dst, dstDev, src, srcDev, sizeBytes): void`
- `launch(kernelFn, gridDim, blockDim, sharedMemBytes, ...args): void`
- `reset(): void`

### NCCL Collectives (1 to 16 GPUs)
- `ncclCommInitRank(nranks: number, rank: number, options?: object): NcclComm`
- `ncclCommInitAll(comms: NcclComm[], nranks: number, options?: object): NcclComm[]`
- `allReduce(sendbufs, recvbufs, count, datatype, op, comm): number`
- `broadcast(sendbufs, recvbufs, count, datatype, root, comm): number`
- `reduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm): number`
- `allGather(sendbufs, recvbufs, sendcount, datatype, comm): number`
- `reduce(sendbufs, recvbufs, count, datatype, op, root, comm): number`
- `allToAll(sendbufs, recvbufs, countPerRank, datatype, comm): number`
- `send(sendbuf, count, datatype, peer, comm): number`
- `recv(recvbuf, count, datatype, peer, comm): number`
- `groupStart(): number`
- `groupEnd(): number`

### Telemetry & Diagnostics
- `getCounters(deviceId?: number): SimCounters`
- `getAggregatedCounters(): SimCounters`
- `getDiagnostics(): SimDiagnostic[]`
- `getLeakReport(): LeakReport`
- `on(eventName: string, callback: Function): this`
- `off(eventName: string, callback: Function): this`
- `getTrace(): TraceEvent[]`
