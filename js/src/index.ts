/**
 * index.ts
 * ────────
 * Main public entry point for Gpubox.
 *
 * Copyright (c) 2026 Aadityansha Verma. All rights reserved.
 */

import type {
  SimConfig,
  Program,
  ProgramAnalysis,
  RunResult,
  StatusVerdict,
  SimCountersReport,
  SimDiagnostic,
  GpuProfileSpec,
  TopologySpec,
  EngineCapabilities,
  ValidationOptions,
  ValidationResult,
  TraceEvent,
  LeakReport
} from './types';

export * from './types';

// Typed error model
export class SimError extends Error {
  code: string;
  category: string;
  location?: { kernel?: string; block?: any; thread?: any; line?: number; rank?: number };
  docsUrl?: string;

  constructor(options: {
    code: string;
    category?: string;
    message: string;
    location?: { kernel?: string; block?: any; thread?: any; line?: number; rank?: number };
    docsUrl?: string;
  }) {
    super(options.message);
    this.name = 'SimError';
    this.code = options.code;
    this.category = options.category || 'RUNTIME_ERROR';
    this.location = options.location;
    this.docsUrl = options.docsUrl || 'https://github.com/aadityansha/gpubox/blob/main/docs/API_REFERENCE.md';
  }
}

// Global runtime references (compatible with Node, browsers, and Web Workers)
function getGlobal(): any {
  if (typeof globalThis !== 'undefined') return globalThis;
  if (typeof window !== 'undefined') return window;
  if (typeof global !== 'undefined') return global;
  return {};
}

const g = getGlobal();

// Dynamically load runtime prerequisites if not already on global
function ensureRuntimeLoaded() {
  if (typeof require === 'function') {
    try {
      const path = require('path');
      const fs = require('fs');
      let demoDir = path.join(__dirname, 'runtime');
      if (!fs.existsSync(path.join(demoDir, 'cuda-runtime.js'))) {
        demoDir = __dirname;
      }
      if (!fs.existsSync(path.join(demoDir, 'cuda-runtime.js'))) {
        demoDir = path.join(path.resolve(__dirname, '..', '..'), 'demo');
      }
      if (!g.GPU_PROFILES) {
        try {
          const mod = require(path.join(demoDir, 'gpu-profiles.js'));
          g.GPU_PROFILES = (mod && mod.GPU_PROFILES) ? mod.GPU_PROFILES : (mod || g.GPU_PROFILES);
        } catch (e) {}
      }
      if (!g.GpuboxFP16) {
        try {
          const mod = require(path.join(demoDir, 'fp16-bf16.js'));
          g.GpuboxFP16 = (mod && mod.GpuboxFP16) ? mod.GpuboxFP16 : (mod || g.GpuboxFP16);
        } catch (e) {}
      }
      if (!g.GpuboxRaceDetector) {
        try {
          const mod = require(path.join(demoDir, 'race-detector.js'));
          g.GpuboxRaceDetector = (mod && mod.GpuboxRaceDetector) ? mod.GpuboxRaceDetector : (mod || g.GpuboxRaceDetector);
        } catch (e) {}
      }
      if (!g.GpuboxSparseMemory) {
        try {
          const mod = require(path.join(demoDir, 'sparse-memory.js'));
          g.GpuboxSparseMemory = (mod && mod.GpuboxSparseMemory) ? mod.GpuboxSparseMemory : (mod || g.GpuboxSparseMemory);
        } catch (e) {}
      }
      if (!g.GpuboxClusterTopology) {
        try {
          const mod = require(path.join(demoDir, 'topology-model.js'));
          g.GpuboxClusterTopology = (mod && mod.GpuboxClusterTopology) ? mod.GpuboxClusterTopology : (mod || g.GpuboxClusterTopology);
        } catch (e) {}
      }
      if (!g.GpuboxDesyncDetector) {
        try {
          const mod = require(path.join(demoDir, 'desync-detector.js'));
          g.GpuboxDesyncDetector = (mod && mod.GpuboxDesyncDetector) ? mod.GpuboxDesyncDetector : (mod || g.GpuboxDesyncDetector);
        } catch (e) {}
      }
      if (!g.CudaTranspiler) {
        try {
          const mod = require(path.join(demoDir, 'cuda-transpiler.js'));
          g.CudaTranspiler = (mod && mod.CudaTranspiler) ? mod.CudaTranspiler : (mod || g.CudaTranspiler);
        } catch (e) {}
      }
      if (!g.CudaRuntime) {
        try {
          const mod = require(path.join(demoDir, 'cuda-runtime.js'));
          g.CudaRuntime = (mod && mod.CudaRuntime) ? mod.CudaRuntime : (mod || g.CudaRuntime);
        } catch (e) {}
      }
      if (!g.GpuboxValidation) {
        try {
          const mod = require(path.join(demoDir, 'validation-harness.js'));
          g.GpuboxValidation = (mod && mod.GpuboxValidation) ? mod.GpuboxValidation : (mod || g.GpuboxValidation);
        } catch (e) {}
      }
    } catch (e) {}
  }
}

ensureRuntimeLoaded();

// Profiles Registry
export function listProfiles(): string[] {
  ensureRuntimeLoaded();
  if (g.GPU_PROFILES && typeof g.GPU_PROFILES.listProfiles === 'function') {
    return g.GPU_PROFILES.listProfiles();
  }
  return ['A100', 'H100', 'RTX4090', 'RTX4050_Laptop', 'RTX3090', 'V100'];
}

export function getProfile(name: string): GpuProfileSpec {
  ensureRuntimeLoaded();
  if (g.GPU_PROFILES && typeof g.GPU_PROFILES.getProfile === 'function') {
    return g.GPU_PROFILES.getProfile(name);
  }
  return {
    name,
    architecture: 'Ampere',
    computeCapability: '8.0',
    clockSpeedMHz: 1410,
    peakTFLOPS: 19.5,
    peakGFLOPS: 19500,
    memBandwidthGBs: 2039,
    smCount: 108,
    maxThreadsPerSM: 2048,
    globalMemLatencyCycles: 400,
    sharedMemLatencyCycles: 28,
    atomicLatencyCycles: 50,
    syncLatencyCycles: 16,
    flopLatencyCycles: 1,
    maxSharedMemPerBlock: 167936,
    maxRegistersPerBlock: 65536
  };
}

export function registerProfile(name: string, spec: GpuProfileSpec): void {
  ensureRuntimeLoaded();
  if (g.GPU_PROFILES && typeof g.GPU_PROFILES.registerProfile === 'function') {
    g.GPU_PROFILES.registerProfile(name, spec);
  }
}

// Topologies Registry
export function listTopologies(): string[] {
  ensureRuntimeLoaded();
  if (g.GpuboxClusterTopology && typeof g.GpuboxClusterTopology.listTopologies === 'function') {
    return g.GpuboxClusterTopology.listTopologies();
  }
  return ['nvswitch_mesh', 'pcie_tree', 'multi_node', 'custom'];
}

export function getTopology(name: string): TopologySpec {
  ensureRuntimeLoaded();
  if (g.GpuboxClusterTopology && typeof g.GpuboxClusterTopology.getTopology === 'function') {
    return g.GpuboxClusterTopology.getTopology(name);
  }
  return {
    name,
    type: 'nvswitch_mesh',
    nodeCount: 1,
    gpusPerNode: 16,
    interGpuBandwidthGBs: 600,
    interNodeBandwidthGBs: 100,
    latencyUs: 0.5
  };
}

export function registerTopology(name: string, spec: TopologySpec): void {
  ensureRuntimeLoaded();
  if (g.GpuboxClusterTopology && typeof g.GpuboxClusterTopology.registerTopology === 'function') {
    g.GpuboxClusterTopology.registerTopology(name, spec);
  }
}

// Capabilities
export function getCapabilities(engine: 'js' | 'wasm' = 'js'): EngineCapabilities {
  ensureRuntimeLoaded();
  if (typeof require === 'function') {
    try {
      const path = require('path');
      const fs = require('fs');
      let compatPath = path.join(__dirname, 'compat.json');
      if (!fs.existsSync(compatPath)) {
        compatPath = path.join(__dirname, '..', 'compat.json');
      }
      if (!fs.existsSync(compatPath)) {
        compatPath = path.resolve(__dirname, '..', '..', 'compat.json');
      }
      if (fs.existsSync(compatPath)) {
        const data = require(compatPath);
        if (data && data.engines && data.engines[engine]) {
          return data.engines[engine];
        }
      }
    } catch (e) {}
  }
  return {
    name: engine === 'wasm' ? 'WebAssembly C++ Analytical Engine' : 'JavaScript SIMT Engine',
    status: 'stable',
    supportsSourceCompilation: engine === 'js',
    maxClusterDevices: 16,
    features: {
      kernelLaunch: { supported: true },
      sharedMemory: { supported: true },
      syncthreads: { supported: true },
      atomics: { supported: true },
      multiGpuCollectives: { supported: true },
      dynamicParallelism: { supported: false, reason: 'Nested kernel launch is unsupported' },
      inlinePtx: { supported: false, reason: 'Inline PTX assembly is unsupported' },
      wmmaTensorCores: { supported: false, reason: 'WMMA tensor cores are unsupported' }
    }
  };
}

// Validation helper
export function validateOutput(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  options?: ValidationOptions
): ValidationResult {
  ensureRuntimeLoaded();
  if (g.GpuboxValidation && typeof g.GpuboxValidation.validateOutput === 'function') {
    return g.GpuboxValidation.validateOutput(actual, expected, options);
  }
  const mode = options?.mode || 'abs';
  const tol = options?.tolerance ?? 1e-5;
  const len = Math.min(actual.length, expected.length);
  for (let i = 0; i < len; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tol) {
      return {
        verdict: 'FAIL',
        totalElements: actual.length,
        mismatchCount: 1,
        firstMismatch: {
          index: i,
          expected: expected[i],
          actual: actual[i],
          absError: diff,
          relError: expected[i] !== 0 ? diff / Math.abs(expected[i]) : diff
        }
      };
    }
  }
  return { verdict: 'PASS', totalElements: actual.length, mismatchCount: 0, firstMismatch: null };
}

// Static Analysis and Compilation
export function analyze(source: string): ProgramAnalysis {
  ensureRuntimeLoaded();
  const unsupported: Array<{ feature: string; reason: string }> = [];
  const diagnostics: SimDiagnostic[] = [];

  if (/\b(?:__)?asm(?:__)?\s*(?:volatile)?\s*\(/i.test(source) || /\b(?:__)?asm(?:__)?\s*\{/i.test(source)) {
    unsupported.push({
      feature: 'inline_ptx',
      reason: 'Inline PTX assembly (__asm__) requires physical NVIDIA SASS assembler and is rejected'
    });
    diagnostics.push({
      category: 'UNSUPPORTED_FEATURE',
      type: 'inline_ptx',
      message: 'Unsupported CUDA feature: Inline PTX assembly (__asm__) is not supported in client-side simulator'
    });
  }

  if (/\b(?:nvcuda::)?wmma\b/.test(source)) {
    unsupported.push({
      feature: 'wmma',
      reason: 'WMMA tensor core intrinsics (nvcuda::wmma) require hardware tensor cores and are rejected'
    });
    diagnostics.push({
      category: 'UNSUPPORTED_FEATURE',
      type: 'wmma',
      message: 'Unsupported CUDA feature: Tensor Core WMMA API (nvcuda::wmma) is not supported in client-side simulator'
    });
  }

  if (/<<<[\s\S]*?>>>/.test(source)) {
    const kernelMatch = source.match(/__global__[\s\S]*?\{([\s\S]*?)\}/);
    if (kernelMatch && /<<<[\s\S]*?>>>/.test(kernelMatch[1])) {
      unsupported.push({
        feature: 'dynamic_parallelism',
        reason: 'Nested kernel launches (dynamic parallelism) are not supported in client-side environment'
      });
      diagnostics.push({
        category: 'UNSUPPORTED_FEATURE',
        type: 'dynamic_parallelism',
        message: 'Unsupported CUDA feature: Dynamic Parallelism (nested kernel launch <<<...>>>) inside kernel'
      });
    }
  }

  const kernelNames: string[] = [];
  const kernelRegex = /__global__\s+void\s+(\w+)\s*\(/g;
  let match;
  while ((match = kernelRegex.exec(source)) !== null) {
    kernelNames.push(match[1]);
  }

  return {
    success: unsupported.length === 0,
    kernels: kernelNames,
    diagnostics,
    unsupportedFeatures: unsupported
  };
}

export function compile(source: string): Program {
  ensureRuntimeLoaded();
  const analysis = analyze(source);
  if (!analysis.success) {
    return {
      source,
      transpiledCode: '',
      kernelNames: analysis.kernels,
      analysis
    };
  }

  let code = '';
  let kernelNames: string[] = [];
  if (g.CudaTranspiler && typeof g.CudaTranspiler.transpile === 'function') {
    const tr = g.CudaTranspiler.transpile(source);
    code = tr.code;
    kernelNames = tr.kernelNames || analysis.kernels;
  }

  return {
    source,
    transpiledCode: code,
    kernelNames,
    analysis
  };
}

// Simulator Implementation
export class GpuboxSimulator {
  private config: SimConfig;
  private runtime: any;
  private listeners: Map<string, Array<(...args: any[]) => void>> = new Map();
  private traces: TraceEvent[] = [];
  private traceIdCounter: number = 0;

  constructor(config: SimConfig = {}) {
    ensureRuntimeLoaded();
    this.config = {
      gpuProfile: config.gpuProfile || 'A100',
      numDevices: Math.max(1, Math.min(16, config.numDevices || 1)),
      topology: config.topology || 'nvswitch_mesh',
      limits: config.limits || {},
      raceDetection: config.raceDetection !== undefined ? config.raceDetection : true,
      traceLevel: config.traceLevel || 'off',
      seed: config.seed || 1337,
      engine: config.engine || 'js'
    };

    const profile = typeof this.config.gpuProfile === 'string'
      ? getProfile(this.config.gpuProfile)
      : this.config.gpuProfile;

    const RuntimeClass = g.CudaRuntime;
    if (!RuntimeClass) {
      throw new SimError({
        code: 'RUNTIME_INIT_FAILED',
        message: 'Could not instantiate Gpubox runtime core'
      });
    }

    this.runtime = new RuntimeClass(profile);
    if (this.config.numDevices && this.config.numDevices > 1) {
      this.runtime.initMultiGPU(this.config.numDevices);
    }
  }

  on(eventName: string, callback: (...args: any[]) => void) {
    if (!this.listeners.has(eventName)) this.listeners.set(eventName, []);
    this.listeners.get(eventName)!.push(callback);
    return this;
  }

  off(eventName: string, callback: (...args: any[]) => void) {
    const arr = this.listeners.get(eventName);
    if (arr) {
      const idx = arr.indexOf(callback);
      if (idx !== -1) arr.splice(idx, 1);
    }
    return this;
  }

  emit(eventName: string, payload: any) {
    if (this.config.traceLevel !== 'off') {
      const ev: TraceEvent = {
        id: ++this.traceIdCounter,
        timestampMs: Date.now(),
        type: eventName as any,
        payload
      };
      this.traces.push(ev);
    }
    const arr = this.listeners.get(eventName);
    if (arr) {
      for (const cb of arr) {
        try { cb(payload); } catch (e) {}
      }
    }
  }

  getTrace(): TraceEvent[] {
    return [...this.traces];
  }

  clearTrace(): void {
    this.traces = [];
  }

  compile(source: string): Program {
    return compile(source);
  }

  analyze(source: string): ProgramAnalysis {
    return analyze(source);
  }

  // Programmatic Driver API
  allocate(sizeBytes: number, deviceId: number = 0) {
    this.runtime.cudaSetDevice(deviceId);
    const ptr = this.runtime.cudaMalloc(sizeBytes);
    const ptrAddr = (ptr && typeof ptr === 'object' && ptr.address !== undefined) ? ptr.address : (typeof ptr === 'number' ? ptr : 0);
    this.emit('alloc', { deviceId, sizeBytes, ptr: ptrAddr });
    return ptr;
  }

  free(ptr: any, deviceId: number = 0) {
    this.runtime.cudaSetDevice(deviceId);
    const res = this.runtime.cudaFree(ptr);
    const ptrAddr = (ptr && typeof ptr === 'object' && ptr.address !== undefined) ? ptr.address : (typeof ptr === 'number' ? ptr : 0);
    this.emit('free', { deviceId, ptr: ptrAddr });
    return res;
  }

  memcpy(dst: any, src: any, sizeBytes: number, kind: string = 'H2D') {
    const res = this.runtime.cudaMemcpy(dst, src, sizeBytes, kind);
    this.emit('memcpy', { kind, sizeBytes });
    return res;
  }

  memcpyPeer(dst: any, dstDev: number, src: any, srcDev: number, sizeBytes: number) {
    const res = this.runtime.cudaMemcpyPeer(dst, dstDev, src, srcDev, sizeBytes);
    this.emit('p2p', { srcDev, dstDev, sizeBytes });
    return res;
  }

  cudaMalloc(sizeBytes: number, deviceId: number = 0) {
    return this.allocate(sizeBytes, deviceId);
  }

  cudaFree(ptr: any, deviceId: number = 0) {
    return this.free(ptr, deviceId);
  }

  cudaMemcpy(dst: any, src: any, sizeBytes: number, kind: string = 'H2D') {
    return this.memcpy(dst, src, sizeBytes, kind);
  }

  cudaMemcpyPeer(dst: any, dstDev: number, src: any, srcDev: number, sizeBytes: number) {
    return this.memcpyPeer(dst, dstDev, src, srcDev, sizeBytes);
  }

  cudaSetDevice(deviceId: number) {
    return this.runtime.cudaSetDevice(deviceId);
  }

  cudaGetDevice() {
    return this.runtime.cudaGetDevice();
  }

  cudaDeviceEnablePeerAccess(peerId: number) {
    return this.runtime.enablePeerAccess ? this.runtime.enablePeerAccess(peerId) : 0;
  }

  cudaDeviceDisablePeerAccess(peerId: number) {
    return this.runtime.cudaDeviceDisablePeerAccess ? this.runtime.cudaDeviceDisablePeerAccess(peerId) : 0;
  }

  launch(kernelFn: any, gridDim: any, blockDim: any, sharedMem: number = 0, ...args: any[]) {
    this.emit('kernel_start', { gridDim, blockDim, sharedMem });
    const res = this.runtime.launch(kernelFn, gridDim, blockDim, sharedMem, ...args);
    this.emit('kernel_end', { gridDim, blockDim });
    return res;
  }

  // Collective API
  ncclCommInitRank(nranks: number, rank: number, options?: any) {
    return this.runtime.ncclCommInitRank(nranks, rank, options);
  }

  ncclCommInitAll(comms: any[], nranks: number, options?: any) {
    return this.runtime.ncclCommInitAll(comms, nranks, options);
  }

  allReduce(sendbufs: any, recvbufs: any, count: number, datatype: string = 'float32', op: string = 'sum', comm?: any) {
    this.emit('collective', { op: 'allreduce', count, reduceOp: op });
    return this.runtime.ncclAllReduce(sendbufs, recvbufs, count, datatype, op, comm);
  }

  broadcast(sendbufs: any, recvbufs: any, count: number, datatype: string = 'float32', root: number = 0, comm?: any) {
    this.emit('collective', { op: 'broadcast', count, root });
    return this.runtime.ncclBroadcast(sendbufs, recvbufs, count, datatype, root, comm);
  }

  reduceScatter(sendbufs: any, recvbufs: any, recvcount: number, datatype: string = 'float32', op: string = 'sum', comm?: any) {
    this.emit('collective', { op: 'reducescatter', count: recvcount, reduceOp: op });
    return this.runtime.ncclReduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm);
  }

  allGather(sendbufs: any, recvbufs: any, sendcount: number, datatype: string = 'float32', comm?: any) {
    this.emit('collective', { op: 'allgather', count: sendcount });
    return this.runtime.ncclAllGather(sendbufs, recvbufs, sendcount, datatype, comm);
  }

  reduce(sendbufs: any, recvbufs: any, count: number, datatype: string = 'float32', op: string = 'sum', root: number = 0, comm?: any) {
    this.emit('collective', { op: 'reduce', count, reduceOp: op, root });
    return this.runtime.ncclReduce(sendbufs, recvbufs, count, datatype, op, root, comm);
  }

  allToAll(sendbufs: any, recvbufs: any, countPerRank: number, datatype: string = 'float32', comm?: any) {
    this.emit('collective', { op: 'alltoall', countPerRank });
    return this.runtime.ncclAllToAll(sendbufs, recvbufs, countPerRank, datatype, comm);
  }

  send(sendbuf: any, count: number, datatype: string = 'float32', peer: number = 0, comm?: any) {
    this.emit('p2p', { type: 'send', peer, count });
    return this.runtime.ncclSend(sendbuf, count, datatype, peer, comm);
  }

  recv(recvbuf: any, count: number, datatype: string = 'float32', peer: number = 0, comm?: any) {
    this.emit('p2p', { type: 'recv', peer, count });
    return this.runtime.ncclRecv(recvbuf, count, datatype, peer, comm);
  }

  groupStart() {
    return this.runtime.ncclGroupStart();
  }

  groupEnd() {
    return this.runtime.ncclGroupEnd();
  }

  ncclAllReduce(sendbufs: any, recvbufs: any, count: number, datatype: string = 'float32', op: string = 'sum', comm?: any) {
    return this.allReduce(sendbufs, recvbufs, count, datatype, op, comm);
  }

  ncclBroadcast(sendbufs: any, recvbufs: any, count: number, datatype: string = 'float32', root: number = 0, comm?: any) {
    return this.broadcast(sendbufs, recvbufs, count, datatype, root, comm);
  }

  ncclReduceScatter(sendbufs: any, recvbufs: any, recvcount: number, datatype: string = 'float32', op: string = 'sum', comm?: any) {
    return this.reduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm);
  }

  ncclAllGather(sendbufs: any, recvbufs: any, sendcount: number, datatype: string = 'float32', comm?: any) {
    return this.allGather(sendbufs, recvbufs, sendcount, datatype, comm);
  }

  ncclReduce(sendbufs: any, recvbufs: any, count: number, datatype: string = 'float32', op: string = 'sum', root: number = 0, comm?: any) {
    return this.reduce(sendbufs, recvbufs, count, datatype, op, root, comm);
  }

  ncclAllToAll(sendbufs: any, recvbufs: any, countPerRank: number, datatype: string = 'float32', comm?: any) {
    return this.allToAll(sendbufs, recvbufs, countPerRank, datatype, comm);
  }

  ncclSend(sendbuf: any, count: number, datatype: string = 'float32', peer: number = 0, comm?: any) {
    return this.send(sendbuf, count, datatype, peer, comm);
  }

  ncclRecv(recvbuf: any, count: number, datatype: string = 'float32', peer: number = 0, comm?: any) {
    return this.recv(recvbuf, count, datatype, peer, comm);
  }

  ncclGroupStart() {
    return this.groupStart();
  }

  ncclGroupEnd() {
    return this.groupEnd();
  }

  getCounters(deviceId?: number) {
    return this.runtime.getCounters(deviceId);
  }

  getAggregatedCounters() {
    return this.runtime.getAggregatedCounters ? this.runtime.getAggregatedCounters() : this.runtime.getCounters();
  }

  getDiagnostics(): SimDiagnostic[] {
    const raw = this.runtime.getDiagnostics ? this.runtime.getDiagnostics() : [];
    return raw.map((d: any) => ({
      category: d.type === 'race' ? 'RACE_CONDITION'
              : d.type === 'oob' ? 'OOB_MEMORY_ACCESS'
              : d.type === 'barrier_divergence' ? 'BARRIER_DIVERGENCE'
              : 'RUNTIME_ERROR',
      type: d.type || 'error',
      message: d.message || String(d),
      location: {
        kernel: d.kernel,
        block: d.block,
        thread: d.thread,
        rank: d.deviceId
      }
    }));
  }

  getLeakReport(): LeakReport {
    return this.runtime.getLeakReport();
  }

  reset() {
    this.runtime.reset();
    this.clearTrace();
  }

  run(programOrSource: Program | string, options: any = {}): RunResult {
    const program = typeof programOrSource === 'string' ? compile(programOrSource) : programOrSource;

    // Check static analysis first
    if (!program.analysis.success) {
      return {
        schemaVersion: '1.0.0',
        engineVersion: '1.0.0',
        status: 'UNSUPPORTED',
        stdout: [],
        outputs: {},
        counters: this.runtime.getCounters().toJSON(),
        diagnostics: program.analysis.diagnostics,
        collectives: [],
        trafficMatrix: Array.from({ length: this.config.numDevices || 1 }, () => new Array(this.config.numDevices || 1).fill(0)),
        estimatedTimeMs: 0,
        leakReport: this.getLeakReport(),
        ...(this.config.traceLevel !== 'off' ? { trace: this.getTrace() } : {})
      };
    }

    try {
      const runner = new Function('__rt', program.transpiledCode);
      runner(this.runtime);

      const counters = (this.runtime.getAggregatedCounters ? this.runtime.getAggregatedCounters() : this.runtime.getCounters()).toJSON();
      const diagnostics = this.getDiagnostics();
      const nGpus = this.config.numDevices || 1;
      const trafficMatrix = Array.from({ length: nGpus }, () => new Array(nGpus).fill(0));

      let validation: ValidationResult | undefined = undefined;
      if (options.expected && options.actual) {
        validation = validateOutput(options.actual, options.expected, options.validationOptions);
      }

      const hasErrorDiags = diagnostics.some(d => d.category === 'RACE_CONDITION' || d.category === 'OOB_MEMORY_ACCESS');
      const status: StatusVerdict = validation && validation.verdict === 'FAIL' ? 'FAIL'
                                 : hasErrorDiags ? 'FAIL'
                                 : 'PASS';

      return {
        schemaVersion: '1.0.0',
        engineVersion: '1.0.0',
        status,
        stdout: [],
        outputs: options.outputs || {},
        counters,
        diagnostics,
        collectives: [],
        trafficMatrix,
        estimatedTimeMs: counters.estimatedTimeMs || 0,
        leakReport: this.getLeakReport(),
        ...(validation ? { validation } : {}),
        ...(this.config.traceLevel !== 'off' ? { trace: this.getTrace() } : {})
      };
    } catch (err: any) {
      return {
        schemaVersion: '1.0.0',
        engineVersion: '1.0.0',
        status: 'ERROR',
        stdout: [],
        outputs: {},
        counters: this.runtime.getCounters().toJSON(),
        diagnostics: [{
          category: 'RUNTIME_ERROR',
          type: err.name || 'Error',
          message: err.message
        }],
        collectives: [],
        trafficMatrix: Array.from({ length: this.config.numDevices || 1 }, () => new Array(this.config.numDevices || 1).fill(0)),
        estimatedTimeMs: 0,
        leakReport: this.getLeakReport(),
        ...(this.config.traceLevel !== 'off' ? { trace: this.getTrace() } : {})
      };
    }
  }
}

// Export Engine alias for GpuboxSimulator
export const Engine = GpuboxSimulator;

// Factory function
export function createSimulator(config?: SimConfig): GpuboxSimulator {
  return new GpuboxSimulator(config);
}

// Top-level one-shot run execution
export function run(programOrSource: Program | string, options: any = {}): RunResult {
  const sim = createSimulator(options);
  return sim.run(programOrSource, options);
}

// Web Worker simulator wrapper
export function createWorkerSimulator(config?: SimConfig) {
  // In environments without Web Worker (e.g. Node), falls back to direct simulation
  const sim = createSimulator(config);
  return {
    run: async (programOrSource: Program | string, options?: any) => {
      return sim.run(programOrSource, options);
    },
    compile: (source: string) => sim.compile(source),
    analyze: (source: string) => sim.analyze(source),
    reset: () => sim.reset(),
    terminate: () => sim.reset()
  };
}

export default {
  createSimulator,
  compile,
  analyze,
  run,
  listProfiles,
  getProfile,
  registerProfile,
  listTopologies,
  getTopology,
  registerTopology,
  getCapabilities,
  validateOutput,
  createWorkerSimulator,
  SimError,
  GpuboxSimulator,
  Engine
};
