/**
 * types.ts
 * ────────
 * Complete public TypeScript interfaces and type definitions for Gpubox.
 *
 * Copyright (c) 2026 Aadityansha Verma. All rights reserved.
 */
export type StatusVerdict = 'PASS' | 'FAIL' | 'UNSUPPORTED' | 'ERROR';
export type DiagnosticCategory = 'OOB_MEMORY_ACCESS' | 'RACE_CONDITION' | 'BARRIER_DIVERGENCE' | 'NCCL_DESYNC' | 'UNSUPPORTED_FEATURE' | 'EXECUTION_TIMEOUT' | 'INVALID_CONFIGURATION' | 'RUNTIME_ERROR';
export interface DiagnosticLocation {
    kernel?: string;
    block?: {
        x: number;
        y: number;
        z: number;
    };
    thread?: {
        x: number;
        y: number;
        z: number;
    };
    line?: number;
    rank?: number;
}
export interface SimDiagnostic {
    category: DiagnosticCategory;
    type: string;
    message: string;
    location?: DiagnosticLocation;
    docsUrl?: string;
}
export interface SimCountersReport {
    threadsLaunched: number;
    blocksLaunched: number;
    globalLoads: number;
    globalStores: number;
    globalLoadBytes: number;
    globalStoreBytes: number;
    globalTransactions: number;
    sharedLoads: number;
    sharedStores: number;
    sharedLoadBytes: number;
    sharedStoreBytes: number;
    bankConflicts: number;
    syncthreadsCount: number;
    divergentBranches: number;
    atomicOps: number;
    atomicMaxContention: number;
    memcpyH2DBytes: number;
    memcpyD2HBytes: number;
    crossGpuBytes: number;
    ncclOps: number;
    flopCount: number;
    estimatedCycles: number;
    estimatedTimeMs: number;
    perDevice?: Array<SimCountersReport & {
        deviceId: number;
    }>;
}
export interface CollectiveRecord {
    sequence: number;
    op: string;
    ranks: number;
    bytes: number;
    modeledTimeMs: number;
}
export interface LeakAllocation {
    deviceId?: number;
    allocId: number;
    sizeBytes: number;
    offset: number;
}
export interface LeakReport {
    totalAllocations: number;
    totalLeakedBytes: number;
    leaks: LeakAllocation[];
}
export interface TraceEvent {
    id: number;
    timestampMs: number;
    type: 'kernel_start' | 'kernel_end' | 'barrier' | 'divergence' | 'alloc' | 'free' | 'memcpy' | 'collective' | 'p2p' | 'diagnostic' | 'printf';
    deviceId?: number;
    payload: any;
}
export interface ValidationMismatch {
    index: number;
    expected: number;
    actual: number;
    absError: number;
    relError: number;
    ulpDiff?: number;
}
export interface ValidationResult {
    verdict: 'PASS' | 'FAIL';
    totalElements: number;
    mismatchCount: number;
    firstMismatch: ValidationMismatch | null;
}
export interface ValidationOptions {
    mode?: 'abs' | 'relative' | 'ulp';
    tolerance?: number;
    absTol?: number;
    relTol?: number;
    maxUlps?: number;
}
export interface ProgramAnalysis {
    success: boolean;
    kernels: string[];
    diagnostics: SimDiagnostic[];
    unsupportedFeatures: Array<{
        feature: string;
        reason: string;
    }>;
}
export interface Program {
    source: string;
    transpiledCode: string;
    kernelNames: string[];
    analysis: ProgramAnalysis;
}
export interface RunResult {
    schemaVersion: '1.0.0';
    engineVersion: string;
    status: StatusVerdict;
    stdout: string[];
    outputs: Record<string, Float32Array | Uint8Array | number[]>;
    counters: SimCountersReport;
    diagnostics: SimDiagnostic[];
    collectives: CollectiveRecord[];
    trafficMatrix: number[][];
    estimatedTimeMs: number;
    leakReport: LeakReport;
    trace?: TraceEvent[];
    validation?: ValidationResult;
}
export interface GpuProfileSpec {
    name: string;
    architecture: string;
    computeCapability: string;
    clockSpeedMHz: number;
    peakTFLOPS: number;
    peakGFLOPS: number;
    memBandwidthGBs: number;
    memoryBandwidthGBs?: number;
    vramBytes?: number;
    smCount: number;
    maxThreadsPerSM: number;
    globalMemLatencyCycles: number;
    sharedMemLatencyCycles: number;
    atomicLatencyCycles: number;
    syncLatencyCycles: number;
    flopLatencyCycles: number;
    maxSharedMemPerBlock: number;
    maxRegistersPerBlock: number;
    status?: string;
}
export interface TopologySpec {
    name: string;
    type: 'nvswitch_mesh' | 'pcie_tree' | 'multi_node' | 'custom';
    nodeCount: number;
    gpusPerNode: number;
    interGpuBandwidthGBs: number;
    interNodeBandwidthGBs: number;
    latencyUs: number;
}
export interface SimulatorLimits {
    maxSteps?: number;
    maxTimeMs?: number;
    maxHeapBytes?: number;
}
export interface SimConfig {
    gpuProfile?: string | GpuProfileSpec;
    numDevices?: number;
    topology?: string | TopologySpec;
    limits?: SimulatorLimits;
    raceDetection?: boolean | {
        enabled: boolean;
        strict: boolean;
    };
    traceLevel?: 'off' | 'summary' | 'events' | 'full';
    seed?: number;
    engine?: 'js' | 'wasm';
}
export interface FeatureCapability {
    supported: boolean;
    notes?: string;
    reason?: string;
}
export interface EngineCapabilities {
    name: string;
    status: string;
    supportsSourceCompilation: boolean;
    maxClusterDevices: number;
    features: Record<string, FeatureCapability>;
}
