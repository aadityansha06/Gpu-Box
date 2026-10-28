/**
 * index.ts
 * ────────
 * Main public entry point for Gpubox.
 *
 * Copyright (c) 2026 Aadityansha Verma. All rights reserved.
 */
import type { SimConfig, Program, ProgramAnalysis, RunResult, SimDiagnostic, GpuProfileSpec, TopologySpec, EngineCapabilities, ValidationOptions, ValidationResult, TraceEvent, LeakReport } from './types';
export * from './types';
export declare class SimError extends Error {
    code: string;
    category: string;
    location?: {
        kernel?: string;
        block?: any;
        thread?: any;
        line?: number;
        rank?: number;
    };
    docsUrl?: string;
    constructor(options: {
        code: string;
        category?: string;
        message: string;
        location?: {
            kernel?: string;
            block?: any;
            thread?: any;
            line?: number;
            rank?: number;
        };
        docsUrl?: string;
    });
}
export declare function listProfiles(): string[];
export declare function getProfile(name: string): GpuProfileSpec;
export declare function registerProfile(name: string, spec: GpuProfileSpec): void;
export declare function listTopologies(): string[];
export declare function getTopology(name: string): TopologySpec;
export declare function registerTopology(name: string, spec: TopologySpec): void;
export declare function getCapabilities(engine?: 'js' | 'wasm'): EngineCapabilities;
export declare function validateOutput(actual: ArrayLike<number>, expected: ArrayLike<number>, options?: ValidationOptions): ValidationResult;
export declare function analyze(source: string): ProgramAnalysis;
export declare function compile(source: string): Program;
export declare class GpuboxSimulator {
    private config;
    private runtime;
    private listeners;
    private traces;
    private traceIdCounter;
    constructor(config?: SimConfig);
    on(eventName: string, callback: (...args: any[]) => void): this;
    off(eventName: string, callback: (...args: any[]) => void): this;
    emit(eventName: string, payload: any): void;
    getTrace(): TraceEvent[];
    clearTrace(): void;
    compile(source: string): Program;
    analyze(source: string): ProgramAnalysis;
    allocate(sizeBytes: number, deviceId?: number): any;
    free(ptr: any, deviceId?: number): any;
    memcpy(dst: any, src: any, sizeBytes: number, kind?: string): any;
    memcpyPeer(dst: any, dstDev: number, src: any, srcDev: number, sizeBytes: number): any;
    cudaMalloc(sizeBytes: number, deviceId?: number): any;
    cudaFree(ptr: any, deviceId?: number): any;
    cudaMemcpy(dst: any, src: any, sizeBytes: number, kind?: string): any;
    cudaMemcpyPeer(dst: any, dstDev: number, src: any, srcDev: number, sizeBytes: number): any;
    cudaSetDevice(deviceId: number): any;
    cudaGetDevice(): any;
    cudaDeviceEnablePeerAccess(peerId: number): any;
    cudaDeviceDisablePeerAccess(peerId: number): any;
    launch(kernelFn: any, gridDim: any, blockDim: any, sharedMem?: number, ...args: any[]): any;
    ncclCommInitRank(nranks: number, rank: number, options?: any): any;
    ncclCommInitAll(comms: any[], nranks: number, options?: any): any;
    allReduce(sendbufs: any, recvbufs: any, count: number, datatype?: string, op?: string, comm?: any): any;
    broadcast(sendbufs: any, recvbufs: any, count: number, datatype?: string, root?: number, comm?: any): any;
    reduceScatter(sendbufs: any, recvbufs: any, recvcount: number, datatype?: string, op?: string, comm?: any): any;
    allGather(sendbufs: any, recvbufs: any, sendcount: number, datatype?: string, comm?: any): any;
    reduce(sendbufs: any, recvbufs: any, count: number, datatype?: string, op?: string, root?: number, comm?: any): any;
    allToAll(sendbufs: any, recvbufs: any, countPerRank: number, datatype?: string, comm?: any): any;
    send(sendbuf: any, count: number, datatype?: string, peer?: number, comm?: any): any;
    recv(recvbuf: any, count: number, datatype?: string, peer?: number, comm?: any): any;
    groupStart(): any;
    groupEnd(): any;
    ncclAllReduce(sendbufs: any, recvbufs: any, count: number, datatype?: string, op?: string, comm?: any): any;
    ncclBroadcast(sendbufs: any, recvbufs: any, count: number, datatype?: string, root?: number, comm?: any): any;
    ncclReduceScatter(sendbufs: any, recvbufs: any, recvcount: number, datatype?: string, op?: string, comm?: any): any;
    ncclAllGather(sendbufs: any, recvbufs: any, sendcount: number, datatype?: string, comm?: any): any;
    ncclReduce(sendbufs: any, recvbufs: any, count: number, datatype?: string, op?: string, root?: number, comm?: any): any;
    ncclAllToAll(sendbufs: any, recvbufs: any, countPerRank: number, datatype?: string, comm?: any): any;
    ncclSend(sendbuf: any, count: number, datatype?: string, peer?: number, comm?: any): any;
    ncclRecv(recvbuf: any, count: number, datatype?: string, peer?: number, comm?: any): any;
    ncclGroupStart(): any;
    ncclGroupEnd(): any;
    getCounters(deviceId?: number): any;
    getAggregatedCounters(): any;
    getDiagnostics(): SimDiagnostic[];
    getLeakReport(): LeakReport;
    reset(): void;
    run(programOrSource: Program | string, options?: any): RunResult;
}
export declare const Engine: typeof GpuboxSimulator;
export declare function createSimulator(config?: SimConfig): GpuboxSimulator;
export declare function run(programOrSource: Program | string, options?: any): RunResult;
export declare function createWorkerSimulator(config?: SimConfig): {
    run: (programOrSource: Program | string, options?: any) => Promise<RunResult>;
    compile: (source: string) => Program;
    analyze: (source: string) => ProgramAnalysis;
    reset: () => void;
    terminate: () => void;
};
declare const _default: {
    createSimulator: typeof createSimulator;
    compile: typeof compile;
    analyze: typeof analyze;
    run: typeof run;
    listProfiles: typeof listProfiles;
    getProfile: typeof getProfile;
    registerProfile: typeof registerProfile;
    listTopologies: typeof listTopologies;
    getTopology: typeof getTopology;
    registerTopology: typeof registerTopology;
    getCapabilities: typeof getCapabilities;
    validateOutput: typeof validateOutput;
    createWorkerSimulator: typeof createWorkerSimulator;
    SimError: typeof SimError;
    GpuboxSimulator: typeof GpuboxSimulator;
    Engine: typeof GpuboxSimulator;
};
export default _default;
