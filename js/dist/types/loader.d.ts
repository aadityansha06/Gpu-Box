/**
 * loader.ts
 * ─────────
 * Lazy WASM loader for the cuda-sim module.
 * Caches the loaded module instance so it's only initialized once.
 *
 * The WASM file is loaded from:
 *   1. A bundled URL (when used as npm package with bundler)
 *   2. jsDelivr CDN URL (when used via <script> tag)
 *   3. Local path (dev)
 */
interface CudaSimWasmModule {
    simReset(): void;
    simGetCounters(): string;
    simGetDiagnostics(): string;
    simRewriteSource(source: string): string;
    simHasErrors(): boolean;
    simResultToJson(passed: boolean): string;
    simGetAllReduceModel(nGpus: number, countFloats: number): string;
}
/** Load and initialize the WASM module. Called once; subsequent calls return the cache. */
export declare function loadWasm(wasmUrl?: string): Promise<CudaSimWasmModule>;
export {};
