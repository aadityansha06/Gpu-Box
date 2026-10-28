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

// The Emscripten-generated module type (partial)
interface CudaSimWasmModule {
  simReset(): void;
  simGetCounters(): string;
  simGetDiagnostics(): string;
  simRewriteSource(source: string): string;
  simHasErrors(): boolean;
  simResultToJson(passed: boolean): string;
  simGetAllReduceModel(nGpus: number, countFloats: number): string;
}

declare function CudaSimModule(opts?: object): Promise<CudaSimWasmModule>;

let _modulePromise: Promise<CudaSimWasmModule> | null = null;

/** Load and initialize the WASM module. Called once; subsequent calls return the cache. */
export async function loadWasm(wasmUrl?: string): Promise<CudaSimWasmModule> {
  if (_modulePromise) return _modulePromise;

  _modulePromise = new Promise<CudaSimWasmModule>((resolve, reject) => {
    const scriptUrl = wasmUrl
      ?? (typeof __CUDA_SIM_WASM_URL__ !== 'undefined'
          ? (__CUDA_SIM_WASM_URL__ as string)
          : 'https://cdn.jsdelivr.net/npm/gpubox@latest/public/cuda_sim.js');

    // Dynamically load the Emscripten JS glue
    if (typeof document !== 'undefined') {
      // Browser
      const script = document.createElement('script');
      script.src = scriptUrl;
      script.onload = async () => {
        try {
          const mod = await (window as any).CudaSimModule({
            locateFile: (file: string) => {
              // Point to WASM file next to JS glue
              const base = scriptUrl.replace(/\/[^/]+$/, '/');
              return base + file;
            }
          });
          resolve(mod as CudaSimWasmModule);
        } catch (e) { reject(e); }
      };
      script.onerror = () => reject(new Error(`Failed to load cuda-sim WASM from ${scriptUrl}`));
      document.head.appendChild(script);
    } else {
      // Node.js / Worker environment
      import(/* webpackIgnore: true */ scriptUrl as string)
        .then(m => m.default({}).then(resolve))
        .catch(reject);
    }
  });

  return _modulePromise;
}

declare const __CUDA_SIM_WASM_URL__: string | undefined;
