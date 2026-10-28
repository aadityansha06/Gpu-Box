// src/lib/engine.js
// Bridges the SvelteKit playground to the real Gpubox simulation engine.
// The engine files (copied from demo/) are served from /sim and loaded as
// classic scripts that attach to `window` (CudaRuntime, CudaTranspiler, GPU_PROFILES).
const SIM_BASE = '/sim';

// Dependency order matters: cuda-runtime.js resolves its helpers (race detector,
// fp16, sparse memory, topology, desync) off `window`, so they must load first.
const DEP_ORDER = [
  'fp16-bf16.js',
  'race-detector.js',
  'sparse-memory.js',
  'topology-model.js',
  'desync-detector.js',
  'gpu-profiles.js',
  'cuda-transpiler.js',
  'cuda-runtime.js',
];

let loaded = false;
let loadPromise = null;
let CudaRuntime;
let CudaTranspiler;
let GPU_PROFILES;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error('Failed to load ' + src));
    document.head.appendChild(el);
  });
}

export async function ensureLoaded() {
  if (loaded) return;
  if (!loadPromise) {
    loadPromise = (async () => {
      for (const file of DEP_ORDER) {
        await loadScript(SIM_BASE + '/' + file);
      }
      CudaRuntime = window.CudaRuntime;
      CudaTranspiler = window.CudaTranspiler;
      GPU_PROFILES = window.GPU_PROFILES;
      if (!CudaRuntime || !CudaTranspiler || !GPU_PROFILES) {
        throw new Error('Gpubox engine failed to initialize');
      }
      loaded = true;
    })();
  }
  return loadPromise;
}

// The transpiler strips `inline` and __forceinline__ but leaves __inline__ as a
// bare identifier, so strip it before transpiling (mirrors scripts/test-presets.cjs).
export function prepareCode(code) {
  return String(code).replace(/\b__inline__\b\s*/g, '').replace(/\b__forceinline__\b\s*/g, '');
}

// Run a snippet of supported CUDA C++ through the real engine.
// Returns { ok, output, errors, counters, stage }.
export async function runCUDA({ code, profile = 'A100', devices = 1 } = {}) {
  await ensureLoaded();
  const src = prepareCode(code);

  let transpileRes;
  try {
    transpileRes = CudaTranspiler.transpile(src);
  } catch (e) {
    return { ok: false, stage: 'transpile', errors: [e.message], output: '', counters: null };
  }
  if (!transpileRes.success) {
    return { ok: false, stage: 'transpile', errors: transpileErrors(transpileRes), output: '', counters: null };
  }

  const rt = new CudaRuntime(GPU_PROFILES.getProfile(profile));
  rt.deviceCount = devices;
  rt.clusterSelectCount = devices;
  if (devices > 1 && typeof rt.initMultiGPU === 'function') {
    rt.initMultiGPU(devices);
  }

  const output = [];
  const origLog = console.log;
  console.log = (...args) => output.push(args.map(String).join(' '));
  let thrown = null;
  try {
    // The transpiled program calls __rt.launch(...) which drives the SIMT scheduler.
    new Function('__rt', transpileRes.code)(rt);
  } catch (e) {
    thrown = e.message;
  } finally {
    console.log = origLog;
  }

  // Surface structured device errors (OOB, barrier divergence, etc.)
  try {
    for (const d of rt.getDiagnostics()) {
      if (d.type === 'error') {
        output.push(`[${d.category || 'error'}] ${d.message || ''}`);
      }
    }
  } catch (e) {
    // diagnostics not available
  }

  const counters = rt.getCounters ? rt.getCounters().toJSON() : null;
  return {
    ok: !thrown,
    stage: thrown ? 'runtime' : 'ok',
    errors: thrown ? [thrown] : [],
    output: output.join('\n'),
    counters,
    kernels: transpileRes.kernelNames || [],
  };
}

function transpileErrors(res) {
  return Array.isArray(res.errors) ? res.errors : [String(res.errors || 'Transpilation failed')];
}

export function formatCounters(c) {
  if (!c) return {};
  return {
    threads: c.threadsLaunched ?? 0,
    blocks: c.blocksLaunched ?? 0,
    cycles: c.estimatedCycles ?? 0,
    timeMs: c.estimatedTimeMs ?? 0,
    flops: c.flopCount ?? 0,
    computeUtil: ((c.computeUtilization ?? 0) * 100).toFixed(2) + '%',
    memUtil: ((c.memoryUtilization ?? 0) * 100).toFixed(2) + '%',
    occupancy: ((c.occupancy ?? 0) * 100).toFixed(2) + '%',
    globalLoads: c.globalLoads ?? 0,
    globalStores: c.globalStores ?? 0,
    sharedLoads: c.sharedLoads ?? 0,
    sharedStores: c.sharedStores ?? 0,
    bankConflicts: c.bankConflicts ?? 0,
    crossGpuBytes: c.crossGpuBytes ?? 0,
    ncclOps: c.ncclOps ?? 0,
  };
}
