// src/lib/engine.js
// Bridges the SvelteKit application to the real Gpubox simulation engine.
// All simulation runtime modules are loaded from /sim.
const SIM_BASE = '/sim';

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

export function prepareCode(code) {
  return String(code).replace(/\b__inline__\b\s*/g, '').replace(/\b__forceinline__\b\s*/g, '');
}

export async function getGPUProfile(profileName = 'A100') {
  await ensureLoaded();
  return GPU_PROFILES ? GPU_PROFILES.getProfile(profileName) : null;
}

export async function runCUDA({ code, profile = 'A100', devices = 1 } = {}) {
  await ensureLoaded();
  const src = prepareCode(code);

  let transpileRes;
  try {
    transpileRes = CudaTranspiler.transpile(src);
  } catch (e) {
    return {
      ok: false,
      stage: 'transpile',
      errors: [e.message],
      output: '',
      transpiledCode: '',
      diagnostics: [],
      counters: null,
      profileSpec: null
    };
  }
  if (!transpileRes.success) {
    return {
      ok: false,
      stage: 'transpile',
      errors: transpileErrors(transpileRes),
      output: '',
      transpiledCode: '',
      diagnostics: [],
      counters: null,
      profileSpec: null
    };
  }

  const prof = GPU_PROFILES.getProfile(profile);
  const rt = new CudaRuntime(prof);
  rt.deviceCount = devices;
  rt.clusterSelectCount = devices;
  if (devices > 1 && typeof rt.initMultiGPU === 'function') {
    rt.initMultiGPU(devices);
  }

  const consoleLogs = [];
  const origLog = console.log;
  console.log = (...args) => consoleLogs.push(args.map(String).join(' '));
  let thrown = null;
  const startPerf = typeof performance !== 'undefined' ? performance.now() : Date.now();
  try {
    new Function('__rt', transpileRes.code)(rt);
  } catch (e) {
    thrown = e.message;
  } finally {
    console.log = origLog;
  }
  const hostDurationMs = ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - startPerf).toFixed(2);

  // Surface structured device diagnostics & printf
  const printfLogs = [];
  const deviceErrors = [];
  try {
    for (const d of rt.getDiagnostics()) {
      if (d.type === 'printf') {
        printfLogs.push(d.message);
      } else if (d.type === 'error') {
        const coord = (d.block && d.thread)
          ? `[Block (${d.block.x},${d.block.y},${d.block.z}) Thread (${d.thread.x},${d.thread.y},${d.thread.z})] `
          : '';
        deviceErrors.push(`${coord}${d.message || d.category || 'Runtime error'}`);
      }
    }
  } catch (e) {
    // diagnostics not available
  }

  const outputParts = [];
  if (printfLogs.length > 0) {
    outputParts.push(printfLogs.join(''));
  }
  if (consoleLogs.length > 0) {
    outputParts.push(consoleLogs.join('\n'));
  }

  const allErrors = [];
  if (thrown) allErrors.push(thrown);
  for (const err of deviceErrors) {
    allErrors.push(err);
  }

  const counters = rt.getCounters ? rt.getCounters().toJSON() : null;

  return {
    ok: allErrors.length === 0,
    stage: allErrors.length > 0 ? 'runtime' : 'ok',
    errors: allErrors,
    output: outputParts.join('\n').trim(),
    transpiledCode: transpileRes.code || '',
    hostDurationMs,
    counters,
    profileSpec: prof,
    kernels: transpileRes.kernelNames || [],
  };
}

function transpileErrors(res) {
  return Array.isArray(res.errors) ? res.errors : [String(res.errors || 'Transpilation failed')];
}

export function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return bytes + ' B';
}

export function computeBankState(counters) {
  if (!counters) {
    return Array.from({ length: 32 }, (_, i) => ({
      bank: i,
      accesses: 0,
      conflicts: 0,
      status: 'idle'
    }));
  }
  const totalShared = (counters.sharedLoads || 0) + (counters.sharedStores || 0);
  const conflicts = counters.bankConflicts || 0;

  return Array.from({ length: 32 }, (_, i) => {
    if (totalShared === 0) {
      return { bank: i, accesses: 0, conflicts: 0, status: 'idle' };
    }
    const baseAccesses = Math.floor(totalShared / 32) + (i < (totalShared % 32) ? 1 : 0);
    const bankConf = conflicts > 0 && (i % 8 === 0 || i % 4 === 0) ? Math.ceil(conflicts / 4) : 0;
    return {
      bank: i,
      accesses: baseAccesses + bankConf,
      conflicts: bankConf,
      status: bankConf > 0 ? 'conflict' : (baseAccesses > 0 ? 'active' : 'idle')
    };
  });
}

export function formatCounters(c) {
  if (!c) return {};
  return {
    threads: c.threadsLaunched ?? 0,
    blocks: c.blocksLaunched ?? 0,
    cycles: (c.estimatedCycles ?? 0).toLocaleString(),
    timeMs: c.estimatedTimeMs < 1 
      ? ((c.estimatedTimeMs ?? 0) * 1000).toFixed(1) + ' µs'
      : (c.estimatedTimeMs ?? 0).toFixed(3) + ' ms',
    flops: (c.flopCount ?? 0).toLocaleString(),
    computeUtil: ((c.computeUtilization ?? 0) * 100).toFixed(2) + '%',
    memUtil: ((c.memoryUtilization ?? 0) * 100).toFixed(2) + '%',
    occupancy: ((c.occupancy ?? 0) * 100).toFixed(2) + '%',
    globalLoads: (c.globalLoads ?? 0).toLocaleString(),
    globalStores: (c.globalStores ?? 0).toLocaleString(),
    globalLoadBytes: formatBytes(c.globalLoadBytes ?? 0),
    globalStoreBytes: formatBytes(c.globalStoreBytes ?? 0),
    globalTransactions: (c.globalTransactions ?? 0).toLocaleString(),
    sharedLoads: (c.sharedLoads ?? 0).toLocaleString(),
    sharedStores: (c.sharedStores ?? 0).toLocaleString(),
    sharedBytes: formatBytes((c.sharedLoadBytes ?? 0) + (c.sharedStoreBytes ?? 0)),
    bankConflicts: c.bankConflicts ?? 0,
    syncthreadsCount: (c.syncthreadsCount ?? 0).toLocaleString(),
    divergentBranches: (c.divergentBranches ?? 0).toLocaleString(),
    atomicOps: (c.atomicOps ?? 0).toLocaleString(),
    atomicMaxContention: c.atomicMaxContention ?? 0,
    crossGpuBytes: formatBytes(c.crossGpuBytes ?? 0),
    ncclOps: (c.ncclOps ?? 0).toLocaleString(),
    memcpyH2DBytes: formatBytes(c.memcpyH2DBytes ?? 0),
    memcpyD2HBytes: formatBytes(c.memcpyD2HBytes ?? 0),
  };
}
