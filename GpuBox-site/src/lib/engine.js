// src/lib/engine.js
// Bridges the SvelteKit application to the real Gpubox simulation engine.
// All simulation runtime modules are loaded from /sim.
const SIM_BASE = '/sim';

const DEP_ORDER = [
  'acorn.js',
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

/**
 * Turns raw simulator counters into a "GPU report": where the bytes went, how much compute
 * ran, what bounds the kernel, and plain-language findings.
 */
export function analyzeRun(c, spec, devices = 1) {
  if (!c) return null;
  const fmtN = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' G' : n >= 1e6 ? (n / 1e6).toFixed(2) + ' M' : n >= 1e3 ? (n / 1e3).toFixed(1) + ' K' : String(Math.round(n)));
  const globalBytes = (c.globalLoadBytes || 0) + (c.globalStoreBytes || 0);
  const sharedBytes = (c.sharedLoadBytes || 0) + (c.sharedStoreBytes || 0);
  const movement = [
    { key: 'h2d', label: 'Host → Device (PCIe)', bytes: c.memcpyH2DBytes || 0, tone: 'host' },
    { key: 'd2h', label: 'Device → Host (PCIe)', bytes: c.memcpyD2HBytes || 0, tone: 'host' },
    { key: 'gread', label: 'Global memory reads', bytes: c.globalLoadBytes || 0, tone: 'dram' },
    { key: 'gwrite', label: 'Global memory writes', bytes: c.globalStoreBytes || 0, tone: 'dram' },
    { key: 'shared', label: 'Shared memory traffic', bytes: sharedBytes, tone: 'shared' },
    { key: 'link', label: devices > 1 ? 'GPU ↔ GPU interconnect' : 'GPU ↔ GPU interconnect', bytes: c.crossGpuBytes || 0, tone: 'link' }
  ];
  const maxBytes = Math.max(1, ...movement.map((m) => m.bytes));
  movement.forEach((m) => { m.pct = (m.bytes / maxBytes) * 100; });

  const flops = c.flopCount || 0;
  const peakTF = (spec && (spec.peakTFLOPS || spec.fp32TFLOPS)) || 10;
  const bwGBs = (spec && (spec.memBandwidthGBs || spec.memoryBandwidthGBs)) || 900;
  const intensity = globalBytes > 0 ? flops / globalBytes : 0;
  const balance = (peakTF * 1000) / bwGBs; // FLOP per byte needed to be compute-bound
  const bound = flops === 0 && globalBytes === 0 ? 'idle' : intensity < balance ? 'memory-bound' : 'compute-bound';

  const idealTx = Math.ceil(globalBytes / 32);
  const coalescing = c.globalTransactions > 0 ? Math.min(1, idealTx / c.globalTransactions) : 1;
  const sharedAccesses = (c.sharedLoads || 0) + (c.sharedStores || 0);
  const conflictRate = sharedAccesses > 0 ? (c.bankConflicts || 0) / sharedAccesses : 0;

  const tb = {};
  const div = devices > 1 ? devices : 1; // aggregated counters sum per-device time; show the per-GPU average
  Object.entries(c.timeBreakdownUs || {}).forEach(([k, v]) => { tb[k] = v / div; });
  const timeParts = [
    { key: 'launch', label: 'Launch overhead', us: tb.launch || 0 },
    { key: 'dram', label: 'Global memory', us: tb.dram || 0 },
    { key: 'compute', label: 'Compute', us: tb.compute || 0 },
    { key: 'shared', label: 'Shared memory', us: tb.shared || 0 },
    { key: 'atomics', label: 'Atomics', us: tb.atomics || 0 },
    { key: 'barriers', label: 'Barriers', us: tb.barriers || 0 },
    { key: 'interconnect', label: 'Interconnect', us: tb.interconnect || 0 }
  ];
  const maxUs = Math.max(1e-9, ...timeParts.map((t) => t.us));
  timeParts.forEach((t) => { t.pct = (t.us / maxUs) * 100; });

  const findings = [];
  if (bound === 'memory-bound') findings.push({ level: 'info', text: `Memory-bound: ${intensity.toFixed(2)} FLOP/byte against a balance point of ${balance.toFixed(1)}. More arithmetic is free; fewer bytes are what speed this up.` });
  if (bound === 'compute-bound') findings.push({ level: 'info', text: `Compute-bound: ${intensity.toFixed(1)} FLOP/byte exceeds the balance point of ${balance.toFixed(1)}.` });
  if (c.globalTransactions > 0 && coalescing < 0.7) findings.push({ level: 'warn', text: `Uncoalesced global access: ${(coalescing * 100).toFixed(0)}% efficiency (${c.globalTransactions.toLocaleString()} 32-byte transactions for ${idealTx.toLocaleString()} ideal). Make neighbouring threads touch neighbouring addresses.` });
  if ((c.bankConflicts || 0) > 0) findings.push({ level: 'warn', text: `${c.bankConflicts.toLocaleString()} shared-memory bank conflicts (${(conflictRate * 100).toFixed(0)}% of accesses). Pad the tile ([32][33]) or change the access stride.` });
  if ((c.divergentBranches || 0) > 0) findings.push({ level: 'warn', text: `${c.divergentBranches.toLocaleString()} divergent branch instances: threads in one warp took different paths and ran serially.` });
  if ((c.atomicMaxContention || 0) > 32) findings.push({ level: 'warn', text: `Atomic hot-spot: one address was updated ${c.atomicMaxContention.toLocaleString()} times. Privatise in shared memory first.` });
  if ((c.occupancy || 0) < 0.25 && (c.threadsLaunched || 0) > 0) findings.push({ level: 'info', text: `Low occupancy (${((c.occupancy || 0) * 100).toFixed(1)}%): the launch is too small to fill this GPU.` });
  if ((c.memcpyH2DBytes || 0) + (c.memcpyD2HBytes || 0) > 0 && globalBytes > 0 && (c.memcpyH2DBytes + c.memcpyD2HBytes) > globalBytes * 4) findings.push({ level: 'info', text: 'Host transfers dwarf on-device traffic: the PCIe copy, not the kernel, dominates this program.' });
  if (!findings.length) findings.push({ level: 'good', text: 'No access-pattern hazards detected: coalesced global traffic, no bank conflicts, no divergence.' });

  return {
    movement, timeParts, flops, flopsLabel: fmtN(flops), intensity, balance, bound, coalescing, conflictRate, findings,
    totalMoved: movement.reduce((a, m) => a + m.bytes, 0), globalBytes, sharedBytes, peakTF, bwGBs
  };
}
