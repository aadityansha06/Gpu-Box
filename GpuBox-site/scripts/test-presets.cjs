// Verifies every playground preset against the real engine files shipped in static/sim.
// Mirrors the execution path used by the browser playground (src/lib/engine.js).
const fs = require('fs');
const path = require('path');

global.window = global;
// Load supporting engine modules in the same order as scripts/run-corpus.js.
// fp16-bf16 defines the __inline__ macro; race-detector defines SharedMemoryRaceDetector,
// etc. cuda-runtime.js depends on them being present as globals on `window` / `global`.
const simDir = path.join(__dirname, '..', 'static', 'sim');
const demoDir = path.join(__dirname, '..', '..', 'demo');
require(path.join(demoDir, 'fp16-bf16.js'));
require(path.join(demoDir, 'race-detector.js'));
require(path.join(demoDir, 'sparse-memory.js'));
require(path.join(demoDir, 'topology-model.js'));
require(path.join(demoDir, 'desync-detector.js'));
require(path.join(simDir, 'gpu-profiles.js'));
require(path.join(simDir, 'cuda-transpiler.js'));
require(path.join(simDir, 'cuda-runtime.js'));

// presets.js is an ES module; evaluate it without a bundler.
const raw = fs.readFileSync(path.join(__dirname, '..', 'src', 'lib', 'presets.js'), 'utf8')
  .replace('export const PRESETS', 'var PRESETS').replace(/export const DEFAULT_PRESET.*/, '');
const PRESETS = new Function(raw + '; return PRESETS;')();

// The transpiler strips `inline` and __forceinline__ but leaves __inline__ as a bare
// identifier, so strip it before transpiling (the site engine does the same).
function prepare(code) {
  return code.replace(/\b__inline__\b\s*/g, '').replace(/\b__forceinline__\b\s*/g, '');
}

const origLog = console.log;
let failed = 0;

for (const p of PRESETS) {
  let stdout = '';
  let thrown = [];
  let note = '';
  try {
    const origLogInner = console.log;
    console.log = (...args) => { stdout += args.join(' ') + '\n'; };
    const tr = global.CudaTranspiler.transpile(prepare(p.code));
    if (!tr.success) throw new Error('transpile failed: ' + tr.errors.join('; '));
    const rt = new global.CudaRuntime(global.GPU_PROFILES.getProfile('A100'));
    rt.deviceCount = p.devices;
    rt.clusterSelectCount = p.devices;
    global.clusterNodes = p.devices;
    if (p.devices > 1) rt.initMultiGPU(p.devices);
    try {
      new Function('__rt', tr.code)(rt);
    } catch (e) {
      thrown.push(e.message);
    }
    console.log = origLogInner;

    for (const d of rt.getDiagnostics()) {
      if (d.type === 'printf') stdout += d.message;
      else if (d.type === 'error') thrown.push((d.category || 'error') + ': ' + d.message);
    }

    const c = rt.getCounters().toJSON();
    note = `${c.threadsLaunched} threads, ${c.estimatedCycles} cycles`;
  } catch (e) {
    thrown.push(e.message);
  } finally {
    console.log = origLog;
  }

  // A "bug caught" means an OOB / illegal-address / divergence diagnostic surfaced.
  // Note: the bug presets still reach their final printf (Gpubox reports the error
  // but does not abort the whole program), so we judge "caught" purely by thrown
  // diagnostics, not by the absence of the trailing printf line.
  const bugCaught = thrown.some((m) =>
    /OOB|out.?of.?bounds|illegal address|diverg|Barrier|race/i.test(m));

  let ok;
  if (p.expect === null) {
    ok = bugCaught;
    note = ok ? 'caught: ' + (thrown[0] || 'diagnostic').slice(0, 110)
      : 'bug was NOT caught (no diagnostic)';
  } else {
    ok = stdout.includes(p.expect) && thrown.length === 0;
    if (!ok) note = `expected "${p.expect}" | thrown=${JSON.stringify(thrown)} | stdout=${JSON.stringify(stdout.slice(0, 160))}`;
  }
  if (!ok) failed++;
  origLog(`${ok ? 'PASS' : 'FAIL'}  ${p.id.padEnd(18)} ${note}`);
}

origLog(`\n${PRESETS.length - failed}/${PRESETS.length} presets verified`);
process.exit(failed ? 1 : 0);
