/**
 * Real-GPU parity check.
 *   node scripts/gpu-parity.js
 * 1. Runs every case from gpu-parity-cases.js on the simulator.
 * 2. Hands the same kernels + inputs + simulator outputs to scripts/gpu-parity.py,
 *    which compiles them with NVRTC, runs them on the physical GPU and compares.
 * Writes test-results-gpu-parity.json.
 */
global.window = global;
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
['fp16-bf16', 'race-detector', 'sparse-memory', 'topology-model', 'desync-detector', 'gpu-profiles', 'cuda-runtime', 'cuda-transpiler']
  .forEach((f) => require('../demo/' + f + '.js'));
const { cases } = require('./gpu-parity-cases.js');

function hostProgram(c) {
  const lines = ['int main(){'];
  const bufNames = [];
  c.args.forEach((a, i) => {
    if (a.t !== 'buf') return;
    const ty = a.dtype === 'i32' ? 'int' : 'float';
    lines.push(`  ${ty} h${i}[${a.data.length}] = {${a.data.join(',')}};`);
    lines.push(`  ${ty}* d${i}; cudaMalloc(&d${i}, ${a.data.length * 4}); cudaMemcpy(d${i}, h${i}, ${a.data.length * 4}, cudaMemcpyHostToDevice);`);
    bufNames.push(i);
  });
  lines.push(`  dim3 g(${c.grid.join(',')}); dim3 b(${c.block.join(',')});`);
  const callArgs = c.args.map((a, i) => (a.t === 'buf' ? `d${i}` : String(a.v))).join(', ');
  lines.push(`  ${c.kernel}<<<g, b>>>(${callArgs});`);
  c.args.forEach((a, i) => { if (a.t === 'buf' && a.out) lines.push(`  cudaMemcpy(h${i}, d${i}, ${a.data.length * 4}, cudaMemcpyDeviceToHost);`); });
  lines.push('  return 0;', '}');
  return lines.join('\n');
}

const spec = [];
let simFail = 0;
for (const c of cases) {
  const code = `#include <cuda_runtime.h>\n${c.src}\n${hostProgram(c)}`;
  const t = window.CudaTranspiler.transpile(code);
  const entry = { name: c.name, kernel: c.kernel, src: c.src, grid: c.grid, block: c.block, tol: c.tol === undefined ? 1e-5 : c.tol, args: c.args, simOut: {} };
  if (!t.success) { entry.simError = t.errors.join('; '); simFail++; spec.push(entry); continue; }
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('RTX_4050_Laptop'));
  let thrown = null;
  try { new Function('__rt', t.code)(rt); } catch (e) { thrown = e.message; }
  const errs = rt.getDiagnostics().filter((d) => d.type === 'error');
  if (thrown || errs.length) { entry.simError = thrown || errs[0].message; simFail++; spec.push(entry); continue; }
  const allocs = Array.from(rt._memory._allocations.values());
  let bi = 0;
  c.args.forEach((a, i) => { if (a.t === 'buf') { if (a.out) entry.simOut[i] = Array.from(new Float32Array(allocs[bi].buffer)).slice(0, a.data.length); bi++; } });
  entry.counters = rt.getCounters().toJSON();
  // analytic sanity checks on the modelled metrics
  const ex = c.expect || {}; const k = entry.counters; const problems = [];
  if (ex.flops !== undefined && Math.abs(k.flopCount - ex.flops) > ex.flops * 0.01) problems.push(`flops ${k.flopCount} != ${ex.flops}`);
  if (ex.minFlops !== undefined && k.flopCount < ex.minFlops) problems.push(`flops ${k.flopCount} < ${ex.minFlops}`);
  if (ex.globalLoadBytes !== undefined && k.globalLoadBytes !== ex.globalLoadBytes) problems.push(`loadBytes ${k.globalLoadBytes} != ${ex.globalLoadBytes}`);
  if (ex.globalStoreBytes !== undefined && k.globalStoreBytes !== ex.globalStoreBytes) problems.push(`storeBytes ${k.globalStoreBytes} != ${ex.globalStoreBytes}`);
  if (ex.h2d !== undefined && k.memcpyH2DBytes !== ex.h2d) problems.push(`H2D ${k.memcpyH2DBytes} != ${ex.h2d}`);
  if (ex.d2h !== undefined && k.memcpyD2HBytes !== ex.d2h) problems.push(`D2H ${k.memcpyD2HBytes} != ${ex.d2h}`);
  if (ex.bankConflicts !== undefined && k.bankConflicts !== ex.bankConflicts) problems.push(`bankConflicts ${k.bankConflicts} != ${ex.bankConflicts}`);
  entry.metricProblems = problems;
  spec.push(entry);
}

const specPath = path.join(__dirname, '..', '.gpu-parity-spec.json');
fs.writeFileSync(specPath, JSON.stringify(spec));
console.log(`Simulator pass done: ${cases.length - simFail}/${cases.length} cases ran cleanly.`);
const py = spawnSync('python', [path.join(__dirname, 'gpu-parity.py'), specPath], { stdio: 'inherit' });
try { fs.unlinkSync(specPath); } catch (e) { /* ignore */ }
process.exit(py.status || (simFail ? 1 : 0));
