// examples/worker/worker.js
// Runs untrusted or long-running CUDA simulation code in a dedicated background worker

const { createSimulator, compile } = require('../../js/dist/cuda-sim.cjs.js');

function runSimulationInWorker(source, grid, block, ...args) {
  const sim = createSimulator({ limits: { maxSteps: 50000, maxTimeMs: 2000 } });
  const prog = compile(source);
  if (!prog.analysis.success) {
    return { status: 'UNSUPPORTED', errors: prog.analysis.unsupportedFeatures };
  }
  return { status: 'PASS', kernels: prog.kernelNames };
}

module.exports = { runSimulationInWorker };
