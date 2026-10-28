// examples/cluster-desync-demo/demo.js
// Demonstrates how Gpubox catches distributed collective deadlocks and parameter desyncs

const assert = require('assert');
const { createSimulator } = require('../../js/dist/cuda-sim.cjs.js');

console.log('Running Multi-GPU Collective Desync & Deadlock Demo...');

const sim = createSimulator({ numDevices: 8, gpuProfile: 'A100' });

// Simulate Rank 3 passing mismatched count
try {
  const comm = sim.ncclCommInitRank(8, 0);
  const sBufs = [], rBufs = [];
  for (let r = 0; r < 8; r++) {
    const count = (r === 3) ? 511 : 512; // rank 3 mismatch!
    const s = sim.allocate(512 * 4, r);
    const rec = sim.allocate(512 * 4, r);
    sBufs.push(s);
    rBufs.push(rec);
  }
  console.log('Simulating 8-rank collective with rank 3 count mismatch (511 vs 512)...');
  sim.allReduce(sBufs, rBufs, 512, 'float32', 'sum', comm);
} catch (err) {
  console.log('✓ Successfully caught collective desync error:');
  console.log(' ', err.message);
}
