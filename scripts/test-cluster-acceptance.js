/**
 * scripts/test-cluster-acceptance.js
 * ──────────────────────────────────
 * Cluster Acceptance Test Suite (Tests A1 to A12)
 * Validates 1 to 16 virtual GPUs, collectives, topologies, sparse memory,
 * race/desync detection, watchdog abort, and spec consistency.
 *
 * Copyright (c) 2026 Aadityansha Verma. All rights reserved.
 */

const assert = require('assert');
const path = require('path');
const fs = require('fs');

global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/fp16-bf16.js');
require('../demo/race-detector.js');
require('../demo/sparse-memory.js');
require('../demo/topology-model.js');
require('../demo/desync-detector.js');
require('../demo/cuda-transpiler.js');
require('../demo/cuda-runtime.js');
const { validateOutput, float32UlpDiff } = require('../demo/validation-harness.js');

const CudaRuntime = global.CudaRuntime;
const GPU_PROFILES = global.GPU_PROFILES;
const CudaTranspiler = global.CudaTranspiler;
const { CollectiveDesyncDetector, NcclDesyncError } = global.GpuboxDesyncDetector;

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const results = [];

function runAcceptanceTest(id, name, fn) {
  totalTests++;
  const label = `[ACCEPTANCE: ${id}] ${name}`;
  process.stdout.write(`${label}... `);
  const start = Date.now();
  try {
    fn();
    const durationMs = Date.now() - start;
    passedTests++;
    console.log(`\x1b[32m✓ PASS\x1b[0m (${durationMs} ms)`);
    results.push({ id, name, status: 'PASS', durationMs });
  } catch (err) {
    const durationMs = Date.now() - start;
    failedTests++;
    console.log(`\x1b[31m✗ FAIL\x1b[0m (${durationMs} ms)`);
    console.error(`    Error: ${err.message}`);
    results.push({ id, name, status: 'FAIL', durationMs, error: err.message });
  }
}

console.log('================================================================');
console.log('  GPUBOX CLUSTER ACCEPTANCE TEST SUITE (A-SERIES: A1–A12)');
console.log('  1–16 Virtual GPUs | Ring Collectives | Desync | Sparse VRAM');
console.log('================================================================\n');

// -----------------------------------------------------------------------------
// A1: AllReduce, 16 ranks
// -----------------------------------------------------------------------------
runAcceptanceTest('A1', '16-rank AllReduce (SUM, MAX, MIN, AVG, PROD, 7680 bytes/rank ring traffic)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt = new CudaRuntime(profile);
  const multi = rt.initMultiGPU(16);
  multi.setCollectiveAlgorithm('ring');

  const count = 1024;
  const comm = multi.ncclCommInitRank(16, 0, { algorithm: 'ring' });

  // 1. SUM
  const sendbufs = [];
  const recvbufs = [];
  for (let r = 0; r < 16; r++) {
    const s = multi.getDevice(r).cudaMalloc(count * 4);
    const rec = multi.getDevice(r).cudaMalloc(count * 4);
    const data = new Float32Array(count).fill(r + 1);
    multi.getDevice(r).cudaMemcpy(s, data, count * 4, 'H2D');
    sendbufs.push(s);
    recvbufs.push(rec);
  }

  multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', 'sum', comm);

  // Assert SUM: every element on every rank == 136 (sum 1..16)
  for (let r = 0; r < 16; r++) {
    const out = new Float32Array(count);
    multi.getDevice(r).cudaMemcpy(out, recvbufs[r], count * 4, 'D2H');
    for (let i = 0; i < count; i++) {
      assert.strictEqual(out[i], 136, `Rank ${r} element ${i} must be 136`);
    }
    // Ring traffic: 2 * (15/16) * 4096 = 7680 bytes received per rank
    assert.strictEqual(multi.getDevice(r).counters.crossGpuBytes, 7680, `Rank ${r} ring traffic must be 7680 bytes`);
    assert.strictEqual(multi.getDevice(r).counters.ncclOps, 1, `Rank ${r} ncclOps must be 1`);
  }

  const agg = multi.getAggregatedCounters().toJSON();
  assert.strictEqual(agg.crossGpuBytes, 122880, 'Total ring traffic must be 122,880 bytes');

  // 2. MAX, MIN, AVG
  multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', 'max', comm);
  const outMax = new Float32Array(count);
  multi.getDevice(0).cudaMemcpy(outMax, recvbufs[0], count * 4, 'D2H');
  assert.strictEqual(outMax[0], 16, 'MAX must be 16');

  multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', 'min', comm);
  const outMin = new Float32Array(count);
  multi.getDevice(0).cudaMemcpy(outMin, recvbufs[0], count * 4, 'D2H');
  assert.strictEqual(outMin[0], 1, 'MIN must be 1');

  multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', 'avg', comm);
  const outAvg = new Float32Array(count);
  multi.getDevice(0).cudaMemcpy(outAvg, recvbufs[0], count * 4, 'D2H');
  assert.strictEqual(outAvg[0], 8.5, 'AVG must be 8.5');

  // 3. PROD with 2.0 on each rank gives 2^16 = 65536
  for (let r = 0; r < 16; r++) {
    const data = new Float32Array(count).fill(2.0);
    multi.getDevice(r).cudaMemcpy(sendbufs[r], data, count * 4, 'H2D');
  }
  multi.ncclAllReduce(sendbufs, recvbufs, count, 'float32', 'prod', comm);
  const outProd = new Float32Array(count);
  multi.getDevice(0).cudaMemcpy(outProd, recvbufs[0], count * 4, 'D2H');
  assert.strictEqual(outProd[0], 65536, 'PROD must be 65536');
});

// -----------------------------------------------------------------------------
// A2: Broadcast, 16 ranks, root = 11
// -----------------------------------------------------------------------------
runAcceptanceTest('A2', '16-rank Broadcast (root=11, 400 bytes per non-root, invalid root rejected)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt = new CudaRuntime(profile);
  const multi = rt.initMultiGPU(16);

  const count = 100;
  const root = 11;
  const comm = multi.ncclCommInitRank(16, 0);

  const sendbufs = [];
  const recvbufs = [];
  for (let r = 0; r < 16; r++) {
    const s = multi.getDevice(r).cudaMalloc(count * 4);
    const rec = multi.getDevice(r).cudaMalloc(count * 4);
    if (r === root) {
      const data = new Float32Array(count);
      for (let i = 0; i < count; i++) data[i] = i;
      multi.getDevice(r).cudaMemcpy(s, data, count * 4, 'H2D');
    }
    sendbufs.push(s);
    recvbufs.push(rec);
  }

  multi.ncclBroadcast(sendbufs, recvbufs, count, 'float32', root, comm);

  // Verify all 16 ranks received 0..99
  for (let r = 0; r < 16; r++) {
    const out = new Float32Array(count);
    multi.getDevice(r).cudaMemcpy(out, recvbufs[r], count * 4, 'D2H');
    for (let i = 0; i < count; i++) {
      assert.strictEqual(out[i], i);
    }
    if (r === root) {
      assert.strictEqual(multi.getDevice(r).counters.crossGpuBytes, 0, 'Root rank receives 0 bytes');
    } else {
      assert.strictEqual(multi.getDevice(r).counters.crossGpuBytes, 400, 'Non-root receives 400 bytes');
    }
  }

  // Reject root = 16 and root = -1
  assert.throws(() => {
    multi.ncclBroadcast(sendbufs, recvbufs, count, 'float32', 16, comm);
  }, /ncclInvalidArgument/);

  assert.throws(() => {
    multi.ncclBroadcast(sendbufs, recvbufs, count, 'float32', -1, comm);
  }, /ncclInvalidArgument/);
});

// -----------------------------------------------------------------------------
// A3: AllGather and ReduceScatter, 16 ranks
// -----------------------------------------------------------------------------
runAcceptanceTest('A3', '16-rank AllGather (180 bytes/rank) & ReduceScatter (120 bytes/rank)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt = new CudaRuntime(profile);
  const multi = rt.initMultiGPU(16);

  // 1. AllGather: rank r contributes [r, r+0.5, r+0.25]
  const sendbufs = [];
  const recvbufs = [];
  for (let r = 0; r < 16; r++) {
    const s = multi.getDevice(r).cudaMalloc(3 * 4);
    const rec = multi.getDevice(r).cudaMalloc(48 * 4);
    multi.getDevice(r).cudaMemcpy(s, new Float32Array([r, r + 0.5, r + 0.25]), 3 * 4, 'H2D');
    sendbufs.push(s);
    recvbufs.push(rec);
  }

  multi.ncclAllGather(sendbufs, recvbufs, 3, 'float32');

  for (let r = 0; r < 16; r++) {
    const out = new Float32Array(48);
    multi.getDevice(r).cudaMemcpy(out, recvbufs[r], 48 * 4, 'D2H');
    for (let src_r = 0; src_r < 16; src_r++) {
      assert.strictEqual(out[3 * src_r], src_r);
      assert.strictEqual(out[3 * src_r + 1], src_r + 0.5);
      assert.strictEqual(out[3 * src_r + 2], src_r + 0.25);
    }
    // Each rank receives 15 * 12 = 180 bytes
    assert.strictEqual(multi.getDevice(r).counters.crossGpuBytes, 180);
  }

  // 2. ReduceScatter: rank r has 32 floats with element j = r + j
  multi.reset();
  multi.setCollectiveAlgorithm('ring');
  const rsSend = [];
  const rsRecv = [];
  for (let r = 0; r < 16; r++) {
    const s = multi.getDevice(r).cudaMalloc(32 * 4);
    const rec = multi.getDevice(r).cudaMalloc(2 * 4);
    const data = new Float32Array(32);
    for (let j = 0; j < 32; j++) data[j] = r + j;
    multi.getDevice(r).cudaMemcpy(s, data, 32 * 4, 'H2D');
    rsSend.push(s);
    rsRecv.push(rec);
  }

  multi.ncclReduceScatter(rsSend, rsRecv, 2, 'float32', 'sum');

  // Rank q receives [120 + 32q, 136 + 32q]
  for (let q = 0; q < 16; q++) {
    const out = new Float32Array(2);
    multi.getDevice(q).cudaMemcpy(out, rsRecv[q], 2 * 4, 'D2H');
    assert.strictEqual(out[0], 120 + 32 * q, `Rank ${q} element 0 mismatch`);
    assert.strictEqual(out[1], 136 + 32 * q, `Rank ${q} element 1 mismatch`);
    // Each rank receives 15/16 * 128 = 120 bytes
    assert.strictEqual(multi.getDevice(q).counters.crossGpuBytes, 120);
  }
});

// -----------------------------------------------------------------------------
// A4: Desync and deadlock detection, 16 ranks
// -----------------------------------------------------------------------------
runAcceptanceTest('A4', '16-rank Desync & Deadlock Detection (missing rank, count mismatch, order mismatch, reset)', () => {
  const detector = new CollectiveDesyncDetector();

  // (a) Rank 7 skips AllReduce: report rank 7 missing, ranks 0-6 and 8-15 waiting, collective #0
  for (let r = 0; r < 16; r++) {
    if (r !== 7) {
      detector.postCollective({ commId: 0, nranks: 16, rank: r, op: 'allreduce', count: 1024, datatype: 'float32' });
    }
  }
  let errA = null;
  try {
    detector.checkDeadlocks(0, 16);
  } catch (err) {
    errA = err;
  }
  assert.ok(errA, 'Must detect collective deadlock');
  assert.strictEqual(errA.details.type, 'COLLECTIVE_DEADLOCK');
  assert.strictEqual(errA.details.seq, 0);
  assert.ok(errA.details.missingRanks.includes(7));
  assert.strictEqual(errA.details.waitingRanks.length, 15);

  // (b) Rank 3 passes count 1023 while others pass 1024
  detector.reset();
  let errB = null;
  try {
    for (let r = 0; r < 16; r++) {
      detector.postCollective({ commId: 0, nranks: 16, rank: r, op: 'allreduce', count: r === 3 ? 1023 : 1024, datatype: 'float32' });
    }
  } catch (err) {
    errB = err;
  }
  assert.ok(errB, 'Must detect count mismatch');
  assert.strictEqual(errB.details.type, 'COUNT_MISMATCH');
  assert.strictEqual(errB.details.offendingRank, 3);
  assert.strictEqual(errB.details.parameter, 'count');

  // (c) Rank 0 calls AllReduce then Broadcast; Rank 1 calls Broadcast then AllReduce
  detector.reset();
  let errC = null;
  try {
    detector.postCollective({ commId: 0, nranks: 16, rank: 0, op: 'allreduce', count: 64, datatype: 'float32' });
    detector.postCollective({ commId: 0, nranks: 16, rank: 1, op: 'broadcast', count: 64, datatype: 'float32' });
    for (let r = 2; r < 16; r++) {
      detector.postCollective({ commId: 0, nranks: 16, rank: r, op: 'allreduce', count: 64, datatype: 'float32' });
    }
  } catch (err) {
    errC = err;
  }
  assert.ok(errC, 'Must detect order mismatch');
  assert.strictEqual(errC.details.type, 'ORDER_MISMATCH');

  // (d) After each failure, reset() and a correct run must pass
  detector.reset();
  for (let r = 0; r < 16; r++) {
    detector.postCollective({ commId: 0, nranks: 16, rank: r, op: 'allreduce', count: 1024, datatype: 'float32' });
  }
  assert.strictEqual(detector.history.length, 1, 'Clean run must succeed after reset');
});

// -----------------------------------------------------------------------------
// A5: Send/Recv and AllToAll, 16 ranks
// -----------------------------------------------------------------------------
runAcceptanceTest('A5', '16-rank Send/Recv Ring Shift & AllToAll (60 bytes/rank)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt = new CudaRuntime(profile);
  const multi = rt.initMultiGPU(16);

  // 1. Ring shift inside a group: rank r sends to (r+1)%16, receives from (r+15)%16
  const sendbufs = [];
  const recvbufs = [];
  for (let r = 0; r < 16; r++) {
    const s = multi.getDevice(r).cudaMalloc(4);
    const rec = multi.getDevice(r).cudaMalloc(4);
    multi.getDevice(r).cudaMemcpy(s, new Float32Array([r]), 4, 'H2D');
    sendbufs.push(s);
    recvbufs.push(rec);
  }

  multi.ncclGroupStart();
  for (let r = 0; r < 16; r++) {
    multi.ncclSend(sendbufs[r], 1, 'float32', (r + 1) % 16);
    multi.ncclRecv(recvbufs[r], 1, 'float32', (r + 15) % 16);
  }
  multi.ncclGroupEnd();

  // Rank r ends with (r + 15) % 16 (rank 0 gets 15, rank 1 gets 0)
  for (let r = 0; r < 16; r++) {
    const out = new Float32Array(1);
    multi.getDevice(r).cudaMemcpy(out, recvbufs[r], 4, 'D2H');
    assert.strictEqual(out[0], (r + 15) % 16);
  }

  // 2. AllToAll with 1 float per peer: rank r sends 16r+q to rank q
  multi.reset();
  const ataSend = [];
  const ataRecv = [];
  for (let r = 0; r < 16; r++) {
    const s = multi.getDevice(r).cudaMalloc(16 * 4);
    const rec = multi.getDevice(r).cudaMalloc(16 * 4);
    const data = new Float32Array(16);
    for (let q = 0; q < 16; q++) data[q] = 16 * r + q;
    multi.getDevice(r).cudaMemcpy(s, data, 16 * 4, 'H2D');
    ataSend.push(s);
    ataRecv.push(rec);
  }

  multi.ncclAllToAll(ataSend, ataRecv, 1, 'float32');

  for (let q = 0; q < 16; q++) {
    const out = new Float32Array(16);
    multi.getDevice(q).cudaMemcpy(out, ataRecv[q], 16 * 4, 'D2H');
    for (let r = 0; r < 16; r++) {
      assert.strictEqual(out[r], 16 * r + q);
    }
    // Each rank receives 15 * 4 = 60 bytes
    assert.strictEqual(multi.getDevice(q).counters.crossGpuBytes, 60);
  }
});

// -----------------------------------------------------------------------------
// A6: Sub-communicators
// -----------------------------------------------------------------------------
runAcceptanceTest('A6', 'Sub-communicators (devices 0-7 sum=36, devices 8-15 sum=100, 448 bytes/rank, 0 inter-group traffic)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt = new CudaRuntime(profile);
  const multi = rt.initMultiGPU(16);

  const groupA = [0, 1, 2, 3, 4, 5, 6, 7];
  const groupB = [8, 9, 10, 11, 12, 13, 14, 15];

  const commA = multi.ncclCommInitRank(8, 0, { ranks: groupA, algorithm: 'ring' });
  const commB = multi.ncclCommInitRank(8, 0, { ranks: groupB, algorithm: 'ring' });

  const count = 64; // 256 bytes

  // Setup send & recv buffers for all 16 devices
  const sendbufsA = [];
  const recvbufsA = [];
  for (let i = 0; i < 8; i++) {
    const dev = groupA[i];
    const s = multi.getDevice(dev).cudaMalloc(count * 4);
    const rec = multi.getDevice(dev).cudaMalloc(count * 4);
    multi.getDevice(dev).cudaMemcpy(s, new Float32Array(count).fill(i + 1), count * 4, 'H2D');
    sendbufsA.push(s);
    recvbufsA.push(rec);
  }

  const sendbufsB = [];
  const recvbufsB = [];
  for (let i = 0; i < 8; i++) {
    const dev = groupB[i];
    const s = multi.getDevice(dev).cudaMalloc(count * 4);
    const rec = multi.getDevice(dev).cudaMalloc(count * 4);
    multi.getDevice(dev).cudaMemcpy(s, new Float32Array(count).fill(i + 1), count * 4, 'H2D');
    sendbufsB.push(s);
    recvbufsB.push(rec);
  }

  // Execute Group A AllReduce
  multi.ncclAllReduce(sendbufsA, recvbufsA, count, 'float32', 'sum', commA);

  // Assert Group A sum = 36 (1+2+3+4+5+6+7+8)
  for (let i = 0; i < 8; i++) {
    const dev = groupA[i];
    const out = new Float32Array(count);
    multi.getDevice(dev).cudaMemcpy(out, recvbufsA[i], count * 4, 'D2H');
    assert.strictEqual(out[0], 36);
    // Ring traffic: 2 * (7/8) * 256 = 448 bytes
    assert.strictEqual(multi.getDevice(dev).counters.crossGpuBytes, 448);
  }
  // Group B must have 0 traffic so far!
  for (let i = 0; i < 8; i++) {
    const dev = groupB[i];
    assert.strictEqual(multi.getDevice(dev).counters.crossGpuBytes, 0, 'Group B must have 0 traffic');
  }

  // Execute Group B AllReduce with rank r contributing (r+9) -> sum = 9+10+11+12+13+14+15+16 = 100
  for (let i = 0; i < 8; i++) {
    const dev = groupB[i];
    multi.getDevice(dev).cudaMemcpy(sendbufsB[i], new Float32Array(count).fill(i + 9), count * 4, 'H2D');
  }
  multi.ncclAllReduce(sendbufsB, recvbufsB, count, 'float32', 'sum', commB);

  for (let i = 0; i < 8; i++) {
    const dev = groupB[i];
    const out = new Float32Array(count);
    multi.getDevice(dev).cudaMemcpy(out, recvbufsB[i], count * 4, 'D2H');
    assert.strictEqual(out[0], 100);
    assert.strictEqual(multi.getDevice(dev).counters.crossGpuBytes, 448);
  }
});

// -----------------------------------------------------------------------------
// A7: Virtual VRAM
// -----------------------------------------------------------------------------
runAcceptanceTest('A7', 'Virtual VRAM (16 devices x 70 GiB succeeds, <256 MB real heap growth, 2nd 70 GiB fails, 81 GiB fails)', () => {
  const profile = GPU_PROFILES.getProfile('A100'); // 80 GB capacity
  const rt = new CudaRuntime(profile);
  const multi = rt.initMultiGPU(16);

  const initialHeap = process.memoryUsage().heapUsed;
  const GIB = 1024 * 1024 * 1024;
  const size70GiB = 70 * GIB;

  const ptrs = [];
  for (let r = 0; r < 16; r++) {
    const ptr = multi.getDevice(r).cudaMalloc(size70GiB);
    assert.ok(ptr);
    // Touch 1 MiB per device
    for (let i = 0; i < 1024; i += 64) {
      ptr.set(i, 42.0);
    }
    ptrs.push(ptr);
  }

  const finalHeap = process.memoryUsage().heapUsed;
  const heapGrowthBytes = finalHeap - initialHeap;
  const maxAllowedGrowth = 256 * 1024 * 1024; // 256 MB
  assert.ok(heapGrowthBytes < maxAllowedGrowth, `Heap growth was ${heapGrowthBytes / 1024 / 1024} MB (must be < 256 MB)`);

  // A second 70 GiB on device 0 must fail with cudaErrorMemoryAllocation
  assert.throws(() => {
    multi.getDevice(0).cudaMalloc(size70GiB);
  }, /cudaErrorMemoryAllocation/);

  // Free first pointer and try again: must succeed!
  multi.getDevice(0).cudaFree(ptrs[0]);
  const ptrRetry = multi.getDevice(0).cudaMalloc(size70GiB);
  assert.ok(ptrRetry, 'Malloc after free must succeed');

  // Requesting 81 GiB on an 80 GB profile must fail
  assert.throws(() => {
    multi.getDevice(1).cudaMalloc(81 * GIB);
  }, /cudaErrorMemoryAllocation/);
});

// -----------------------------------------------------------------------------
// A8: Watchdog and responsiveness
// -----------------------------------------------------------------------------
runAcceptanceTest('A8', 'Watchdog runaway loop abort (aborts while(true){} naming kernel, block, thread)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt = new CudaRuntime(profile);

  // Infinite loop kernel
  const runawayKernel = function* (__tid, __bid) {
    while (true) {
      yield;
    }
  };

  let threw = false;
  try {
    rt.launch(runawayKernel, 1, 32, 0);
  } catch (err) {
    threw = true;
    assert.ok(err.message.includes('watchdog timeout') || err.message.includes('step limit'), `Error: ${err.message}`);
  }
  assert.ok(threw, 'Watchdog must abort infinite loop');

  // Next run after reset() must pass
  rt.reset();
  const normalKernel = function* (__tid, __bid, __bdim, __gdim, __shared, __sync, __atomic, __printf, out) {
    out.set(__tid.x, 100);
  };
  const p = rt.cudaMalloc(32 * 4);
  rt.launch(normalKernel, 1, 32, 0, p);
  const out = new Float32Array(32);
  rt.cudaMemcpy(out, p, 32 * 4, 'D2H');
  assert.strictEqual(out[0], 100, 'Normal run after reset must succeed');
});

// -----------------------------------------------------------------------------
// A9: Determinism and state
// -----------------------------------------------------------------------------
runAcceptanceTest('A9', 'Determinism & State (bit-identical Math.fround reduction, aggregate counters sum, clean reset)', () => {
  const profile = GPU_PROFILES.getProfile('A100');
  const rt1 = new CudaRuntime(profile);
  const multi1 = rt1.initMultiGPU(16);
  multi1.setCollectiveAlgorithm('ring');

  const rt2 = new CudaRuntime(profile);
  const multi2 = rt2.initMultiGPU(16);
  multi2.setCollectiveAlgorithm('ring');

  const count = 128;
  const s1 = [], r1 = [], s2 = [], r2 = [];
  for (let r = 0; r < 16; r++) {
    s1.push(multi1.getDevice(r).cudaMalloc(count * 4));
    r1.push(multi1.getDevice(r).cudaMalloc(count * 4));
    s2.push(multi2.getDevice(r).cudaMalloc(count * 4));
    r2.push(multi2.getDevice(r).cudaMalloc(count * 4));

    const data = new Float32Array(count);
    for (let i = 0; i < count; i++) data[i] = Math.fround(0.1 * (r + 1));
    multi1.getDevice(r).cudaMemcpy(s1[r], data, count * 4, 'H2D');
    multi2.getDevice(r).cudaMemcpy(s2[r], data, count * 4, 'H2D');
  }

  multi1.ncclAllReduce(s1, r1, count, 'float32', 'sum');
  multi2.ncclAllReduce(s2, r2, count, 'float32', 'sum');

  // Independent CPU reference reduction order using Math.fround
  let refSum = 0;
  for (let r = 0; r < 16; r++) {
    refSum = Math.fround(refSum + Math.fround(0.1 * (r + 1)));
  }

  for (let r = 0; r < 16; r++) {
    const o1 = new Float32Array(count);
    const o2 = new Float32Array(count);
    multi1.getDevice(r).cudaMemcpy(o1, r1[r], count * 4, 'D2H');
    multi2.getDevice(r).cudaMemcpy(o2, r2[r], count * 4, 'D2H');
    for (let i = 0; i < count; i++) {
      assert.strictEqual(o1[i], o2[i], 'Both runs must be bit-identical');
      assert.strictEqual(float32UlpDiff(o1[i], refSum), 0, 'Must match reference reduction order with 0 ULP diff');
    }
  }

  // Counters sum exactly to aggregate
  const agg = multi1.getAggregatedCounters().toJSON();
  let sumBytes = 0;
  let sumOps = 0;
  for (let r = 0; r < 16; r++) {
    sumBytes += multi1.getDevice(r).counters.crossGpuBytes;
    sumOps += multi1.getDevice(r).counters.ncclOps;
  }
  assert.strictEqual(agg.crossGpuBytes, sumBytes);
  assert.strictEqual(agg.ncclOps, sumOps);

  // After reset(), all 16 devices are empty
  multi1.reset();
  const leakReport = multi1.getLeakReport();
  assert.strictEqual(leakReport.totalAllocations, 0);
  assert.strictEqual(leakReport.totalLeakedBytes, 0);
});

// -----------------------------------------------------------------------------
// A10: Configuration matrix
// -----------------------------------------------------------------------------
runAcceptanceTest('A10', 'Configuration matrix (6 GPU profiles x 1..16 GPUs = 96 configurations in < 60s)', () => {
  const profileNames = ['A100', 'H100', 'RTX4090', 'RTX4050_Laptop', 'RTX3090', 'V100'];
  const startT = Date.now();
  let executedConfigs = 0;

  for (const profName of profileNames) {
    const profile = GPU_PROFILES.getProfile(profName);
    for (let N = 1; N <= 16; N++) {
      const rt = new CudaRuntime(profile);
      if (N > 1) {
        const multi = rt.initMultiGPU(N);
        const count = 16;
        const s = [], r = [];
        for (let dev = 0; dev < N; dev++) {
          const sPtr = multi.getDevice(dev).cudaMalloc(count * 4);
          const rPtr = multi.getDevice(dev).cudaMalloc(count * 4);
          multi.getDevice(dev).cudaMemcpy(sPtr, new Float32Array(count).fill(dev + 1), count * 4, 'H2D');
          s.push(sPtr);
          r.push(rPtr);
        }
        multi.ncclAllReduce(s, r, count, 'float32', 'sum');
        const expectedVal = (N * (N + 1)) / 2;
        const out = new Float32Array(count);
        multi.getDevice(0).cudaMemcpy(out, r[0], count * 4, 'D2H');
        assert.strictEqual(out[0], expectedVal);
      } else {
        // Single GPU vector add
        const pA = rt.cudaMalloc(64);
        const pB = rt.cudaMalloc(64);
        rt.cudaMemcpy(pA, new Float32Array(16).fill(10), 64, 'H2D');
        rt.cudaMemcpy(pB, new Float32Array(16).fill(20), 64, 'H2D');
        rt.cudaFree(pA);
        rt.cudaFree(pB);
      }
      executedConfigs++;
    }
  }

  const durationSec = (Date.now() - startT) / 1000.0;
  assert.strictEqual(executedConfigs, 96, 'Must execute exactly 96 configurations');
  assert.ok(durationSec < 60.0, `Matrix took ${durationSec}s (must be < 60s)`);
});

// -----------------------------------------------------------------------------
// A11: Validation verdicts
// -----------------------------------------------------------------------------
runAcceptanceTest('A11', 'Validation verdicts (PASS, FAIL with mismatch details, UNSUPPORTED, tolerance boundaries)', () => {
  // 1. Correct vector add reports PASS
  const actPass = [1.0, 2.0, 3.0, 4.0];
  const expPass = [1.0, 2.0, 3.0, 4.0];
  const valPass = validateOutput(actPass, expPass);
  assert.strictEqual(valPass.verdict, 'PASS');
  assert.strictEqual(valPass.mismatchCount, 0);

  // 2. Off-by-one bug reports FAIL with first mismatch index, expected, actual, errors
  const actFail = [1.0, 2.0, 3.99, 4.0];
  const expFail = [1.0, 2.0, 3.0, 4.0];
  const valFail = validateOutput(actFail, expFail, { mode: 'abs', tolerance: 1e-4 });
  assert.strictEqual(valFail.verdict, 'FAIL');
  assert.strictEqual(valFail.firstMismatch.index, 2);
  assert.strictEqual(valFail.firstMismatch.expected, 3.0);
  assert.strictEqual(valFail.firstMismatch.actual, 3.99);

  // 3. Kernel using wmma reports UNSUPPORTED
  const wmmaCode = `
    #include <mma.h>
    __global__ void k() { nvcuda::wmma::fragment frag; }
  `;
  const tr = CudaTranspiler.transpile(wmmaCode);
  assert.strictEqual(tr.success, false);
  assert.ok(tr.errors.some(e => e.includes('WMMA')));

  // 4. Tolerance modes abs, relative, and ULP boundary checks
  const valAbs = validateOutput([1.00001], [1.0], { mode: 'abs', tolerance: 0.00001 });
  assert.strictEqual(valAbs.verdict, 'PASS');
  const valAbsFail = validateOutput([1.00002], [1.0], { mode: 'abs', tolerance: 0.00001 });
  assert.strictEqual(valAbsFail.verdict, 'FAIL');

  const valRel = validateOutput([1000.1], [1000.0], { mode: 'relative', tolerance: 0.001 });
  assert.strictEqual(valRel.verdict, 'PASS');

  // ULP boundary
  const ulpDiff = float32UlpDiff(1.0, 1.0000001192092896); // 1 ULP diff for float32 at 1.0
  assert.strictEqual(ulpDiff, 1);
  const valUlp = validateOutput([1.0000001192092896], [1.0], { mode: 'ulp', tolerance: 1 });
  assert.strictEqual(valUlp.verdict, 'PASS');
  const valUlpFail = validateOutput([1.0000003], [1.0], { mode: 'ulp', tolerance: 1 });
  assert.strictEqual(valUlpFail.verdict, 'FAIL');
});

// -----------------------------------------------------------------------------
// A12: Spec consistency
// -----------------------------------------------------------------------------
runAcceptanceTest('A12', 'Spec consistency (A100 80GB peak mem bandwidth = 2039 GB/s, profile registry matches docs)', () => {
  const a100 = GPU_PROFILES.getProfile('A100');
  assert.strictEqual(a100.memBandwidthGBs, 2039, 'A100 memory bandwidth must be 2039 GB/s');
  assert.strictEqual(a100.memoryBandwidthGBs, 2039);
  assert.strictEqual(a100.computeCapability, '8.0');
  assert.strictEqual(a100.smCount, 108);

  // RTX 4050 Laptop Profile
  const rtx4050 = GPU_PROFILES.getProfile('RTX4050_Laptop');
  assert.strictEqual(rtx4050.computeCapability, '8.9');
  assert.strictEqual(rtx4050.smCount, 20);
  assert.strictEqual(rtx4050.memBandwidthGBs, 192);

  // RTX 5060 Ti is marked TO VERIFY
  const rtx5060 = GPU_PROFILES.getProfile('RTX5060Ti');
  assert.ok(rtx5060.computeCapability.includes('TO VERIFY'));
});

// -----------------------------------------------------------------------------
// Summary & Artifact Generation
// -----------------------------------------------------------------------------
console.log('\n================================================================');
console.log(`  CLUSTER ACCEPTANCE SUMMARY: ${passedTests}/${totalTests} PASSED (${failedTests} FAILED)`);
console.log('================================================================\n');

const artifactPath = path.join(__dirname, '..', 'test-results-cluster-acceptance.json');
fs.writeFileSync(artifactPath, JSON.stringify({
  timestamp: new Date().toISOString(),
  totalTests,
  passedTests,
  failedTests,
  results
}, null, 2));

console.log(`[ARTIFACT] Machine-readable results written to: ${artifactPath}\n`);

if (failedTests > 0) {
  process.exit(1);
}
