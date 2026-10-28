// Test generator SIMT execution
function* testKernel(tid, shared, sync, shfl) {
  let val = tid + 1; // thread 0 has 1, thread 1 has 2, ...
  console.log(`Thread ${tid} started with val = ${val}`);

  // Warp shuffle down by 1
  val += yield* shfl(val, 1);
  console.log(`Thread ${tid} after shfl: val = ${val}`);

  // Write to shared memory
  shared[tid] = val;

  // Barrier
  yield* sync();

  // Read neighbor from shared memory
  let neighbor = shared[(tid + 1) % 4];
  console.log(`Thread ${tid} read neighbor: ${neighbor}`);
}

// SIMT Runner
function runSimtBlock(numThreads) {
  const shared = new Float32Array(numThreads);
  const warps = new Map();

  function* shfl(val, delta) {
    const res = yield { type: 'shfl', val, delta };
    return res;
  }

  function* sync() {
    yield { type: 'sync' };
  }

  const threads = [];
  for (let i = 0; i < numThreads; i++) {
    const tid = i;
    threads.push({
      tid,
      done: false,
      gen: testKernel(tid, shared, sync, shfl),
      pendingShfl: null
    });
  }

  let running = true;
  while (running) {
    running = false;
    // Step all active threads
    for (const th of threads) {
      if (!th.done && !th.blocked) {
        const res = th.gen.next(th.nextVal);
        th.nextVal = undefined;
        if (res.done) {
          th.done = true;
        } else {
          running = true;
          if (res.value && res.value.type === 'sync') {
            th.blocked = true;
          } else if (res.value && res.value.type === 'shfl') {
            th.pendingShfl = res.value;
          }
        }
      }
    }

    // Process shuffles for threads that yielded shfl
    const shflThreads = threads.filter(t => t.pendingShfl);
    if (shflThreads.length > 0) {
      // Collect values by lane (assuming 1 warp of 4 threads here)
      const vals = new Float32Array(numThreads);
      for (const th of shflThreads) {
        vals[th.tid] = th.pendingShfl.val;
      }
      for (const th of shflThreads) {
        const delta = th.pendingShfl.delta;
        const srcLane = th.tid + delta;
        th.nextVal = srcLane < numThreads ? vals[srcLane] : th.pendingShfl.val;
        th.pendingShfl = null;
      }
    }

    // Check barrier sync: if all running threads are at sync, wake them all
    const activeThreads = threads.filter(t => !t.done);
    if (activeThreads.length > 0 && activeThreads.every(t => t.blocked)) {
      for (const th of activeThreads) th.blocked = false;
    }
  }
}

runSimtBlock(4);
