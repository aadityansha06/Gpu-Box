/**
 * scripts/patch-runtime-full.js
 * ─────────────────────────────
 * Complete implementation of Phase 3 and Phase 4 features in demo/cuda-runtime.js.
 */

const fs = require('fs');
const path = require('path');

const RUNTIME_PATH = path.join(__dirname, '..', 'demo', 'cuda-runtime.js');
let code = fs.readFileSync(RUNTIME_PATH, 'utf8');

// 1. DevicePointer: add sparse buffer delegates
if (!code.includes('this._sparseBuffer')) {
  // Update get(index)
  code = code.replace(
    'get(index) {',
    `get(index) {
      if (this._sparseBuffer) {
        if (this._isFreed) throw new Error('cudaErrorIllegalAddress: use-after-free detected');
        const absIdx = this._offset + index;
        if (this._counters) this._counters.recordGlobalLoad(this._elementSize, absIdx * this._elementSize);
        return this._sparseBuffer.get(absIdx);
      }`
  );

  // Update set(index, value)
  code = code.replace(
    'set(index, value) {',
    `set(index, value) {
      if (this._sparseBuffer) {
        if (this._isFreed) throw new Error('cudaErrorIllegalAddress: use-after-free detected');
        const absIdx = this._offset + index;
        if (this._counters) this._counters.recordGlobalStore(this._elementSize, absIdx * this._elementSize);
        this._sparseBuffer.set(absIdx, value);
        return;
      }`
  );

  // Update _rawGet(index)
  code = code.replace(
    '_rawGet(index) {',
    `_rawGet(index) {
      if (this._sparseBuffer) return this._sparseBuffer.get(this._offset + index);`
  );

  // Update _rawSet(index, value)
  code = code.replace(
    '_rawSet(index, value) {',
    `_rawSet(index, value) {
      if (this._sparseBuffer) { this._sparseBuffer.set(this._offset + index, value); return; }`
  );

  console.log('[patch-full] DevicePointer sparse buffer support added.');
}

// 2. SimMemory: capacity checks and sparse virtual allocations
if (!code.includes('this._capacityBytes')) {
  code = code.replace(
    'class SimMemory {',
    `class SimMemory {
    constructor(counters, deviceId = 0, profile) {
      this._counters = counters;
      this._deviceId = deviceId;
      this._profile = profile || (typeof GPU_PROFILES !== 'undefined' ? GPU_PROFILES.getProfile('A100') : { globalMemoryGB: 80 });
      this._capacityBytes = (this._profile.globalMemoryGB || 80) * 1024 * 1024 * 1024;
      this._allocatedBytes = 0;
      /** @type {Map<number, {buffer: ArrayBuffer|null, sparse: any, sizeBytes: number}>} */
      this._allocations = new Map();
    }`
  );

  code = code.replace(
    'constructor(counters, deviceId = 0) {',
    '// constructor replaced'
  );

  // Update cudaMalloc
  const oldMalloc = `    cudaMalloc(sizeBytes) {
      if (typeof sizeBytes !== 'number' || sizeBytes < 0) {
        throw new Error('cudaErrorMemoryAllocation: invalid allocation size ' + sizeBytes);
      }
      // Round up to 4-byte alignment
      const aligned = Math.max(0, Math.ceil(sizeBytes / 4) * 4);
      const buffer = new ArrayBuffer(aligned);
      const allocId = _nextAllocId++;
      const elementSize = 4;
      const allocLength = aligned / elementSize;

      this._allocations.set(allocId, { buffer, sizeBytes: aligned });

      const ptr = new DevicePointer(buffer, 0, elementSize, allocId, this._counters, allocLength);
      ptr._deviceId = this._deviceId !== undefined ? this._deviceId : 0;
      return ptr;
    }`;

  const newMalloc = `    cudaMalloc(sizeBytes) {
      if (typeof sizeBytes !== 'number' || sizeBytes < 0) {
        throw new Error('cudaErrorMemoryAllocation: invalid allocation size ' + sizeBytes);
      }
      // Round up to 4-byte alignment
      const aligned = Math.max(0, Math.ceil(sizeBytes / 4) * 4);
      if (this._allocatedBytes + aligned > this._capacityBytes) {
        throw new Error('cudaErrorMemoryAllocation: out of memory on device ' + this._deviceId + ' (requested ' + aligned + ' bytes, allocated ' + this._allocatedBytes + ' bytes, capacity ' + this._capacityBytes + ' bytes)');
      }

      const allocId = _nextAllocId++;
      const elementSize = 4;
      const allocLength = aligned / elementSize;
      const maxDirectBytes = 16 * 1024 * 1024; // 16 MB

      if (aligned > maxDirectBytes) {
        const sparse = new SparseMemModule.SparseVirtualBuffer(aligned, this._deviceId);
        this._allocations.set(allocId, { buffer: null, sparse, sizeBytes: aligned });
        this._allocatedBytes += aligned;
        const ptr = new DevicePointer(null, 0, elementSize, allocId, this._counters, allocLength);
        ptr._sparseBuffer = sparse;
        ptr._deviceId = this._deviceId !== undefined ? this._deviceId : 0;
        return ptr;
      }

      const buffer = new ArrayBuffer(aligned);
      this._allocations.set(allocId, { buffer, sparse: null, sizeBytes: aligned });
      this._allocatedBytes += aligned;

      const ptr = new DevicePointer(buffer, 0, elementSize, allocId, this._counters, allocLength);
      ptr._deviceId = this._deviceId !== undefined ? this._deviceId : 0;
      return ptr;
    }`;

  code = code.replace(oldMalloc, newMalloc);

  // Update cudaFree
  const oldFree = `    cudaFree(ptr) {
      if (ptr && ptr._allocId !== undefined) {
        if (!this._allocations.has(ptr._allocId)) {
          throw new Error('cudaErrorInvalidValue: double free or invalid pointer in cudaFree');
        }
        this._allocations.delete(ptr._allocId);
        ptr._isFreed = true;
      }
    }`;

  const newFree = `    cudaFree(ptr) {
      if (ptr && ptr._allocId !== undefined) {
        const alloc = this._allocations.get(ptr._allocId);
        if (!alloc) {
          throw new Error('cudaErrorInvalidValue: double free detected in cudaFree');
        }
        this._allocatedBytes = Math.max(0, this._allocatedBytes - alloc.sizeBytes);
        if (alloc.sparse) {
          alloc.sparse.free();
        }
        this._allocations.delete(ptr._allocId);
        ptr._isFreed = true;
      }
    }`;

  code = code.replace(oldFree, newFree);
  console.log('[patch-full] SimMemory capacity & sparse virtual VRAM added.');
}

// 3. SharedMemory: Race Detection
if (!code.includes('this._raceDetector')) {
  // Add race detector instantiation in SharedMemory constructor
  code = code.replace(
    'this._staticArrays = new Map();',
    `this._staticArrays = new Map();
      this._raceDetector = new RaceDetectorModule.SharedMemoryRaceDetector();
      this._currentCtx = null;`
  );

  // Add setActiveContext in SharedMemory
  code = code.replace(
    'getArray(name, size, type = \'float32\') {',
    `setActiveContext(ctx) {
      this._currentCtx = ctx;
    }

    getArray(name, size, type = 'float32') {`
  );

  // Instrument static array get & set
  code = code.replace(
    'self._counters.recordSharedLoad(idx * 4);',
    `self._counters.recordSharedLoad(idx * 4);
              if (self._raceDetector && self._currentCtx) self._raceDetector.recordAccess(name, idx, false, self._currentCtx);`
  );

  code = code.replace(
    'self._counters.recordSharedStore(idx * 4);',
    `self._counters.recordSharedStore(idx * 4);
              if (self._raceDetector && self._currentCtx) self._raceDetector.recordAccess(name, idx, true, self._currentCtx);`
  );

  // Instrument 2D static array get & set
  code = code.replace(
    'self._counters.recordSharedLoad((rowIdx * c + colIdx) * 4);',
    `self._counters.recordSharedLoad((rowIdx * c + colIdx) * 4);
                if (self._raceDetector && self._currentCtx) self._raceDetector.recordAccess(name, rowIdx * c + colIdx, false, self._currentCtx);`
  );

  code = code.replace(
    'self._counters.recordSharedStore((rowIdx * c + colIdx) * 4);',
    `self._counters.recordSharedStore((rowIdx * c + colIdx) * 4);
                if (self._raceDetector && self._currentCtx) self._raceDetector.recordAccess(name, rowIdx * c + colIdx, true, self._currentCtx);`
  );

  // Instrument KernelLauncher to set active context and advance barrier
  code = code.replace(
    'counters.syncthreadsCount++;',
    `counters.syncthreadsCount++;
                if (sharedMem && sharedMem._raceDetector) sharedMem._raceDetector.advanceBarrier();`
  );

  code = code.replace(
    'counters.setActiveContext(ctx.warpId, ctx.linearIdx);',
    `counters.setActiveContext(ctx.warpId, ctx.linearIdx);
                if (sharedMem && sharedMem.setActiveContext) sharedMem.setActiveContext(ctx);`
  );

  console.log('[patch-full] SharedMemory barrier-epoch race detection added.');
}

// 4. KernelLauncher Watchdog & Step Limit
if (!code.includes('Watchdog timeout exceeded')) {
  code = code.replace(
    'let isGenerator = false;',
    `let isGenerator = false;
            let stepCount = 0;
            const maxSteps = this.maxSteps || 5000000;
            const maxTimeMs = this.maxTimeMs || 5000;
            const launchStartTime = Date.now();`
  );

  code = code.replace(
    'const item = threadContexts[curIdx];',
    `stepCount++;
              if (stepCount > maxSteps || (stepCount % 5000 === 0 && (Date.now() - launchStartTime) > maxTimeMs)) {
                throw new Error('cudaErrorLaunchFailure: Watchdog timeout exceeded (' + stepCount + ' steps). Runaway loop detected in kernel "' + kName + '" at block (' + currentBlock.x + ',' + currentBlock.y + ',' + currentBlock.z + '), thread (' + currentThread.x + ',' + currentThread.y + ',' + currentThread.z + ')');
              }
              const item = threadContexts[curIdx];`
  );

  console.log('[patch-full] KernelLauncher watchdog timeout & step limit added.');
}

fs.writeFileSync(RUNTIME_PATH, code, 'utf8');
console.log('[patch-full] Patches 1-4 applied successfully.');
