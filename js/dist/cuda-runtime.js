/**
 * cuda-runtime.js — Complete GPU Simulation Runtime
 *
 * A browser-based CUDA GPU simulator that executes transpiled CUDA kernels
 * in JavaScript. Provides accurate thread scheduling, memory hierarchy,
 * atomic operations, warp-level primitives, multi-GPU support, and
 * performance counter tracking.
 *
 * Exports: window.CudaRuntime
 */
(function () {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════════
  // Constants
  // ═══════════════════════════════════════════════════════════════════════════

  const WARP_SIZE = 32;
  const SHARED_MEM_BANKS = 32;
  const SECTOR_SIZE_BYTES = 32;
  const POISON_VALUE = 0xDEADBEEF;

  const CUDA_MEMCPY_H2D = 'H2D';
  const CUDA_MEMCPY_D2H = 'D2H';
  const CUDA_MEMCPY_D2D = 'D2D';
  const CUDA_MEMCPY_H2H = 'H2H';

  // CUDA math constants
  const CUDA_INFINITY = Infinity;
  const CUDA_NAN = NaN;
  const CUDA_FLT_MAX = 3.4028235e+38;
  const CUDA_FLT_MIN = 1.1754944e-38;
  const CUDA_FLT_EPSILON = 1.1920929e-7;
  const CUDA_M_PI = Math.PI;
  const CUDA_M_PI_2 = Math.PI / 2;
  const CUDA_M_SQRT2 = Math.SQRT2;

  // ═══════════════════════════════════════════════════════════════════════════
  // Dim3 helper
  // ═══════════════════════════════════════════════════════════════════════════

  /** Normalize a grid/block dimension to {x, y, z}. Accepts number, [x,y,z], or object. */
  function dim3(v) {
    const toInt = (n) => Math.max(1, Math.floor(Number(n) || 1));
    if (typeof v === 'number') return { x: toInt(v), y: 1, z: 1 };
    if (Array.isArray(v)) return { x: toInt(v[0]), y: toInt(v[1]), z: toInt(v[2]) };
    if (v && typeof v === 'object') return { x: toInt(v.x), y: toInt(v.y), z: toInt(v.z) };
    return { x: 1, y: 1, z: 1 };
  }

  /** Total number of elements in a dim3 */
  function dim3Volume(d) {
    return d.x * d.y * d.z;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SimCounters — performance counter tracking
  // ═══════════════════════════════════════════════════════════════════════════

  class SimCounters {
    constructor(gpuProfile) {
      this._gpuProfile = gpuProfile || {};
      this.reset();
    }

    reset() {
      // Launch counters
      this.threadsLaunched = 0;
      this.blocksLaunched = 0;

      // Global memory
      this.globalLoads = 0;
      this.globalStores = 0;
      this.globalLoadBytes = 0;
      this.globalStoreBytes = 0;
      this.globalTransactions = 0;

      // Shared memory
      this.sharedLoads = 0;
      this.sharedStores = 0;
      this.sharedLoadBytes = 0;
      this.sharedStoreBytes = 0;
      this.bankConflicts = 0;

      // Synchronization
      this.syncthreadsCount = 0;

      // Branch divergence
      this.divergentBranches = 0;

      // Atomics
      this.atomicOps = 0;
      this.atomicMaxContention = 0;

      // Memory copy
      this.memcpyH2DBytes = 0;
      this.memcpyD2HBytes = 0;

      // Multi-GPU / NCCL
      this.crossGpuBytes = 0;
      this.ncclOps = 0;

      // Instruction counters for utilization
      this.flopCount = 0;

      // Warp-level tracking: per-warp address sets for coalescing analysis
      this._warpGlobalAccesses = [];
      this._warpAccessWindows = new Map();
      this._threadAccessSeq = new Map();
      this._activeWarpId = undefined;
      this._activeThreadId = undefined;
      // Per-warp shared bank tracking
      this._warpSharedBanks = [];
      // Atomic contention tracking per kernel launch
      this._atomicContentionMap = new Map();
    }

    /** Set active thread and warp context for SIMT analysis */
    setActiveContext(warpId, threadId) {
      this._activeWarpId = warpId;
      this._activeThreadId = threadId;
    }

    /** Clear active thread context */
    clearActiveContext() {
      this._activeWarpId = undefined;
      this._activeThreadId = undefined;
    }

    /** Record a global memory load */
    recordGlobalLoad(elementBytes, byteAddr) {
      this.globalLoads++;
      this.globalLoadBytes += (elementBytes || 4);
      this._recordGlobalAccess(byteAddr);
    }

    /** Record a global memory store */
    recordGlobalStore(elementBytes, byteAddr) {
      this.globalStores++;
      this.globalStoreBytes += (elementBytes || 4);
      this._recordGlobalAccess(byteAddr);
    }

    /** Record internal address access for 32-byte transaction coalescing */
    _recordGlobalAccess(byteAddr) {
      const addr = byteAddr !== undefined ? byteAddr : 0;
      if (this._activeWarpId !== undefined && this._activeThreadId !== undefined) {
        const threadKey = this._activeWarpId + ':' + this._activeThreadId;
        const seq = this._threadAccessSeq.get(threadKey) || 0;
        this._threadAccessSeq.set(threadKey, seq + 1);
        const windowKey = this._activeWarpId + ':' + seq;
        let window = this._warpAccessWindows.get(windowKey);
        if (!window) {
          window = [];
          this._warpAccessWindows.set(windowKey, window);
        }
        window.push(addr);
      } else {
        this.globalTransactions++;
      }
    }

    /** Flush all pending warp global access windows and compute transaction counts */
    flushWarpGlobalAccesses() {
      if (!this._warpAccessWindows || this._warpAccessWindows.size === 0) return;
      for (const [_, addresses] of this._warpAccessWindows) {
        if (!addresses || addresses.length === 0) continue;
        const sectors = new Set();
        for (let i = 0; i < addresses.length; i++) {
          sectors.add(Math.floor(addresses[i] / SECTOR_SIZE_BYTES));
        }
        this.globalTransactions += sectors.size;
      }
      this._warpAccessWindows.clear();
      this._threadAccessSeq.clear();
    }

    /** Record branch divergence */
    recordDivergentBranch() {
      this.divergentBranches++;
    }

    /** Record a shared memory load */
    recordSharedLoad(byteAddr) {
      this.sharedLoads++;
      this.sharedLoadBytes += 4;
      const bank = Math.floor(byteAddr / 4) % SHARED_MEM_BANKS;
      this._warpSharedBanks.push(bank);
    }

    /** Record a shared memory store */
    recordSharedStore(byteAddr) {
      this.sharedStores++;
      this.sharedStoreBytes += 4;
      const bank = Math.floor(byteAddr / 4) % SHARED_MEM_BANKS;
      this._warpSharedBanks.push(bank);
    }

    /** Flush per-warp shared bank accesses and count conflicts */
    flushSharedBankAccesses() {
      if (this._warpSharedBanks.length <= 1) {
        this._warpSharedBanks = [];
        return;
      }
      // Count duplicate bank accesses in this warp-step
      const bankCounts = new Int32Array(SHARED_MEM_BANKS);
      for (let i = 0; i < this._warpSharedBanks.length; i++) {
        bankCounts[this._warpSharedBanks[i]]++;
      }
      let conflicts = 0;
      for (let b = 0; b < SHARED_MEM_BANKS; b++) {
        if (bankCounts[b] > 1) {
          conflicts += bankCounts[b] - 1;
        }
      }
      this.bankConflicts += conflicts;
      this._warpSharedBanks = [];
    }

    /** Record a warp's global memory access for transaction/coalescing analysis */
    recordWarpGlobalAccess(byteAddresses) {
      // Count 32-byte sectors touched
      const sectors = new Set();
      for (let i = 0; i < byteAddresses.length; i++) {
        sectors.add(Math.floor(byteAddresses[i] / SECTOR_SIZE_BYTES));
      }
      this.globalTransactions += sectors.size;
    }

    /** Record an atomic operation and track contention */
    recordAtomic(allocId, index) {
      this.atomicOps++;
      const key = allocId + ':' + index;
      const prev = this._atomicContentionMap.get(key) || 0;
      const next = prev + 1;
      this._atomicContentionMap.set(key, next);
      if (next > this.atomicMaxContention) {
        this.atomicMaxContention = next;
      }
    }

    /** Record a floating-point operation */
    recordFlop(count) {
      this.flopCount += (count || 1);
    }

    // ── Derived metrics ──

    /** Estimated cycles based on GPU profile latencies */
    get estimatedCycles() {
      const prof = this._gpuProfile;
      const globalLatency = prof.globalMemLatencyCycles || 400;
      const sharedLatency = prof.sharedMemLatencyCycles || 20;
      const atomicLatency = prof.atomicLatencyCycles || 50;
      const syncLatency = prof.syncLatencyCycles || 5;
      const flopLatency = prof.flopLatencyCycles || 1;

      return (this.globalLoads + this.globalStores) * globalLatency +
        (this.sharedLoads + this.sharedStores) * sharedLatency +
        this.atomicOps * atomicLatency +
        this.syncthreadsCount * syncLatency +
        this.flopCount * flopLatency;
    }

    /** Estimated wall-clock time in milliseconds */
    get estimatedTimeMs() {
      const clockHz = (this._gpuProfile.clockSpeedMHz || 1500) * 1e6;
      return (this.estimatedCycles / clockHz) * 1000;
    }

    /** Compute utilization: FLOPs / peak FLOPs over the estimated time */
    get computeUtilization() {
      const peakGFLOPS = this._gpuProfile.peakTFLOPS
        ? this._gpuProfile.peakTFLOPS * 1000
        : (this._gpuProfile.peakGFLOPS || 10000);
      const timeSec = this.estimatedTimeMs / 1000;
      if (timeSec === 0) return 0;
      const actualGFLOPS = (this.flopCount / 1e9) / timeSec;
      return Math.min(actualGFLOPS / peakGFLOPS, 1.0);
    }

    /** Memory bandwidth utilization */
    get memoryUtilization() {
      const peakBWGBs = this._gpuProfile.memBandwidthGBs || 900;
      const totalBytes = this.globalLoadBytes + this.globalStoreBytes;
      const timeSec = this.estimatedTimeMs / 1000;
      if (timeSec === 0) return 0;
      const actualGBs = (totalBytes / 1e9) / timeSec;
      return Math.min(actualGBs / peakBWGBs, 1.0);
    }

    /** Occupancy: threads / maxThreadsPerSM */
    get occupancy() {
      const maxThreadsPerSM = this._gpuProfile.maxThreadsPerSM || 2048;
      const smCount = this._gpuProfile.smCount || 80;
      if (smCount === 0) return 0;
      return Math.min(this.threadsLaunched / (maxThreadsPerSM * smCount), 1.0);
    }

    get totalBytesRead() { return this.globalLoadBytes; }
    get totalBytesWritten() { return this.globalStoreBytes; }
    get coalescedTransactions() { return this.globalTransactions; }
    get dramReadBytes() { return this.globalLoadBytes; }
    get dramWriteBytes() { return this.globalStoreBytes; }

    toJSON() {
      const res = {
        threadsLaunched: this.threadsLaunched,
        blocksLaunched: this.blocksLaunched,
        globalLoads: this.globalLoads,
        globalStores: this.globalStores,
        globalLoadBytes: this.globalLoadBytes,
        globalStoreBytes: this.globalStoreBytes,
        totalBytesRead: this.globalLoadBytes,
        totalBytesWritten: this.globalStoreBytes,
        globalTransactions: this.globalTransactions,
        coalescedTransactions: this.globalTransactions,
        dramReadBytes: this.globalLoadBytes,
        dramWriteBytes: this.globalStoreBytes,
        sharedLoads: this.sharedLoads,
        sharedStores: this.sharedStores,
        sharedLoadBytes: this.sharedLoadBytes,
        sharedStoreBytes: this.sharedStoreBytes,
        bankConflicts: this.bankConflicts,
        syncthreadsCount: this.syncthreadsCount,
        divergentBranches: this.divergentBranches,
        atomicOps: this.atomicOps,
        atomicMaxContention: this.atomicMaxContention,
        memcpyH2DBytes: this.memcpyH2DBytes,
        memcpyD2HBytes: this.memcpyD2HBytes,
        crossGpuBytes: this.crossGpuBytes,
        ncclOps: this.ncclOps,
        flopCount: this.flopCount,
        estimatedCycles: this.estimatedCycles,
        estimatedTimeMs: this.estimatedTimeMs,
        computeUtilization: this.computeUtilization,
        memoryUtilization: this.memoryUtilization,
        occupancy: this.occupancy
      };
      if (this.perDevice) {
        res.perDevice = this.perDevice;
      }
      return res;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // DevicePointer — typed view into the global memory arena
  // ═══════════════════════════════════════════════════════════════════════════

  let _nextAllocId = 1;

  class DevicePointer {
    /**
     * @param {ArrayBuffer} buffer   – Backing ArrayBuffer
     * @param {number}      offset   – Element offset within the typed view
     * @param {number}      elementSize – Bytes per element (1, 2, 4)
     * @param {number}      allocId  – Unique allocation identifier
     * @param {SimCounters} counters – Runtime counters for tracking
     * @param {number}      allocLength – Total elements in this allocation
     */
    constructor(buffer, offset, elementSize, allocId, counters, allocLength) {
      this._buffer = buffer;
      this._offset = offset;
      this._elementSize = elementSize;
      this._allocId = allocId;
      this._counters = counters;
      this._allocLength = allocLength;

      // Build the appropriate typed view
      if (elementSize === 4) {
        this._f32 = new Float32Array(buffer);
        this._i32 = new Int32Array(buffer);
      }
      this._u8 = new Uint8Array(buffer);

      return new Proxy(this, {
        get(target, prop, receiver) {
          if (typeof prop === 'string' && /^\d+$/.test(prop)) {
            return target.get(parseInt(prop, 10));
          }
          return Reflect.get(target, prop, receiver);
        },
        set(target, prop, value, receiver) {
          if (typeof prop === 'string' && /^\d+$/.test(prop)) {
            target.set(parseInt(prop, 10), value);
            return true;
          }
          return Reflect.set(target, prop, value, receiver);
        }
      });
    }

    /** Read element at logical index (relative to this pointer's offset) */
    get(index) {
      const absIdx = this._offset + index;
      if (absIdx < 0 || absIdx >= this._allocLength) {
        throw new RangeError(
          'DevicePointer OOB read: index ' + absIdx + ' (alloc size ' + this._allocLength + ')'
        );
      }
      if (this._counters) {
        this._counters.recordGlobalLoad(this._elementSize, absIdx * this._elementSize);
      }
      if (this._elementSize === 4) {
        return this._f32[absIdx];
      }
      return this._u8[absIdx];
    }

    /** Write element at logical index */
    set(index, value) {
      const absIdx = this._offset + index;
      if (absIdx < 0 || absIdx >= this._allocLength) {
        throw new RangeError(
          'DevicePointer OOB write: index ' + absIdx + ' (alloc size ' + this._allocLength + ')'
        );
      }
      if (this._counters) {
        this._counters.recordGlobalStore(this._elementSize, absIdx * this._elementSize);
      }
      if (this._elementSize === 4) {
        this._f32[absIdx] = value;
      } else {
        this._u8[absIdx] = value;
      }
    }

    /** Raw read (bypasses counter tracking) – used for atomics, memcpy */
    _rawGet(index) {
      const absIdx = this._offset + index;
      if (this._elementSize === 4) return this._f32[absIdx];
      return this._u8[absIdx];
    }

    /** Raw write (bypasses counter tracking) */
    _rawSet(index, value) {
      const absIdx = this._offset + index;
      if (this._elementSize === 4) this._f32[absIdx] = value;
      else this._u8[absIdx] = value;
    }

    /** Raw i32 read at absolute index (for atomics) */
    _rawGetI32(index) {
      return this._i32[this._offset + index];
    }

    /** Raw i32 write at absolute index (for atomics) */
    _rawSetI32(index, value) {
      this._i32[this._offset + index] = value;
    }

    /** Pointer arithmetic: advance by n elements */
    add(n) {
      return new DevicePointer(
        this._buffer, this._offset + n, this._elementSize,
        this._allocId, this._counters, this._allocLength
      );
    }

    /** Pointer arithmetic: retreat by n elements */
    sub(n) {
      return new DevicePointer(
        this._buffer, this._offset - n, this._elementSize,
        this._allocId, this._counters, this._allocLength
      );
    }

    /** Reinterpret as float pointer (4 bytes per element) */
    castFloat() {
      const byteOffset = this._offset * this._elementSize;
      const newOffset = Math.floor(byteOffset / 4);
      const newLength = Math.floor((this._allocLength * this._elementSize) / 4);
      return new DevicePointer(
        this._buffer, newOffset, 4, this._allocId, this._counters, newLength
      );
    }

    /** Reinterpret as int pointer (4 bytes per element) */
    castInt() {
      return this.castFloat(); // same layout, i32 view is always available
    }

    /** Read a float4 (4 floats) at float4-index (16-byte aligned) */
    getFloat4(index) {
      const base = (index || 0) * 4;
      return {
        x: this.get(base),
        y: this.get(base + 1),
        z: this.get(base + 2),
        w: this.get(base + 3),
      };
    }

    /** Write a float4 at float4-index */
    setFloat4(index, v) {
      const base = (index || 0) * 4;
      const val = v || {};
      this.set(base, val.x !== undefined ? val.x : 0);
      this.set(base + 1, val.y !== undefined ? val.y : 0);
      this.set(base + 2, val.z !== undefined ? val.z : 0);
      this.set(base + 3, val.w !== undefined ? val.w : 0);
    }

    /** Read a float2 (2 floats) */
    getFloat2(index) {
      const base = (index || 0) * 2;
      return { x: this.get(base), y: this.get(base + 1) };
    }

    /** Write a float2 */
    setFloat2(index, v) {
      const base = (index || 0) * 2;
      const val = v || {};
      this.set(base, val.x !== undefined ? val.x : 0);
      this.set(base + 1, val.y !== undefined ? val.y : 0);
    }

    /** Read an int4 */
    getInt4(index) {
      return this.getFloat4(index);
    }

    /** Write an int4 */
    setInt4(index, v) {
      this.setFloat4(index, v);
    }

    /** Copy count elements from device to a new host Float32Array */
    toFloat32Array(count) {
      const result = new Float32Array(count);
      const f32 = new Float32Array(this._buffer);
      for (let i = 0; i < count; i++) {
        result[i] = f32[this._offset + i];
      }
      return result;
    }

    /** Copy a host Float32Array into device memory starting at this pointer */
    fromFloat32Array(arr) {
      const f32 = new Float32Array(this._buffer);
      for (let i = 0; i < arr.length; i++) {
        f32[this._offset + i] = arr[i];
      }
    }

    /** Byte address of element at index (for coalescing analysis) */
    _byteAddress(index) {
      return (this._offset + index) * this._elementSize;
    }

    /** Returns the underlying ArrayBuffer */
    get buffer() { return this._buffer; }

    /** Returns the element offset */
    get offset() { return this._offset; }

    /** Returns the element size in bytes */
    get elementSize() { return this._elementSize; }

    /** Returns the allocation ID */
    get allocId() { return this._allocId; }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SimMemory — global device memory arena
  // ═══════════════════════════════════════════════════════════════════════════

  class SimMemory {
    constructor(counters) {
      this._counters = counters;
      /** @type {Map<number, {buffer: ArrayBuffer, sizeBytes: number}>} */
      this._allocations = new Map();
    }

    /**
     * Allocate device memory.
     * @param {number} sizeBytes – Number of bytes to allocate
     * @returns {DevicePointer} pointer to the allocation (float32 view by default)
     */
    cudaMalloc(sizeBytes) {
      // Round up to 4-byte alignment
      const aligned = Math.ceil(sizeBytes / 4) * 4;
      const buffer = new ArrayBuffer(aligned);
      const allocId = _nextAllocId++;
      const elementSize = 4;
      const allocLength = aligned / elementSize;

      this._allocations.set(allocId, { buffer, sizeBytes: aligned });

      return new DevicePointer(buffer, 0, elementSize, allocId, this._counters, allocLength);
    }

    /**
     * Free a device allocation.
     * @param {DevicePointer} ptr
     */
    cudaFree(ptr) {
      if (ptr && ptr._allocId) {
        this._allocations.delete(ptr._allocId);
      }
    }

    /**
     * Copy data between host and device.
     * @param {DevicePointer|Float32Array|Int32Array|Uint8Array} dst
     * @param {DevicePointer|Float32Array|Int32Array|Uint8Array} src
     * @param {number} sizeBytes
     * @param {string} kind – 'H2D', 'D2H', 'D2D', 'H2H'
     */
    cudaMemcpy(dst, src, sizeBytes, kind) {
      if (!kind || kind === 'DEFAULT' || kind === 'cudaMemcpyDefault' || kind === 0) {
        if (dst instanceof DevicePointer && !(src instanceof DevicePointer)) kind = CUDA_MEMCPY_H2D;
        else if (!(dst instanceof DevicePointer) && src instanceof DevicePointer) kind = CUDA_MEMCPY_D2H;
        else if (dst instanceof DevicePointer && src instanceof DevicePointer) kind = CUDA_MEMCPY_D2D;
        else kind = CUDA_MEMCPY_H2H;
      }
      switch (kind) {
        case CUDA_MEMCPY_H2D: {
          // src is host typed array, dst is DevicePointer
          const dstU8 = new Uint8Array(dst._buffer, dst._offset * dst._elementSize);
          if (ArrayBuffer.isView(src)) {
            const copyLen = Math.min(sizeBytes, src.byteLength);
            const srcU8 = new Uint8Array(src.buffer, src.byteOffset, copyLen);
            dstU8.set(srcU8.subarray(0, copyLen));
          } else if (Array.isArray(src)) {
            const f32 = new Float32Array(src);
            const copyLen = Math.min(sizeBytes, f32.byteLength);
            const srcU8 = new Uint8Array(f32.buffer, 0, copyLen);
            dstU8.set(srcU8);
          } else if (typeof src === 'number') {
            new Float32Array(dst._buffer, dst._offset * dst._elementSize, 1)[0] = src;
          } else if (typeof src === 'string') {
            const nums = src.split(',').map(Number).filter(n => !Number.isNaN(n));
            if (nums.length > 0) {
              const f32 = new Float32Array(nums);
              const copyLen = Math.min(sizeBytes, f32.byteLength);
              const srcU8 = new Uint8Array(f32.buffer, 0, copyLen);
              dstU8.set(srcU8);
            }
          }
          this._counters.memcpyH2DBytes += sizeBytes;
          break;
        }
        case CUDA_MEMCPY_D2H: {
          // src is DevicePointer, dst is host typed array
          const dstLen = dst && dst.byteLength !== undefined ? dst.byteLength : sizeBytes;
          const availableSrcBytes = Math.max(0, (src._buffer ? src._buffer.byteLength : 0) - (src._offset * src._elementSize));
          const copyLen = Math.min(sizeBytes, dstLen, availableSrcBytes);
          const srcU8 = new Uint8Array(src._buffer, src._offset * src._elementSize, copyLen);
          if (ArrayBuffer.isView(dst)) {
            const dstU8 = new Uint8Array(dst.buffer, dst.byteOffset);
            dstU8.set(srcU8.subarray(0, copyLen));
          } else if (Array.isArray(dst)) {
            const f32 = new Float32Array(src._buffer, src._offset * src._elementSize, Math.floor(copyLen / 4));
            for (let i = 0; i < f32.length; i++) dst[i] = f32[i];
          }
          this._counters.memcpyD2HBytes += copyLen;
          break;
        }
        case CUDA_MEMCPY_D2D: {
          // Both are DevicePointers
          const availableSrcBytes = Math.max(0, (src._buffer ? src._buffer.byteLength : 0) - (src._offset * src._elementSize));
          const availableDstBytes = Math.max(0, (dst._buffer ? dst._buffer.byteLength : 0) - (dst._offset * dst._elementSize));
          const copyLen = Math.min(sizeBytes, availableSrcBytes, availableDstBytes);
          const srcU8 = new Uint8Array(src._buffer, src._offset * src._elementSize, copyLen);
          const dstU8 = new Uint8Array(dst._buffer, dst._offset * dst._elementSize);
          dstU8.set(srcU8.subarray(0, copyLen));
          break;
        }
        case CUDA_MEMCPY_H2H: {
          // Both are host typed arrays
          const srcU8 = new Uint8Array(src.buffer, src.byteOffset, sizeBytes);
          const dstU8 = new Uint8Array(dst.buffer, dst.byteOffset);
          dstU8.set(srcU8.subarray(0, sizeBytes));
          break;
        }
        default:
          throw new Error('Unknown cudaMemcpy kind: ' + kind);
      }
    }

    /**
     * Set device memory to a byte value.
     * @param {DevicePointer} ptr
     * @param {number} value – Byte value (0-255)
     * @param {number} sizeBytes
     */
    cudaMemset(ptr, value, sizeBytes) {
      const u8 = new Uint8Array(ptr._buffer, ptr._offset * ptr._elementSize);
      for (let i = 0; i < sizeBytes; i++) {
        u8[i] = value & 0xFF;
      }
    }

    /** Destroy all allocations */
    destroy() {
      this._allocations.clear();
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SharedMemory — per-block shared memory with bank conflict tracking
  // ═══════════════════════════════════════════════════════════════════════════

  class SharedMemory {
    /**
     * @param {number} sizeBytes – Size in bytes
     * @param {SimCounters} counters
     */
    constructor(sizeBytes, counters) {
      this._sizeBytes = sizeBytes;
      this._counters = counters;
      // Backing store as float32 (most common use)
      const aligned = Math.max(1024, Math.ceil((sizeBytes || 1024) / 4) * 4);
      this._buffer = new ArrayBuffer(aligned);
      this._f32 = new Float32Array(this._buffer);
      this._i32 = new Int32Array(this._buffer);
      this._u8 = new Uint8Array(this._buffer);

      // Static shared arrays map: name -> typed array
      this._staticArrays = new Map();

      // Poison-fill to detect uninitialized reads
      this._i32.fill(POISON_VALUE);
      // Track which indices have been written
      this._written = new Uint8Array(this._f32.length);

      return new Proxy(this, {
        get(target, prop, receiver) {
          if (typeof prop === 'string' && /^\d+$/.test(prop)) {
            const idx = parseInt(prop, 10);
            return target.get(idx);
          }
          return Reflect.get(target, prop, receiver);
        },
        set(target, prop, value, receiver) {
          if (typeof prop === 'string' && /^\d+$/.test(prop)) {
            const idx = parseInt(prop, 10);
            target.set(idx, value);
            return true;
          }
          return Reflect.set(target, prop, value, receiver);
        }
      });
    }

    /**
     * Get or create a named static shared array shared across all threads in the block.
     * @param {string} name
     * @param {number} size
     * @param {string} type
     * @returns {Float32Array|Int32Array}
     */
    getArray(name, size, type = 'float32') {
      if (!this._staticArrays.has(name)) {
        const count = size || 256;
        const raw = (type === 'int' || type === 'int32') ? new Int32Array(count) : new Float32Array(count);
        const self = this;
        const proxy = new Proxy(raw, {
          get(target, prop, receiver) {
            if (typeof prop === 'string' && /^\d+$/.test(prop)) {
              const idx = parseInt(prop, 10);
              self._counters.recordSharedLoad(idx * 4);
              return target[idx];
            }
            return Reflect.get(target, prop, receiver);
          },
          set(target, prop, value, receiver) {
            if (typeof prop === 'string' && /^\d+$/.test(prop)) {
              const idx = parseInt(prop, 10);
              self._counters.recordSharedStore(idx * 4);
              target[idx] = value;
              return true;
            }
            return Reflect.set(target, prop, value, receiver);
          }
        });
        this._staticArrays.set(name, proxy);
      }
      return this._staticArrays.get(name);
    }

    /**
     * Get or create a 2D static shared array (e.g. As[TILE][TILE]) shared across all threads in the block.
     * @param {string} name
     * @param {number} rows
     * @param {number} cols
     * @param {string} type
     * @returns {Array<Float32Array|Int32Array>}
     */
    get2DArray(name, rows, cols, type = 'float32') {
      if (!this._staticArrays.has(name)) {
        const r = rows || 16;
        const c = cols || 16;
        const grid = [];
        const self = this;
        for (let i = 0; i < r; i++) {
          const rawRow = (type === 'int' || type === 'int32') ? new Int32Array(c) : new Float32Array(c);
          const rowIdx = i;
          const rowProxy = new Proxy(rawRow, {
            get(target, prop, receiver) {
              if (typeof prop === 'string' && /^\d+$/.test(prop)) {
                const colIdx = parseInt(prop, 10);
                self._counters.recordSharedLoad((rowIdx * c + colIdx) * 4);
                return target[colIdx];
              }
              return Reflect.get(target, prop, receiver);
            },
            set(target, prop, value, receiver) {
              if (typeof prop === 'string' && /^\d+$/.test(prop)) {
                const colIdx = parseInt(prop, 10);
                self._counters.recordSharedStore((rowIdx * c + colIdx) * 4);
                target[colIdx] = value;
                return true;
              }
              return Reflect.set(target, prop, value, receiver);
            }
          });
          grid.push(rowProxy);
        }
        this._staticArrays.set(name, grid);
      }
      return this._staticArrays.get(name);
    }

    /** Read float at index */
    get(index) {
      if (index < 0 || index >= this._f32.length) {
        throw new RangeError('SharedMemory OOB read at index ' + index);
      }
      const byteAddr = index * 4;
      this._counters.recordSharedLoad(byteAddr);
      return this._f32[index];
    }

    /** Write float at index */
    set(index, value) {
      if (index < 0 || index >= this._f32.length) {
        throw new RangeError('SharedMemory OOB write at index ' + index);
      }
      const byteAddr = index * 4;
      this._counters.recordSharedStore(byteAddr);
      this._f32[index] = value;
      this._written[index] = 1;
    }

    /** Read int32 at index */
    getInt(index) {
      if (index < 0 || index >= this._i32.length) {
        throw new RangeError('SharedMemory OOB i32 read at index ' + index);
      }
      const byteAddr = index * 4;
      this._counters.recordSharedLoad(byteAddr);
      return this._i32[index];
    }

    /** Write int32 at index */
    setInt(index, value) {
      if (index < 0 || index >= this._i32.length) {
        throw new RangeError('SharedMemory OOB i32 write at index ' + index);
      }
      const byteAddr = index * 4;
      this._counters.recordSharedStore(byteAddr);
      this._i32[index] = value;
      this._written[index] = 1;
    }

    /** Check if an index was written (to detect uninitialized reads) */
    isInitialized(index) {
      return this._written[index] === 1;
    }

    /** Get the size in floats */
    get length() {
      return this._f32.length;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Atomic Operations
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * All atomics operate on DevicePointer or SharedMemory.
   * They return the old value (CAS-style semantics).
   * Since JS is single-threaded, these are inherently atomic.
   */

  function atomicAdd(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._f32[index];
      ptr._f32[index] = old + val;
    } else {
      old = ptr._f32[absIdx];
      ptr._f32[absIdx] = old + val;
    }
    counters.recordFlop(1);
    return old;
  }

  function atomicSub(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._f32[index];
      ptr._f32[index] = old - val;
    } else {
      old = ptr._f32[absIdx];
      ptr._f32[absIdx] = old - val;
    }
    return old;
  }

  function atomicMax(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._i32[index];
      ptr._i32[index] = Math.max(old, val);
    } else {
      old = ptr._i32[absIdx];
      ptr._i32[absIdx] = Math.max(old, val);
    }
    return old;
  }

  function atomicMin(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._i32[index];
      ptr._i32[index] = Math.min(old, val);
    } else {
      old = ptr._i32[absIdx];
      ptr._i32[absIdx] = Math.min(old, val);
    }
    return old;
  }

  function atomicCAS(ptr, index, compare, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._i32[index];
      if (old === compare) ptr._i32[index] = val;
    } else {
      old = ptr._i32[absIdx];
      if (old === compare) ptr._i32[absIdx] = val;
    }
    return old;
  }

  function atomicExch(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._f32[index];
      ptr._f32[index] = val;
    } else {
      old = ptr._f32[absIdx];
      ptr._f32[absIdx] = val;
    }
    return old;
  }

  function atomicAnd(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._i32[index];
      ptr._i32[index] = old & val;
    } else {
      old = ptr._i32[absIdx];
      ptr._i32[absIdx] = old & val;
    }
    return old;
  }

  function atomicOr(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._i32[index];
      ptr._i32[index] = old | val;
    } else {
      old = ptr._i32[absIdx];
      ptr._i32[absIdx] = old | val;
    }
    return old;
  }

  function atomicXor(ptr, index, val, counters) {
    counters.recordAtomic(ptr._allocId || 0, index);
    const absIdx = (ptr._offset !== undefined) ? ptr._offset + index : index;
    let old;
    if (ptr instanceof SharedMemory) {
      old = ptr._i32[index];
      ptr._i32[index] = old ^ val;
    } else {
      old = ptr._i32[absIdx];
      ptr._i32[absIdx] = old ^ val;
    }
    return old;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Warp-Level Primitives
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Warp shuffle operations operate on a virtual "warp register file".
   * Since we execute threads sequentially, we buffer warp results and
   * replay shuffles from the register file.
   */

  class WarpState {
    constructor() {
      /** Per-lane register values for shuffle: warpId → laneId → value */
      this._registers = new Map();
      /** Per-lane ballot predicate values */
      this._ballots = new Map();
    }

    /**
     * Store a thread's register value for shuffle ops.
     * @param {number} warpId
     * @param {number} laneId
     * @param {*} val
     */
    setRegister(warpId, laneId, val) {
      if (!this._registers.has(warpId)) {
        this._registers.set(warpId, new Float64Array(WARP_SIZE));
      }
      this._registers.get(warpId)[laneId] = val;
    }

    getRegister(warpId, laneId) {
      const regs = this._registers.get(warpId);
      if (!regs) return 0;
      return regs[laneId];
    }

    clear() {
      this._registers.clear();
      this._ballots.clear();
    }
  }

  /**
   * Shuffle down: each lane reads from lane + delta.
   * Used in warp-level reductions.
   */
  function __shfl_down_sync(mask, val, delta, width, warpState, warpId, laneId) {
    const w = width || WARP_SIZE;
    const segment = Math.floor(laneId / w) * w;
    const srcLane = laneId + delta;
    if (srcLane < segment + w) {
      warpState.setRegister(warpId, laneId, val);
      return warpState.getRegister(warpId, srcLane);
    }
    return val;
  }

  /** Shuffle up: each lane reads from lane - delta */
  function __shfl_up_sync(mask, val, delta, width, warpState, warpId, laneId) {
    const w = width || WARP_SIZE;
    const segment = Math.floor(laneId / w) * w;
    const srcLane = laneId - delta;
    if (srcLane >= segment) {
      warpState.setRegister(warpId, laneId, val);
      return warpState.getRegister(warpId, srcLane);
    }
    return val;
  }

  /** Shuffle xor: each lane reads from lane ^ laneMask */
  function __shfl_xor_sync(mask, val, laneMask, width, warpState, warpId, laneId) {
    warpState.setRegister(warpId, laneId, val);
    const w = width || WARP_SIZE;
    const segment = Math.floor(laneId / w) * w;
    const srcLane = segment + ((laneId - segment) ^ laneMask);
    if (srcLane < segment + w) {
      return warpState.getRegister(warpId, srcLane);
    }
    return val;
  }

  /** Shuffle: each lane reads from srcLane */
  function __shfl_sync(mask, val, srcLane, width, warpState, warpId, laneId) {
    warpState.setRegister(warpId, laneId, val);
    const w = width || WARP_SIZE;
    const segment = Math.floor(laneId / w) * w;
    const effSrc = segment + (srcLane % w);
    return warpState.getRegister(warpId, effSrc);
  }

  /** Ballot: returns bitmask of lanes with predicate true */
  function __ballot_sync(mask, predicate, warpState, warpId, laneId) {
    // Store this lane's predicate
    if (!warpState._ballots.has(warpId)) {
      warpState._ballots.set(warpId, new Uint32Array(1));
    }
    const bitmask = warpState._ballots.get(warpId);
    if (predicate) {
      bitmask[0] |= (1 << laneId);
    } else {
      bitmask[0] &= ~(1 << laneId);
    }
    return bitmask[0] & mask;
  }

  /** All sync: returns true if all lanes in mask have predicate true */
  function __all_sync(mask, predicate, warpState, warpId, laneId) {
    const ballot = __ballot_sync(mask, predicate, warpState, warpId, laneId);
    return (ballot & mask) === mask;
  }

  /** Any sync: returns true if any lane in mask has predicate true */
  function __any_sync(mask, predicate, warpState, warpId, laneId) {
    const ballot = __ballot_sync(mask, predicate, warpState, warpId, laneId);
    return (ballot & mask) !== 0;
  }

  /** Population count – number of set bits */
  function __popc(val) {
    let v = val >>> 0;
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
  }

  /** Count leading zeros */
  function __clz(val) {
    return Math.clz32(val >>> 0);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CUDA Math Functions
  // ═══════════════════════════════════════════════════════════════════════════

  /** Build the math context object for kernel execution */
  function buildMathContext() {
    return {
      // Standard math functions
      expf: Math.exp,
      logf: Math.log,
      sqrtf: Math.sqrt,
      rsqrtf: function (x) { return 1.0 / Math.sqrt(x); },
      fabsf: Math.abs,
      fmaxf: Math.max,
      fminf: Math.min,
      powf: Math.pow,
      sinf: Math.sin,
      cosf: Math.cos,
      tanf: Math.tan,
      tanhf: Math.tanh,
      ceilf: Math.ceil,
      floorf: Math.floor,
      roundf: Math.round,

      // Floating-point classification & sign
      isinf: function (x) { return !Number.isFinite(x) && !Number.isNaN(x); },
      isinff: function (x) { return !Number.isFinite(x) && !Number.isNaN(x); },
      __isinf: function (x) { return !Number.isFinite(x) && !Number.isNaN(x); },
      __isinff: function (x) { return !Number.isFinite(x) && !Number.isNaN(x); },
      isnan: Number.isNaN,
      isnanf: Number.isNaN,
      __isnan: Number.isNaN,
      __isnanf: Number.isNaN,
      isfinite: Number.isFinite,
      isfinitef: Number.isFinite,
      __finite: Number.isFinite,
      __finitef: Number.isFinite,
      signbit: function (x) { return Math.sign(x) < 0 || Object.is(x, -0); },
      signbitf: function (x) { return Math.sign(x) < 0 || Object.is(x, -0); },

      // Additional math operations
      hypot: Math.hypot,
      hypotf: Math.hypot,
      cbrt: Math.cbrt,
      cbrtf: Math.cbrt,
      expm1: Math.expm1,
      expm1f: Math.expm1,
      log1p: Math.log1p,
      log1pf: Math.log1p,
      exp2: function (x) { return Math.pow(2, x); },
      exp2f: function (x) { return Math.pow(2, x); },
      exp10: function (x) { return Math.pow(10, x); },
      exp10f: function (x) { return Math.pow(10, x); },
      __exp10f: function (x) { return Math.pow(10, x); },
      fmod: function (a, b) { return a % b; },
      fmodf: function (a, b) { return a % b; },

      // Fast-math intrinsics (same implementations for simulation)
      __expf: Math.exp,
      __logf: Math.log,
      __fdividef: function (a, b) { return a / b; },
      __fmaf_rn: function (a, b, c) { return Math.fround(a * b + c); },
      __fsqrt_rn: function (x) { return Math.fround(Math.sqrt(x)); },

      // FMA
      fma: function (a, b, c) { return a * b + c; },
      fmaf: function (a, b, c) { return Math.fround(a * b + c); },
      copysignf: function (mag, sgn) {
        return Math.sign(sgn) === 0 ? mag : Math.abs(mag) * Math.sign(sgn);
      },

      // Constants
      INFINITY: CUDA_INFINITY,
      NAN: CUDA_NAN,
      FLT_MAX: CUDA_FLT_MAX,
      FLT_MIN: CUDA_FLT_MIN,
      FLT_EPSILON: CUDA_FLT_EPSILON,
      M_PI: CUDA_M_PI,
      M_PI_2: CUDA_M_PI_2,
      M_SQRT2: CUDA_M_SQRT2
    };
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // KernelLauncher — executes transpiled CUDA kernels
  // ═══════════════════════════════════════════════════════════════════════════

  class KernelLauncher {
    /**
     * @param {SimCounters} counters
     * @param {object}      gpuProfile
     */
    constructor(counters, gpuProfile) {
      this._counters = counters;
      this._gpuProfile = gpuProfile;
      this._diagnostics = [];
      this._warpState = new WarpState();
      this._mathContext = buildMathContext();
    }

    /**
     * Launch a kernel across the full grid.
     *
     * The transpiled kernel function signature:
     *   function(threadIdx, blockIdx, blockDim, gridDim, __shared, __syncthreads,
     *            __atomicAdd, __printf, ...userArgs)
     *
     * Additionally, warp intrinsics, math functions, and other atomics are
     * injected via the execution context.
     *
     * @param {Function} kernelFn  – Transpiled kernel function
     * @param {object}   gridDim   – Grid dimensions {x, y, z}
     * @param {object}   blockDim  – Block dimensions {x, y, z}
     * @param {number}   sharedMemBytes – Dynamic shared memory in bytes
     * @param {...*}     kernelArgs – User kernel arguments
     */
    launch(kernelFn, gridDim, blockDim, sharedMemBytes, ...kernelArgs) {
      const grid = dim3(gridDim);
      const block = dim3(blockDim);
      const totalBlocks = dim3Volume(grid);
      const threadsPerBlock = dim3Volume(block);
      const counters = this._counters;
      const diag = this._diagnostics;
      const warpState = this._warpState;

      counters.blocksLaunched += totalBlocks;
      counters.threadsLaunched += totalBlocks * threadsPerBlock;

      // Pre-allocated shared memory size (minimum 256 bytes)
      const shmemSize = Math.max(sharedMemBytes || 256, 256);

      // Iterate over every block in the grid
      for (let bz = 0; bz < grid.z; bz++) {
        for (let by = 0; by < grid.y; by++) {
          for (let bx = 0; bx < grid.x; bx++) {
            const blockIdx = { x: bx, y: by, z: bz };

            // Allocate shared memory for this block
            const sharedMem = new SharedMemory(shmemSize, counters);

            // Sync barrier tracking: kernels that call __syncthreads
            // are run in phases. We collect per-thread continuations.
            // Since JS is single-threaded and we can't truly do barriers
            // within a single function call, we use a multi-pass approach:
            // Phase 1: run all threads until first __syncthreads
            // Phase 2: run all threads until next __syncthreads (or end)
            // etc.

            // We'll execute threads using a coroutine-like approach via
            // generators if the kernel yields at __syncthreads. For
            // simple kernels without __syncthreads, direct execution.

            // Collect warp-level global access addresses per warp for
            // coalescing analysis
            const warpAccesses = new Map(); // warpId → [byteAddresses]

            // Build the execution context that threads receive
            const self = this;

            // __syncthreads sentinel
            const SYNC_SENTINEL = Symbol('__syncthreads');

            // Detect if the kernel uses generators (__syncthreads)
            // We'll try generator-based execution first, fall back to direct.
            let useGenerators = false;
            let threadGens = [];

            // === Phase-based execution with __syncthreads support ===

            // First, determine the linear thread ordering
            const threadIndices = [];
            for (let tz = 0; tz < block.z; tz++) {
              for (let ty = 0; ty < block.y; ty++) {
                for (let tx = 0; tx < block.x; tx++) {
                  threadIndices.push({ x: tx, y: ty, z: tz });
                }
              }
            }

            // Build intrinsics for each thread
            const buildThreadContext = (threadIdx) => {
              const linearIdx = threadIdx.x + threadIdx.y * block.x + threadIdx.z * block.x * block.y;
              const warpId = Math.floor(linearIdx / WARP_SIZE);
              const laneId = linearIdx % WARP_SIZE;

              // Printf for kernel debugging
              const __printf = function (...args) {
                if (args.length === 0) return;
                let fmt = String(args[0]);
                let argIdx = 1;
                let formatted = fmt.replace(/%([0-9]*\.?[0-9]*[dfsuxXgG%])/g, (match, spec) => {
                  if (spec === '%') return '%';
                  if (argIdx >= args.length) return match;
                  const val = args[argIdx++];
                  if (spec.endsWith('f') || spec.endsWith('g') || spec.endsWith('G')) {
                    const precision = spec.includes('.') ? parseInt(spec.split('.')[1]) : 6;
                    return Number(val).toFixed(precision);
                  }
                  if (spec.endsWith('d') || spec.endsWith('i') || spec.endsWith('u')) {
                    return parseInt(val, 10);
                  }
                  if (spec.endsWith('x') || spec.endsWith('X')) {
                    return (val >>> 0).toString(16);
                  }
                  return String(val);
                });
                if (argIdx < args.length) {
                  formatted += ' ' + args.slice(argIdx).join(' ');
                }
                diag.push({
                  type: 'printf',
                  message: formatted,
                  block: { ...blockIdx },
                  thread: { ...threadIdx }
                });
              };

              // Atomic operations bound to this runtime's counters
              const __atomicAdd = (ptr, idx, val) => atomicAdd(ptr, idx, val, counters);
              const __atomicSub = (ptr, idx, val) => atomicSub(ptr, idx, val, counters);
              const __atomicMax = (ptr, idx, val) => atomicMax(ptr, idx, val, counters);
              const __atomicMin = (ptr, idx, val) => atomicMin(ptr, idx, val, counters);
              const __atomicCAS = (ptr, idx, cmp, val) => atomicCAS(ptr, idx, cmp, val, counters);
              const __atomicExch = (ptr, idx, val) => atomicExch(ptr, idx, val, counters);
              const __atomicAnd = (ptr, idx, val) => atomicAnd(ptr, idx, val, counters);
              const __atomicOr = (ptr, idx, val) => atomicOr(ptr, idx, val, counters);
              const __atomicXor = (ptr, idx, val) => atomicXor(ptr, idx, val, counters);

              // Warp primitives bound to this thread's warp/lane
              const __shfl_down = (mask, val, delta, width) =>
                __shfl_down_sync(mask, val, delta, width, warpState, warpId, laneId);
              const __shfl_up = (mask, val, delta, width) =>
                __shfl_up_sync(mask, val, delta, width, warpState, warpId, laneId);
              const __shfl_xor = (mask, val, laneMask, width) =>
                __shfl_xor_sync(mask, val, laneMask, width, warpState, warpId, laneId);
              const __shfl = (mask, val, srcLane, width) =>
                __shfl_sync(mask, val, srcLane, width, warpState, warpId, laneId);
              const __ballot = (mask, pred) =>
                __ballot_sync(mask, pred, warpState, warpId, laneId);
              const __all = (mask, pred) =>
                __all_sync(mask, pred, warpState, warpId, laneId);
              const __any = (mask, pred) =>
                __any_sync(mask, pred, warpState, warpId, laneId);

              const __atomic = function (ptr, idx, val) { return atomicAdd(ptr, idx, val, counters); };
              __atomic.add = (ptr, idx, val) => atomicAdd(ptr, idx, val, counters);
              __atomic.sub = (ptr, idx, val) => atomicSub(ptr, idx, val, counters);
              __atomic.max = (ptr, idx, val) => atomicMax(ptr, idx, val, counters);
              __atomic.min = (ptr, idx, val) => atomicMin(ptr, idx, val, counters);
              __atomic.cas = (ptr, idx, cmp, val) => atomicCAS(ptr, idx, cmp, val, counters);
              __atomic.exch = (ptr, idx, val) => atomicExch(ptr, idx, val, counters);
              __atomic.and = (ptr, idx, val) => atomicAnd(ptr, idx, val, counters);
              __atomic.or = (ptr, idx, val) => atomicOr(ptr, idx, val, counters);
              __atomic.xor = (ptr, idx, val) => atomicXor(ptr, idx, val, counters);

              return {
                __printf,
                __atomic,
                __atomicAdd, __atomicSub, __atomicMax, __atomicMin,
                __atomicCAS, __atomicExch, __atomicAnd, __atomicOr, __atomicXor,
                __shfl_down_sync: __shfl_down,
                __shfl_up_sync: __shfl_up,
                __shfl_xor_sync: __shfl_xor,
                __shfl_sync: __shfl,
                __ballot_sync: __ballot,
                __all_sync: __all,
                __any_sync: __any,
                __popc,
                __clz,
                linearIdx,
                warpId,
                laneId
              };
            };

            // Try executing with sync barriers (phase-based)
            // We use a simple model: all threads run between barriers.
            // __syncthreads is passed as a callback that records the phase.

            // Since generators aren't guaranteed, we use a simpler model:
            // Collect all sync points by running each thread in sequence,
            // and use a phase counter to coordinate.

            // Execute block threads via SIMT coroutines or sequential fallback
            const threadContexts = [];
            let isGenerator = false;

            // Check for block thread boundary divergence
            if (threadsPerBlock % WARP_SIZE !== 0) {
              counters.recordDivergentBranch();
            }

            for (let t = 0; t < threadIndices.length; t++) {
              const threadIdx = threadIndices[t];
              const ctx = buildThreadContext(threadIdx);

              const __syncthreads = function* () {
                counters.syncthreadsCount++;
                counters.flushSharedBankAccesses();
                counters.flushWarpGlobalAccesses();
                yield { type: 'sync' };
              };

              // Warp primitives that yield for warp-wide lane exchange
              const __warp_shfl_down = function* (mask, val, delta, width) {
                return yield { type: 'shfl', op: 'down', mask, val, delta, width, warpId: ctx.warpId, laneId: ctx.laneId };
              };
              const __warp_shfl = function* (mask, val, srcLane, width) {
                return yield { type: 'shfl', op: 'sync', mask, val, srcLane, width, warpId: ctx.warpId, laneId: ctx.laneId };
              };
              const __warp_shfl_up = function* (mask, val, delta, width) {
                return yield { type: 'shfl', op: 'up', mask, val, delta, width, warpId: ctx.warpId, laneId: ctx.laneId };
              };
              const __warp_shfl_xor = function* (mask, val, laneMask, width) {
                return yield { type: 'shfl', op: 'xor', mask, val, laneMask, width, warpId: ctx.warpId, laneId: ctx.laneId };
              };
              const __warp_ballot = function* (mask, pred) {
                return yield { type: 'ballot', mask, pred, warpId: ctx.warpId, laneId: ctx.laneId };
              };

              const boundThis = Object.assign({}, self._mathContext, ctx, {
                __sync: __syncthreads,
                __syncthreads: __syncthreads,
                __shfl_down_sync: __warp_shfl_down,
                __shfl_sync: __warp_shfl,
                __shfl_up_sync: __warp_shfl_up,
                __shfl_xor_sync: __warp_shfl_xor,
                __ballot_sync: __warp_ballot,
              });

              try {
                counters.setActiveContext(ctx.warpId, ctx.linearIdx);
                const res = kernelFn.call(
                  boundThis,
                  threadIdx,
                  blockIdx,
                  block,
                  grid,
                  sharedMem,
                  __syncthreads,
                  ctx.__atomic,
                  ctx.__printf,
                  ...kernelArgs
                );
                counters.clearActiveContext();

                if (res && typeof res.next === 'function') {
                  isGenerator = true;
                  threadContexts.push({
                    t,
                    threadIdx,
                    ctx,
                    gen: res,
                    done: false,
                    blocked: false,
                    pendingOp: null,
                    nextVal: undefined
                  });
                }
              } catch (err) {
                counters.clearActiveContext();
                diag.push({
                  type: 'error',
                  message: err.message + '\n' + err.stack,
                  block: { ...blockIdx },
                  thread: { ...threadIdx }
                });
              }
            }

            if (isGenerator) {
              // SIMT Coroutine Engine: Run all threads in lockstep
              let running = true;
              while (running) {
                running = false;

                // Step each unblocked thread
                for (const th of threadContexts) {
                  if (!th.done && !th.blocked) {
                    try {
                      counters.setActiveContext(th.ctx.warpId, th.ctx.linearIdx);
                      const step = th.gen.next(th.nextVal);
                      counters.clearActiveContext();
                      th.nextVal = undefined;
                      if (step.done) {
                        th.done = true;
                      } else {
                        running = true;
                        const v = step.value;
                        if (v && v.type === 'sync') {
                          th.blocked = true;
                        } else if (v && (v.type === 'shfl' || v.type === 'ballot')) {
                          th.pendingOp = v;
                        }
                      }
                    } catch (err) {
                      counters.clearActiveContext();
                      th.done = true;
                      diag.push({
                        type: 'error',
                        message: err.message + '\n' + err.stack,
                        block: { ...blockIdx },
                        thread: { ...th.threadIdx }
                      });
                    }
                  }
                }

                // Resolve pending warp operations for this step
                const warpPending = new Map();
                for (const th of threadContexts) {
                  if (th.pendingOp) {
                    const wId = th.ctx.warpId;
                    if (!warpPending.has(wId)) warpPending.set(wId, []);
                    warpPending.get(wId).push(th);
                  }
                }

                for (const [wId, thList] of warpPending.entries()) {
                  // Buffer registers across this warp
                  const warpRegisters = new Float64Array(WARP_SIZE);
                  for (const th of thList) {
                    warpRegisters[th.ctx.laneId] = th.pendingOp.val;
                  }

                  // Deliver shuffle / ballot results
                  for (const th of thList) {
                    const op = th.pendingOp;
                    const lane = th.ctx.laneId;
                    const w = op.width || WARP_SIZE;
                    const segment = Math.floor(lane / w) * w;

                    if (op.op === 'down') {
                      const srcLane = lane + op.delta;
                      th.nextVal = (srcLane < segment + w) ? warpRegisters[srcLane] : op.val;
                    } else if (op.op === 'up') {
                      const srcLane = lane - op.delta;
                      th.nextVal = (srcLane >= segment) ? warpRegisters[srcLane] : op.val;
                    } else if (op.op === 'xor') {
                      const srcLane = segment + ((lane - segment) ^ op.laneMask);
                      th.nextVal = (srcLane < segment + w) ? warpRegisters[srcLane] : op.val;
                    } else if (op.op === 'sync') {
                      const srcLane = segment + (op.srcLane % w);
                      th.nextVal = warpRegisters[srcLane];
                    } else if (op.type === 'ballot') {
                      let mask = 0;
                      for (const otherTh of thList) {
                        if (otherTh.pendingOp.pred) mask |= (1 << otherTh.ctx.laneId);
                      }
                      const expectedMask = op.mask || 0xFFFFFFFF;
                      if (mask !== 0 && mask !== expectedMask) {
                        counters.recordDivergentBranch();
                      }
                      th.nextVal = mask & expectedMask;
                    }
                    th.pendingOp = null;
                  }
                }

                // Advance barrier: if all living threads are blocked at sync, wake them all
                const living = threadContexts.filter(t => !t.done);
                if (living.length > 0 && living.every(t => t.blocked)) {
                  for (const th of living) th.blocked = false;
                }
              }
            }

            // Flush remaining shared bank and warp global analyses
            counters.flushSharedBankAccesses();
            counters.flushWarpGlobalAccesses();
            warpState.clear();
          }
        }
      }
    }

    /** Get diagnostics (printf output, errors) from the last launch */
    getDiagnostics() {
      return this._diagnostics;
    }

    /** Clear diagnostics */
    clearDiagnostics() {
      this._diagnostics = [];
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MultiGPURuntime — multi-device simulation with NCCL collectives
  // ═══════════════════════════════════════════════════════════════════════════

  class NcclComm {
    /**
     * @param {number} nranks – Number of ranks in the communicator
     * @param {number} rank   – This rank's index
     * @param {MultiGPURuntime} runtime
     * @param {object} [clique] - Shared collective state among group
     */
    constructor(nranks, rank, runtime, clique) {
      this.nranks = nranks;
      this.rank = rank;
      this._runtime = runtime;
      this._clique = clique || {
        nranks,
        pendingAllReduce: [],
        pendingBroadcast: [],
        pendingReduceScatter: [],
        pendingAllGather: []
      };
    }
  }

  class MultiGPURuntime {
    /**
     * @param {number} numDevices – Number of simulated GPUs
     * @param {object} gpuProfile
     */
    constructor(numDevices, gpuProfile) {
      this._numDevices = numDevices;
      this._currentDevice = 0;
      this._gpuProfile = gpuProfile;

      /** Per-device state: memory arenas, counters, launchers */
      this._devices = [];
      for (let i = 0; i < numDevices; i++) {
        const counters = new SimCounters(gpuProfile);
        const memory = new SimMemory(counters);
        const launcher = new KernelLauncher(counters, gpuProfile);
        this._devices.push({ counters, memory, launcher });
      }

      /** Peer access enabled pairs: "src:dst" → true */
      this._peerAccess = new Set();
    }

    /** cudaSetDevice */
    cudaSetDevice(id) {
      if (id < 0 || id >= this._numDevices) {
        throw new Error('Invalid device id: ' + id);
      }
      this._currentDevice = id;
    }

    /** cudaGetDevice */
    cudaGetDevice() {
      return this._currentDevice;
    }

    /** Get the current device's state */
    _dev(id) {
      const i = (id !== undefined) ? id : this._currentDevice;
      return this._devices[i];
    }

    /** cudaDeviceEnablePeerAccess */
    cudaDeviceEnablePeerAccess(peer, flags) {
      this._peerAccess.add(this._currentDevice + ':' + peer);
    }

    /**
     * cudaMemcpyPeer — copy between devices.
     * @param {DevicePointer} dst
     * @param {number} dstDev
     * @param {DevicePointer} src
     * @param {number} srcDev
     * @param {number} sizeBytes
     */
    cudaMemcpyPeer(dst, dstDev, src, srcDev, sizeBytes) {
      const srcU8 = new Uint8Array(src._buffer, src._offset * src._elementSize, sizeBytes);
      const dstU8 = new Uint8Array(dst._buffer, dst._offset * dst._elementSize);
      dstU8.set(srcU8.subarray(0, sizeBytes));

      // Track cross-GPU bytes
      this._devices[srcDev].counters.crossGpuBytes += sizeBytes;
      this._devices[dstDev].counters.crossGpuBytes += sizeBytes;
    }

    /** ncclCommInitRank */
    ncclCommInitRank(nranks, rank) {
      return new NcclComm(nranks, rank, this);
    }

    /** ncclCommInitAll */
    ncclCommInitAll(comms, nranks) {
      const clique = {
        nranks,
        pendingAllReduce: [],
        pendingBroadcast: [],
        pendingReduceScatter: [],
        pendingAllGather: []
      };
      for (let r = 0; r < nranks; r++) {
        const comm = new NcclComm(nranks, r, this, clique);
        if (comms && (Array.isArray(comms) || typeof comms === 'object')) {
          comms[r] = comm;
        }
      }
      return comms;
    }

    /**
     * ncclAllReduce — all-reduce across devices.
     * @param {DevicePointer[]|DevicePointer} sendbufs – Per-rank send buffers (or single DevicePointer for current rank)
     * @param {DevicePointer[]|DevicePointer} recvbufs – Per-rank receive buffers (or single DevicePointer for current rank)
     * @param {number} count – Number of float elements
     * @param {string} datatype – 'float32'
     * @param {string} op – 'sum', 'prod', 'max', 'min', 'avg'
     * @param {NcclComm} comm
     */
    ncclAllReduce(sendbufs, recvbufs, count, datatype, op, comm) {
      if (count < 0) throw new Error('ncclInvalidArgument: count must be non-negative');
      op = (typeof op === 'string' ? op.toLowerCase() : 'sum');

      // If called per-rank with a single buffer, stage until all ranks arrive
      if (!Array.isArray(sendbufs)) {
        const nranks = (comm && comm.nranks) ? comm.nranks : this._numDevices;
        const rank = (comm && comm.rank !== undefined) ? comm.rank : this._currentDevice;
        if (rank < 0 || rank >= nranks) throw new Error('ncclInvalidArgument: invalid rank ' + rank + ' for comm with ' + nranks + ' ranks');
        const clique = (comm && comm._clique) ? comm._clique : (this._defaultClique || (this._defaultClique = {
          nranks,
          pendingAllReduce: [],
          pendingBroadcast: [],
          pendingReduceScatter: [],
          pendingAllGather: []
        }));
        clique.pendingAllReduce[rank] = { sendbuf: sendbufs, recvbuf: recvbufs, count, datatype, op, comm };
        let ready = true;
        for (let r = 0; r < nranks; r++) {
          if (!clique.pendingAllReduce[r]) { ready = false; break; }
        }
        if (ready) {
          const staged = clique.pendingAllReduce;
          clique.pendingAllReduce = [];
          const allSends = staged.map(s => s.sendbuf);
          const allRecvs = staged.map(s => s.recvbuf);
          return this.ncclAllReduce(allSends, allRecvs, count, datatype, op, comm || { nranks });
        }
        return 0;
      }

      const nranks = (comm && comm.nranks) ? comm.nranks : (Array.isArray(sendbufs) ? sendbufs.length : this._numDevices);
      if (sendbufs.length !== nranks) throw new Error('ncclInvalidArgument: sendbufs length (' + sendbufs.length + ') must match comm.nranks (' + nranks + ')');
      if (recvbufs.length !== nranks) throw new Error('ncclInvalidArgument: recvbufs length (' + recvbufs.length + ') must match comm.nranks (' + nranks + ')');

      if (count === 0) {
        for (let r = 0; r < nranks; r++) this._devices[r].counters.ncclOps++;
        return 0;
      }

      // Collect all values
      const reduced = new Float32Array(count);

      // Initialize with identity element
      if (op === 'sum' || op === 'avg') reduced.fill(0);
      else if (op === 'prod') reduced.fill(1);
      else if (op === 'max') reduced.fill(-Infinity);
      else if (op === 'min') reduced.fill(Infinity);

      // Reduce across ranks
      for (let r = 0; r < nranks; r++) {
        const src = sendbufs[r];
        for (let i = 0; i < count; i++) {
          const val = src._rawGet(i);
          if (op === 'sum' || op === 'avg') reduced[i] += val;
          else if (op === 'prod') reduced[i] *= val;
          else if (op === 'max') reduced[i] = Math.max(reduced[i], val);
          else if (op === 'min') reduced[i] = Math.min(reduced[i], val);
        }
      }

      if (op === 'avg') {
        for (let i = 0; i < count; i++) reduced[i] /= nranks;
      }

      // Scatter result to all receive buffers
      const bytesPerRank = count * 4;
      for (let r = 0; r < nranks; r++) {
        const dst = recvbufs[r];
        for (let i = 0; i < count; i++) {
          dst._rawSet(i, reduced[i]);
        }
        if (nranks > 1) {
          this._devices[r].counters.crossGpuBytes += bytesPerRank * (nranks - 1);
        }
        this._devices[r].counters.ncclOps++;
      }
      return 0;
    }

    /**
     * ncclBroadcast — broadcast from root to all ranks.
     */
    ncclBroadcast(sendbufs, recvbufs, count, datatype, root, comm) {
      if (count < 0) throw new Error('ncclInvalidArgument: count must be non-negative');
      root = (typeof root === 'number' ? root : 0);

      // If called per-rank with a single buffer, stage until all ranks arrive
      if (!Array.isArray(sendbufs)) {
        const nranks = (comm && comm.nranks) ? comm.nranks : this._numDevices;
        const rank = (comm && comm.rank !== undefined) ? comm.rank : this._currentDevice;
        if (rank < 0 || rank >= nranks) throw new Error('ncclInvalidArgument: invalid rank ' + rank + ' for comm with ' + nranks + ' ranks');
        if (root < 0 || root >= nranks) throw new Error('ncclInvalidArgument: invalid root rank ' + root + ' for comm with ' + nranks + ' ranks');
        const clique = (comm && comm._clique) ? comm._clique : (this._defaultClique || (this._defaultClique = {
          nranks,
          pendingAllReduce: [],
          pendingBroadcast: [],
          pendingReduceScatter: [],
          pendingAllGather: []
        }));
        clique.pendingBroadcast[rank] = { sendbuf: sendbufs, recvbuf: recvbufs, count, datatype, root, comm };
        let ready = true;
        for (let r = 0; r < nranks; r++) {
          if (!clique.pendingBroadcast[r]) { ready = false; break; }
        }
        if (ready) {
          const staged = clique.pendingBroadcast;
          clique.pendingBroadcast = [];
          const allSends = staged.map(s => s.sendbuf);
          const allRecvs = staged.map(s => s.recvbuf);
          return this.ncclBroadcast(allSends, allRecvs, count, datatype, root, comm || { nranks });
        }
        return 0;
      }

      const nranks = (comm && comm.nranks) ? comm.nranks : (Array.isArray(sendbufs) ? sendbufs.length : this._numDevices);
      if (root < 0 || root >= nranks) throw new Error('ncclInvalidArgument: invalid root rank ' + root + ' for communicator of size ' + nranks);
      if (sendbufs.length !== nranks) throw new Error('ncclInvalidArgument: sendbufs length (' + sendbufs.length + ') must match comm.nranks (' + nranks + ')');
      if (recvbufs.length !== nranks) throw new Error('ncclInvalidArgument: recvbufs length (' + recvbufs.length + ') must match comm.nranks (' + nranks + ')');

      if (count === 0) {
        for (let r = 0; r < nranks; r++) this._devices[r].counters.ncclOps++;
        return 0;
      }

      const src = sendbufs[root];
      const bytesPerRank = count * 4;

      for (let r = 0; r < nranks; r++) {
        const dst = recvbufs[r];
        for (let i = 0; i < count; i++) {
          dst._rawSet(i, src._rawGet(i));
        }
        if (nranks > 1 && r !== root) {
          this._devices[r].counters.crossGpuBytes += bytesPerRank;
        }
        this._devices[r].counters.ncclOps++;
      }
      return 0;
    }

    /**
     * ncclReduceScatter — reduce then scatter.
     * Each rank receives recvcount elements of the reduction.
     */
    ncclReduceScatter(sendbufs, recvbufs, recvcount, datatype, op, comm) {
      if (recvcount < 0) throw new Error('ncclInvalidArgument: recvcount must be non-negative');
      op = (typeof op === 'string' ? op.toLowerCase() : 'sum');

      // If called per-rank with a single buffer, stage until all ranks arrive
      if (!Array.isArray(sendbufs)) {
        const nranks = (comm && comm.nranks) ? comm.nranks : this._numDevices;
        const rank = (comm && comm.rank !== undefined) ? comm.rank : this._currentDevice;
        if (rank < 0 || rank >= nranks) throw new Error('ncclInvalidArgument: invalid rank ' + rank + ' for comm with ' + nranks + ' ranks');
        const clique = (comm && comm._clique) ? comm._clique : (this._defaultClique || (this._defaultClique = {
          nranks,
          pendingAllReduce: [],
          pendingBroadcast: [],
          pendingReduceScatter: [],
          pendingAllGather: []
        }));
        clique.pendingReduceScatter[rank] = { sendbuf: sendbufs, recvbuf: recvbufs, recvcount, datatype, op, comm };
        let ready = true;
        for (let r = 0; r < nranks; r++) {
          if (!clique.pendingReduceScatter[r]) { ready = false; break; }
        }
        if (ready) {
          const staged = clique.pendingReduceScatter;
          clique.pendingReduceScatter = [];
          const allSends = staged.map(s => s.sendbuf);
          const allRecvs = staged.map(s => s.recvbuf);
          return this.ncclReduceScatter(allSends, allRecvs, recvcount, datatype, op, comm || { nranks });
        }
        return 0;
      }

      const nranks = (comm && comm.nranks) ? comm.nranks : (Array.isArray(sendbufs) ? sendbufs.length : this._numDevices);
      if (sendbufs.length !== nranks) throw new Error('ncclInvalidArgument: sendbufs length (' + sendbufs.length + ') must match comm.nranks (' + nranks + ')');
      if (recvbufs.length !== nranks) throw new Error('ncclInvalidArgument: recvbufs length (' + recvbufs.length + ') must match comm.nranks (' + nranks + ')');

      if (recvcount === 0) {
        for (let r = 0; r < nranks; r++) this._devices[r].counters.ncclOps++;
        return 0;
      }

      const totalCount = recvcount * nranks;

      // Full reduce first
      const reduced = new Float32Array(totalCount);
      if (op === 'sum' || op === 'avg') reduced.fill(0);
      else if (op === 'prod') reduced.fill(1);
      else if (op === 'max') reduced.fill(-Infinity);
      else if (op === 'min') reduced.fill(Infinity);

      for (let r = 0; r < nranks; r++) {
        const src = sendbufs[r];
        for (let i = 0; i < totalCount; i++) {
          const val = src._rawGet(i);
          if (op === 'sum' || op === 'avg') reduced[i] += val;
          else if (op === 'prod') reduced[i] *= val;
          else if (op === 'max') reduced[i] = Math.max(reduced[i], val);
          else if (op === 'min') reduced[i] = Math.min(reduced[i], val);
        }
      }

      if (op === 'avg') {
        for (let i = 0; i < totalCount; i++) reduced[i] /= nranks;
      }

      // Scatter chunks
      for (let r = 0; r < nranks; r++) {
        const dst = recvbufs[r];
        const offset = r * recvcount;
        for (let i = 0; i < recvcount; i++) {
          dst._rawSet(i, reduced[offset + i]);
        }
        if (nranks > 1) {
          this._devices[r].counters.crossGpuBytes += recvcount * 4 * (nranks - 1);
        }
        this._devices[r].counters.ncclOps++;
      }
      return 0;
    }

    /**
     * ncclAllGather — gather data from all ranks to all ranks.
     * Each rank contributes sendcount elements; each receives nranks*sendcount.
     */
    ncclAllGather(sendbufs, recvbufs, sendcount, datatype, comm) {
      if (sendcount < 0) throw new Error('ncclInvalidArgument: sendcount must be non-negative');

      // If called per-rank with a single buffer, stage until all ranks arrive
      if (!Array.isArray(sendbufs)) {
        const nranks = (comm && comm.nranks) ? comm.nranks : this._numDevices;
        const rank = (comm && comm.rank !== undefined) ? comm.rank : this._currentDevice;
        if (rank < 0 || rank >= nranks) throw new Error('ncclInvalidArgument: invalid rank ' + rank + ' for comm with ' + nranks + ' ranks');
        const clique = (comm && comm._clique) ? comm._clique : (this._defaultClique || (this._defaultClique = {
          nranks,
          pendingAllReduce: [],
          pendingBroadcast: [],
          pendingReduceScatter: [],
          pendingAllGather: []
        }));
        clique.pendingAllGather[rank] = { sendbuf: sendbufs, recvbuf: recvbufs, sendcount, datatype, comm };
        let ready = true;
        for (let r = 0; r < nranks; r++) {
          if (!clique.pendingAllGather[r]) { ready = false; break; }
        }
        if (ready) {
          const staged = clique.pendingAllGather;
          clique.pendingAllGather = [];
          const allSends = staged.map(s => s.sendbuf);
          const allRecvs = staged.map(s => s.recvbuf);
          return this.ncclAllGather(allSends, allRecvs, sendcount, datatype, comm || { nranks });
        }
        return 0;
      }

      const nranks = (comm && comm.nranks) ? comm.nranks : (Array.isArray(sendbufs) ? sendbufs.length : this._numDevices);
      if (sendbufs.length !== nranks) throw new Error('ncclInvalidArgument: sendbufs length (' + sendbufs.length + ') must match comm.nranks (' + nranks + ')');
      if (recvbufs.length !== nranks) throw new Error('ncclInvalidArgument: recvbufs length (' + recvbufs.length + ') must match comm.nranks (' + nranks + ')');

      if (sendcount === 0) {
        for (let r = 0; r < nranks; r++) this._devices[r].counters.ncclOps++;
        return 0;
      }

      for (let r = 0; r < nranks; r++) {
        const dst = recvbufs[r];
        for (let src_r = 0; src_r < nranks; src_r++) {
          const src = sendbufs[src_r];
          const dstOffset = src_r * sendcount;
          for (let i = 0; i < sendcount; i++) {
            dst._rawSet(dstOffset + i, src._rawGet(i));
          }
        }
        if (nranks > 1) {
          this._devices[r].counters.crossGpuBytes += sendcount * 4 * (nranks - 1);
        }
        this._devices[r].counters.ncclOps++;
      }
      return 0;
    }

    allReduce(sendbuf, recvbuf, count, datatype, op, comm) {
      return this.ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    get deviceCount() {
      return this._numDevices;
    }

    get numDevices() {
      return this._numDevices;
    }

    /**
     * Get per-device interface wrapper
     * @param {number} id
     */
    getDevice(id) {
      const dev = this._dev(id);
      return {
        cudaMalloc: (sizeBytes) => dev.memory.cudaMalloc(sizeBytes),
        cudaMemcpy: (dst, src, sizeBytes, kind) => {
          dev.memory.cudaMemcpy(dst, src, sizeBytes, kind);
          return 0;
        },
        cudaFree: (ptr) => {
          dev.memory.cudaFree(ptr);
          return 0;
        },
        counters: dev.counters,
        memory: dev.memory,
        launcher: dev.launcher
      };
    }

    /** Get counters for a specific device */
    getCounters(deviceId) {
      const idx = deviceId !== undefined ? deviceId : this._currentDevice;
      if (idx < 0 || idx >= this._numDevices) {
        throw new Error('Invalid device id: ' + idx);
      }
      return this._devices[idx].counters;
    }

    /** Get aggregated counters across all multi-GPU nodes */
    getAggregatedCounters() {
      const agg = new SimCounters(this._gpuProfile);
      for (let i = 0; i < this._numDevices; i++) {
        const c = this._devices[i].counters;
        c.flushWarpGlobalAccesses();
        c.flushSharedBankAccesses();
        agg.threadsLaunched += c.threadsLaunched;
        agg.blocksLaunched += c.blocksLaunched;
        agg.globalLoads += c.globalLoads;
        agg.globalStores += c.globalStores;
        agg.globalLoadBytes += c.globalLoadBytes;
        agg.globalStoreBytes += c.globalStoreBytes;
        agg.globalTransactions += c.globalTransactions;
        agg.sharedLoads += c.sharedLoads;
        agg.sharedStores += c.sharedStores;
        agg.sharedLoadBytes += c.sharedLoadBytes;
        agg.sharedStoreBytes += c.sharedStoreBytes;
        agg.bankConflicts += c.bankConflicts;
        agg.syncthreadsCount += c.syncthreadsCount;
        agg.divergentBranches += c.divergentBranches;
        agg.atomicOps += c.atomicOps;
        agg.atomicMaxContention = Math.max(agg.atomicMaxContention, c.atomicMaxContention);
        agg.memcpyH2DBytes += c.memcpyH2DBytes;
        agg.memcpyD2HBytes += c.memcpyD2HBytes;
        agg.crossGpuBytes += c.crossGpuBytes;
        agg.ncclOps += c.ncclOps;
        agg.flopCount += c.flopCount;
      }
      agg.perDevice = this._devices.map((d, idx) => ({
        deviceId: idx,
        ...d.counters.toJSON()
      }));
      return agg;
    }

    /** Reset all devices */
    reset() {
      for (const dev of this._devices) {
        dev.counters.reset();
        dev.launcher.clearDiagnostics();
      }
    }

    /** Destroy all devices */
    destroy() {
      for (const dev of this._devices) {
        dev.memory.destroy();
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CudaRuntime — the main API that ties everything together
  // ═══════════════════════════════════════════════════════════════════════════

  class CudaRuntime {
    /**
     * @param {object} [gpuProfile] – GPU hardware profile for metric estimation.
     *   Fields: clockSpeedMHz, peakTFLOPS, peakGFLOPS, memBandwidthGBs,
     *           smCount, maxThreadsPerSM, globalMemLatencyCycles,
     *           sharedMemLatencyCycles, atomicLatencyCycles, syncLatencyCycles,
     *           flopLatencyCycles, maxSharedMemPerBlock, maxRegistersPerBlock
     */
    constructor(gpuProfile) {
      this._gpuProfile = gpuProfile || {
        clockSpeedMHz: 1500,
        peakTFLOPS: 19.5,
        peakGFLOPS: 19500,
        memBandwidthGBs: 900,
        smCount: 80,
        maxThreadsPerSM: 2048,
        globalMemLatencyCycles: 400,
        sharedMemLatencyCycles: 20,
        atomicLatencyCycles: 50,
        syncLatencyCycles: 5,
        flopLatencyCycles: 1,
        maxSharedMemPerBlock: 49152,
        maxRegistersPerBlock: 65536
      };

      this._counters = new SimCounters(this._gpuProfile);
      this._memory = new SimMemory(this._counters);
      this._launcher = new KernelLauncher(this._counters, this._gpuProfile);

      // Multi-GPU support
      this._multiGPU = null;
      this._currentDevice = 0;
    }

    // ── Memory API ──

    /**
     * Allocate device memory.
     * @param {number} sizeBytes
     * @returns {DevicePointer}
     */
    cudaMalloc(sizeBytes) {
      if (this._multiGPU) {
        return this._multiGPU.getDevice(this._currentDevice).cudaMalloc(sizeBytes);
      }
      return this._memory.cudaMalloc(sizeBytes);
    }

    /**
     * Free device memory.
     * @param {DevicePointer} ptr
     */
    cudaFree(ptr) {
      if (this._multiGPU) {
        return this._multiGPU.getDevice(this._currentDevice).cudaFree(ptr);
      }
      this._memory.cudaFree(ptr);
      return 0;
    }

    /**
     * Copy memory between host and device.
     * @param {DevicePointer|TypedArray} dst
     * @param {DevicePointer|TypedArray} src
     * @param {number} sizeBytes
     * @param {string} kind – 'H2D', 'D2H', 'D2D', 'H2H'
     */
    cudaMemcpy(dst, src, sizeBytes, kind) {
      if (this._multiGPU) {
        return this._multiGPU.getDevice(this._currentDevice).cudaMemcpy(dst, src, sizeBytes, kind);
      }
      this._memory.cudaMemcpy(dst, src, sizeBytes, kind);
      return 0;
    }

    /** Copy scalar float/int from device memory */
    cudaMemcpyD2H_Scalar(src, sizeBytes) {
      const bytes = (sizeBytes || 4);
      if (this._multiGPU) {
        this._multiGPU.getCounters(this._currentDevice).memcpyD2HBytes += bytes;
      } else {
        this._counters.memcpyD2HBytes += bytes;
      }
      return src && src._rawGet ? src._rawGet(0) : (src && src.get ? src.get(0) : (Array.isArray(src) || ArrayBuffer.isView(src) ? src[0] : src));
    }

    /** Copy scalar value into device memory */
    cudaMemcpyH2D_Scalar(dst, val, sizeBytes) {
      const bytes = (sizeBytes || 4);
      if (this._multiGPU) {
        this._multiGPU.getCounters(this._currentDevice).memcpyH2DBytes += bytes;
      } else {
        this._counters.memcpyH2DBytes += bytes;
      }
      const num = Number(val) || 0;
      if (dst && dst._rawSet) dst._rawSet(0, num);
      else if (dst && dst.set) dst.set(0, num);
      else if (Array.isArray(dst) || ArrayBuffer.isView(dst)) dst[0] = num;
    }

    /**
     * Set device memory.
     * @param {DevicePointer} ptr
     * @param {number} value
     * @param {number} sizeBytes
     */
    cudaMemset(ptr, value, sizeBytes) {
      if (this._multiGPU) {
        const mem = this._multiGPU.getDevice(this._currentDevice).memory;
        if (ptr && ptr._allocId) {
          const entry = mem._allocations.get(ptr._allocId);
          if (entry) {
            new Uint8Array(entry.buffer).fill(value & 0xFF, 0, Math.min(sizeBytes, entry.sizeBytes));
          }
        }
        return 0;
      }
      this._memory.cudaMemset(ptr, value, sizeBytes);
      return 0;
    }

    /** Synchronize device (no-op in simulation — all launches are sync) */
    cudaDeviceSynchronize() {
      return 0;
    }

    // ── Launch API ──

    /**
     * Launch a kernel.
     * @param {Function} kernelFn – Transpiled kernel function
     * @param {object|number|number[]} grid – Grid dimensions
     * @param {object|number|number[]} block – Block dimensions
     * @param {number} [sharedMem=0] – Dynamic shared memory bytes
     * @param {...*} args – Kernel arguments
     */
    launch(kernelFn, grid, block, sharedMem, ...args) {
      if (this._multiGPU) {
        this._multiGPU.getDevice(this._currentDevice).launcher.launch(kernelFn, grid, block, sharedMem || 0, ...args);
        return;
      }
      this._launcher.launch(kernelFn, grid, block, sharedMem || 0, ...args);
    }

    // ── Multi-GPU ──

    /**
     * Initialize multi-GPU support.
     * @param {number} numDevices
     */
    initMultiGPU(numDevices) {
      const n = Math.max(1, Number(numDevices) || 2);
      this._multiGPU = new MultiGPURuntime(n, this._gpuProfile);
      return this._multiGPU;
    }

    /** Set the active GPU device */
    setDevice(id) {
      const targetId = Number(id) || 0;
      if (!this._multiGPU && targetId > 0) {
        this.initMultiGPU(Math.max(targetId + 1, 2));
      } else if (this._multiGPU && targetId >= this._multiGPU.deviceCount) {
        this.initMultiGPU(targetId + 1);
      }
      if (this._multiGPU) {
        this._multiGPU.cudaSetDevice(targetId);
      }
      this._currentDevice = targetId;
      return 0;
    }

    /** Get the active GPU device */
    getDevice() {
      return this._currentDevice;
    }

    /** Get total device count */
    getDeviceCount() {
      if (this._multiGPU) return this._multiGPU.deviceCount;
      if (this.deviceCount && this.deviceCount > 1) return this.deviceCount;
      return (typeof window !== 'undefined' && window.clusterNodes) || this.deviceCount || 2;
    }

    cudaGetDeviceCount() {
      return this.getDeviceCount();
    }

    enablePeerAccess(peerId) {
      if (this._multiGPU) {
        this._multiGPU.cudaDeviceEnablePeerAccess(peerId, 0);
      }
      return 0;
    }

    /**
     * Copy between GPU devices.
     */
    memcpyPeer(dst, dstDev, src, srcDev, sizeBytes) {
      const maxDev = Math.max(Number(dstDev) || 0, Number(srcDev) || 0);
      if (!this._multiGPU || maxDev >= this._multiGPU.deviceCount) {
        this.initMultiGPU(Math.max(maxDev + 1, 2));
      }
      this._multiGPU.cudaMemcpyPeer(dst, dstDev, src, srcDev, sizeBytes);
      return 0;
    }

    ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm) {
      if (!this._multiGPU) {
        const nranks = (comm && comm.nranks) ? comm.nranks : 2;
        this.initMultiGPU(nranks);
      }
      return this._multiGPU.allReduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    // ── Query ──

    /** Get the current performance counters (aggregated across multi-GPU or for specific deviceId) */
    getCounters(deviceId) {
      if (this._multiGPU) {
        if (deviceId !== undefined) {
          return this._multiGPU.getCounters(deviceId);
        }
        return this._multiGPU.getAggregatedCounters();
      }
      this._counters.flushWarpGlobalAccesses();
      this._counters.flushSharedBankAccesses();
      return this._counters;
    }

    /** Get the GPU profile */
    getProfile() {
      return this._gpuProfile;
    }

    /** Get diagnostics from kernel execution */
    getDiagnostics() {
      if (this._multiGPU) {
        const seen = new Set();
        const all = [];
        const addDiag = (item, devId) => {
          if (!seen.has(item)) {
            seen.add(item);
            all.push(devId !== undefined && item.deviceId === undefined ? Object.assign({ deviceId: devId }, item) : item);
          }
        };

        for (const item of this._launcher.getDiagnostics()) {
          addDiag(item, this._currentDevice);
        }
        for (let d = 0; d < this._multiGPU.deviceCount; d++) {
          const devDiag = this._multiGPU.getDevice(d).launcher.getDiagnostics();
          for (const item of devDiag) {
            addDiag(item, d);
          }
        }
        return all;
      }
      return this._launcher.getDiagnostics();
    }

    /** Get the multi-GPU runtime (if initialized) */
    getMultiGPU() {
      return this._multiGPU;
    }

    // ── Lifecycle ──

    /** Print message from host code or kernel */
    printf(...args) {
      if (args.length === 0) return;
      let fmt = String(args[0]);
      let argIdx = 1;
      let formatted = fmt.replace(/%([0-9]*\.?[0-9]*[dfsuxXgG%])/g, (match, spec) => {
        if (spec === '%') return '%';
        if (argIdx >= args.length) return match;
        const val = args[argIdx++];
        if (spec.endsWith('f') || spec.endsWith('g') || spec.endsWith('G')) {
          const precision = spec.includes('.') ? parseInt(spec.split('.')[1]) : 6;
          return Number(val).toFixed(precision);
        }
        if (spec.endsWith('d') || spec.endsWith('i') || spec.endsWith('u')) {
          return parseInt(val, 10);
        }
        if (spec.endsWith('x') || spec.endsWith('X')) {
          return (val >>> 0).toString(16);
        }
        return String(val);
      });
      if (argIdx < args.length) {
        formatted += ' ' + args.slice(argIdx).join(' ');
      }
      const diagItem = {
        type: 'printf',
        message: formatted,
        block: { x: 0, y: 0, z: 0 },
        thread: { x: 0, y: 0, z: 0 },
        deviceId: this._currentDevice
      };
      this._launcher.getDiagnostics().push(diagItem);
      console.log('[CUDA printf]', formatted);
    }

    /** Multi-GPU NCCL Comm Init Rank */
    ncclCommInitRank(nranks, rank) {
      if (!this._multiGPU) this.initMultiGPU(nranks);
      return this._multiGPU.ncclCommInitRank(nranks, rank);
    }

    /** Multi-GPU NCCL Comm Init All */
    ncclCommInitAll(comms, nranks) {
      if (!this._multiGPU) this.initMultiGPU(nranks);
      return this._multiGPU.ncclCommInitAll(comms, nranks);
    }

    /** Multi-GPU NCCL AllReduce */
    ncclAllReduce(...args) {
      if (!this._multiGPU) this.initMultiGPU(this._numDevices || 2);
      return this._multiGPU.ncclAllReduce(...args);
    }

    /** Multi-GPU NCCL Broadcast */
    ncclBroadcast(...args) {
      if (!this._multiGPU) this.initMultiGPU(this._numDevices || 2);
      return this._multiGPU.ncclBroadcast(...args);
    }

    /** Multi-GPU NCCL ReduceScatter */
    ncclReduceScatter(...args) {
      if (!this._multiGPU) this.initMultiGPU(this._numDevices || 2);
      return this._multiGPU.ncclReduceScatter(...args);
    }

    /** Multi-GPU NCCL AllGather */
    ncclAllGather(...args) {
      if (!this._multiGPU) this.initMultiGPU(this._numDevices || 2);
      return this._multiGPU.ncclAllGather(...args);
    }

    /** Reset all counters and diagnostics across all devices */
    reset() {
      this._counters.reset();
      this._launcher.clearDiagnostics();
      if (this._multiGPU) {
        this._multiGPU.reset();
      }
    }

    /** Destroy the runtime and free all memory */
    destroy() {
      this._memory.destroy();
      if (this._multiGPU) {
        this._multiGPU.destroy();
        this._multiGPU = null;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Export everything on window.CudaRuntime
  // ═══════════════════════════════════════════════════════════════════════════

  window.CudaRuntime = CudaRuntime;

  // Also export sub-classes and utilities for advanced use
  window.CudaRuntime.DevicePointer = DevicePointer;
  window.CudaRuntime.SimMemory = SimMemory;
  window.CudaRuntime.SharedMemory = SharedMemory;
  window.CudaRuntime.SimCounters = SimCounters;
  window.CudaRuntime.KernelLauncher = KernelLauncher;
  window.CudaRuntime.MultiGPURuntime = MultiGPURuntime;
  window.CudaRuntime.NcclComm = NcclComm;
  window.CudaRuntime.WarpState = WarpState;

  // Export standalone functions
  window.CudaRuntime.atomicAdd = atomicAdd;
  window.CudaRuntime.atomicSub = atomicSub;
  window.CudaRuntime.atomicMax = atomicMax;
  window.CudaRuntime.atomicMin = atomicMin;
  window.CudaRuntime.atomicCAS = atomicCAS;
  window.CudaRuntime.atomicExch = atomicExch;
  window.CudaRuntime.atomicAnd = atomicAnd;
  window.CudaRuntime.atomicOr = atomicOr;
  window.CudaRuntime.atomicXor = atomicXor;

  // Warp intrinsics
  window.CudaRuntime.__shfl_down_sync = __shfl_down_sync;
  window.CudaRuntime.__shfl_up_sync = __shfl_up_sync;
  window.CudaRuntime.__shfl_xor_sync = __shfl_xor_sync;
  window.CudaRuntime.__shfl_sync = __shfl_sync;
  window.CudaRuntime.__ballot_sync = __ballot_sync;
  window.CudaRuntime.__all_sync = __all_sync;
  window.CudaRuntime.__any_sync = __any_sync;
  window.CudaRuntime.__popc = __popc;
  window.CudaRuntime.__clz = __clz;

  // ═══════════════════════════════════════════════════════════════════════════
  // InferenceCluster — Multi-GPU LLM Inference & Traffic Overload Simulator
  // ═══════════════════════════════════════════════════════════════════════════

  class InferenceCluster {
    /**
     * @param {object} [config]
     */
    constructor(config = {}) {
      this.gpuModel = config.gpuModel || 'A100-SXM4-80GB';
      this.gpuCount = config.gpuCount || 8;
      this.tensorParallel = config.tensorParallel || Math.min(8, this.gpuCount);
      this.pipelineParallel = config.pipelineParallel || Math.max(1, Math.floor(this.gpuCount / this.tensorParallel));
      this.dataParallelReplicas = Math.max(1, Math.floor(this.gpuCount / (this.tensorParallel * this.pipelineParallel)));
      this.modelParamsB = config.modelParamsB || 70;
      this.kvCacheDtypeBytes = config.kvCacheDtypeBytes || 2;
      this.gpuVramGB = config.gpuVramGB || (this.gpuModel.includes('4090') ? 24 : 80);
      this.peakTFLOPS = config.peakTFLOPS || (this.gpuModel.includes('H100') ? 989 : (this.gpuModel.includes('A100') ? 312 : 82.6));
      this.memBandwidthGBs = config.memBandwidthGBs || (this.gpuModel.includes('H100') ? 3350 : (this.gpuModel.includes('A100') ? 2039 : 1008));
      this.interconnectBandwidthGBs = config.interconnectBandwidthGBs || (this.gpuModel.includes('H100') ? 900 : (this.gpuModel.includes('A100') ? 600 : 64));
      this.numLayers = config.numLayers || (this.modelParamsB >= 70 ? 80 : 32);
      this.hiddenSize = config.hiddenSize || (this.modelParamsB >= 70 ? 8192 : 4096);
      this.numHeads = config.numHeads || (this.modelParamsB >= 70 ? 64 : 32);
      this.numKvHeads = config.numKvHeads || (this.modelParamsB >= 70 ? 8 : 8);

      // Memory breakdown
      const totalWeightGB = this.modelParamsB * 2;
      this.weightPerGpuGB = totalWeightGB / (this.tensorParallel * this.pipelineParallel);
      this.systemReserveGB = 4.0;
      this.availableKvCacheGBPerGpu = Math.max(0, this.gpuVramGB - this.weightPerGpuGB - this.systemReserveGB);

      // KV cache bytes per token across all layers
      const kvDim = (this.hiddenSize * this.numKvHeads) / this.numHeads;
      this.kvBytesPerTokenTotal = 2 * this.numLayers * kvDim * this.kvCacheDtypeBytes;
      this.kvBytesPerTokenPerGpu = this.kvBytesPerTokenTotal / this.tensorParallel;

      // Max concurrent tokens across cluster
      this.maxConcurrentTokensPerGpu = Math.floor((this.availableKvCacheGBPerGpu * 1024 * 1024 * 1024) / Math.max(1, this.kvBytesPerTokenPerGpu));
      this.maxConcurrentTokensCluster = this.maxConcurrentTokensPerGpu * this.tensorParallel * this.dataParallelReplicas;
    }

    /**
     * Compute single request baseline latencies (uncongested)
     */
    estimateRequestLatencies(promptLen = 512, genLen = 128) {
      // 1. TTFT (Prefill Phase):
      const prefillFlops = 2 * (this.modelParamsB * 1e9) * promptLen;
      const clusterComputeTFLOPS = this.tensorParallel * this.pipelineParallel * this.peakTFLOPS * 1e12;
      const tPrefillComputeMs = (prefillFlops / clusterComputeTFLOPS) * 1000 * 2.2; // 45% MFU

      // NVLink TP AllReduce comm per layer
      const commBytesPrefillLayer = 2 * ((this.tensorParallel - 1) / this.tensorParallel) * (promptLen * this.hiddenSize * 2);
      const totalCommBytesPrefill = commBytesPrefillLayer * this.numLayers * 2;
      const tPrefillCommMs = (totalCommBytesPrefill / (this.interconnectBandwidthGBs * 1e9)) * 1000;

      const ttftMs = Math.max(1.5, tPrefillComputeMs + tPrefillCommMs);

      // 2. ITL (Decode Phase per token):
      const tReadWeightsMs = ((this.weightPerGpuGB * 1e9) / (this.memBandwidthGBs * 1e9)) * 1000;
      const commBytesDecodeStep = 2 * ((this.tensorParallel - 1) / this.tensorParallel) * (1 * this.hiddenSize * 2) * this.numLayers * 2;
      const tDecodeCommMs = (commBytesDecodeStep / (this.interconnectBandwidthGBs * 1e9)) * 1000;

      const itlMs = Math.max(0.8, tReadWeightsMs + tDecodeCommMs);
      const totalGenTimeMs = ttftMs + (genLen * itlMs);

      return {
        ttftMs: Number(ttftMs.toFixed(2)),
        itlMs: Number(itlMs.toFixed(2)),
        totalGenTimeMs: Number(totalGenTimeMs.toFixed(2)),
        tPrefillComputeMs: Number(tPrefillComputeMs.toFixed(2)),
        tPrefillCommMs: Number(tPrefillCommMs.toFixed(2)),
        tDecodeCommMs: Number(tDecodeCommMs.toFixed(2))
      };
    }

    /**
     * Simulate cluster performance under traffic load & burst spikes.
     */
    simulateTraffic(workload = {}) {
      const targetRps = workload.targetRps || 50;
      const durationSec = workload.durationSec || 10;
      const burstMultiplier = workload.burstMultiplier || 1.0;
      const avgPromptLen = workload.avgPromptLen || 512;
      const avgGenLen = workload.avgGenLen || 128;
      const totalTokensPerReq = avgPromptLen + avgGenLen;

      const base = this.estimateRequestLatencies(avgPromptLen, avgGenLen);
      const peakRps = targetRps * burstMultiplier;
      const totalRequests = Math.round(peakRps * durationSec);
      const maxConcurrentRequests = Math.max(1, Math.floor(this.maxConcurrentTokensCluster / totalTokensPerReq));
      const maxBatchSize = workload.maxBatchSize || 256;
      const effectiveMaxConcurrent = Math.min(maxConcurrentRequests, maxBatchSize * this.dataParallelReplicas);

      const queueDepthHistory = [];
      const kvSaturationHistory = [];
      let currentActiveRequests = 0;
      let currentQueueDepth = 0;
      let droppedRequests = 0;
      let servedRequests = 0;
      let partialCompleted = 0.0;
      const latencies = [];

      const timeSteps = 50;
      const dt = durationSec / timeSteps;
      const reqsPerSlice = (peakRps * dt);

      for (let step = 0; step < timeSteps; step++) {
        const arriving = Math.round(reqsPerSlice * (0.9 + Math.random() * 0.2));

        for (let i = 0; i < arriving; i++) {
          if (currentActiveRequests < effectiveMaxConcurrent) {
            currentActiveRequests++;
          } else {
            if (currentQueueDepth < effectiveMaxConcurrent * 2) {
              currentQueueDepth++;
            } else {
              droppedRequests++;
            }
          }
        }

        const currentBatch = Math.max(1, currentActiveRequests);
        const batchDecodeStepMs = base.itlMs + (currentBatch * 0.012);
        const tokensGeneratedSlice = (currentActiveRequests * ((dt * 1000) / batchDecodeStepMs));
        const requestsCompletedFraction = tokensGeneratedSlice / avgGenLen;
        partialCompleted += requestsCompletedFraction;

        const completed = Math.min(currentActiveRequests, Math.floor(partialCompleted));
        partialCompleted -= completed;
        currentActiveRequests -= completed;
        servedRequests += completed;

        const slotsAvailable = effectiveMaxConcurrent - currentActiveRequests;
        const admittedFromQueue = Math.min(currentQueueDepth, slotsAvailable);
        currentQueueDepth -= admittedFromQueue;
        currentActiveRequests += admittedFromQueue;

        const currentActiveTokens = (currentActiveRequests * totalTokensPerReq);
        const kvSaturationPct = Math.min(100, (currentActiveTokens / this.maxConcurrentTokensCluster) * 100);

        queueDepthHistory.push(currentQueueDepth);
        kvSaturationHistory.push(Number(kvSaturationPct.toFixed(1)));

        const batchDegradation = 1.0 + Math.pow(kvSaturationPct / 100, 2) * 1.5;
        const queueWaitTimeMs = (currentQueueDepth / Math.max(1, peakRps)) * 1000;
        const effectiveTtftMs = base.ttftMs * (1.0 + currentQueueDepth * 0.04) + queueWaitTimeMs;
        const effectiveItlMs = batchDecodeStepMs * batchDegradation;
        const totalLatencyMs = effectiveTtftMs + (avgGenLen * effectiveItlMs);

        for (let k = 0; k < completed; k++) {
          const jitter = (Math.random() - 0.5) * 0.12 * totalLatencyMs;
          latencies.push(Math.max(base.totalGenTimeMs, totalLatencyMs + jitter));
        }
      }

      while (latencies.length < servedRequests) {
        latencies.push(base.totalGenTimeMs * (1.0 + (queueDepthHistory[queueDepthHistory.length - 1] / Math.max(1, peakRps))));
      }

      latencies.sort((a, b) => a - b);
      const p50 = latencies[Math.floor(latencies.length * 0.50)] || base.totalGenTimeMs;
      const p90 = latencies[Math.floor(latencies.length * 0.90)] || (base.totalGenTimeMs * 1.5);
      const p95 = latencies[Math.floor(latencies.length * 0.95)] || (base.totalGenTimeMs * 2.0);
      const p99 = latencies[Math.floor(latencies.length * 0.99)] || (base.totalGenTimeMs * 3.5);

      const maxQueueDepth = Math.max(...queueDepthHistory, 0);
      const peakKvSaturationPct = Math.max(...kvSaturationHistory, 0);
      const totalTokensGenerated = servedRequests * avgGenLen;
      const tokenThroughputTps = totalTokensGenerated / durationSec;

      return {
        clusterConfig: {
          gpuModel: this.gpuModel,
          gpuCount: this.gpuCount,
          tensorParallel: this.tensorParallel,
          pipelineParallel: this.pipelineParallel,
          dataParallelReplicas: this.dataParallelReplicas,
          modelParamsB: this.modelParamsB,
          weightPerGpuGB: Number(this.weightPerGpuGB.toFixed(2)),
          availableKvCacheGBPerGpu: Number(this.availableKvCacheGBPerGpu.toFixed(2)),
          maxConcurrentTokensCluster: this.maxConcurrentTokensCluster,
          maxConcurrentRequests: maxConcurrentRequests
        },
        trafficMetrics: {
          targetRps,
          burstMultiplier,
          peakRps,
          durationSec,
          totalArrivedRequests: totalRequests,
          servedRequests,
          droppedRequests,
          dropRatePct: Number(((droppedRequests / Math.max(1, totalRequests)) * 100).toFixed(2)),
          tokenThroughputTps: Number(tokenThroughputTps.toFixed(1))
        },
        baselineLatency: base,
        congestedLatency: {
          p50Ms: Number(p50.toFixed(2)),
          p90Ms: Number(p90.toFixed(2)),
          p95Ms: Number(p95.toFixed(2)),
          p99Ms: Number(p99.toFixed(2)),
          maxQueueDepth,
          peakKvSaturationPct
        },
        history: {
          queueDepth: queueDepthHistory,
          kvSaturationPct: kvSaturationHistory
        }
      };
    }
  }

  // Math context builder
  window.CudaRuntime.buildMathContext = buildMathContext;
  window.CudaRuntime.InferenceCluster = InferenceCluster;
  window.InferenceCluster = InferenceCluster;

  // Helpers
  window.CudaRuntime.dim3 = dim3;

  // Memory copy constants
  window.CudaRuntime.cudaMemcpyHostToDevice = CUDA_MEMCPY_H2D;
  window.CudaRuntime.cudaMemcpyDeviceToHost = CUDA_MEMCPY_D2H;
  window.CudaRuntime.cudaMemcpyDeviceToDevice = CUDA_MEMCPY_D2D;
  window.CudaRuntime.cudaMemcpyHostToHost = CUDA_MEMCPY_H2H;

  // Constants
  window.CudaRuntime.WARP_SIZE = WARP_SIZE;

  // Register standard C math helpers globally for bare calls in generated code
  const _g = typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : global);
  if (_g) {
    _g.isinf = function (x) { return !Number.isFinite(x) && !Number.isNaN(x); };
    _g.isinff = _g.isinf;
    _g.__isinf = _g.isinf;
    _g.__isinff = _g.isinf;
    _g.isnan = Number.isNaN;
    _g.isnanf = Number.isNaN;
    _g.__isnan = Number.isNaN;
    _g.__isnanf = Number.isNaN;
    _g.isfinite = Number.isFinite;
    _g.isfinitef = Number.isFinite;
    _g.__finite = Number.isFinite;
    _g.__finitef = Number.isFinite;
    _g.signbit = function (x) { return Math.sign(x) < 0 || Object.is(x, -0); };
    _g.signbitf = _g.signbit;
    _g.rsqrtf = function (x) { return 1.0 / Math.sqrt(x); };
    _g.rsqrt = _g.rsqrtf;
    _g.__rsqrtf = _g.rsqrtf;
    _g.fmaxf = Math.max;
    _g.fmax = Math.max;
    _g.fminf = Math.min;
    _g.fmin = Math.min;
    _g.fabsf = Math.abs;
    _g.fabs = Math.abs;
    _g.copysign = function (mag, sgn) { return Math.sign(sgn) === 0 ? Math.abs(mag) : Math.abs(mag) * Math.sign(sgn); };
    _g.copysignf = _g.copysign;
    _g.fmaf = function (a, b, c) { return Math.fround(a * b + c); };
    _g.fma = function (a, b, c) { return a * b + c; };

    // CUDA & NCCL return codes and enum constants
    _g.cudaSuccess = 0;
    _g.ncclSuccess = 0;
    _g.ncclFloat = 'float32';
    _g.ncclFloat32 = 'float32';
    _g.ncclHalf = 'float16';
    _g.ncclFloat16 = 'float16';
    _g.ncclInt = 'int32';
    _g.ncclInt32 = 'int32';
    _g.ncclSum = 'sum';
    _g.ncclProd = 'prod';
    _g.ncclMax = 'max';
    _g.ncclMin = 'min';
    _g.ncclAvg = 'avg';
  }

})();
