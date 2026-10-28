# CUDA Simulator Compatibility Report & Specification

**Package:** `@trentorch/cuda-sim` (v0.1.0)  
**Specification Version:** 1.2 (Post-Verification Audit)  
**Date:** October 28, 2026

---

## 1. Compatibility Overview

This document provides a feature-by-feature evaluation of supported, limited, and unsupported CUDA APIs in `@trentorch/cuda-sim`. 

### Compatibility Classification Legend
- **SUPPORTED & TESTED (ST):** Full semantic simulation supported; verified by automated tests in this repository.
- **SUPPORTED WITH LIMITATIONS (SL):** Supported under specific execution constraints, modeled approximations, or synchronous emulation.
- **UNSUPPORTED (U):** Not implemented; calling will fail at transpilation or raise an explicit runtime exception.
- **NOT YET VERIFIED (NV):** Implemented in codebase but lacks end-to-end regression test coverage.

---

## 2. Feature-by-Feature Compatibility Matrix

### 2.1 Kernel Execution & Syntax

| Feature / API | Status | Limitations / Constraints | Tested In | Source File |
| :--- | :---: | :--- | :--- | :--- |
| `__global__` functions | **ST** | Transpiled to ES6 generator functions (`function*`) | All suites | `cuda-transpiler.js` |
| `__device__` functions | **ST** | Inlined or transformed to callable device generators | Attention tests | `cuda-transpiler.js` |
| `__shared__` memory arrays | **ST** | 32-bank modeled memory with bank conflict tracking | Suite 2, 3, 10 | `cuda-runtime.js` |
| `__shared__` scalars | **ST** | Tracked as single-element shared memory locations | Suite 10 | `cuda-transpiler.js` |
| `<<<grid, block, shmem>>>` | **ST** | Dim3 and scalar integer arguments supported; strictly validates `threadsPerBlock <= 1024` and `totalBlocks > 0` | All suites, Neg Suite 4 | `cuda-transpiler.js`, `cuda-runtime.js` |
| Dynamic Parallelism (`cudaLaunchDevice`) | **U (Tested)** | Nested kernel launches explicitly diagnosed and rejected | Neg Suite 6 | `cuda-transpiler.js` |
| Inline PTX Assembly (`asm volatile`) | **U (Tested)** | PTX assembly explicitly diagnosed and rejected | Neg Suite 6 | `cuda-transpiler.js` |
| CUDA Graph API (`cudaGraphCreate`, `cudaGraphLaunch`) | **U (Tested)** | Graph capture explicitly diagnosed and rejected | Neg Suite 6 | `cuda-transpiler.js` |

### 2.2 Memory Operations & Management

| Feature / API | Status | Limitations / Constraints | Tested In | Source File |
| :--- | :---: | :--- | :--- | :--- |
| `cudaMalloc` | **ST** | Allocates from simulated virtual device arena; rejects negative sizes (`cudaErrorMemoryAllocation`) | All suites, Neg Suite 3 | `cuda-runtime.js` |
| `cudaFree` | **ST** | Releases allocation, catches double-free (`cudaErrorInvalidValue`), and flags use-after-free (`cudaErrorIllegalAddress`) | Suite 1, Neg Suite 3 | `cuda-runtime.js` |
| `cudaMemset` | **ST** | Sets memory to zero or byte pattern | Histogram test | `cuda-runtime.js` |
| `cudaMemcpy(H2D)` | **ST** | Copies host array into virtual device buffer | All suites | `cuda-runtime.js` |
| `cudaMemcpy(D2H)` | **ST** | Safely clamped to minimum of size, dest, and source | All suites | `cuda-runtime.js` |
| `cudaMemcpy(D2D)` | **ST** | Copies within device arena or between device pointers | Suite 1 | `cuda-runtime.js` |
| `cudaMemcpyPeer` | **ST** | Cross-device memory transfer; records `crossGpuBytes` | Suite 7 (Reg) | `cuda-runtime.js` |
| `cudaDeviceEnablePeerAccess` / `cudaDeviceDisablePeerAccess` | **ST** | Enables/disables UVA direct peer memory access; flags unauthorized cross-device reads/writes | Neg Suite 5 | `cuda-runtime.js` |
| `cudaHostAlloc` / `cudaMallocHost` | **ST** | Transpiled to pinned host typed array | Metric test | `cuda-transpiler.js` |
| Hardware ECC / Parity Error Injection | **U** | Simulator operates in linear heap; no soft errors | None | N/A |

### 2.3 Synchronization, Warp Primitives & Error Handling

| Feature / API | Status | Limitations / Constraints | Tested In | Source File |
| :--- | :---: | :--- | :--- | :--- |
| `__syncthreads()` | **ST** | Block barrier synchronization across all warps | Suite 2, 3, 10 | `cuda-runtime.js` |
| Barrier Divergence Hazard Detection | **ST** | Flags `BarrierDivergenceError` if `__syncthreads` is called conditionally | Neg Suite 4 | `cuda-runtime.js` |
| `__shfl_down_sync` | **ST** | Intra-warp register exchange across 32 lanes | Suite 2, Warp test | `cuda-runtime.js` |
| `__shfl_up_sync` | **ST** | Intra-warp register exchange | Warp test | `cuda-runtime.js` |
| `__shfl_sync` | **ST** | Broadcast lane value across warp | Warp test | `cuda-runtime.js` |
| `__shfl_xor_sync` | **ST** | Butterfly permutation exchange across warp | Warp test | `cuda-runtime.js` |
| `cudaDeviceSynchronize()` | **ST** | Host waits for all device queues to complete | All suites | `cuda-runtime.js` |
| `cudaGetLastError()` | **ST** | Returns last error code and resets error status | Neg Suite 7 | `cuda-runtime.js` |
| `cudaPeekAtLastError()` | **ST** | Returns last error code without resetting | Neg Suite 7 | `cuda-runtime.js` |
| `cudaGetErrorString(err)` | **ST** | Returns descriptive string for CUDA error code | Neg Suite 7 | `cuda-runtime.js` |
| Async Stream Concurrency | **SL** | Streams execute sequentially; overlap is modeled | Suite 1 | `cuda-runtime.js` |

### 2.4 Atomics & Arithmetic Intrinsics

| Feature / API | Status | Limitations / Constraints | Tested In | Source File |
| :--- | :---: | :--- | :--- | :--- |
| `atomicAdd` (int, float) | **ST** | Atomic memory update; tracks contention | Histogram test | `cuda-runtime.js` |
| `atomicMin` / `atomicMax` | **ST** | Atomic reduction update | Regression test | `cuda-runtime.js` |
| `atomicExch` | **ST** | Atomic swap | Regression test | `cuda-runtime.js` |
| `sqrtf`, `expf`, `fabsf`, `fmaf` | **ST** | IEEE-754 single precision float arithmetic | Suite 2, 4, 10 | `cuda-runtime.js` |
| `isinf`, `isnan`, `INFINITY` | **ST** | Conforms to IEEE-754 infinity and NaN handling | Suite 10, isinf test | `cuda-runtime.js` |
| Hardware Tensor Cores (WMMA / MMA) | **U** | Matrix multiply executes in SIMT; no systolic array | Suite 3 | N/A |

### 2.5 Multi-GPU & NCCL Collectives

| Feature / API | Status | Limitations / Constraints | Tested In | Source File |
| :--- | :---: | :--- | :--- | :--- |
| `cudaSetDevice(id)` | **ST** | Switches active context for memory and launches | Suite 4, 5, 6, 9 | `cuda-runtime.js` |
| `cudaGetDeviceCount(&count)` | **ST** | Returns configured virtual cluster size (1..16) | Suite 4, 9 | `cuda-runtime.js` |
| `ncclCommInitAll` | **ST** | Initializes communicator clique across $N$ ranks | Suite 5, 6, 7, 8, 9 | `cuda-runtime.js` |
| `ncclCommInitRank` | **ST** | Initializes individual rank communicator | Suite 4 (Reg) | `cuda-runtime.js` |
| `ncclAllReduce` (SUM, MAX, MIN, PROD, AVG) | **ST** | Supports array mode and per-rank SPMG loop mode | Suite 5, 9, 11 | `cuda-runtime.js` |
| `ncclBroadcast` | **ST** | Validates root rank; models non-root inbound bytes | Suite 8, 11 | `cuda-runtime.js` |
| `ncclReduceScatter` | **ST** | Full reduction followed by rank chunk scatter | Suite 7, 11 | `cuda-runtime.js` |
| `ncclAllGather` | **ST** | Gathers rank shards in order $[0..N-1]$ | Suite 6, 11 | `cuda-runtime.js` |
| Collective Argument Validation | **ST** | Rejects invalid roots, mismatched buffers, negative sizes | Suite 11 | `cuda-runtime.js` |

---

## 3. Explicitly Documented Unsupported Features & Rationale

1. **CUDA Dynamic Parallelism (`cudaLaunchDevice`)**:
   - *Rationale:* Client-side execution in JavaScript and WebAssembly cannot recursively spin up new thread grids from worker threads without substantial runtime overhead and stack expansion.
2. **CUDA Graph API**:
   - *Rationale:* The graph node capture engine (`cudaStreamBeginCapture`, `cudaGraphInstantiate`) requires static kernel graph compilation not suitable for the immediate-mode SIMT interpreter.
3. **Inline PTX Assembly**:
   - *Rationale:* Web browsers execute WebAssembly bytecode or JavaScript JIT; physical NVVM PTX instructions (`@%p0 st.global.f32...`) cannot be executed without an NVIDIA driver and physical GPU silicon.
4. **Hardware Systolic Tensor Cores (WMMA/MMA)**:
   - *Rationale:* FP16/BF16 tensor core instructions (`nvcuda::wmma::load_matrix_sync`) operate on specialized physical microarchitectures. The simulator models performance FLOPs and computes mathematically identical FP32 results, but does not simulate hardware systolic arrays.
5. **Hardware ECC / Memory Scrubbing Errors**:
   - *Rationale:* The simulator allocates virtual arenas within browser heap memory (`ArrayBuffer`), which is memory-safe and cannot reproduce hardware DRAM cosmic ray bit-flips or ECC scrubbing interrupts.
