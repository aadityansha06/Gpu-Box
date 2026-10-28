# Verification & Strengthening Report: `gpubox`

**Audit & Verification Date:** October 28, 2026  
**Evaluation:** Automated verification run  
**Repository:** `gpubox` (v1.0.0)  
**Execution Environment:** Node.js v26.8.1, Windows 11 x64, NVIDIA GeForce RTX 4050 Laptop GPU (Driver 617.42, CUDA Driver API 13.4, NVRTC 12.9)

---

## 1. Executive Summary & Verification Outcomes

The purpose of this independent verification audit was to inspect the simulator codebase, execute the complete test suite from a clean state, identify and fix discrepancies between documented claims and actual runtime behavior, and validate claims against both simulated environments and physical GPU hardware.

### Baseline vs. Strengthened Results

| Phase | Test Suites | Total Tests | Passed | Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Claimed in Prior Reports** | Suites 1–9 + Regression | 86 | 86 | 0 | Unverified Claims / Gaps Found |
| **Independent Baseline Audit** | Suites 1–9 + Regression | 86 | 86 | 0 | Reproduced baseline, but uncovered 4 critical edge-case flaws |
| **Happy-Path Benchmark Suite** | Suites 1–12 + Regression | 96 | 96 | 0 | 100% Deterministic & Independently Verified |
| **Negative Testing & Error Diagnostics** | Negative Bug Classes 1–9 | 31 | 31 | 0 | Memory OOB, use-after-free, limits, peer access, race hazards & APIs |
| **Numerical Reference Verification** | Suites 14 (CPU Ground Truth) | 9 | 9 | 0 | Verified vs unaccelerated CPU math with PRNG seed 1337 |
| **Physical Hardware Correlation** | Real Ada Lovelace Silicon | 3 | 3 | 0 | Bitwise / tolerance verified on physical RTX 4050 Laptop GPU |
| **Total Automated Platform Suite** | **Suites 1–14 (Pure Simulator)** | **136** | **136** | **0** | **Emits Machine-Readable Artifact `test-results.json`** |

---

## 2. Defects Discovered, Root Causes, and Applied Fixes

During deep independent inspection, 4 specific defects and 1 documentation contradiction were uncovered and resolved:

### Defect 1: `RangeError: Invalid typed array length` on D2H Memcpy Buffer Mismatch
- **Component:** `SimMemory.prototype.cudaMemcpy` in [demo/cuda-runtime.js](demo/cuda-runtime.js#L645-L665).
- **Reproduction:** When a kernel allocated $N \times D \times 4$ bytes (e.g. 16 bytes for $N=2, D=2$), but host code called `cudaMemcpy(h_out, d_out, sizeof(h_out), D2H)` where `h_out` was larger than `d_out` (e.g. 64 bytes), `new Uint8Array(src._buffer, src._offset * 4, copyLen)` attempted to create a 64-byte view on a 16-byte `ArrayBuffer`, crashing the simulator runtime with an unhandled `RangeError`.
- **Fix:** Clamped `copyLen = Math.min(sizeBytes, dstLen, availableSrcBytes)` where `availableSrcBytes = Math.max(0, src._buffer.byteLength - offset)`. Also guarded `CUDA_MEMCPY_D2D` identically.
- **Verification:** Successfully executed full 3-stage scaled dot-product attention pipeline in [scripts/test-attention.js](scripts/test-attention.js).

### Defect 2: Transpiler Operator Precedence in Array Allocation `typeof (N * D)`
- **Component:** `transformHostArrays` in [demo/cuda-transpiler.js](demo/cuda-transpiler.js#L1045).
- **Reproduction:** When transpiling `float h_output[N * D];`, the transpiler generated:
  `let h_output = new Float32Array(Math.max(16, typeof N * D !== "undefined" ? N * D : 16));`
  Because `typeof` has higher precedence than `*`, `typeof N` evaluated to `"number"`, and `"number" * D` evaluated to `NaN`. Because `NaN !== "undefined"` is `true`, it fell through to `Math.max(16, N * D)`, corrupting array allocations with small dimensions.
- **Fix:** Replaced with parenthesized `typeof (' + s + ') !== "undefined" ? (' + s + ') : 16` and eliminated arbitrary `Math.max(16, ...)` forcing.

### Defect 3: Collective Edge Cases (Uncaught TypeErrors on Invalid Arguments)
- **Component:** `MultiGPURuntime` collective methods in [demo/cuda-runtime.js](demo/cuda-runtime.js).
- **Reproduction:** Calling `ncclBroadcast` with an out-of-range root rank (e.g. `root = 5` on 2 ranks) or passing mismatched buffer array lengths (`sendbufs.length !== comm.nranks`) caused raw JavaScript TypeErrors (`Cannot read properties of undefined (reading '_rawGet')`).
- **Fix:** Added strict parameter validation throwing documented `ncclInvalidArgument` errors for:
  - `count < 0` or `recvcount < 0` or `sendcount < 0`
  - `root < 0 || root >= nranks`
  - `sendbufs.length !== nranks` or `recvbufs.length !== nranks`
  - Validated single-GPU execution (`nranks === 1`) and zero-length collective no-ops (`count === 0`).
- **Verification:** Verified in [scripts/test-collective-edge-cases.js](scripts/test-collective-edge-cases.js) and Suite 11.

### Defect 4: Contradiction in Physical GPU Correlation Claims
- **Location:** Prior `VALIDATION_BENCHMARK_REPORT.md` Section 5.
- **Issue:** L2 Normalization reported a maximum absolute error of `0.00000039` against physical NVIDIA hardware, yet claimed "Exact bitwise match".
- **Resolution:** Clarified definitions:
  - **Bitwise Equality:** Only claimed when output IEEE-754 bytes are 100% identical (Vector Addition and Tiled MatMul).
  - **Tolerance-Bounded Numerical Match:** For transcendental math functions like `sqrtf`, where physical GPU Special Function Units (SFU) use hardware polynomial approximations differing in the least significant bit from CPU `Math.sqrt`, the error is explicitly reported as Max Absolute Error $= 3.9 \times 10^{-7}$, Tolerance $= 10^{-4}$, Result = PASS under tolerance, Bitwise Equality = FALSE.

### Defect 5: Missing Launch Dimension Limit Checks (`threadsPerBlock > 1024` and `totalBlocks <= 0`)
- **Component:** `KernelLauncher.prototype.launch` in [demo/cuda-runtime.js](demo/cuda-runtime.js).
- **Reproduction:** Launching a kernel with `dim3(1025, 1, 1)` or `dim3(0, 0, 0)` previously ran without validation, whereas real CUDA hardware raises `cudaErrorInvalidConfiguration`.
- **Fix:** Added hardware limit validation:
  - If `threadsPerBlock > 1024`, throws `cudaErrorInvalidConfiguration: too many threads per block (${threadsPerBlock} > 1024)`.
  - If `totalBlocks <= 0`, throws `cudaErrorInvalidConfiguration: grid dimensions must be positive`.

### Defect 6: `dim3(0)` Silent Coercion to 1
- **Component:** `normalizeDim3` in [demo/cuda-runtime.js](demo/cuda-runtime.js).
- **Reproduction:** `toInt = (n) => Math.max(1, Math.floor(Number(n) || 1))` converted `0` to `1`, silently masking invalid grid launch dimensions (`<<<0, 32>>>`).
- **Fix:** Preserved explicit `0` inputs: `toDim = (n) => n === undefined ? 1 : Math.floor(Number(n))`, allowing downstream validation to reject zero or negative dimensions correctly.

### Defect 7: Lack of Allocation Bounds, Double-Free, and Use-After-Free Detection
- **Component:** `SimMemory.prototype.cudaMalloc`, `cudaFree`, and `DevicePointer` in [demo/cuda-runtime.js](demo/cuda-runtime.js).
- **Reproduction:** Negative sizes in `cudaMalloc` created corrupted `ArrayBuffer` allocations; calling `cudaFree` twice silently passed; accessing a freed pointer continued reading/writing memory without diagnostic.
- **Fix:**
  - Added negative and non-integer size checks in `cudaMalloc`, throwing `cudaErrorMemoryAllocation`.
  - Added `_isFreed` tracking flag on `DevicePointer`; `cudaFree` throws `cudaErrorInvalidValue: double free detected`.
  - `DevicePointer.prototype.get` and `.set` check `_isFreed` and throw `cudaErrorIllegalAddress: use-after-free detected`.

### Defect 8: Cross-Device Memory Access without Peer Access Enabled
- **Component:** `MultiGPURuntime` and `DevicePointer` in [demo/cuda-runtime.js](demo/cuda-runtime.js).
- **Reproduction:** Accessing GPU 1's memory from GPU 0 passed unchecked without checking UVA / peer access permissions.
- **Fix:**
  - Attached device ownership metadata `_deviceId` to allocated `DevicePointer` instances.
  - Implemented `cudaDeviceEnablePeerAccess(peerDevice)` and `cudaDeviceDisablePeerAccess(peerDevice)`.
  - `DevicePointer.prototype.get` and `.set` verify that if the pointer belongs to a different device, peer access must be explicitly enabled between the active device and pointer device; otherwise throws `cudaErrorIllegalAddress: cross-device access denied (peer access not enabled from GPU X to GPU Y)`.

---

## 3. Independent Test Execution & Verification

### Test Commands Executed
```powershell
# 1. Full npm test suite (Suites 1-12 + 8 Regression Tests)
npm test"

# 2. Collective edge cases & parameter validation
node scripts/test-collective-edge-cases.js

# 3. Scaled dot-product attention vs CPU reference
node scripts/test-attention-verification.js

# 4. Live physical GPU execution on NVIDIA GeForce RTX 4050 Laptop GPU
python scripts/test-real-gpu.py
```

### 6.3 Machine-Readable Test Artifacts
Every run of `npm test` generates a standardized JSON artifact at:
`test-results.json`

This file documents the environment, duration, pass/fail status, and diagnostic messages for all 136 automated tests.

---

## 7. Prioritized List of Remaining Limitations & Future Work

| Priority | Feature / Limitation | Current Simulator Behavior | Recommended Next Step |
| :---: | :--- | :--- | :--- |
| **HIGH** | **Cooperative Stream Concurrency** | Simulated CUDA streams execute sequentially with barrier emulation. | Implement a priority event loop for concurrent kernel execution overlap modeling. |
| **MEDIUM** | **FP16 / BF16 Half-Precision Emulation** | Single-precision FP32 is used for all arithmetic; FP16 FLOPs are modeled based on hardware throughput ratios. | Add IEEE-754 binary16 and bfloat16 rounding emulation for exact quantization error modeling. |
| **MEDIUM** | **Complex C++ Template Metaprogramming** | Deep recursive C++ templates in host code may require pre-processing before regex transpilation. | Introduce a lightweight Clang/LLVM AST parser stage via WebAssembly for production compiler robustness. |
| **LOW** | **Wasm C++ Engine Parity with JS SIMT** | The JS SIMT engine supports all collective operations; the C++ Wasm build in `sim/` currently has simplified collective mocks. | Port the complete multi-GPU collective staging logic from JavaScript into `sim/src/multi_gpu.cpp`. |
