# Engine Gap Analysis: WASM Engine vs. JavaScript Engine vs. Playground

**Date:** October 28, 2026  
**Audited Components:**  
1. **WASM C++ Engine:** `sim/src/` (`api.cpp`, `multi_gpu.cpp`, `runtime.cpp`, `scheduler.cpp`, `counters.cpp`, `detectors.cpp`, `rewriter.cpp`)  
2. **JavaScript Engine:** `demo/` (`cuda-runtime.js`, `demo/cuda-transpiler.js`, `demo/gpu-profiles.js`)  
3. **npm Package Entry:** `js/src/` (`index.ts`, `loader.ts`, `types.ts`)  
4. **Browser Playground:** `demo/index.html`  

---

## 1. Feature & Capability Comparison Matrix

| Feature / Capability | JavaScript SIMT Engine (`demo/cuda-runtime.js`) | WASM Core Engine (`sim/src/`) | npm Package (`js/src/index.ts`) | Playground (`demo/index.html`) |
| :--- | :---: | :---: | :---: | :---: |
| **Arbitrary CUDA Source Execution** | **Supported** (transpiles C++ to ES6 generators) | **Not Supported** (requires pre-compiled C++ kernels in binary) | **Stubbed** (calls precompiled kernels or fails) | Calls JS transpiler directly |
| **Warp Scheduler & SIMT (32 lanes)** | **Supported** (generators advance in lockstep) | **Supported** (fiber/coroutine scheduler) | Exposes WASM counters only | Uses JS scheduler |
| **Warp Shuffles (`__shfl_*_sync`)** | **Supported** (down, up, sync, xor across 32 lanes) | **Partial** (`__shfl_down_sync` in `runtime.cpp`) | Not directly accessible | Uses JS implementation |
| **Shared Memory Bank Conflicts** | **Supported** (32 banks tracked via Proxy) | **Supported** (tracked in `detectors.cpp`) | Exposes counter | Renders JS counter |
| **Barrier Divergence Hazard Detection** | **Supported** (throws `BarrierDivergenceError`) | **Partial** (`detectors.cpp` records mismatch) | Exposes diagnostic string | Renders in console |
| **Device Pointer Bounds Checking** | **Supported** (OOB read/write throws `RangeError`) | **Supported** (tracked via `detectors.cpp`) | Exposes diagnostic string | Renders in console |
| **Allocation & Use-After-Free Detection** | **Supported** (catches double-free & use-after-free) | **Not Supported** (basic heap arena) | Not exposed | Relies on JS runtime |
| **Multi-GPU Cluster Sizes** | **1–16 Virtual GPUs** (tested at 2, 4, 8, 16) | **1–16 Virtual GPUs** (arena array allocated) | Exposes analytical model only | Fixed dropdown (1, 2, 4, 8, 16) |
| **Native NCCL `AllReduce`** | **Supported** (SUM, MAX, MIN, PROD, AVG across ranks) | **Mocked** (only local `memcpy` send->recv; no inter-rank reduction) | Exposes analytical model only | Direct JS execution |
| **Native NCCL `Broadcast`** | **Supported** (validates root rank 0..N-1, distributes data) | **Not Implemented** | Not exposed | Direct JS execution |
| **Native NCCL `ReduceScatter`** | **Supported** (reduces and scatters chunks) | **Not Implemented** | Not exposed | Direct JS execution |
| **Native NCCL `AllGather`** | **Supported** (gathers rank shards in order) | **Not Implemented** | Not exposed | Direct JS execution |
| **UVA Peer Memory Isolation** | **Supported** (`cudaDeviceEnablePeerAccess` enforced) | **Partial** (`peer_access_enabled` flag present, unchecked on read/write) | Not exposed | Direct JS execution |
| **Collective Desync / Deadlock Detector** | **Not Implemented** (relies on sequential execution) | **Not Implemented** | Not exposed | None |
| **Shared Memory Race Detector** | **Not Implemented** (only bank conflicts tracked) | **Not Implemented** (only bank conflicts tracked) | Not exposed | None |
| **FP16 / BF16 Arithmetic** | **Modeled** (FP32 precision with throughput weighting) | **Modeled** (FP32 precision) | Not exposed | Renders modeled FLOPs |
| **Headless Public API Integration** | No public packaging (lives in `demo/`) | Packaged via Emscripten | Incomplete bridge | Direct DOM / global script coupling |

---

## 2. Inventory of Playground Private Back-Doors & Couplings

Inspection of `demo/index.html` reveals that the playground is currently tightly coupled to private global state and bypasses `js/src/index.ts`:

1. **Script Inclusions:**
   - `<script src="gpu-profiles.js"></script>`
   - `<script src="cuda-runtime.js"></script>`
   - `<script src="cuda-transpiler.js"></script>`
   - No inclusion of `dist/cuda-sim.js`, `cuda_sim.wasm`, or standard ES module entry points.
2. **Private Runtime Access:**
   - Instantiates `new CudaTranspiler()` and calls `transpile(code)`.
   - Executes host code using `new Function('__rt', 'SimMemory', '__kernel_*', ...)` in the window global context.
   - Directly inspects internal properties of `__rt`, including `__rt._gpus`, `__rt.metrics`, `__rt.activeDevice`.
   - Manually mutates DOM elements based on ad-hoc global callbacks rather than consuming a structured `RunResult` object.
3. **Profile Inconsistencies:**
   - The UI displays hardcoded specs (e.g. A100 memory bandwidth showing 900 GB/s in certain labels vs 2039 GB/s in `gpu-profiles.js`).

---

## 3. WASM Engine (`sim/src/`) Limitations & Code Gaps

1. **Execution Model:**
   - WASM is compiled from C++ using Emscripten. Because there is no client-side C++ compiler inside WASM, user CUDA C++ kernels cannot be compiled to WASM bytecode on the fly without a heavy compiler like Clang/LLVM compiled to WASM.
   - The current WASM engine relies on kernels being compiled into the binary or uses `simRewriteSource` to emit transformed C++ for offline compilation.
2. **Collective Staging Deficiencies:**
   - In `sim/src/multi_gpu.cpp`, `__sim_nccl_allreduce` only computes cross-GPU byte counters and copies local buffers (`memcpy(recvbuff, sendbuff)`). It does not perform actual reductions across multiple virtual GPU instances.
   - NCCL collectives `ncclBroadcast`, `ncclReduceScatter`, and `ncclAllGather` are completely missing from `sim/src/multi_gpu.cpp`.
3. **Memory Limits:**
   - Virtual memory is allocated as `std::vector<uint8_t> arena`. It does not support sparse virtual allocations; allocating full simulated capacities (e.g. 80 GB) would crash the browser or 32-bit WASM linear memory address space (limited to 2 GB - 4 GB).

---

## 4. Architectural Target: Engine-First SDK

To resolve these discrepancies and fulfill the product principle:
1. **Unified `Engine` Interface**: A TypeScript/JavaScript-first headless SDK that provides 100% of simulator features (transpilation, SIMT execution, 1–16 GPU cluster, collectives, telemetry, and diagnostics) with zero DOM dependencies.
2. **WASM Alignment**: Expose the analytical models, counters, and high-performance kernel simulation routines through the unified interface, honestly reporting in `getCapabilities()` which capabilities run in JS vs WASM.
3. **Thin Client Playground**: Rebuild `demo/index.html` to consume exclusively the public SDK (`import { createSimulator, compile, run } from 'gpubox'`).
