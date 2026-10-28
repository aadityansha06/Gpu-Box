# Gpubox — Browser-Based CUDA Simulation & Pre-Validation Environment

`gpubox` is a client-side CUDA and multi-GPU simulation and pre-validation library that helps developers test supported CUDA algorithms, compare kernel outputs against references, inspect modeled execution metrics, and experiment with virtual 1–16 GPU clusters before paying for physical hardware.

> **Client-side execution via WebAssembly & JavaScript SIMT. No cloud GPU allocation required for early logic pre-validation.**

```html
<!-- CDN Import via jsDelivr -->
<script src="https://cdn.jsdelivr.net/npm/gpubox@latest/dist/cuda-sim.min.js"></script>
```

```bash
# Node.js / Bundler Installation
npm install gpubox
```

> [!NOTE]
> **Trademark Notice**: Not affiliated with or endorsed by NVIDIA. CUDA and NVIDIA are trademarks of NVIDIA Corporation; used here only to describe compatibility.

---

## Documentation

- **[User Guide & Quick Start](docs/USAGE.md)** — Step-by-step developer guide, workflows, diagnostic tools, and quick start.
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)** — Guide for embedding Gpubox in web playgrounds, EdTech LMS, and CI pipelines.
- **[API Reference](docs/API_REFERENCE.md)** — Complete typed API reference (`createSimulator`, `compile`, `run`, collectives).
- **[Recipes & Runnables](docs/RECIPES.md)** — 8 tested, runnable integration recipes.
- **[Material Limitations](docs/LIMITATIONS.md)** — Exact boundaries between supported and unsupported CUDA features.
- **[Security & CSP Guide](docs/SECURITY_AND_CSP.md)** — Content Security Policy requirements and worker isolation.
- **[Performance Guide](docs/PERFORMANCE.md)** — Optimization strategies for browser and Node execution.
- **[Versioning & Stability](docs/VERSIONING.md)** — Semantic versioning and schema guarantees.

---

## Core Capabilities & Intended Benefits

1. **Earlier Detection of Supported Logic Errors**:
   - Write or paste supported CUDA C++ kernels and host orchestration code (`cudaMalloc`, `cudaMemcpy`, `<<<grid, block>>>`, `dim3`, etc.).
   - Catches off-by-one boundary violations, out-of-bounds global and shared memory accesses, divergent `__syncthreads()` deadlocks, double-free operations, and uninitialized reads with structured diagnostics identifying the kernel, thread, and block coordinates.
2. **Shared Memory Race Detection & Watchdog**:
   - Barrier-epoch hazard detector flags RAW, WAR, and WAW shared memory hazards when `__syncthreads()` is omitted.
   - Runaway loop watchdog aborts infinite loops (`while(true) {}`) and names the offending kernel, block, and thread.
3. **Virtual Multi-GPU & NCCL Collectives (1–16 GPUs)**:
   - Simulates clusters of 1, 2, 4, 8, and 16 virtual GPU nodes with rank-isolated memory spaces and virtual sparse VRAM up to 80 GiB per device.
   - Supports ring collectives (`ncclAllReduce` with SUM/MAX/MIN/PROD/AVG, `ncclBroadcast`, `ncclReduceScatter`, `ncclAllGather`, `ncclAllToAll`, and point-to-point `ncclSend`/`ncclRecv`).
   - Automated desync detector catches rank omission, count mismatches, and collective ordering deadlock before running on physical GPU clusters.
4. **Hardware Profiles & Topologies**:
   - Models published specs for NVIDIA A100 SXM4 (2039 GB/s), H100 SXM5 (3350 GB/s), RTX 4090, RTX 4050 Laptop, RTX 5060 Ti (`sm_120 [TO VERIFY]`), RTX 3090, and V100.
   - Supported cluster topologies: `nvswitch_mesh`, `ring`, `torus_2d`, and `pcie_tree`.
5. **Dual Simulation Engines**:
   - **JavaScript SIMT Generator Engine**: Direct AST transpilation to ES6 generators with dynamic step-by-step lockstep coroutine scheduling.
   - **WebAssembly Engine (WASM)**: C++17 core compiled via Emscripten for fixed problem harnesses.
6. **Rigorous Validation**:
   - **175/175 Automated Tests Passing**: Benchmarks (88), Metrics regression (8), Negative diagnostics (31), Numerical reference checks (9), External kernel corpus (21), Cluster acceptance A1–A12 (12), and SDK acceptance B1–B6 (6).
   - **6 Real-GPU Hardware Checks**: Verified bit-exact and accurate against physical NVIDIA GeForce RTX 4050 Laptop GPU (AD107, sm_89).

---

## Important Scope & Material Limitations

`cuda-sim` is designed to accelerate early development and pre-validation. It **does not replace physical GPU testing** and does not guarantee identical hardware behavior:

* **Hardware Microarchitecture:** The simulator does not model physical DRAM refresh timings, hardware ECC memory scrubbing, thermal throttling, or PCIe bus contention.
* **Floating-Point Arithmetic:** Calculations follow IEEE-754 single-precision float32 rules (`Math.fround`). Physical GPU Special Function Units (SFU) use hardware polynomial approximations (e.g., `rsqrt.approx.f32`) that may differ by a least-significant bit from CPU math implementations.
* **Concurrency & Scheduling:** Simulated warps step in deterministic lockstep coroutines; physical GPU hardware warp schedulers interleave warps asynchronously across physical SMs.
* **Unsupported CUDA Features:** Dynamic Parallelism (`cudaLaunchDevice`), Inline PTX assembly (`asm volatile`), CUDA Graph APIs (`cudaGraphCreate`), and hardware tensor core systolic arrays (WMMA/MMA) are explicitly unsupported and rejected during transpilation.
* **Security & Execution Isolation:** When running untrusted user code via the JavaScript engine, execution relies on dynamic function evaluation (`new Function`). Running untrusted code in production requires external sandboxing (such as secure iframes, restricted Web Workers with execution timeouts, and strict Content Security Policies), as Web Workers alone do not prevent CPU exhaustion or infinite loops.

---

## Architecture & Deployment Documentation

For an in-depth breakdown of the simulator internals, WebAssembly compilation, CDN distribution, and known constraints, see:
**[ARCHITECTURE.md](ARCHITECTURE.md)**

---

## Directory Structure

```
cuda-sim/
├── ARCHITECTURE.md       ← Comprehensive architecture, WASM, CDN, and limitations guide
├── README.md             ← Project overview & quick start
├── demo/                 ← Interactive browser playground UI
│   ├── index.html        ← Free-form Monaco CUDA editor & hardware profiler
│   ├── gpu-profiles.js   ← Real GPU specifications (A100, H100, RTX 4090, 5060 Ti...)
│   ├── cuda-runtime.js   ← Virtual GPU runtime (SIMT scheduler, memory, NCCL)
│   └── cuda-transpiler.js← CUDA C++ to JavaScript transpiler
├── sim/                  ← C++17 simulator core (compiled to WebAssembly)
│   ├── include/
│   │   └── cuda_sim.h    ← Full CUDA API shim header
│   ├── src/              ← Runtime, scheduler, counters, detectors, multi-GPU
│   └── build-wasm/       ← Compiled cuda_sim.wasm (192 KB) & cuda_sim.js
├── js/                   ← NPM package (@trentorch/cuda-sim)
│   ├── dist/             ← Bundled ESM, CJS, UMD, and types
│   ├── public/           ← Static WASM assets
│   └── src/              ← TypeScript API wrapper
└── scripts/
    ├── install-emsdk.ps1 ← Automated Emscripten SDK installer
    ├── build-wasm.ps1    ← WASM compilation script
    └── server.js         ← Local development server (http://localhost:3456/demo/)
```

---

## Quick Start

### 1. Launch the Playground
```powershell
node scripts\server.js
# Open http://localhost:3456/demo/
```

### 2. Run Any CUDA Code
In the editor, type or paste any CUDA C++ code and click **▶ Run Simulation** (or press `Ctrl + Enter`):
```cpp
__global__ void saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int n = 1024;
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    
    saxpy<<<4, 256>>>(2.0f, d_x, d_y, n);
    cudaDeviceSynchronize();
    return 0;
}
```

### 3. Rebuilding the WebAssembly Core (Optional)
If modifying the C++ core:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build-wasm.ps1
```

### 4. Building the NPM Package
```powershell
cd js
npm run build
```

### 5. Running the Complete Verification & Test Suite
```powershell
# 1. Complete automated test suite (175 Automated Tests)
cd js
npm test

# 2. Run individual test suites
npm run test:benchmarks   # 88 validation benchmarks (1, 2, 4, 8, 16 virtual GPUs)
npm run test:metrics      # 8 GPU metrics regression tests
npm run test:negative     # 31 negative testing & error diagnostic tests
npm run test:numerical    # 9 independent CPU ground-truth numerical tests
node ../scripts/run-corpus.js                # 21 external CUDA kernels (llm.c, matrix, reductions)
node ../scripts/test-cluster-acceptance.js   # 12 cluster acceptance tests (A1–A12)
node ../scripts/test-sdk-acceptance.js       # 6 SDK & integrator acceptance tests (B1–B6)

# 3. Physical NVIDIA GPU Hardware Verification (NVIDIA GPU + NVRTC)
npm run test:real-gpu     # 6 physical GPU kernel execution checks (Ada Lovelace RTX 4050)
```

---

## Technical Reports & Specifications

Comprehensive architectural, benchmarking, compatibility, and verification reports are available:

- **[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md)** — Automated verification run results, 175/175 automated tests, negative testing diagnostics, and physical RTX 4050 GPU execution.
- **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)** — Comprehensive benchmark matrix covering 1, 2, 4, 8, and 16 GPUs, edge cases, negative tests, and hardware correlation.
- **[CUDA_COMPATIBILITY.md](CUDA_COMPATIBILITY.md)** — Feature-by-feature specification of supported, limited, and unsupported CUDA APIs.
- **[AUDIT_REPORT.md](AUDIT_REPORT.md)** — Deep architectural audit of transpilation, SIMT scheduling, and collective communication paths.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Commercial deployment, WebAssembly core, and client-side distribution design.
- **[BUSINESS_IDEA.md](BUSINESS_IDEA.md)** — Product positioning, target customer segments, buyer pain points, and commercialization strategy.

---

## License

Commercial License — Copyright (c) 2026 Aadityansha Verma. All rights reserved.
