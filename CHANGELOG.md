# Changelog

All notable changes to the **Gpubox** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-28

### Added
- **Unified Typed Public API**: Headless entry points `createSimulator`, `compile`, `analyze`, `run`, `validateOutput`, `createWorkerSimulator`, `getCapabilities`, `listProfiles`, `getProfile`, `listTopologies`, `getTopology`.
- **Public Integrator Guides & Usage Documentation**:
  - `docs/USAGE.md`: Comprehensive user manual with quick starts, memory allocation, multi-GPU collectives, race detection, and watchdog.
  - `docs/INTEGRATION_GUIDE.md`: Production integration patterns for online playgrounds, LMS platforms, and CI runners.
  - `docs/API_REFERENCE.md`: Full typed API signatures and option matrices.
  - `docs/RECIPES.md`: 8 verified, runnable real-world integration recipes.
  - `docs/LIMITATIONS.md`: Exhaustive boundaries of supported vs unsupported CUDA features.
  - `docs/SECURITY_AND_CSP.md`: Content Security Policy and Web Worker isolation guidelines.
  - `docs/PERFORMANCE.md`: Practical execution performance guidelines.
  - `docs/VERSIONING.md`: Versioning scheme and RunResult JSON schema guarantees.
- **External Real-Kernel Corpus (21 Kernels)**: Sourced and verified 21 real CUDA C++ kernels in `corpus/`, including 6 LLM forward-pass kernels from Karpathy's `llm.c` (`layernorm_forward`, `matmul_forward`, `attention_forward`, `gelu_forward`, `residual_forward`, `cross_entropy_forward`), 12 algorithmic patterns, and 3 unsupported boundary tests.
- **Physical GPU Hardware Verification**: Automated PyCUDA / NVRTC runner (`scripts/test-real-gpu.py`) verifying 6 kernels bit-exact or numerically matched on physical NVIDIA GeForce RTX 4050 Laptop GPU (AD107, sm_89, Driver 617.42).
- **Virtual Multi-GPU Subsystem (1–16 GPUs)**:
  - Ring-based NCCL collectives: `ncclAllReduce` (SUM, MAX, MIN, PROD, AVG), `ncclBroadcast`, `ncclReduceScatter`, `ncclAllGather`, `ncclAllToAll`, `ncclSend`, `ncclRecv`, `ncclGroupStart`, `ncclGroupEnd`.
  - Analytical ring traffic calculation: $2 \times \frac{N-1}{N} \times \text{bytesPerRank}$.
  - Sub-communicator partitioning (`ncclCommSplit`).
- **Safety, Diagnostic, and Hazard Detection**:
  - **Shared Memory Race Detector**: Epoch-based barrier hazard tracking for RAW, WAR, and WAW hazards.
  - **Multi-GPU Desync Detector**: Identifies rank omission, count mismatches, and deadlocks before hardware execution.
  - **Runaway Watchdog**: Execution timeout and step limit watchdog preventing browser hangs.
  - **Virtual Sparse VRAM**: 64 KB paged allocation enabling virtual allocations up to 80 GiB per device without inflating host RAM.
  - **Memory Leak Detection**: `getLeakReport()` tracking unfreed device allocations.
- **Numerical Reference Engine**:
  - IEEE-754 binary16 (FP16) and bfloat16 round-to-nearest-even conversions and arithmetic (`demo/fp16-bf16.js`).
  - Multi-mode validation harness (`validation-harness.js`) supporting absolute, relative, and ULP tolerance comparisons.
- **Rigorous Acceptance Test Suites**:
  - Cluster Acceptance Suite A1–A12 (12/12 passing).
  - SDK Acceptance Suite B1–B6 (6/6 passing).
  - Total automated suite: 175 tests passing (0 failed, 0 skipped).
- **CI / CD Pipeline**: GitHub Actions workflow (`.github/workflows/ci.yml`) testing on Node.js 18, 20, and 22 with automated report consistency drift verification.

### Changed
- Rebranded product and package to **Gpubox** (`gpubox`).
- Assigned copyright to **Aadityansha Verma** under Commercial License.
- Replaced ambiguous "Arbitrary CUDA" claims across documentation and UI with accurate "Supported CUDA GPU & Cluster Simulator".
- Labeled LLM cluster inference simulation as `[EXPERIMENTAL - analytical model, not validated]`.
- Corrected NVIDIA A100 SXM4 peak memory bandwidth specification to published 2039 GB/s across all profiles, reports, and models.
- Added explicit rejection and compiler diagnostics for unsupported CUDA features: Dynamic Parallelism (`cudaLaunchDevice`), Inline PTX (`asm volatile`), and Tensor Core WMMA (`nvcuda::wmma`).
- Removed duplicate documentation files and synchronized all report numbers to `test-results.json`.

---

Copyright (c) 2026 Aadityansha Verma. All rights reserved.
