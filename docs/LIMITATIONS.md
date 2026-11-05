# Gpubox Known Limitations & Boundary Constraints

This document is derived directly from `compat.json` and details the architectural and feature boundaries of the Gpubox simulation platform.

## Supported vs. Unsupported Feature Matrix

| Feature | Support Tier | Behavior on Usage | Recommendation |
|---|---|---|---|
| Arbitrary Grid/Block Dims | Supported | Full execution up to 1024 threads per block | Standard CUDA block launch |
| Static `__shared__` Memory | Supported | Modeled with bank conflict and race detection | Use standard 1D/2D array declarations |
| Dynamic `extern __shared__` | Supported | Managed via launcher allocation | Pass 3rd launch argument |
| `__syncthreads()` Barrier | Supported | Lockstep coroutine sync with divergence detection | Avoid calling inside diverged branches |
| CUDA Atomics | Supported | `atomicAdd`, `atomicSub`, `atomicMax`, `atomicMin`, `atomicCAS`, etc. | Standard atomic primitives |
| Warp Intrinsics | Supported | `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync`, `__shfl_sync`, `__ballot_sync` | Standard warp-level primitives |
| FP16 & BF16 Math | Supported | Bit-exact IEEE-754 binary16 & bfloat16 round-to-nearest-even | Use `__half` and `__nv_bfloat16` intrinsics |
| NCCL Collectives (1-16 GPUs) | Supported | Native simulation of AllReduce, Broadcast, ReduceScatter, AllGather, Reduce, AllToAll | Use standard NCCL calls |
| P2P & Sub-Communicators | Supported | Point-to-point Send/Recv and rank sub-cliques | Use `ncclGroupStart`/`ncclGroupEnd` |
| Sparse Virtual VRAM | Supported | Up to 80 GiB per device across 16 GPUs with <256 MB real heap | Pre-validate memory scaling |
| **Dynamic Parallelism** | **Unsupported** | Deterministically rejected at compilation | Flatten nested kernel launches |
| **Inline PTX Assembly (`__asm__`)** | **Unsupported** | Deterministically rejected at compilation | Use high-level CUDA C++ primitives |
| **WMMA Tensor Cores (`wmma`)** | **Unsupported** | Deterministically rejected at compilation | Use SIMT matrix multiply kernels |
| **CUDA Graph API (`cudaGraph*`)** | **Unsupported** | Deterministically rejected at compilation | Launch kernels sequentially via stream |

## Analytical vs. Physical Correlation

1. **Cycle Accuracy:** Gpubox uses an analytical memory and instruction latency model based on GPU profile datasheets. It models global memory latency, shared bank conflicts, warp divergence, and collective network ring transfers. It does not model micro-architectural hardware cache thrashing or physical thermal throttling.
2. **Safety First:** Any unsupported feature triggers an explicit compilation error. Gpubox never silently ignores code it cannot faithfully simulate.

## Numerics, integers and timing (verified against an RTX 4050 Laptop GPU)

- **Arithmetic precision:** kernels run on JavaScript doubles and are rounded to FP32 when stored, so results differ from a GPU at roughly 1e-7 relative error (11/11 parity kernels match within 4e-6; see `node scripts/gpu-parity.js`).
- **Integers in device memory** are stored as float32 values: exact up to 2^24 (16,777,216). Register/local 32-bit integer math (`int` multiply wrap, `unsigned` shifts and stores) follows C semantics.
- **Integer division** between integer-typed operands truncates like C; mixed or untyped expressions follow JavaScript.
- **Time is a roofline model** (launch + max(DRAM, compute, shared) + atomics + barriers + interconnect), calibrated on an RTX 4050 Laptop GPU. Expect 2-3x error on micro-kernels; bandwidth-bound kernels track better.
- **Not a C++ front end:** semantic errors such as type mismatches are not diagnosed. Syntax the translator cannot express is reported as a compile error rather than run incorrectly.
- **Collective deadlocks** (a rank that never joins a collective, mismatched counts or ops) are reported when the program ends.
