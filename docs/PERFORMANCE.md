# Gpubox Performance & Tuning Guide

This document outlines practical problem sizes, tuning parameters, and measured execution performance for the Gpubox simulation engine.

## Recommended Problem Sizes for Browser Simulation

Because Gpubox simulates SIMT hardware threads in software with full cycle instrumentation, bank conflict tracking, and barrier synchronization:

| Problem Domain | Recommended Grid / Problem Size | Execution Time (Typical) | Memory Overhead |
|---|---|---|---|
| Vector Arithmetic | N = 1 to 16,384 elements | < 100 ms | < 2 MB |
| Tiled Matrix Multiply | 16x16 to 64x64 matrices | 50 – 400 ms | < 5 MB |
| Block Reductions | N = 32 to 2,048 elements | 20 – 150 ms | < 2 MB |
| LLM Token Layers | LayerNorm / GELU (C = 64..512) | 30 – 120 ms | < 4 MB |
| Multi-GPU Collectives | 1 to 16 virtual GPUs (count = 1024) | 1 – 15 ms | < 10 MB |
| Virtual VRAM Scaling | Up to 80 GiB per device (1.28 TiB cluster) | Instant (< 5 ms) | < 30 MB (Sparse) |

## Tuning for High-Throughput CI

When running in CI pipelines (e.g. testing hundreds of student kernels or pre-commit checks):

1. **Disable Unnecessary Traces:** Set `traceLevel: 'off'` in `SimConfig`. This disables event allocations and keeps runs fast.
2. **Selective Race Detection:** Shared memory race detection tracks barrier epochs. For production regression passes, enable `raceDetection: true`; for rapid throughput testing, disable when races are already verified.
3. **Use Sparse Virtual Buffers:** Large tensor shapes can be tested without allocating full arrays in Node or browser memory.
