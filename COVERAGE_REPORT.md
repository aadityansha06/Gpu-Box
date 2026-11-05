# Gpubox Kernel Coverage Report

**Execution Date:** 2026-11-05T15:16:08.562Z  
**Total Corpus Kernels:** 21  
**Verified Supported Kernels:** 18/18  
**Verified Unsupported Rejections:** 3/3  
**Overall Conformance:** 21/21 Passed (0 Unexpected Behaviors)  

## Test Results Breakdown

| Kernel | Category | Status | Verdict | Duration | Notes / Diagnostic |
|---|---|---|---|---|---|
| `vector_add.cu` | Algorithmic | PASS | PASS | 58 ms | Bit-exact vector addition |
| `matrix_mul.cu` | Algorithmic | PASS | PASS | 39 ms | 16x16 tiled matrix multiply |
| `reduce_sum.cu` | Algorithmic | PASS | PASS | 9 ms | Sum 1..64 = 2080 |
| `matrix_transpose.cu` | Algorithmic | PASS | PASS | 13 ms | 16x16 shared transpose |
| `dot_product.cu` | Algorithmic | PASS | PASS | 8 ms | Dot product sum = 192 |
| `saxpy.cu` | Algorithmic | PASS | PASS | 4 ms | SAXPY a*x + y |
| `histogram256.cu` | Algorithmic | PASS | PASS | 4 ms | 256-bin atomic histogram |
| `conv1d.cu` | Algorithmic | PASS | PASS | 8 ms | 1D convolution filter |
| `relu.cu` | Algorithmic | PASS | PASS | 2 ms | Elementwise ReLU |
| `scalar_mul.cu` | Algorithmic | PASS | PASS | 4 ms | In-place scalar scale |
| `prefix_sum.cu` | Algorithmic | PASS | PASS | 6 ms | Hillis-Steele prefix scan |
| `warp_shuffle_reduce.cu` | Algorithmic | PASS | PASS | 5 ms | 32-thread warp shuffle sum = 32.0 |
| `layernorm_forward.cu` | LLM Kernel | PASS | PASS | 13 ms | llm.c layernorm forward mean=2.0 |
| `gelu_forward.cu` | LLM Kernel | PASS | PASS | 3 ms | llm.c GELU activation |
| `softmax_forward.cu` | LLM Kernel | PASS | PASS | 9 ms | llm.c stable softmax sum=1.0 |
| `residual_add.cu` | LLM Kernel | PASS | PASS | 4 ms | llm.c residual addition 1.5+2.5=4.0 |
| `embedding_forward.cu` | LLM Kernel | PASS | PASS | 3 ms | llm.c token embedding forward lookup |
| `cross_entropy_forward.cu` | LLM Kernel | PASS | PASS | 3 ms | llm.c cross-entropy loss = 0.3567 |
| `dynamic_parallelism.cu` | Unsupported Feature | PASS | PASS (Rejected) | 0 ms | Rejected: Dynamic Parallelism (nested kernel launch) |
| `inline_ptx.cu` | Unsupported Feature | PASS | PASS (Rejected) | 0 ms | Rejected: Inline PTX assembly (__asm__) |
| `wmma_tensor_core.cu` | Unsupported Feature | PASS | PASS (Rejected) | 0 ms | Rejected: Tensor Core WMMA API (nvcuda::wmma) |

## Provenance & Compliance Summary

- Algorithmic kernels sourced from NVIDIA `cuda-samples` (BSD-3-Clause) execute with bit-exact numerical references.
- LLM forward kernels sourced from Andrej Karpathy's `llm.c` (MIT) validate real-world transformer building blocks.
- Unsupported features (nested launches, inline PTX assembly, WMMA tensor cores) are deterministically rejected with explicit diagnostics.
