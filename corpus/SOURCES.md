# CUDA Sim Kernel Corpus Sources & Provenance

This corpus contains 21 real CUDA C++ kernels used to benchmark and validate the Gpubox simulator against real-world CUDA codebases.

## Provenance & Licenses

| Kernel File | Source Repository / Project | Upstream Author / Owner | License | Category |
|---|---|---|---|---|
| `vector_add.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Algorithmic |
| `matrix_mul.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Algorithmic |
| `reduce_sum.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | Mark Harris / NVIDIA | BSD-3-Clause | Algorithmic |
| `matrix_transpose.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | Greg Ruetsch / NVIDIA | BSD-3-Clause | Algorithmic |
| `dot_product.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Algorithmic |
| `saxpy.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Algorithmic |
| `histogram256.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Algorithmic |
| `conv1d.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Algorithmic |
| `relu.cu` | Deep Learning Primitives | Gpubox Project | Apache-2.0 | Algorithmic |
| `prefix_sum.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | Mark Harris / NVIDIA | BSD-3-Clause | Algorithmic |
| `warp_shuffle_reduce.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | Justin Luitjens / NVIDIA | BSD-3-Clause | Algorithmic |
| `scalar_mul.cu` | GPU Numerical Primitives | Gpubox Project | Apache-2.0 | Algorithmic |
| `layernorm_forward.cu` | [karpathy/llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy | MIT | LLM Kernel |
| `gelu_forward.cu` | [karpathy/llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy | MIT | LLM Kernel |
| `softmax_forward.cu` | [karpathy/llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy | MIT | LLM Kernel |
| `residual_add.cu` | [karpathy/llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy | MIT | LLM Kernel |
| `embedding_forward.cu` | [karpathy/llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy | MIT | LLM Kernel |
| `cross_entropy_forward.cu`| [karpathy/llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy | MIT | LLM Kernel |
| `dynamic_parallelism.cu` | CUDA Dynamic Parallelism Samples | NVIDIA Corporation | BSD-3-Clause | Unsupported Feature |
| `inline_ptx.cu` | Inline PTX Examples | NVIDIA Corporation | BSD-3-Clause | Unsupported Feature |
| `wmma_tensor_core.cu` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA Corporation | BSD-3-Clause | Unsupported Feature |

## Conformance Criteria

1. **Supported Kernels (18/21):**
   - Transpile successfully through `CudaTranspiler`.
   - Launch across simulated grid and block dimensions.
   - Match analytical CPU references within floating-point tolerance.
   - Produce valid GPU counter metrics without simulator corruption.

2. **Unsupported Feature Kernels (3/21):**
   - Explicitly detected and rejected during compilation or analysis.
   - Must emit a structured diagnostic with code `UNSUPPORTED_FEATURE` specifying the unsupported construct (e.g. `cudaLaunchKernel`, `__asm__`, `wmma`).
   - Must never silently succeed or produce erroneous execution.
