// src/lib/cuda-highlighter.js
// Syntax highlighting for CUDA C++ and JavaScript using Prism.js
import Prism from 'prismjs';
import 'prismjs/components/prism-c.js';
import 'prismjs/components/prism-cpp.js';

// Extend C++ syntax with CUDA qualifiers, primitives, and vector types
Prism.languages.cuda = Prism.languages.extend('cpp', {
  'cuda-qualifier': {
    pattern: /\b(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__forceinline__|__inline__|__noinline__)\b/,
    alias: 'keyword'
  },
  'cuda-builtin': {
    pattern: /\b(?:threadIdx|blockIdx|blockDim|gridDim|warpSize)\b(?:\.[xyz])?/,
    alias: 'variable'
  },
  'cuda-function': {
    pattern: /\b(?:cudaMalloc|cudaFree|cudaMemcpy|cudaMemset|cudaDeviceSynchronize|cudaSetDevice|cudaGetDevice|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|cudaMemcpyDeviceToDevice|__syncthreads|__syncthreads_count|__syncthreads_and|__syncthreads_or|__syncwarp|__shfl_sync|__shfl_up_sync|__shfl_down_sync|__shfl_xor_sync|atomicAdd|atomicSub|atomicExch|atomicMin|atomicMax|atomicCAS|ncclCommInitAll|ncclAllReduce|ncclBcast|ncclAllGather|ncclReduceScatter)\b/,
    alias: 'function'
  },
  'cuda-type': {
    pattern: /\b(?:dim3|cudaError_t|ncclComm_t|ncclResult_t|half|__half|__nv_bfloat16|float[234]|int[234]|uint[234]|double2)\b/,
    alias: 'class-name'
  }
});

export function highlightCUDA(code) {
  if (!code) return '';
  try {
    return Prism.highlight(String(code), Prism.languages.cuda, 'cuda');
  } catch (err) {
    return escapeHtml(code);
  }
}

export function highlightJS(code) {
  if (!code) return '';
  try {
    return Prism.highlight(String(code), Prism.languages.javascript, 'javascript');
  } catch (err) {
    return escapeHtml(code);
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
