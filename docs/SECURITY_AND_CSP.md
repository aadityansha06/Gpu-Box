# Gpubox Security & Content Security Policy (CSP) Guide

This guide details the security model, Content Security Policy (CSP) directives, and safe evaluation patterns for running Gpubox in production applications.

## Content Security Policy Requirements

When integrating Gpubox into browser applications, the following CSP directives apply depending on the execution engine:

### 1. WebAssembly Engine (Analytical Core)

```http
Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; object-src 'none';
```
- The WebAssembly engine compiles and executes pre-built WASM binaries without requiring JavaScript code generation.
- Modern browsers allow WASM compilation with `'wasm-unsafe-eval'` without needing `'unsafe-eval'`.

### 2. JavaScript Transpiler & Dynamic JIT Engine

```http
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-eval'; object-src 'none';
```
- The client-side CUDA C++ transpiler transforms CUDA syntax into JavaScript generator functions and executes them via `new Function('__rt', code)`.
- If your application enforces a strict CSP forbidding `'unsafe-eval'`, you must isolate dynamic CUDA execution into a **Sandboxed Web Worker**.

## Sandboxed Web Worker Isolation (Zero Main-Thread CSP Risk)

To safely execute untrusted student or user CUDA code in enterprise environments:

1. Create a dedicated Web Worker on a separate origin or sandboxed subdomain.
2. Grant `'unsafe-eval'` or `'wasm-unsafe-eval'` exclusively to the worker thread.
3. Keep your main application UI strictly protected under `default-src 'self'`.

```javascript
// main.js
const worker = new Worker('/workers/gpubox-sandbox.js');
worker.postMessage({ type: 'RUN', code: userCode });
worker.onmessage = (e) => {
  renderResults(e.data.result);
};
```

## Watchdog & Resource Limits

To prevent runaway user loops or denial-of-service from infinite loops:
- Gpubox enforces an internal step watchdog (default: 50,000 steps).
- It checks time limits (default: 2,000 ms).
- It enforces strict virtual VRAM capacity checks (e.g. 80 GiB per device) without allocating real host physical memory.
