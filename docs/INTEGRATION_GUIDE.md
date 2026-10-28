# Gpubox Integration Guide

A production guide for integrating **Gpubox** into web applications, developer playgrounds, EdTech assessment platforms, and CI/CD pipelines.

---

## Installation & Imports

### Node.js & Bundlers (npm / yarn / pnpm)
```bash
npm install gpubox
```

### Browser Script Tag (CDN / Local Hosting)
```html
<script src="https://cdn.jsdelivr.net/npm/gpubox@latest/dist/cuda-sim.min.js"></script>
<script>
  const { createSimulator, compile, run, validateOutput } = window.Gpubox;
</script>
```

### ES Modules (Vite, Webpack, Next.js, Rollup)
```javascript
import { createSimulator, compile, analyze, run, validateOutput } from 'gpubox';
```

---

## Production Integration Patterns

### 1. Online Coding Playground & IDE Integration

Gpubox can be embedded in browser code editors (such as Monaco Editor or CodeMirror) to run and debug CUDA kernels client-side with zero backend server compute.

```javascript
import { createSimulator, compile, validateOutput } from 'gpubox';

// Initialize 16-rank virtual cluster
const sim = createSimulator({
  gpuProfile: 'A100',
  numDevices: 16,
  topology: 'nvswitch_mesh',
  traceLevel: 'events'
});

// Hook into simulation events to update the UI in real time
sim.on('alloc', (ev) => {
  updateMemoryBar(ev.deviceId, ev.sizeBytes);
});

sim.on('collective', (ev) => {
  renderCollectiveAnimation(ev.op, ev.count);
});

// Run user code from the editor
async function handleRunCode(userCudaSource) {
  try {
    const prog = compile(userCudaSource);
    const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_main;')();
    
    // Allocate device memory and execute
    const d_out = sim.allocate(1024 * 4, 0);
    sim.launch(kernelFn, 4, 256, 0, d_out, 1024);

    // Retrieve and display telemetry
    const counters = sim.getCounters(0).toJSON();
    renderMetricsTable(counters);
  } catch (err) {
    displayErrorDiagnostics(sim.getDiagnostics());
  }
}
```

---

### 2. EdTech Assessment & Autograding Platform

Universities and online platforms can grade student CUDA assignments in the browser without allocating cloud GPUs:

```javascript
const { createSimulator, compile, validateOutput } = require('gpubox');

function gradeStudentKernel(studentCudaCode) {
  const sim = createSimulator({
    gpuProfile: 'A100',
    limits: { maxSteps: 500000, maxTimeMs: 3000 }
  });

  const prog = compile(studentCudaCode);
  const kernelFn = new Function('__rt', prog.transpiledCode + '; return __kernel_solution;')();

  const N = 512;
  const d_in = sim.allocate(N * 4);
  const d_out = sim.allocate(N * 4);
  sim.memcpy(d_in, testInputData, N * 4, 'H2D');

  // Launch student kernel
  sim.launch(kernelFn, 2, 256, 0, d_out, d_in, N);

  // Validate output against reference CPU solution
  const studentResult = new Float32Array(N);
  sim.memcpy(studentResult, d_out, N * 4, 'D2H');

  const check = validateOutput(studentResult, cpuReferenceSolution, { tolerance: 1e-4 });
  const leaks = sim.getLeakReport();

  return {
    passed: check.verdict === 'PASS' && leaks.totalLeakedBytes === 0,
    mismatchCount: check.mismatchCount,
    leakedBytes: leaks.totalLeakedBytes,
    diagnostics: sim.getDiagnostics()
  };
}
```

---

### 3. CI/CD Pre-Commit Gate (Headless CLI)

Catching algorithmic logic and collective synchronization bugs in GitHub Actions before deploying jobs to physical NVIDIA clusters:

```javascript
// test-kernels.ci.js
const { run, validateOutput } = require('gpubox');
const fs = require('fs');

const kernelSource = fs.readFileSync('kernels/attention.cu', 'utf8');

const result = run(kernelSource, {
  gpuProfile: 'A100',
  numDevices: 8,
  validation: {
    expected: referenceAttentionOutput,
    tolerance: 1e-4
  }
});

if (result.status !== 'PASS') {
  console.error('Validation failed: detected logic or collective bugs!');
  console.error(result.diagnostics);
  process.exit(1);
}

console.log('Kernel validated successfully. Safe for cluster deployment.');
```

---

### 4. Web Worker Isolation (Background Execution)

To prevent UI thread stutter during large kernel simulations, use `createWorkerSimulator()`:

```javascript
import { createWorkerSimulator } from 'gpubox';

const sim = createWorkerSimulator({
  gpuProfile: 'H100',
  numDevices: 4,
  limits: { maxSteps: 1000000, maxTimeMs: 5000 }
});

const result = await sim.run(`
  __global__ void matrixProcess(float* out, const float* in, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) out[idx] = in[idx] * 2.0f;
  }
`);

console.log('Simulation complete:', result.status);
```

---

## Typed Error Handling

All simulator exceptions inherit from `SimError`, providing structured error categories, source location, and direct documentation links:

```javascript
const { createSimulator, SimError } = require('gpubox');
const sim = createSimulator();

try {
  sim.allocate(-1024);
} catch (err) {
  if (err instanceof SimError) {
    console.error(`Error Code: ${err.code}`);
    console.error(`Category: ${err.category}`);
    console.error(`Message: ${err.message}`);
    console.error(`Docs: ${err.docsUrl}`);
  }
}
```

---

## Commercial Distribution & EaaS Architecture

Gpubox is designed for commercial deployment as a licensed, client-side simulation engine (Engine-as-a-Service):

- **Zero Cloud Infrastructure Cost**: 100% of the simulation compute executes on the client device (browser or on-premises Node.js process). The platform provider does not pay backend cloud GPU fees.
- **Data Privacy**: Customer CUDA kernels never leave the user machine or corporate VPC.
- **IP Protection**: The C++ core is compiled to binary WebAssembly (`.wasm`), and JavaScript runtimes are minified to protect proprietary logic.
- **Enterprise Offline Licensing**: Supports cryptographic license key verification for corporate seat-based or domain-based agreements without internet phone-homes.

---

Copyright (c) 2026 Aadityansha Verma. All rights reserved. Commercial License.
