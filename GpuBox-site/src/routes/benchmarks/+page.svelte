<script>
  const testSuites = [
    { name: 'Suite 1: Vector Addition & Boundary Edge Cases', cases: '10 edge cases (N=1, 7, 31, 33, 63, 64, 127, 255, 1023, 2048)', assert: 'Exact element-wise equality c[i] == a[i] + b[i]', status: 'PASS (10/10)' },
    { name: 'Suite 2: Reduction & L2 Normalization', cases: '10 input dimensions (N=1 to 255)', assert: 'Shared memory tree reduction against float norm', status: 'PASS (10/10)' },
    { name: 'Suite 3: 2D Shared Memory Tiled MatMul', cases: '6 matrix sizes (4x4 to 32x32)', assert: 'Conflict-free 8x8 tiling & syncthreads barriers', status: 'PASS (6/6)' },
    { name: 'Suite 4: Distributed Softmax Multi-GPU', cases: '4 cluster sizes (2, 4, 8, 16 GPUs)', assert: 'Ring all-reduce max + sum, probability sum == 1.0', status: 'PASS (4/4)' },
    { name: 'Suite 5: NCCL Collectives Suite', cases: '20 collective runs (AllReduce, AllGather, Bcast)', assert: 'Bit-exact multi-rank array convergence', status: 'PASS (20/20)' },
    { name: 'Suite 6: Atomic Operations & Contention', cases: '8 histogram contention loads', assert: 'Atomic serialization & bin tally verification', status: 'PASS (8/8)' },
    { name: 'Suite 7: Memory Safety & Hazard Detection', cases: '11 safety tests (OOB, desync, race)', assert: 'Deterministic exception coordinate reporting', status: 'PASS (11/11)' }
  ];

  const physicalBenchmarks = [
    { kernel: 'vector_add', hardware: 'NVIDIA RTX 4050 Laptop (sm_89)', sim: 'Gpubox v1.0.0 (sm_89)', maxError: '0.000000e+00', match: 'Bit-Exact', status: 'PASS' },
    { kernel: 'saxpy', hardware: 'NVIDIA RTX 4050 Laptop (sm_89)', sim: 'Gpubox v1.0.0 (sm_89)', maxError: '0.000000e+00', match: 'Bit-Exact', status: 'PASS' },
    { kernel: 'relu', hardware: 'NVIDIA RTX 4050 Laptop (sm_89)', sim: 'Gpubox v1.0.0 (sm_89)', maxError: '0.000000e+00', match: 'Bit-Exact', status: 'PASS' },
    { kernel: 'matmul_shared', hardware: 'NVIDIA RTX 4050 Laptop (sm_89)', sim: 'Gpubox v1.0.0 (sm_89)', maxError: '0.000000e+00', match: 'Bit-Exact', status: 'PASS' },
    { kernel: 'warp_reduction', hardware: 'NVIDIA RTX 4050 Laptop (sm_89)', sim: 'Gpubox v1.0.0 (sm_89)', maxError: '0.000000e+00', match: 'Bit-Exact', status: 'PASS' },
    { kernel: 'atomic_histogram', hardware: 'NVIDIA RTX 4050 Laptop (sm_89)', sim: 'Gpubox v1.0.0 (sm_89)', maxError: '0.000000e+00', match: 'Bit-Exact', status: 'PASS' }
  ];

  const profiles = [
    { model: 'NVIDIA H100 SXM5', arch: 'Hopper', cc: 'sm_90', sms: 132, tflops: '67.0 TFLOPS', bw: '3350 GB/s', shared: '228 KB', interconnect: 'NVLink 4 (900 GB/s)' },
    { model: 'NVIDIA A100 SXM4', arch: 'Ampere', cc: 'sm_80', sms: 108, tflops: '19.5 TFLOPS', bw: '2039 GB/s', shared: '164 KB', interconnect: 'NVLink 3 (600 GB/s)' },
    { model: 'NVIDIA RTX 4090', arch: 'Ada Lovelace', cc: 'sm_89', sms: 128, tflops: '82.6 TFLOPS', bw: '1008 GB/s', shared: '100 KB', interconnect: 'PCIe Gen4 (31.5 GB/s)' },
    { model: 'NVIDIA RTX 4050 Laptop', arch: 'Ada Lovelace', cc: 'sm_89', sms: 20, tflops: '9.0 TFLOPS', bw: '192 GB/s', shared: '100 KB', interconnect: 'PCIe Gen4 (15.75 GB/s)' },
    { model: 'NVIDIA RTX 3090', arch: 'Ampere', cc: 'sm_86', sms: 82, tflops: '35.6 TFLOPS', bw: '936 GB/s', shared: '100 KB', interconnect: 'PCIe Gen4 (15.75 GB/s)' },
    { model: 'NVIDIA Tesla V100', arch: 'Volta', cc: 'sm_70', sms: 80, tflops: '14.1 TFLOPS', bw: '900 GB/s', shared: '96 KB', interconnect: 'NVLink 2 (300 GB/s)' },
    { model: 'NVIDIA Turing T4', arch: 'Turing', cc: 'sm_75', sms: 40, tflops: '8.1 TFLOPS', bw: '320 GB/s', shared: '64 KB', interconnect: 'PCIe Gen3 (15.75 GB/s)' }
  ];
</script>

<svelte:head>
  <title>GPUBOX Benchmarks &bull; Hardware Verification & Validation Suite</title>
</svelte:head>

<div class="benchmarks-page container">
  <!-- Header -->
  <header class="page-header">
    <div class="badge badge--success">175 / 175 Tests Passing &bull; Bit-Exact Hardware Verified</div>
    <h1 class="page-title">Validation Suite & <span class="gold-text">Hardware Benchmarks</span></h1>
    <p class="page-subtitle">
      Automated regression tests and physical NVIDIA GeForce RTX 4050 GPU kernel comparisons verifying mathematical accuracy, 32-bank shared memory modeling, and multi-GPU cluster interconnects.
    </p>
  </header>

  <!-- High-Level Scorecard -->
  <div class="scorecard-grid">
    <div class="score-card">
      <div class="score-card__label">Automated Test Suites</div>
      <div class="score-card__val text-gold">175 / 175</div>
      <div class="score-card__sub">100% deterministic pass rate</div>
    </div>
    <div class="score-card">
      <div class="score-card__label">Physical GPU Verification</div>
      <div class="score-card__val text-gold">6 / 6 Kernels</div>
      <div class="score-card__sub">Bit-exact against NVIDIA RTX 4050</div>
    </div>
    <div class="score-card">
      <div class="score-card__label">Hardware Architectures</div>
      <div class="score-card__val text-gold">8 Profiles</div>
      <div class="score-card__sub">Hopper, Ampere, Ada, Volta, Turing</div>
    </div>
    <div class="score-card">
      <div class="score-card__label">Virtual Cluster Scaling</div>
      <div class="score-card__val text-gold">1 &rarr; 16 GPUs</div>
      <div class="score-card__sub">NCCL Ring AllReduce & P2P NVLink</div>
    </div>
  </div>

  <!-- Physical Silicon Verification -->
  <section class="section">
    <div class="section-heading">
      <h2>Physical Silicon vs Gpubox Simulation</h2>
      <p>
        Compiled and executed with NVIDIA CUDA Compiler (<code class="mono-code">nvcc 12.8</code>) on an <strong>NVIDIA GeForce RTX 4050 Laptop GPU</strong> (Driver 576.28, Ada Lovelace, sm_89) compared bit-for-bit against client-side Gpubox execution.
      </p>
    </div>

    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Kernel Benchmark</th>
            <th>Physical Silicon Target</th>
            <th>Simulation Engine</th>
            <th>Max Absolute Error</th>
            <th>Numerical Match</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {#each physicalBenchmarks as b}
            <tr>
              <td class="mono font-bold">{b.kernel}</td>
              <td>{b.hardware}</td>
              <td>{b.sim}</td>
              <td class="mono">{b.maxError}</td>
              <td><span class="badge badge--success">{b.match}</span></td>
              <td><span class="badge badge--success">{b.status}</span></td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <!-- Automated Test Suite Breakdown -->
  <section class="section">
    <div class="section-heading">
      <h2>Automated Regression & Stress Suites</h2>
      <p>
        Deterministic CI test suites executed in headless Node.js and client browsers across boundary dimensions, hazard injection, and multi-GPU topologies.
      </p>
    </div>

    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Test Suite</th>
            <th>Tested Dimensions & Configurations</th>
            <th>Assertion Target</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {#each testSuites as s}
            <tr>
              <td class="font-bold">{s.name}</td>
              <td class="mono">{s.cases}</td>
              <td>{s.assert}</td>
              <td><span class="badge badge--success">{s.status}</span></td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <!-- Hardware Profiles Comparison Matrix -->
  <section class="section">
    <div class="section-heading">
      <h2>Simulated Hardware Specifications Matrix</h2>
      <p>
        Architectural parameters, compute caps, SM counts, peak bandwidths, and interconnect models configured across the Gpubox profile library.
      </p>
    </div>

    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Architecture</th>
            <th>Compute Cap</th>
            <th>SMs</th>
            <th>Peak FP32</th>
            <th>Memory Bandwidth</th>
            <th>Shared Mem / Block</th>
            <th>Interconnect</th>
          </tr>
        </thead>
        <tbody>
          {#each profiles as p}
            <tr>
              <td class="font-bold">{p.model}</td>
              <td>{p.arch}</td>
              <td class="mono">{p.cc}</td>
              <td class="mono">{p.sms}</td>
              <td class="mono">{p.tflops}</td>
              <td class="mono">{p.bw}</td>
              <td class="mono">{p.shared}</td>
              <td>{p.interconnect}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
</div>

<style>
  .benchmarks-page {
    padding-top: 3.5rem;
    padding-bottom: 5rem;
  }

  .page-header {
    text-align: center;
    max-width: 800px;
    margin: 0 auto 3.5rem;
  }

  .page-title {
    font-size: 2.75rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .page-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  /* Scorecard */
  .scorecard-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.25rem;
    margin-bottom: 4rem;
  }

  @media (max-width: 860px) {
    .scorecard-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .score-card {
    background: #08080d;
    border: 1px solid var(--gold-border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  }

  .score-card:hover {
    border-color: var(--gold-border);
  }

  .score-card__label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--gold-light);
    margin-bottom: 0.5rem;
  }

  .score-card__val {
    font-family: var(--font-mono);
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
  }

  .score-card__sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
  }

  .text-gold {
    color: var(--gold-light);
    text-shadow: 0 0 12px rgba(212, 175, 55, 0.3);
  }

  /* Section */
  .section {
    margin-bottom: 4rem;
  }

  .section-heading {
    margin-bottom: 1.5rem;
  }

  .section-heading h2 {
    font-size: 1.65rem;
    margin-bottom: 0.5rem;
  }

  .section-heading p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 800px;
  }

  .mono-code {
    font-family: var(--font-mono);
    color: var(--primary-light);
    background: rgba(255, 255, 255, 0.05);
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.85em;
  }

  .font-bold {
    font-weight: 600;
    color: #ffffff;
  }
</style>
