<script>
  import { onMount, onDestroy } from 'svelte';
  import { startHero3D } from '../lib/hero3d.js';
  import { PRESETS } from '../lib/presets.js';
  import { runCUDA, formatCounters, formatBytes } from '../lib/engine.js';
  import { LINKS, PRICING } from '../lib/config.js';
  import MemoryBankGrid from '../lib/MemoryBankGrid.svelte';
  import CodeEditor from '../lib/CodeEditor.svelte';
  import EnterpriseDiagrams from '../lib/EnterpriseDiagrams.svelte';
  import MetricsPanel from '../lib/MetricsPanel.svelte';
  import parity from '../lib/parity.json';

  let canvasRef;
  let stop3D;

  // Playground state
  let selectedPresetId = PRESETS[0].id;
  let selectedProfile = 'A100';
  let selectedDevices = 1;
  let code = PRESETS[0].code;

  let isRunning = false;
  let runStatus = 'IDLE'; // IDLE | RUNNING | PASS | FAIL | ERROR
  let runOutput = '';
  let runErrors = [];
  let rawCounters = null;
  let telemetry = null;
  let executionTimeMs = 0;
  let activeTab = 'report';
  let profileSpec = null;

  let pkgTab = 'npm';
  const installCommands = {
    npm: 'npm install gpubox',
    pnpm: 'pnpm add gpubox',
    cdn: '<script src="https://cdn.jsdelivr.net/npm/gpubox@1.0.0/dist/gpubox.min.js"><' + '/script>'
  };
  let copiedInstall = false;

  function copyInstallCmd() {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(installCommands[pkgTab]);
      copiedInstall = true;
      setTimeout(() => copiedInstall = false, 2000);
    }
  }

  const profiles = [
    { id: 'A100', label: 'NVIDIA A100 SXM4 (80GB, 2039 GB/s)' },
    { id: 'H100', label: 'NVIDIA H100 SXM5 (80GB, 3350 GB/s)' },
    { id: 'RTX 4090', label: 'NVIDIA RTX 4090 (24GB, 1008 GB/s)' },
    { id: 'RTX 4050 Laptop', label: 'NVIDIA RTX 4050 Laptop (6GB, 192 GB/s, Hardware Verified)' },
    { id: 'RTX 5060 Ti', label: 'NVIDIA RTX 5060 Ti (16GB, projected specs)' },
    { id: 'RTX 3090', label: 'NVIDIA RTX 3090 (24GB, 936 GB/s)' },
    { id: 'V100', label: 'NVIDIA Tesla V100 (16GB, 900 GB/s)' }
  ];

  const deviceCounts = [1, 2, 4, 8, 16];

  onMount(() => {
    if (canvasRef) {
      stop3D = startHero3D(canvasRef);
    }
    // Run the default example once so the report is alive when the visitor scrolls to it.
    const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 600));
    idle(() => handleRun());
  });

  onDestroy(() => {
    if (stop3D) stop3D();
  });

  function selectPreset(preset) {
    selectedPresetId = preset.id;
    code = preset.code;
    selectedDevices = preset.devices || 1;
    runStatus = 'IDLE';
    runOutput = '';
    runErrors = [];
    rawCounters = null;
    telemetry = null;
  }

  async function handleRun() {
    if (isRunning) return;
    isRunning = true;
    runStatus = 'RUNNING';
    runOutput = 'Transpiling and executing on virtual GPU...';
    runErrors = [];
    rawCounters = null;
    telemetry = null;

    const tStart = performance.now();
    try {
      const res = await runCUDA({
        code,
        profile: selectedProfile,
        devices: selectedDevices
      });

      executionTimeMs = Math.round(performance.now() - tStart);
      runOutput = res.output || (res.ok ? 'Kernel execution completed successfully (no console output).' : '');
      runErrors = res.errors || [];
      rawCounters = res.counters;
      profileSpec = res.profileSpec;
      telemetry = formatCounters(res.counters);

      if (res.ok && runErrors.length === 0) {
        runStatus = 'PASS';
      } else {
        runStatus = 'FAIL';
      }
    } catch (err) {
      executionTimeMs = Math.round(performance.now() - tStart);
      runStatus = 'ERROR';
      runOutput = '';
      runErrors = [err.message || String(err)];
    } finally {
      isRunning = false;
    }
  }

  function handleEditorKeydown(e) {
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.target;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      code = code.substring(0, start) + '    ' + code.substring(end);
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 4;
      }, 0);
    }
  }
</script>

<svelte:head>
  <title>GPUBOX &bull; Browser-Based CUDA Simulation & Pre-Validation Engine</title>
  <meta name="description" content="Validate supported CUDA C++ algorithms and 1 to 16 virtual GPU clusters directly in the browser. Zero cloud GPU spend. Bit-exact verification against real NVIDIA hardware." />
</svelte:head>

<div class="page">
  <!-- Hero Section -->
  <section class="hero">
    <div class="hero__canvas-wrap">
      <canvas bind:this={canvasRef} class="hero__canvas"></canvas>
      <div class="hero__overlay"></div>
    </div>

    <div class="container hero__content">
      <div class="badge">CUDA pre-flight for AI teams</div>
      <h1>
        Catch the CUDA bug
        <span class="gold-text">before it costs a GPU hour.</span>
      </h1>
      <p class="lead">
        Run real CUDA C++ and multi-GPU NCCL code in your browser. Gpubox tells you what the hardware would do: the bytes moved at every level of memory, the FLOPs executed, the bank conflicts, the divergence, the deadlock. Nothing leaves your machine, and results match a physical RTX 4050.
      </p>

      <div class="hero__actions">
        <a href="/playground" class="btn btn--primary">Launch Playground &rarr;</a>
        <a href="/docs" class="btn btn--secondary">Technical Documentation</a>
      </div>

      <div class="hero__install-box">
        <div class="install-tabs">
          <button class="install-tab {pkgTab === 'npm' ? 'install-tab--active' : ''}" on:click={() => pkgTab = 'npm'}>npm</button>
          <button class="install-tab {pkgTab === 'pnpm' ? 'install-tab--active' : ''}" on:click={() => pkgTab = 'pnpm'}>pnpm</button>
          <button class="install-tab {pkgTab === 'cdn' ? 'install-tab--active' : ''}" on:click={() => pkgTab = 'cdn'}>cdn</button>
        </div>
        <div class="install-code-row">
          <code class="install-code">{installCommands[pkgTab]}</code>
          <button class="copy-install-btn" on:click={copyInstallCmd}>
            {copiedInstall ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      <div class="hero__chip">
        <span class="hero__chip-dot"></span>
        <span><strong>11 / 11</strong> kernels match a physical RTX 4050</span>
        <span class="hero__chip-sep">&bull;</span>
        <span><strong>205</strong> automated checks</span>
        <span class="hero__chip-sep">&bull;</span>
        <span>Zero data egress</span>
      </div>
    </div>
  </section>

  <!-- What every run reports -->
  <section class="section" id="report">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">What you get from every run</span>
        <h2>Not just "it ran". Exactly what the GPU would have done.</h2>
        <p>Every execution produces a profiler-style report you can act on before you ever touch hardware.</p>
      </div>
      <div class="grid grid--4 report-tiles">
        <div class="card"><div class="tile-num">01</div><h3>Data movement</h3><p>Host to device, global reads and writes, shared memory, and GPU to GPU interconnect bytes, each counted separately and charged per 32-byte DRAM sector.</p></div>
        <div class="card"><div class="tile-num">02</div><h3>Compute executed</h3><p>FP32 operations and transcendental calls counted per thread, with arithmetic intensity and a memory-bound or compute-bound verdict for the GPU you pick.</p></div>
        <div class="card"><div class="tile-num">03</div><h3>Hazards, located</h3><p>Out-of-bounds access, shared-memory races, divergent barriers, bank conflicts, uncoalesced loads, each with kernel, block and thread coordinates.</p></div>
        <div class="card"><div class="tile-num">04</div><h3>Cluster behaviour</h3><p>1 to 16 virtual GPUs with ring collectives, per-rank traffic, interconnect time, and a deadlock report when a rank skips a collective.</p></div>
      </div>
    </div>
  </section>

  <!-- Enterprise Architectural Showcase Section with Handcrafted SVGs -->
  <section class="section">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Virtual Silicon Architecture</span>
        <h2>Deterministic SIMT Pipeline & Cluster Topography</h2>
        <p>A zero-server WebAssembly and SIMT execution environment engineered to reproduce real GPU memory hierarchies and interconnect topologies.</p>
      </div>

      <!-- Main Micro-Architecture Pipeline SVG -->
      <div class="architecture-diagram-card">
        <EnterpriseDiagrams diagram="architecture" />
      </div>

      <!-- Two-Column Architecture Showcase with SVGs -->
      <div class="grid grid--2" style="margin-top: 2rem;">
        <div class="card card--diagram">
          <div class="card__head-tag">Cluster Fabric</div>
          <h3>Multi-GPU Ring Collective Topography</h3>
          <p>Models NVLink 600 GB/s bidirectional ring throughput and peer-to-peer memory transactions across 1 to 16 virtual GPU nodes with NCCL AllReduce, Broadcast, and AllGather.</p>
          <div class="diagram-mini-wrap">
            <EnterpriseDiagrams diagram="cluster" />
          </div>
        </div>

        <div class="card card--diagram">
          <div class="card__head-tag">Enterprise Governance</div>
          <h3>100% Zero-Data-Egress Sandbox</h3>
          <p>Proprietary kernel code, intellectual property, and custom quantization algorithms run exclusively within the local developer environment. Zero external server calls or cloud telemetry.</p>
          <div class="diagram-mini-wrap">
            <EnterpriseDiagrams diagram="security" />
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Interactive Live Playground Section -->
  <section class="section" id="playground">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Live Interactive Sandbox</span>
        <h2>Run it. Break it. Read the report.</h2>
        <p>Seventeen working examples, from SAXPY to an 8-GPU gradient AllReduce, plus deliberate bugs that Gpubox catches. Edit anything, or paste your own kernel.</p>
      </div>

      <div class="playground-card">
        <!-- Presets Toolbar -->
        <div class="presets-bar">
          <span class="presets-bar__label">Presets:</span>
          <div class="presets-bar__scroll">
            {#each PRESETS as p}
              <button
                class="preset-btn {selectedPresetId === p.id ? 'preset-btn--active' : ''}"
                on:click={() => selectPreset(p)}
              >
                {p.title}
                {#if p.tag}
                  <span class="preset-btn__tag">{p.tag}</span>
                {/if}
              </button>
            {/each}
          </div>
        </div>

        <!-- Controls Toolbar -->
        <div class="controls-bar">
          <div class="controls-bar__left">
            <div class="control-group">
              <label for="profileSelect">Hardware Profile:</label>
              <select id="profileSelect" bind:value={selectedProfile} class="select-input">
                {#each profiles as prof}
                  <option value={prof.id}>{prof.label}</option>
                {/each}
              </select>
            </div>

            <div class="control-group">
              <label for="deviceSelect">Virtual Cluster:</label>
              <select id="deviceSelect" bind:value={selectedDevices} class="select-input select-input--sm">
                {#each deviceCounts as count}
                  <option value={count}>{count} {count === 1 ? 'GPU' : 'GPUs'}</option>
                {/each}
              </select>
            </div>
          </div>

          <button
            class="btn btn--primary btn--run {isRunning ? 'btn--running' : ''}"
            on:click={handleRun}
            disabled={isRunning}
          >
            {#if isRunning}
              <span class="spinner"></span> Running Simulation...
            {:else}
              Run Simulation
            {/if}
          </button>
        </div>

        <!-- Playground Grid: Editor & Output -->
        <div class="playground-grid">
          <!-- Code Editor Area with Real-Time Syntax Highlighting -->
          <div class="editor-pane">
            <div class="pane-header">
              <span class="pane-header__title">CUDA C++ Kernel Source</span>
              <span class="pane-header__meta">{code.split('\n').length} lines &bull; Syntax Highlighted</span>
            </div>
            <div class="editor-body">
              <CodeEditor
                bind:code
                language="cuda"
                minHeight="420px"
                placeholder="// Paste or write supported CUDA C++ code here..."
                on:run={handleRun}
              />
            </div>
          </div>

          <!-- Simulation Output & Telemetry Area -->
          <div class="output-pane">
            <div class="pane-header">
              <div class="pane-header__tabs">
                <button
                  class="pane-tab {activeTab === 'report' ? 'pane-tab--active' : ''}"
                  on:click={() => activeTab = 'report'}
                >
                  GPU Report
                </button>
                <button
                  class="pane-tab {activeTab === 'terminal' ? 'pane-tab--active' : ''}"
                  on:click={() => activeTab = 'terminal'}
                >
                  Console
                  {#if runStatus === 'PASS'}
                    <span class="tab-dot tab-dot--pass"></span>
                  {:else if runStatus === 'FAIL' || runStatus === 'ERROR'}
                    <span class="tab-dot tab-dot--fail"></span>
                  {/if}
                </button>
                <button
                  class="pane-tab {activeTab === 'banks' ? 'pane-tab--active' : ''}"
                  on:click={() => activeTab = 'banks'}
                >
                  32-Bank Memory Grid
                  {#if rawCounters?.bankConflicts > 0}
                    <span class="tab-tag-warn">!{rawCounters.bankConflicts}</span>
                  {/if}
                </button>
              </div>

              <div class="pane-header__right">
                <a href="/playground" class="fullscreen-link" title="Open Fullscreen IDE">
                  Fullscreen IDE &rarr;
                </a>
              </div>
            </div>

            {#if activeTab === 'report'}
              <div class="report-wrap">
                <MetricsPanel counters={rawCounters} spec={profileSpec} devices={selectedDevices} profileName={selectedProfile} />
              </div>
            {:else if activeTab === 'terminal'}
              <!-- Terminal output -->
              <div class="terminal-area">
                {#if runErrors.length > 0}
                  <div class="terminal-errors">
                    <div class="terminal-errors__title">Diagnostics & Errors Caught:</div>
                    {#each runErrors as err}
                      <div class="terminal-error-item">{err}</div>
                    {/each}
                  </div>
                {/if}

                {#if runOutput}
                  <pre class="terminal-stdout">{runOutput}</pre>
                {:else if runErrors.length === 0}
                  <div class="terminal-placeholder">
                    Click "Run Simulation" to compile and execute kernel.
                  </div>
                {/if}
              </div>

              <!-- Telemetry Metrics -->
              {#if telemetry}
                <div class="telemetry-bar">
                  <div class="telemetry-item">
                    <span class="telemetry-label">Threads:</span>
                    <span class="telemetry-val">{telemetry.threads.toLocaleString()}</span>
                  </div>
                  <div class="telemetry-item">
                    <span class="telemetry-label">Est. Time:</span>
                    <span class="telemetry-val">{telemetry.timeMs}</span>
                  </div>
                  <div class="telemetry-item">
                    <span class="telemetry-label">Global Loads:</span>
                    <span class="telemetry-val">{telemetry.globalLoads.toLocaleString()}</span>
                  </div>
                  <div class="telemetry-item">
                    <span class="telemetry-label">Global Stores:</span>
                    <span class="telemetry-val">{telemetry.globalStores.toLocaleString()}</span>
                  </div>
                  <div class="telemetry-item">
                    <span class="telemetry-label">Bank Conflicts:</span>
                    <span class="telemetry-val">{telemetry.bankConflicts}</span>
                  </div>
                  {#if selectedDevices > 1}
                    <div class="telemetry-item">
                      <span class="telemetry-label">Cluster Traffic:</span>
                      <span class="telemetry-val">{telemetry.crossGpuBytes}</span>
                    </div>
                  {/if}
                </div>
              {/if}
            {:else if activeTab === 'banks'}
              <div class="embedded-bank-wrap">
                <MemoryBankGrid counters={rawCounters} />
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Features Section -->
  <section class="section section--alt" id="features">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Core Capabilities</span>
        <h2>What Gpubox Simulates and Detects</h2>
        <p>Built from the ground up for reliable algorithmic pre-validation.</p>
      </div>

      <div class="grid grid--3">
        <div class="card">
          <h3>Client-Side SIMT Execution</h3>
          <p>AST transpiler translates supported CUDA kernels to ES6 generator coroutines. Executes with true SIMT semantics, warp lockstep scheduling, and branch divergence tracking.</p>
        </div>
        <div class="card">
          <h3>Multi-GPU NCCL Collectives</h3>
          <p>Simulates 1 to 16 virtual GPU nodes with isolated rank memory spaces. Native ring collectives: AllReduce (SUM/MAX/MIN/PROD/AVG), Broadcast, ReduceScatter, AllGather, and AllToAll.</p>
        </div>
        <div class="card">
          <h3>Shared Memory Race Detection</h3>
          <p>Barrier-epoch hazard detector monitors shared memory access patterns across threads, flagging Read-After-Write (RAW), Write-After-Read (WAR), and Write-After-Write (WAW) races.</p>
        </div>
        <div class="card">
          <h3>Collective Desync & Deadlock Guard</h3>
          <p>Real-time rank sequence tracking flags communicator mismatches, missing ranks, count discrepancies, and collective ordering bugs before they hang real hardware.</p>
        </div>
        <div class="card">
          <h3>Virtual 80 GiB Sparse VRAM</h3>
          <p>64 KB paged virtual VRAM model supports up to 80 GiB allocations per GPU (1.28 TiB cluster total) while consuming less than 256 MB of actual host heap memory.</p>
        </div>
        <div class="card">
          <h3>Runaway Watchdog & Leak Reports</h3>
          <p>Watchdog automatically aborts infinite loops (<code>while(true)&#123;&#125;</code>) and pinpoints the offending kernel, block, and thread coordinates. <code>getLeakReport()</code> tracks unfreed allocations.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Benchmarks & Hardware Correlation Section -->
  <section class="section" id="benchmarks">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Hardware verification</span>
        <h2>Same kernels, same inputs, run on a physical RTX 4050</h2>
        <p>Every row is the same CUDA source compiled with NVRTC and executed on a real RTX 4050 Laptop GPU (sm_89), then compared element by element with the simulator's output. Error is relative; the small residue is JavaScript double intermediates versus GPU single precision.</p>
      </div>

      <div class="benchmark-table-wrap">
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>Kernel</th>
              <th>Max relative error</th>
              <th>FLOPs counted</th>
              <th>Global bytes</th>
              <th>Bank conflicts</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {#each parity.results as r}
              <tr>
                <td><code>{r.name}</code></td>
                <td class="mono">{r.maxRelErr === 0 ? 'exact' : r.maxRelErr.toExponential(1)}</td>
                <td class="mono">{r.simFlops.toLocaleString()}</td>
                <td class="mono">{r.simGlobalBytes.toLocaleString()}</td>
                <td class="mono">{r.simBankConflicts}</td>
                <td><span class="badge badge--success">{r.status === 'PASS' ? 'MATCH' : r.status}</span></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <p class="fine-print">Reproduce on your own NVIDIA GPU: <code>node scripts/gpu-parity.js</code>. Modeled kernel times are roofline estimates, typically within 2 to 3x on micro-kernels and tighter on bandwidth-bound ones.</p>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-card__val">205</div>
          <div class="stat-card__label">Automated checks passing</div>
          <div class="stat-card__sub">175 regression + 30 engine stress tests (reductions, softmax, matmul, NCCL, compile errors)</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__val">{parity.passed} / {parity.kernelsTested}</div>
          <div class="stat-card__label">Kernels matching physical GPU</div>
          <div class="stat-card__sub">Ada Lovelace RTX 4050 Laptop GPU (AD107, sm_89)</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__val">16</div>
          <div class="stat-card__label">Virtual GPUs, ring collectives</div>
          <div class="stat-card__sub">AllReduce, AllGather, ReduceScatter, AllToAll, P2P with deadlock detection</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__val">$0</div>
          <div class="stat-card__label">Provider Server Spend</div>
          <div class="stat-card__sub">100% client-side compute execution</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Honest boundaries -->
  <section class="section" id="limits">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Know the boundaries</span>
        <h2>What Gpubox is not</h2>
        <p>A pre-flight check is only useful if you know where it stops. These are the limits, stated plainly.</p>
      </div>
      <div class="grid grid--3 limits-grid">
        <div class="card"><h3>Not a replacement for the real GPU</h3><p>Gpubox filters logic, memory and collective bugs early. Final performance tuning, tensor cores and driver behaviour still need silicon.</p></div>
        <div class="card"><h3>Rejected, never faked</h3><p>Inline PTX, WMMA tensor cores, dynamic parallelism and CUDA Graphs are refused at compile time with a clear error instead of silently mis-simulated.</p></div>
        <div class="card"><h3>Numerics and timing</h3><p>Arithmetic runs in JavaScript doubles and is stored as FP32, so results differ from a GPU at the 1e-7 level. Integers in device memory are exact up to 2<sup>24</sup>. Time is a roofline model, not a cycle-accurate one.</p></div>
      </div>
    </div>
  </section>

  <!-- Pricing Section -->
  <section class="section section--alt" id="pricing">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Licensing Plans</span>
        <h2>Commercial Engine-as-a-Service (EaaS) Licensing</h2>
        <p>Distribute Gpubox inside your platform, LMS, or internal engineering workflows with zero server compute bills.</p>
      </div>

      <div class="grid grid--3">
        <!-- Education / Academic Plan -->
        <div class="card pricing-card">
          <div class="pricing-card__head">
            <span class="badge">Education Tier</span>
            <h3>Academic & Bootcamps</h3>
            <div class="pricing-card__price">
              <span class="pricing-card__num">${PRICING.education.price}</span>
              <span class="pricing-card__period">/{PRICING.education.period}</span>
            </div>
            <p>For universities, coding bootcamps, and online courses teaching CUDA programming.</p>
          </div>
          <ul class="pricing-card__features">
            <li>Unlimited student and learner seats</li>
            <li>Zero cloud GPU hosting bills</li>
            <li>Client-side browser playground integration</li>
            <li>Automated assignment autograding runner</li>
            <li>All 7 GPU hardware profiles included</li>
            <li>Full documentation and recipe access</li>
          </ul>
          <div class="pricing-card__footer">
            <a href={LINKS.buyEducation} class="btn btn--ghost btn--full">Acquire Education License</a>
          </div>
        </div>

        <!-- Commercial Developer Plan -->
        <div class="card pricing-card pricing-card--featured">
          <div class="pricing-card__head">
            <span class="badge badge--success">Most Popular</span>
            <h3>Commercial Team</h3>
            <div class="pricing-card__price">
              <span class="pricing-card__num">${PRICING.commercial.price}</span>
              <span class="pricing-card__period">/{PRICING.commercial.period}</span>
            </div>
            <p>For AI startups, developer platforms, and engineering teams validating CUDA kernels.</p>
          </div>
          <ul class="pricing-card__features">
            <li>Commercial deployment rights (EULA)</li>
            <li>Full minified JS + WebAssembly bundle</li>
            <li>Offline cryptographic license key</li>
            <li>1 to 16 Virtual GPU cluster simulation</li>
            <li>Shared memory race & desync detector</li>
            <li>Pre-commit CI/CD automated validation</li>
            <li>Technical support & version updates</li>
          </ul>
          <div class="pricing-card__footer">
            <a href={LINKS.buyCommercial} class="btn btn--primary btn--full">Acquire Commercial License</a>
          </div>
        </div>

        <!-- Enterprise Custom Plan -->
        <div class="card pricing-card">
          <div class="pricing-card__head">
            <span class="badge">Enterprise</span>
            <h3>Custom Enterprise</h3>
            <div class="pricing-card__price">
              <span class="pricing-card__num">Custom</span>
            </div>
            <p>For large enterprise AI engineering teams, custom ASIC modeling, and on-premises VPCs.</p>
          </div>
          <ul class="pricing-card__features">
            <li>Air-gapped on-premises VPC deployment</li>
            <li>Custom GPU / Accelerator profile modeling</li>
            <li>Private scoped npm registry distribution</li>
            <li>Dedicated SLA & priority engineering support</li>
            <li>Custom topology interconnect modeling</li>
            <li>Direct engineer-to-engineer onboarding</li>
          </ul>
          <div class="pricing-card__footer">
            <a href={LINKS.contact} class="btn btn--ghost btn--full">Contact Enterprise Sales</a>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  .page {
    position: relative;
    overflow-x: hidden;
  }

  /* Hero Section */
  .hero {
    position: relative;
    min-height: 82vh;
    display: flex;
    align-items: center;
    padding: 6rem 0 4rem;
  }

  .hero__canvas-wrap {
    position: absolute;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
  }

  .hero__canvas {
    width: 100%;
    height: 100%;
    display: block;
    opacity: 0.65;
  }

  .hero__overlay {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 50%, rgba(7, 8, 12, 0.4) 0%, rgba(7, 8, 12, 0.95) 85%);
  }

  .hero__content {
    position: relative;
    z-index: 1;
    max-width: 820px;
    margin: 0 auto;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .hero__content h1 {
    font-family: var(--font-hero);
    font-size: clamp(2.4rem, 5.2vw, 4.2rem);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.025em;
    margin: 1.5rem 0 1.25rem;
  }

  .hero__actions {
    display: flex;
    gap: 1rem;
    margin: 2rem 0 1.5rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  /* Install Command Box */
  .hero__install-box {
    background: #000000;
    border: 1px solid var(--border-gold);
    border-radius: var(--radius-md);
    margin-bottom: 2rem;
    overflow: hidden;
    max-width: 540px;
    width: 100%;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
  }

  .install-tabs {
    display: flex;
    background: #000000;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
  }

  .install-tab {
    padding: 0.35rem 0.85rem;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--text-muted);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
  }

  .install-tab--active {
    color: var(--gold-light);
    border-bottom-color: var(--gold-primary);
    background: rgba(212, 175, 55, 0.08);
  }

  .install-code-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 1rem;
    gap: 1rem;
    background: #000000;
  }

  .install-code {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--gold-light);
    overflow-x: auto;
    white-space: nowrap;
  }

  .copy-install-btn {
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.25);
    color: var(--gold-light);
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: all 0.15s ease;
    flex-shrink: 0;
  }

  .copy-install-btn:hover {
    color: #ffffff;
    background: rgba(212, 175, 55, 0.2);
  }

  .hero__chip {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.2);
    padding: 0.5rem 1.15rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .hero__chip-dot {
    width: 8px;
    height: 8px;
    background: var(--gold-primary);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--gold-primary);
  }

  .hero__chip-sep {
    color: var(--gold-muted);
  }

  /* Architectural Diagrams Cards */
  .architecture-diagram-card {
    background: #000000;
    border: 1px solid var(--gold-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
  }

  .card--diagram {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.18);
  }

  .diagram-mini-wrap {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(212, 175, 55, 0.12);
  }

  .card__head-tag {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--gold-light);
    margin-bottom: 0.5rem;
  }

  /* Playground Section */
  .playground-card {
    background: #000000;
    border: 1px solid var(--border-gold);
    border-radius: var(--radius-xl);
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.85);
  }

  .presets-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #000000;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    padding: 0.75rem 1.25rem;
  }

  .presets-bar__label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--gold-muted);
    flex-shrink: 0;
  }

  .presets-bar__scroll {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding-bottom: 2px;
  }

  .preset-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(212, 175, 55, 0.15);
    color: var(--text-secondary);
    padding: 0.35rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
  }

  .preset-btn:hover {
    background: rgba(212, 175, 55, 0.1);
    color: var(--gold-light);
    border-color: var(--gold-primary);
  }

  .preset-btn--active {
    background: var(--gold-surface);
    color: var(--gold-light);
    border-color: var(--gold-primary);
  }

  .preset-btn__tag {
    font-size: 0.65rem;
    background: rgba(0, 0, 0, 0.5);
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-weight: 600;
    color: var(--gold-light);
  }

  /* Controls Bar */
  .controls-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.25rem;
    background: #000000;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    flex-wrap: wrap;
    gap: 1rem;
  }

  .controls-bar__left {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    flex-wrap: wrap;
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .control-group label {
    font-size: 0.82rem;
    color: var(--gold-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
  }

  .select-input {
    background: #000000;
    color: #ffffff;
    border: 1px solid rgba(212, 175, 55, 0.25);
    padding: 0.4rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    outline: none;
    cursor: pointer;
  }

  .select-input:focus {
    border-color: var(--gold-primary);
  }

  .select-input--sm {
    padding-right: 1.5rem;
  }

  .btn--run {
    min-width: 150px;
  }

  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #ffffff;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Playground Grid */
  .playground-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    min-height: 460px;
    background: #000000;
  }

  @media (max-width: 860px) {
    .playground-grid {
      grid-template-columns: 1fr;
    }
  }

  .editor-pane {
    border-right: 1px solid rgba(212, 175, 55, 0.18);
    display: flex;
    flex-direction: column;
    background: #000000;
  }

  .output-pane {
    display: flex;
    flex-direction: column;
    background: #000000;
  }

  .pane-header {
    height: 2.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0.75rem;
    background: #000000;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
  }

  .pane-header__tabs {
    display: flex;
    align-items: center;
    height: 100%;
    gap: 0.25rem;
  }

  .pane-tab {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0 0.75rem;
    height: 100%;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    transition: all 0.15s ease;
  }

  .pane-tab:hover {
    color: #ffffff;
  }

  .pane-tab--active {
    color: var(--gold-light);
    border-bottom-color: var(--gold-primary);
    background: rgba(212, 175, 55, 0.08);
  }

  .tab-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }
  .tab-dot--pass { background: var(--success); }
  .tab-dot--fail { background: var(--danger); }

  .tab-tag-warn {
    font-size: 0.65rem;
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    background: var(--warning-bg);
    color: var(--warning);
    font-family: var(--font-mono);
  }

  .fullscreen-link {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gold-light);
    transition: color 0.15s ease;
  }

  .fullscreen-link:hover {
    color: #ffffff;
  }

  .embedded-bank-wrap {
    padding: 1rem;
    overflow-y: auto;
    max-height: 440px;
    background: #000000;
  }

  .pane-header__title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--gold-light);
  }

  .pane-header__meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  /* Terminal Area */
  .terminal-area {
    flex: 1;
    padding: 1rem;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    line-height: 1.5;
    overflow-y: auto;
    max-height: 400px;
    background: #000000;
  }

  .terminal-placeholder {
    color: var(--text-muted);
    font-style: italic;
    padding: 1.5rem 0;
  }

  .terminal-stdout {
    color: #fce59f;
    white-space: pre-wrap;
  }

  .terminal-errors {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: var(--radius-sm);
    padding: 0.75rem;
    margin-bottom: 0.75rem;
  }

  .terminal-errors__title {
    color: #f87171;
    font-weight: 700;
    margin-bottom: 0.4rem;
  }

  .terminal-error-item {
    color: #fca5a5;
    font-size: 0.82rem;
  }

  /* Telemetry Bar */
  .telemetry-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: #000000;
    border-top: 1px solid rgba(212, 175, 55, 0.15);
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }

  .telemetry-item {
    display: flex;
    gap: 0.35rem;
  }

  .telemetry-label {
    color: var(--text-muted);
  }

  .telemetry-val {
    color: var(--gold-light);
    font-weight: 600;
  }

  /* Benchmarks Table */
  .benchmark-table-wrap {
    overflow-x: auto;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: 2.5rem;
  }

  .benchmark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    text-align: left;
  }

  .benchmark-table th {
    background: rgba(255, 255, 255, 0.03);
    padding: 0.85rem 1.25rem;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .benchmark-table td {
    padding: 0.85rem 1.25rem;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .benchmark-table tr:last-child td {
    border-bottom: none;
  }

  .benchmark-table tr:hover td {
    background: rgba(255, 255, 255, 0.015);
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.5rem;
  }

  @media (max-width: 860px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    text-align: center;
  }

  .stat-card__val {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    font-family: var(--font-mono);
    margin-bottom: 0.35rem;
  }

  .stat-card__label {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
  }

  .stat-card__sub {
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  /* Pricing Cards */
  .pricing-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .pricing-card--featured {
    border-color: var(--gold-border);
    background: linear-gradient(180deg, rgba(212, 175, 55, 0.08) 0%, #0d0d14 100%);
    box-shadow: 0 0 40px rgba(212, 175, 55, 0.2);
  }

  .pricing-card__head {
    margin-bottom: 1.5rem;
  }

  .pricing-card__head h3 {
    margin: 0.75rem 0 0.5rem;
  }

  .pricing-card__price {
    display: flex;
    align-items: baseline;
    margin: 1rem 0 0.75rem;
  }

  .pricing-card__num {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    font-family: var(--font-mono);
  }

  .pricing-card__period {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-left: 0.25rem;
  }

  .pricing-card__features {
    list-style: none;
    margin-bottom: 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .pricing-card__features li {
    font-size: 0.88rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .pricing-card__features li::before {
    content: "\2022";
    color: var(--primary-light);
    font-weight: bold;
    font-size: 1.2rem;
  }

  .btn--full {
    width: 100%;
  }
</style>
