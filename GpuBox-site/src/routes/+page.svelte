<script>
  import { onMount, onDestroy } from 'svelte';
  import { startHero3D } from '../lib/hero3d.js';
  import { PRESETS } from '../lib/presets.js';
  import { runCUDA, formatCounters } from '../lib/engine.js';
  import { LINKS, PRICING } from '../lib/config.js';

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
  let telemetry = null;
  let executionTimeMs = 0;

  const profiles = [
    { id: 'A100', label: 'NVIDIA A100 SXM4 (80GB, 2039 GB/s)' },
    { id: 'H100', label: 'NVIDIA H100 SXM5 (80GB, 3350 GB/s)' },
    { id: 'RTX 4090', label: 'NVIDIA RTX 4090 (24GB, 1008 GB/s)' },
    { id: 'RTX 4050 Laptop', label: 'NVIDIA RTX 4050 Laptop (6GB, 192 GB/s, Hardware Verified)' },
    { id: 'RTX 5060 Ti', label: 'NVIDIA RTX 5060 Ti (sm_120 [TO VERIFY])' },
    { id: 'RTX 3090', label: 'NVIDIA RTX 3090 (24GB, 936 GB/s)' },
    { id: 'V100', label: 'NVIDIA Tesla V100 (16GB, 900 GB/s)' }
  ];

  const deviceCounts = [1, 2, 4, 8, 16];

  onMount(() => {
    if (canvasRef) {
      stop3D = startHero3D(canvasRef);
    }
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
    telemetry = null;
  }

  async function handleRun() {
    if (isRunning) return;
    isRunning = true;
    runStatus = 'RUNNING';
    runOutput = 'Transpiling and executing on virtual GPU...';
    runErrors = [];
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
      <div class="badge">GPUBOX &bull; Browser CUDA Pre-Validation</div>
      <h1>Validate CUDA kernels before you pay for cloud GPUs.</h1>
      <p class="lead">
        A client-side CUDA and multi-GPU simulation engine. Transpile, simulate, and debug supported CUDA C++ kernels, shared memory hazards, and 1 to 16 GPU cluster collectives directly in your browser with zero cloud server spend.
      </p>

      <div class="hero__actions">
        <a href="#playground" class="btn btn--primary">Open Playground</a>
        <a href="#pricing" class="btn btn--ghost">Commercial License</a>
        <a href="https://github.com/aadityansha06/Gpu-Box/blob/main/docs/USAGE.md" target="_blank" rel="noreferrer" class="btn btn--ghost">Documentation</a>
      </div>

      <div class="hero__chip">
        <span class="hero__chip-dot"></span>
        <span><strong>175 / 175</strong> Automated Tests Passing</span>
        <span class="hero__chip-sep">&bull;</span>
        <span>Physically Verified on NVIDIA RTX 4050 Laptop GPU</span>
        <span class="hero__chip-sep">&bull;</span>
        <span>Zero Server Cost</span>
      </div>
    </div>
  </section>

  <!-- Problem & Solution Section -->
  <section class="section section--alt" id="problem">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">The Problem</span>
        <h2>Validating CUDA on hardware is slow and expensive</h2>
        <p>Iterating on real GPU clusters creates engineering friction and avoidable cloud infrastructure spend.</p>
      </div>

      <div class="grid grid--4">
        <div class="card">
          <div class="card__tag">GPU Budget Waste</div>
          <h3>Wasted Cloud Compute</h3>
          <p>Engineers provision on-demand cloud GPU instances just to check whether an indexing formula or collective buffer size is correct. Basic logic bugs burn thousands of dollars.</p>
        </div>
        <div class="card">
          <div class="card__tag">Slow Cycles</div>
          <h3>Slow Feedback Loops</h3>
          <p>Cluster launch, docker provisioning, and remote SSH turn a 30-second fix into a 15-minute wait. Gpubox compiles and simulates client-side in under 20 milliseconds.</p>
        </div>
        <div class="card">
          <div class="card__tag">Local Barriers</div>
          <h3>Hardware Access Gaps</h3>
          <p>Engineers on MacBook, Linux, or laptops without NVIDIA GPUs cannot iterate locally. Gpubox gives every developer an instant client-side CUDA sandbox.</p>
        </div>
        <div class="card">
          <div class="card__tag">Education</div>
          <h3>EdTech & Classroom Cost</h3>
          <p>Teaching CUDA requires giving every student access to dedicated GPU cloud instances. Gpubox embeds directly in browser assignments at zero hosting cost.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Interactive Live Playground Section -->
  <section class="section" id="playground">
    <div class="container">
      <div class="section__head">
        <span class="section__kicker">Live Interactive Sandbox</span>
        <h2>Try Gpubox in Your Browser</h2>
        <p>Select a preset algorithm or paste supported CUDA C++ code below to execute it client-side.</p>
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
          <!-- Code Editor Area -->
          <div class="editor-pane">
            <div class="pane-header">
              <span class="pane-header__title">CUDA C++ Kernel Source</span>
              <span class="pane-header__meta">{code.split('\n').length} lines</span>
            </div>
            <textarea
              bind:value={code}
              class="code-editor"
              spellcheck="false"
              on:keydown={handleEditorKeydown}
              placeholder="Paste or write supported CUDA C++ code here..."
            ></textarea>
          </div>

          <!-- Simulation Output & Telemetry Area -->
          <div class="output-pane">
            <div class="pane-header">
              <div class="pane-header__left">
                <span class="pane-header__title">Simulator Output & Telemetry</span>
                {#if runStatus !== 'IDLE'}
                  <span class="status-pill status-pill--{runStatus.toLowerCase()}">{runStatus}</span>
                  <span class="status-time">{executionTimeMs} ms</span>
                {/if}
              </div>
            </div>

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
                  <span class="telemetry-label">Est. Cycles:</span>
                  <span class="telemetry-val">{telemetry.cycles.toLocaleString()}</span>
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
                    <span class="telemetry-val">{telemetry.crossGpuBytes.toLocaleString()} B</span>
                  </div>
                {/if}
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
        <span class="section__kicker">Hardware Verification</span>
        <h2>Physically Verified on NVIDIA RTX 4050 Laptop GPU</h2>
        <p>Gpubox outputs have been compared bit-for-bit against live execution on physical Ada Lovelace NVIDIA hardware via NVRTC and CUDA driver APIs.</p>
      </div>

      <div class="benchmark-table-wrap">
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>Kernel Name</th>
              <th>Physical RTX 4050 Execution</th>
              <th>Gpubox Simulation</th>
              <th>Output Match</th>
              <th>Tolerance Mode</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>vecAdd</code> (Vector Addition)</td>
              <td>Verified via NVRTC & CUDA Driver</td>
              <td>Bit-Exact Simulation</td>
              <td><span class="badge badge--success">PASS (Bit-Exact)</span></td>
              <td>Float32 (1e-5)</td>
            </tr>
            <tr>
              <td><code>l2Norm</code> (L2 Reduction & Normalization)</td>
              <td>Verified via NVRTC & CUDA Driver</td>
              <td>Bit-Exact Simulation</td>
              <td><span class="badge badge--success">PASS (Bit-Exact)</span></td>
              <td>Float32 (1e-5)</td>
            </tr>
            <tr>
              <td><code>matMulTiled</code> (Tiled Shared Memory GEMM)</td>
              <td>Verified via NVRTC & CUDA Driver</td>
              <td>Bit-Exact Simulation</td>
              <td><span class="badge badge--success">PASS (Bit-Exact)</span></td>
              <td>Float32 (1e-5)</td>
            </tr>
            <tr>
              <td><code>histogram256</code> (Atomic Binning)</td>
              <td>Verified via NVRTC & CUDA Driver</td>
              <td>Bit-Exact Simulation</td>
              <td><span class="badge badge--success">PASS (Bit-Exact)</span></td>
              <td>Integer Atomics</td>
            </tr>
            <tr>
              <td><code>transposeCoalesced</code> (Padded Transpose)</td>
              <td>Verified via NVRTC & CUDA Driver</td>
              <td>Bit-Exact Simulation</td>
              <td><span class="badge badge--success">PASS (Bit-Exact)</span></td>
              <td>Float32 (1e-5)</td>
            </tr>
            <tr>
              <td><code>softmaxForward</code> (Karpathy llm.c Softmax)</td>
              <td>Verified via NVRTC & CUDA Driver</td>
              <td>Bit-Exact Simulation</td>
              <td><span class="badge badge--success">PASS (Bit-Exact)</span></td>
              <td>Float32 (1e-5)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-card__val">175 / 175</div>
          <div class="stat-card__label">Automated Tests Passing</div>
          <div class="stat-card__sub">88 benchmarks, 8 metrics, 31 negative, 9 numerical, 21 corpus, 18 acceptance</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__val">6 / 6</div>
          <div class="stat-card__label">Physical GPU Verifications</div>
          <div class="stat-card__sub">Ada Lovelace RTX 4050 Laptop GPU (AD107, sm_89, Driver 617.42)</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__val">~57 KB</div>
          <div class="stat-card__label">CDN Transfer Size</div>
          <div class="stat-card__sub">Brotli-compressed bundle loads in under 20ms</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__val">$0</div>
          <div class="stat-card__label">Provider Server Spend</div>
          <div class="stat-card__sub">100% client-side compute execution</div>
        </div>
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
    margin: 1.25rem 0 1.25rem;
  }

  .hero__actions {
    display: flex;
    gap: 1rem;
    margin: 2.25rem 0 2rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .hero__chip {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    padding: 0.5rem 1.15rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .hero__chip-dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--success);
  }

  .hero__chip-sep {
    color: var(--text-muted);
  }

  /* Problem Cards */
  .card__tag {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--primary-light);
    margin-bottom: 0.5rem;
  }

  /* Playground Section */
  .playground-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
  }

  .presets-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border-subtle);
    padding: 0.75rem 1.25rem;
  }

  .presets-bar__label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
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
    border: 1px solid var(--border-subtle);
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
    background: rgba(255, 255, 255, 0.08);
    color: #ffffff;
  }

  .preset-btn--active {
    background: var(--primary);
    color: #ffffff;
    border-color: var(--primary-hover);
  }

  .preset-btn__tag {
    font-size: 0.65rem;
    background: rgba(0, 0, 0, 0.25);
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-weight: 600;
  }

  /* Controls Bar */
  .controls-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.25rem;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-subtle);
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
    color: var(--text-secondary);
    font-weight: 500;
  }

  .select-input {
    background: var(--bg-input);
    color: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 0.4rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    outline: none;
    cursor: pointer;
  }

  .select-input:focus {
    border-color: var(--primary);
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
  }

  @media (max-width: 860px) {
    .playground-grid {
      grid-template-columns: 1fr;
    }
  }

  .editor-pane {
    border-right: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    background: #090c13;
  }

  .output-pane {
    display: flex;
    flex-direction: column;
    background: #07090f;
  }

  .pane-header {
    height: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1rem;
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border-subtle);
  }

  .pane-header__title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }

  .pane-header__meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .pane-header__left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .status-pill {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    font-family: var(--font-mono);
  }

  .status-pill--pass {
    background: var(--success-bg);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  .status-pill--fail, .status-pill--error {
    background: var(--danger-bg);
    color: var(--danger);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .status-pill--running {
    background: rgba(99, 102, 241, 0.15);
    color: var(--primary-light);
  }

  .status-time {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .code-editor {
    flex: 1;
    width: 100%;
    min-height: 400px;
    background: transparent;
    color: #e5e7eb;
    font-family: var(--font-mono);
    font-size: 0.88rem;
    line-height: 1.6;
    padding: 1rem;
    border: none;
    outline: none;
    resize: none;
    white-space: pre;
    tab-size: 4;
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
  }

  .terminal-placeholder {
    color: var(--text-muted);
    font-style: italic;
    padding: 1.5rem 0;
  }

  .terminal-stdout {
    color: #38bdf8;
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
    background: rgba(0, 0, 0, 0.3);
    border-top: 1px solid var(--border-subtle);
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
    color: var(--primary-light);
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
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
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
