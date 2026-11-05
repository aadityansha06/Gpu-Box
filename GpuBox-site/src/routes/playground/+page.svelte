<script>
  import { PRESETS } from '../../lib/presets.js';
  import { runCUDA, formatCounters, formatBytes, getGPUProfile } from '../../lib/engine.js';
  import MetricsPanel from '../../lib/MetricsPanel.svelte';
  import { onMount } from 'svelte';
  import { makeShareUrl, readSharedState } from '../../lib/share.js';
  import MemoryBankGrid from '../../lib/MemoryBankGrid.svelte';
  import CodeEditor from '../../lib/CodeEditor.svelte';

  let selectedPresetId = PRESETS[0].id;
  let selectedProfile = 'A100';
  let selectedDevices = 1;
  let code = PRESETS[0].code;

  let mode = 'preset'; // 'preset' | 'custom'
  let isRunning = false;
  let runStatus = 'IDLE';
  let runOutput = '';
  let runErrors = [];
  let transpiledCode = '';
  let rawCounters = null;
  let telemetry = null;
  let profileSpec = null;
  let executionTimeMs = 0;
  let activeTab = 'hardware';

  let fileInputRef;

  const profiles = [
    { id: 'A100', name: 'NVIDIA A100 SXM4 (80GB, 2039 GB/s, Ampere)' },
    { id: 'H100', name: 'NVIDIA H100 SXM5 (80GB, 3350 GB/s, Hopper)' },
    { id: 'RTX 4090', name: 'NVIDIA RTX 4090 (24GB, 1008 GB/s, Ada)' },
    { id: 'RTX 4050 Laptop', name: 'NVIDIA RTX 4050 Laptop (6GB, 192 GB/s, Ada - Hardware Verified)' },
    { id: 'RTX 5060 Ti', name: 'NVIDIA RTX 5060 Ti (16GB, Blackwell, projected specs)' },
    { id: 'RTX 3090', name: 'NVIDIA RTX 3090 (24GB, 936 GB/s, Ampere)' },
    { id: 'V100', name: 'NVIDIA Tesla V100 (16GB, 900 GB/s, Volta)' },
    { id: 'T4', name: 'NVIDIA Turing T4 (16GB, 320 GB/s, Turing)' }
  ];

  const deviceCounts = [1, 2, 4, 8, 16];

  function selectPreset(preset) {
    selectedPresetId = preset.id;
    code = preset.code;
    selectedDevices = preset.devices || 1;
    if (preset.id === 'custom') {
      mode = 'custom';
    } else {
      mode = 'preset';
    }
    runStatus = 'IDLE';
    runOutput = '';
    runErrors = [];
    transpiledCode = '';
    rawCounters = null;
    telemetry = null;
  }

  async function handleRun() {
    if (isRunning) return;
    isRunning = true;
    runStatus = 'RUNNING';
    runOutput = 'Transpiling and executing on virtual GPU cluster...';
    runErrors = [];
    transpiledCode = '';
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
      transpiledCode = res.transpiledCode || '';
      rawCounters = res.counters;
      telemetry = formatCounters(res.counters);
      profileSpec = res.profileSpec || await getGPUProfile(selectedProfile);

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

  let shareLabel = 'Share link';
  async function share() {
    const url = await makeShareUrl('/playground', { code, profile: selectedProfile, devices: selectedDevices });
    history.replaceState(null, '', url);
    try { await navigator.clipboard.writeText(url); shareLabel = 'Link copied'; } catch (e) { shareLabel = 'Copy from address bar'; }
    setTimeout(() => (shareLabel = 'Share link'), 2200);
  }

  onMount(async () => {
    const shared = await readSharedState();
    if (shared) {
      code = shared.code;
      if (shared.profile) selectedProfile = shared.profile;
      if (shared.devices) selectedDevices = shared.devices;
      mode = 'custom';
      selectedPresetId = 'custom';
    }
    setTimeout(handleRun, 150);
  });

  function handleClear() {
    runOutput = '// Console cleared.';
    runErrors = [];
    transpiledCode = '';
    rawCounters = null;
    telemetry = null;
    runStatus = 'IDLE';
  }

  function copyCode() {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(code);
    }
  }

  function copyTranspiled() {
    if (navigator.clipboard && transpiledCode) {
      navigator.clipboard.writeText(transpiledCode);
    }
  }

  function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      code = event.target?.result || '';
      mode = 'custom';
      selectedPresetId = 'custom';
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function handleDownload() {
    const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'kernel.cu';
    a.click();
    URL.revokeObjectURL(url);
  }

  function resetCustomTemplate() {
    const customPreset = PRESETS.find(p => p.id === 'custom');
    if (customPreset) {
      code = customPreset.code;
      selectedPresetId = 'custom';
      mode = 'custom';
    }
  }
</script>

<svelte:head>
  <title>GPUBOX Playground &bull; Syntax-Highlighted Multi-GPU CUDA Simulation</title>
</svelte:head>

<div class="playground-layout">
  <!-- Top Control Header -->
  <header class="pg-header">
    <div class="pg-header__left">
      <!-- Mode Toggle -->
      <div class="mode-toggle">
        <button
          class="mode-btn {mode === 'preset' ? 'mode-btn--active' : ''}"
          on:click={() => mode = 'preset'}
        >
          Preset Workloads
        </button>
        <button
          class="mode-btn {mode === 'custom' ? 'mode-btn--active' : ''}"
          on:click={() => {
            mode = 'custom';
            if (selectedPresetId !== 'custom') resetCustomTemplate();
          }}
        >
          Custom Run
        </button>
      </div>

      {#if mode === 'preset'}
        <div class="control-unit">
          <label for="pg-preset">Preset:</label>
          <select
            id="pg-preset"
            class="pg-select"
            bind:value={selectedPresetId}
            on:change={() => {
              const p = PRESETS.find(x => x.id === selectedPresetId);
              if (p) selectPreset(p);
            }}
          >
            {#each PRESETS as p}
              <option value={p.id}>{p.title} ({p.tag})</option>
            {/each}
          </select>
        </div>
      {/if}

      <div class="control-unit">
        <label for="pg-profile">Hardware Profile:</label>
        <select id="pg-profile" class="pg-select" bind:value={selectedProfile}>
          {#each profiles as prof}
            <option value={prof.id}>{prof.name}</option>
          {/each}
        </select>
      </div>

      <div class="control-unit">
        <label for="pg-nodes">Virtual Cluster:</label>
        <select id="pg-nodes" class="pg-select" bind:value={selectedDevices}>
          {#each deviceCounts as count}
            <option value={count}>{count} GPU{count > 1 ? 's' : ''} ({count >= 8 ? 'SuperPOD / Ring NVLink' : (count > 1 ? 'NVLink Ring' : 'Single Node')})</option>
          {/each}
        </select>
      </div>
    </div>

    <div class="pg-header__actions">
      <!-- Hidden file input for Upload -->
      <input
        type="file"
        accept=".cu,.cpp,.c"
        bind:this={fileInputRef}
        style="display:none"
        on:change={handleFileUpload}
      />

      <button class="btn btn--secondary btn--sm" on:click={() => fileInputRef.click()} title="Upload .cu or .cpp file">
        Upload .cu
      </button>

      <button class="btn btn--secondary btn--sm" on:click={share} title="Copy a link that reproduces this exact run">
        {shareLabel}
      </button>

      <button class="btn btn--secondary btn--sm" on:click={handleDownload} title="Download kernel as .cu file">
        Save .cu
      </button>

      {#if mode === 'custom'}
        <button class="btn btn--secondary btn--sm" on:click={resetCustomTemplate} title="Reset to clean CUDA template">
          Template
        </button>
      {/if}

      <button
        class="btn btn--primary btn--sm"
        disabled={isRunning}
        on:click={handleRun}
        title="Execute supported CUDA code (Ctrl + Enter)"
      >
        {#if isRunning}
          Running...
        {:else}
          Run Simulation
        {/if}
      </button>

      <button class="btn btn--secondary btn--sm" on:click={handleClear}>
        Clear
      </button>
    </div>
  </header>

  <!-- Split IDE Workspace -->
  <div class="pg-workspace">
    <!-- Left Editor Pane with Syntax Highlighting -->
    <section class="pg-pane pg-pane--editor">
      <div class="pg-pane__header">
        <div class="file-tag">
          <span class="file-dot"></span>
          <span class="file-name">kernel.cu</span>
          <span class="file-lang">CUDA C++ &bull; Syntax Highlighted</span>
        </div>

        <div class="pane-actions">
          <span class="shortcut-hint">Ctrl + Enter to run</span>
          <button class="icon-btn" on:click={copyCode} title="Copy CUDA code">
            Copy
          </button>
        </div>
      </div>

      <div class="editor-wrap">
        <CodeEditor
          bind:code
          language="cuda"
          placeholder="// Paste or write supported CUDA C++ code here..."
          on:run={handleRun}
        />
      </div>
    </section>

    <!-- Right Inspector Pane -->
    <section class="pg-pane pg-pane--inspector">
      <div class="pg-tabs">
        <button
          class="pg-tab {activeTab === 'output' ? 'pg-tab--active' : ''}"
          on:click={() => activeTab = 'output'}
        >
          Output & Logs
          {#if runStatus === 'PASS'}
            <span class="tab-indicator tab-indicator--pass"></span>
          {:else if runStatus === 'FAIL' || runStatus === 'ERROR'}
            <span class="tab-indicator tab-indicator--fail"></span>
          {/if}
        </button>

        <button
          class="pg-tab {activeTab === 'hardware' ? 'pg-tab--active' : ''}"
          on:click={() => activeTab = 'hardware'}
        >
          GPU Report
        </button>

        <button
          class="pg-tab {activeTab === 'memory' ? 'pg-tab--active' : ''}"
          on:click={() => activeTab = 'memory'}
        >
          32-Bank Memory
          {#if rawCounters?.bankConflicts > 0}
            <span class="tab-badge-warn">!{rawCounters.bankConflicts}</span>
          {/if}
        </button>

        <button
          class="pg-tab {activeTab === 'multigpu' ? 'pg-tab--active' : ''}"
          on:click={() => activeTab = 'multigpu'}
        >
          Cluster & NCCL ({selectedDevices} GPU{selectedDevices > 1 ? 's' : ''})
        </button>

        <button
          class="pg-tab {activeTab === 'transpiled' ? 'pg-tab--active' : ''}"
          on:click={() => activeTab = 'transpiled'}
        >
          Transpiled JS
        </button>
      </div>

      <div class="pg-tab-content">
        <!-- Tab 1: Output & Logs -->
        {#if activeTab === 'output'}
          <div class="output-tab-view">
            <div class="output-status-bar">
              <span class="status-chip status-chip--{runStatus.toLowerCase()}">
                Status: {runStatus}
              </span>
              {#if executionTimeMs > 0}
                <span class="status-time">Simulation duration: {executionTimeMs} ms</span>
              {/if}
            </div>

            {#if runErrors.length > 0}
              <div class="error-banner">
                <div class="error-banner__title">Diagnostics / Runtime Hazards Detected:</div>
                <ul>
                  {#each runErrors as err}
                    <li>{err}</li>
                  {/each}
                </ul>
              </div>
            {/if}

            <pre class="terminal-body">{runOutput || '// Press "Run Simulation" (Ctrl+Enter) to compile and execute...'}</pre>
          </div>
        {/if}

        <!-- Tab 2: GPU Metrics & Hardware Specs -->
        {#if activeTab === 'hardware'}
          <div class="hardware-tab-view">
            <MetricsPanel counters={rawCounters} spec={profileSpec} devices={selectedDevices} profileName={selectedProfile} />

            <div class="spec-section">
              <div class="spec-section__title">Active Hardware Specification Profile</div>
              <table class="data-table">
                <tbody>
                  <tr><td>GPU Model</td><td class="mono">{selectedProfile}</td></tr>
                  <tr><td>Architecture</td><td class="mono">{profileSpec?.architecture || 'Ampere'}</td></tr>
                  <tr><td>Compute Capability</td><td class="mono">sm_{profileSpec?.computeCapability || '80'}</td></tr>
                  <tr><td>Streaming Multiprocessors (SM)</td><td class="mono">{profileSpec?.smCount || 108} SMs</td></tr>
                  <tr><td>Peak FP32 Compute</td><td class="mono">{profileSpec?.fp32TFLOPS || profileSpec?.peakTFLOPS || 19.5} TFLOPS</td></tr>
                  <tr><td>Peak Memory Bandwidth</td><td class="mono">{profileSpec?.memoryBandwidthGBs || profileSpec?.memBandwidthGBs || 2039} GB/s</td></tr>
                  <tr><td>Max Shared Memory / Block</td><td class="mono">{formatBytes(profileSpec?.maxSharedMemPerBlock || 49152)}</td></tr>
                  <tr><td>Interconnect Topology</td><td class="mono">{profileSpec?.interconnect || 'NVLink (600 GB/s bidirectional)'}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        {/if}

        <!-- Tab 3: Memory & 32-Bank Visualizer -->
        {#if activeTab === 'memory'}
          <div class="memory-tab-view">
            <MemoryBankGrid counters={rawCounters} />
          </div>
        {/if}

        <!-- Tab 4: Cluster & NCCL (1 to 16 GPUs) -->
        {#if activeTab === 'multigpu'}
          <div class="cluster-tab-view">
            <div class="kpi-grid">
              <div class="kpi-card">
                <span class="kpi-label">Cluster GPU Nodes</span>
                <span class="kpi-val">{selectedDevices} Virtual Devices</span>
                <span class="kpi-sub">Interconnect: {selectedDevices >= 8 ? 'NVLink 4 NVSwitch Mesh' : (selectedDevices > 1 ? 'NVLink Ring' : 'Single PCIe')}</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Cross-GPU Traffic</span>
                <span class="kpi-val">{telemetry?.crossGpuBytes || '0 B'}</span>
                <span class="kpi-sub">{telemetry?.ncclOps || '0'} NCCL collective calls</span>
              </div>
            </div>

            <div class="cluster-nodes-list">
              <div class="spec-section__title">Virtual Cluster Topology & Device Ranks (0 to {selectedDevices - 1})</div>
              {#if selectedDevices > 1}
                <div class="nodes-grid nodes-grid--{selectedDevices > 8 ? 'dense' : 'normal'}">
                  {#each Array(selectedDevices) as _, rank}
                    <div class="node-card">
                      <div class="node-card__header">
                        <span class="node-title">Rank #{rank}: {selectedProfile}</span>
                        <span class="badge badge--success">Online</span>
                      </div>
                      <div class="node-stats">
                        <span class="node-stat-item">Device ID: cudaSetDevice({rank})</span>
                        <span class="node-stat-item">Peer Fabric: NVLink Bidirectional</span>
                        <span class="node-stat-item">Rank Status: Shard Active</span>
                      </div>
                    </div>
                  {/each}
                </div>
              {:else}
                <div class="single-node-notice">
                  Single GPU selected. Increase "Virtual Cluster" to 2, 4, 8, or 16 GPUs in the top bar to simulate multi-GPU NCCL collectives, ring AllReduce, and peer interconnect latency.
                </div>
              {/if}
            </div>
          </div>
        {/if}

        <!-- Tab 5: Transpiled JavaScript with Syntax Highlighting -->
        {#if activeTab === 'transpiled'}
          <div class="transpiled-tab-view">
            <div class="transpiled-header">
              <span>Client-Side Generated JavaScript (Executed via SIMT scheduler)</span>
              <button class="icon-btn" on:click={copyTranspiled} disabled={!transpiledCode}>
                Copy JS
              </button>
            </div>
            <div class="transpiled-editor-wrap">
              <CodeEditor
                code={transpiledCode || '// Transpiled JavaScript code will appear here after clicking "Run Simulation"...'}
                language="javascript"
                readOnly={true}
              />
            </div>
          </div>
        {/if}
      </div>
    </section>
  </div>
</div>

<style>
  .playground-layout {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 4.25rem);
    background: #000000;
    overflow: hidden;
  }

  .pg-header {
    background: #000000;
    border-bottom: 1px solid var(--border-gold);
    padding: 0.65rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    z-index: 10;
  }

  .pg-header__left {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  /* Mode Toggle */
  .mode-toggle {
    display: flex;
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.25);
    border-radius: var(--radius-sm);
    padding: 2px;
  }

  .mode-btn {
    background: transparent;
    border: none;
    padding: 0.3rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .mode-btn:hover {
    color: #ffffff;
  }

  .mode-btn--active {
    background: var(--gold-glow);
    color: var(--gold-primary);
  }

  .control-unit {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  .control-unit label {
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    color: var(--gold-muted);
  }

  .pg-select {
    background: #000000;
    color: var(--text-primary);
    border: 1px solid rgba(212, 175, 55, 0.25);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 500;
    outline: none;
    cursor: pointer;
  }

  .pg-select:focus {
    border-color: var(--gold-primary);
  }

  .pg-header__actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* Workspace */
  .pg-workspace {
    display: flex;
    flex: 1;
    overflow: hidden;
    background: #000000;
  }

  @media (max-width: 960px) {
    .pg-workspace {
      flex-direction: column;
    }
  }

  .pg-pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #000000;
  }

  .pg-pane--editor {
    border-right: 1px solid rgba(212, 175, 55, 0.18);
    background: #000000;
  }

  .pg-pane--inspector {
    background: #000000;
  }

  .pg-pane__header {
    background: #000000;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    padding: 0.5rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .file-tag {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }

  .file-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--gold-primary);
    box-shadow: 0 0 6px var(--gold-glow-strong);
  }

  .file-name {
    color: #ffffff;
    font-weight: 600;
  }

  .file-lang {
    color: var(--text-muted);
    font-size: 0.72rem;
  }

  .pane-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .shortcut-hint {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .icon-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    cursor: pointer;
    font-family: var(--font-mono);
  }

  .icon-btn:hover {
    color: var(--gold-light);
    border-color: var(--gold-primary);
    background: var(--gold-glow);
  }

  .editor-wrap {
    flex: 1;
    display: flex;
    position: relative;
    overflow: hidden;
    background: #000000;
  }

  /* Inspector Tabs */
  .pg-tabs {
    display: flex;
    background: #000000;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    overflow-x: auto;
  }

  .pg-tab {
    padding: 0.65rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.15s ease;
  }

  .pg-tab:hover {
    color: var(--text-primary);
  }

  .pg-tab--active {
    color: var(--gold-light);
    border-bottom-color: var(--gold-primary);
    background: rgba(212, 175, 55, 0.08);
  }

  .tab-indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .tab-indicator--pass { background: var(--success); }
  .tab-indicator--fail { background: var(--danger); }

  .tab-badge-warn {
    font-size: 0.68rem;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    background: var(--warning-bg);
    color: var(--warning);
    font-family: var(--font-mono);
  }

  .pg-tab-content {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 1.25rem;
    background: #000000;
  }

  /* Tab Views */
  .output-tab-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    gap: 0.75rem;
  }

  .output-status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.75rem;
  }

  .status-chip {
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.72rem;
  }

  .status-chip--idle { background: rgba(255, 255, 255, 0.05); color: var(--text-muted); }
  .status-chip--running { background: rgba(212, 175, 55, 0.15); color: var(--gold-light); }
  .status-chip--pass { background: var(--success-bg); color: var(--success); }
  .status-chip--fail, .status-chip--error { background: var(--danger-bg); color: var(--danger); }

  .status-time {
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .error-banner {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: var(--radius-sm);
    padding: 0.75rem;
    font-size: 0.8rem;
    color: #fca5a5;
  }

  .error-banner__title {
    font-weight: 700;
    margin-bottom: 0.35rem;
  }

  .error-banner ul {
    margin-left: 1.25rem;
  }

  .terminal-body {
    flex: 1;
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.18);
    border-radius: var(--radius-sm);
    padding: 0.85rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    line-height: 1.6;
    color: #e2e8f0;
    white-space: pre-wrap;
    word-break: break-all;
    overflow-y: auto;
    min-height: 240px;
  }

  /* Hardware KPI */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
  }

  .kpi-card {
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.18);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
  }

  .kpi-label {
    display: block;
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
  }

  .kpi-val {
    display: block;
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
  }

  .kpi-sub {
    display: block;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
  }

  .spec-section__title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--gold-light);
    margin-bottom: 0.6rem;
  }

  .single-node-notice {
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.18);
    border-radius: var(--radius-md);
    padding: 1rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  .nodes-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.65rem;
  }

  .nodes-grid--dense {
    grid-template-columns: repeat(4, 1fr);
  }

  @media (max-width: 900px) {
    .nodes-grid--dense {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 600px) {
    .nodes-grid, .nodes-grid--dense {
      grid-template-columns: 1fr;
    }
  }

  .node-card {
    background: #000000;
    border: 1px solid rgba(212, 175, 55, 0.18);
    border-radius: var(--radius-md);
    padding: 0.75rem;
  }

  .node-card__header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
  }

  .node-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #ffffff;
    font-family: var(--font-mono);
  }

  .node-stats {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .transpiled-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
  }

  .transpiled-editor-wrap {
    height: 440px;
    border: 1px solid rgba(212, 175, 55, 0.18);
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: #000000;
  }
</style>
