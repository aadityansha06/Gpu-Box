<script>
  import { PRESETS } from '../../lib/presets.js';
  import { runCUDA, formatCounters } from '../../lib/engine.js';

  let selectedPresetId = PRESETS[0].id;
  let selectedProfile = 'A100';
  let selectedDevices = 1;
  let code = PRESETS[0].code;

  let isRunning = false;
  let runStatus = 'IDLE';
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
  <title>GPUBOX Playground &bull; Full-Screen CUDA Simulation</title>
</svelte:head>

<div class="playground-page">
  <div class="playground-topbar">
    <div class="topbar-presets">
      <span class="topbar-label">Presets:</span>
      {#each PRESETS as p}
        <button
          class="preset-pill {selectedPresetId === p.id ? 'preset-pill--active' : ''}"
          on:click={() => selectPreset(p)}
        >
          {p.title}
        </button>
      {/each}
    </div>

    <div class="topbar-controls">
      <select bind:value={selectedProfile} class="select-box">
        {#each profiles as prof}
          <option value={prof.id}>{prof.label}</option>
        {/each}
      </select>

      <select bind:value={selectedDevices} class="select-box select-box--sm">
        {#each deviceCounts as count}
          <option value={count}>{count} {count === 1 ? 'GPU' : 'GPUs'}</option>
        {/each}
      </select>

      <button
        class="btn btn--primary btn--sm"
        on:click={handleRun}
        disabled={isRunning}
      >
        {isRunning ? 'Running...' : 'Run Simulation'}
      </button>
    </div>
  </div>

  <div class="playground-body">
    <div class="editor-column">
      <div class="column-header">
        <span>CUDA C++ Kernel Source</span>
        <span>{code.split('\n').length} lines</span>
      </div>
      <textarea
        bind:value={code}
        class="fullscreen-editor"
        spellcheck="false"
        on:keydown={handleEditorKeydown}
      ></textarea>
    </div>

    <div class="output-column">
      <div class="column-header">
        <div class="column-header__left">
          <span>Simulation Output</span>
          {#if runStatus !== 'IDLE'}
            <span class="status-tag status-tag--{runStatus.toLowerCase()}">{runStatus}</span>
            <span class="status-time">{executionTimeMs} ms</span>
          {/if}
        </div>
      </div>

      <div class="terminal-view">
        {#if runErrors.length > 0}
          <div class="error-box">
            <strong>Diagnostics & Errors:</strong>
            {#each runErrors as err}
              <div>{err}</div>
            {/each}
          </div>
        {/if}

        {#if runOutput}
          <pre class="stdout-text">{runOutput}</pre>
        {:else if runErrors.length === 0}
          <div class="stdout-placeholder">Click "Run Simulation" to execute on virtual GPU.</div>
        {/if}
      </div>

      {#if telemetry}
        <div class="telemetry-footer">
          <div>Threads: <strong>{telemetry.threads.toLocaleString()}</strong></div>
          <div>Cycles: <strong>{telemetry.cycles.toLocaleString()}</strong></div>
          <div>Loads: <strong>{telemetry.globalLoads.toLocaleString()}</strong></div>
          <div>Stores: <strong>{telemetry.globalStores.toLocaleString()}</strong></div>
          <div>Bank Conflicts: <strong>{telemetry.bankConflicts}</strong></div>
          {#if selectedDevices > 1}
            <div>Traffic: <strong>{telemetry.crossGpuBytes.toLocaleString()} B</strong></div>
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .playground-page {
    height: calc(100vh - 4.25rem);
    display: flex;
    flex-direction: column;
    background: #07090e;
  }

  .playground-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.25rem;
    background: #0b0e17;
    border-bottom: 1px solid var(--border-subtle);
    gap: 1rem;
    flex-wrap: wrap;
  }

  .topbar-presets {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    overflow-x: auto;
  }

  .topbar-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-right: 0.25rem;
  }

  .preset-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
  }

  .preset-pill:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.08);
  }

  .preset-pill--active {
    background: var(--primary);
    color: #ffffff;
    border-color: var(--primary-hover);
  }

  .topbar-controls {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .select-box {
    background: #0f1422;
    color: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 0.35rem 0.65rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    outline: none;
  }

  .select-box--sm {
    padding-right: 1.2rem;
  }

  .playground-body {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    min-height: 0;
  }

  .editor-column, .output-column {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .editor-column {
    border-right: 1px solid var(--border-subtle);
    background: #090c13;
  }

  .output-column {
    background: #07090f;
  }

  .column-header {
    height: 2.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1rem;
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
  }

  .column-header__left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .status-tag {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-family: var(--font-mono);
  }

  .status-tag--pass { background: var(--success-bg); color: var(--success); }
  .status-tag--fail, .status-tag--error { background: var(--danger-bg); color: var(--danger); }
  .status-tag--running { background: rgba(99, 102, 241, 0.2); color: var(--primary-light); }

  .status-time {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .fullscreen-editor {
    flex: 1;
    width: 100%;
    background: transparent;
    color: #e5e7eb;
    font-family: var(--font-mono);
    font-size: 0.88rem;
    line-height: 1.6;
    padding: 1rem;
    border: none;
    outline: none;
    resize: none;
    tab-size: 4;
  }

  .terminal-view {
    flex: 1;
    padding: 1rem;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    overflow-y: auto;
  }

  .stdout-text {
    color: #38bdf8;
    white-space: pre-wrap;
  }

  .stdout-placeholder {
    color: var(--text-muted);
    font-style: italic;
  }

  .error-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 0.65rem;
    border-radius: 4px;
    color: #fca5a5;
    margin-bottom: 0.75rem;
    font-size: 0.82rem;
  }

  .telemetry-footer {
    display: flex;
    gap: 1rem;
    padding: 0.6rem 1rem;
    background: rgba(0, 0, 0, 0.4);
    border-top: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    flex-wrap: wrap;
  }

  .telemetry-footer strong {
    color: var(--primary-light);
  }
</style>
