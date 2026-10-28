<script>
  import { computeBankState, formatBytes } from './engine.js';

  export let counters = null;

  $: bankState = computeBankState(counters);
  $: totalSharedLoads = counters?.sharedLoads ?? 0;
  $: totalSharedStores = counters?.sharedStores ?? 0;
  $: totalSharedBytes = (counters?.sharedLoadBytes ?? 0) + (counters?.sharedStoreBytes ?? 0);
  $: bankConflicts = counters?.bankConflicts ?? 0;
  $: globalLoads = counters?.globalLoads ?? 0;
  $: globalStores = counters?.globalStores ?? 0;
  $: globalTransactions = counters?.globalTransactions ?? 0;
  $: syncthreadsCount = counters?.syncthreadsCount ?? 0;
  $: divergentBranches = counters?.divergentBranches ?? 0;
  $: atomicOps = counters?.atomicOps ?? 0;
  $: atomicMaxContention = counters?.atomicMaxContention ?? 0;

  let hoveredBank = null;
</script>

<div class="memory-profiler">
  <!-- Top Statistics Summary -->
  <div class="stats-row">
    <div class="stat-box">
      <div class="stat-label">Shared Loads / Stores</div>
      <div class="stat-val">{totalSharedLoads.toLocaleString()} / {totalSharedStores.toLocaleString()}</div>
      <div class="stat-sub">{formatBytes(totalSharedBytes)} total volume</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Bank Conflicts</div>
      <div class="stat-val {bankConflicts > 0 ? 'text-warn' : 'text-success'}">
        {bankConflicts} {bankConflicts === 1 ? 'conflict' : 'conflicts'}
      </div>
      <div class="stat-sub">{bankConflicts > 0 ? 'Warp serializations detected' : 'Full bandwidth utilization'}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Coalesced Transactions</div>
      <div class="stat-val">{globalTransactions.toLocaleString()}</div>
      <div class="stat-sub">{globalLoads + globalStores} global accesses</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Barrier & Branch Hazards</div>
      <div class="stat-val">{syncthreadsCount} syncs / {divergentBranches} diverged</div>
      <div class="stat-sub">{atomicOps} atomics (max {atomicMaxContention}x)</div>
    </div>
  </div>

  <!-- 32-Bank Shared Memory Interactive Grid -->
  <div class="bank-grid-container">
    <div class="bank-grid-header">
      <div>
        <span class="bank-grid-title">32-Bank Shared Memory Topography</span>
        <span class="bank-grid-desc">4-Byte interleaved banks across 32 warp lanes</span>
      </div>
      <div class="bank-grid-legend">
        <span class="legend-item"><span class="legend-dot idle"></span>Idle</span>
        <span class="legend-item"><span class="legend-dot active"></span>Active</span>
        <span class="legend-item"><span class="legend-dot conflict"></span>Conflict</span>
      </div>
    </div>

    <div class="bank-grid-32">
      {#each bankState as b}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="bank-cell status-{b.status} {hoveredBank === b.bank ? 'is-hovered' : ''}"
          on:mouseenter={() => hoveredBank = b.bank}
          on:mouseleave={() => hoveredBank = null}
        >
          <span class="bank-num">B{b.bank.toString().padStart(2, '0')}</span>
          <span class="bank-stat">
            {#if b.conflicts > 0}
              !{b.conflicts}
            {:else if b.accesses > 0}
              {b.accesses > 999 ? (b.accesses / 1000).toFixed(0) + 'k' : b.accesses}
            {:else}
              -
            {/if}
          </span>
        </div>
      {/each}
    </div>

    <!-- Hovered Bank Detailed Inspector -->
    <div class="bank-inspector">
      {#if hoveredBank !== null}
        {@const selected = bankState[hoveredBank]}
        <div class="inspector-detail">
          <span class="mono-highlight">Bank {selected.bank}</span>:
          {#if selected.status === 'conflict'}
            <span class="text-warn">Bank Conflict ({selected.conflicts} warp collision{selected.conflicts > 1 ? 's' : ''})</span>. Shared memory requests to distinct words in this bank must serialize.
          {:else if selected.status === 'active'}
            <span class="text-success">Active ({selected.accesses} word accesses)</span>. Conflict-free parallel access across active warp threads.
          {:else}
            <span class="text-muted">Idle (0 accesses in simulated block)</span>.
          {/if}
        </div>
      {:else}
        <div class="inspector-hint">
          Hover over any bank cell above to inspect lane collisions, word alignments, and serialization events.
        </div>
      {/if}
    </div>
  </div>

  <!-- Memory Hierarchy Breakdown Tables -->
  <div class="memory-tables-grid">
    <div class="mem-subcard">
      <div class="mem-subcard-header">Global Memory Access Profile</div>
      <table class="counter-table">
        <tbody>
          <tr><td>Global Loads</td><td class="mono">{globalLoads.toLocaleString()}</td></tr>
          <tr><td>Global Stores</td><td class="mono">{globalStores.toLocaleString()}</td></tr>
          <tr><td>Bytes Read (HBM/GDDR)</td><td class="mono">{formatBytes(counters?.globalLoadBytes ?? 0)}</td></tr>
          <tr><td>Bytes Written (HBM/GDDR)</td><td class="mono">{formatBytes(counters?.globalStoreBytes ?? 0)}</td></tr>
          <tr><td>32-Byte Sector Transactions</td><td class="mono">{globalTransactions.toLocaleString()}</td></tr>
        </tbody>
      </table>
    </div>

    <div class="mem-subcard">
      <div class="mem-subcard-header">Shared Memory & Warp Operations</div>
      <table class="counter-table">
        <tbody>
          <tr><td>Shared Memory Loads</td><td class="mono">{totalSharedLoads.toLocaleString()}</td></tr>
          <tr><td>Shared Memory Stores</td><td class="mono">{totalSharedStores.toLocaleString()}</td></tr>
          <tr><td>__syncthreads() Barriers</td><td class="mono">{syncthreadsCount.toLocaleString()}</td></tr>
          <tr><td>Warp Branch Divergences</td><td class="mono">{divergentBranches.toLocaleString()}</td></tr>
          <tr><td>Atomic Operations (atomicAdd)</td><td class="mono">{atomicOps.toLocaleString()}</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<style>
  .memory-profiler {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
  }

  @media (max-width: 900px) {
    .stats-row {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .stat-box {
    background: #090d15;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
  }

  .stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
  }

  .stat-val {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
  }

  .stat-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
  }

  .text-success { color: var(--success); }
  .text-warn { color: var(--warning); }
  .text-muted { color: var(--text-muted); }

  .bank-grid-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #ffffff;
    display: block;
  }

  .bank-grid-desc {
    font-size: 0.72rem;
    color: var(--text-muted);
  }

  .bank-inspector {
    background: #06090f;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.85rem;
    font-size: 0.78rem;
    min-height: 38px;
    display: flex;
    align-items: center;
  }

  .inspector-hint {
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .mono-highlight {
    font-family: var(--font-mono);
    font-weight: 700;
    color: var(--primary-light);
  }

  .memory-tables-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
  }

  @media (max-width: 768px) {
    .memory-tables-grid {
      grid-template-columns: 1fr;
    }
  }

  .mem-subcard {
    background: #090d15;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .mem-subcard-header {
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border-subtle);
    padding: 0.5rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }

  .counter-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }

  .counter-table td {
    padding: 0.5rem 0.85rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    color: var(--text-secondary);
  }

  .counter-table td:last-child {
    text-align: right;
  }

  .counter-table tr:last-child td {
    border-bottom: none;
  }

  .mono {
    font-family: var(--font-mono);
    color: var(--text-primary);
  }
</style>
