<script>
  import { analyzeRun, formatBytes } from './engine.js';

  export let counters = null;
  export let spec = null;
  export let devices = 1;
  export let profileName = '';

  $: report = analyzeRun(counters, spec, devices);
  $: timeLabel = counters
    ? (counters.estimatedTimeMs < 1 ? (counters.estimatedTimeMs * 1000).toFixed(1) + ' µs' : counters.estimatedTimeMs.toFixed(3) + ' ms')
    : '—';
</script>

{#if !report}
  <div class="empty">
    <div class="empty__glyph">◈</div>
    <p>Run a kernel to see its GPU report: bytes moved at every level of the memory hierarchy, FLOPs executed, where the time goes, and what limits it.</p>
  </div>
{:else}
  <div class="report">
    <div class="headline">
      <div class="stat">
        <span class="stat__label">Modeled time</span>
        <span class="stat__val">{timeLabel}</span>
        <span class="stat__sub">{profileName}</span>
      </div>
      <div class="stat">
        <span class="stat__label">Compute executed</span>
        <span class="stat__val">{report.flopsLabel}<small>FLOP</small></span>
        <span class="stat__sub">{(counters.computeUtilization * 100).toFixed(2)}% of peak</span>
      </div>
      <div class="stat">
        <span class="stat__label">Data moved</span>
        <span class="stat__val">{formatBytes(report.totalMoved)}</span>
        <span class="stat__sub">{formatBytes(report.globalBytes)} through DRAM</span>
      </div>
      <div class="stat">
        <span class="stat__label">Limiter</span>
        <span class="stat__val stat__val--{report.bound}">{report.bound}</span>
        <span class="stat__sub">{report.intensity.toFixed(2)} FLOP/B vs balance {report.balance.toFixed(1)}</span>
      </div>
    </div>

    <div class="cols">
      <section>
        <h4>Where the bytes went</h4>
        {#each report.movement as m}
          <div class="row">
            <span class="row__label">{m.label}</span>
            <div class="bar"><i class="bar__fill bar__fill--{m.tone}" style="width: {Math.max(m.bytes > 0 ? 2 : 0, m.pct)}%"></i></div>
            <span class="row__val">{formatBytes(m.bytes)}</span>
          </div>
        {/each}
      </section>

      <section>
        <h4>Where the time went</h4>
        {#each report.timeParts as t}
          <div class="row">
            <span class="row__label">{t.label}</span>
            <div class="bar"><i class="bar__fill bar__fill--time" style="width: {Math.max(t.us > 0 ? 2 : 0, t.pct)}%"></i></div>
            <span class="row__val">{t.us < 0.01 ? '0' : t.us.toFixed(2)} µs</span>
          </div>
        {/each}
        <p class="note">Time = launch + max(memory, compute, shared) + atomics + barriers + interconnect.</p>
      </section>
    </div>

    <div class="gauges">
      <div class="gauge">
        <span class="gauge__label">Coalescing efficiency</span>
        <div class="bar bar--thick"><i class="bar__fill {report.coalescing < 0.7 ? 'bar__fill--bad' : 'bar__fill--good'}" style="width: {report.coalescing * 100}%"></i></div>
        <span class="gauge__val">{(report.coalescing * 100).toFixed(0)}%</span>
      </div>
      <div class="gauge">
        <span class="gauge__label">Bank-conflict rate</span>
        <div class="bar bar--thick"><i class="bar__fill {report.conflictRate > 0.1 ? 'bar__fill--bad' : 'bar__fill--good'}" style="width: {Math.min(100, report.conflictRate * 100)}%"></i></div>
        <span class="gauge__val">{(report.conflictRate * 100).toFixed(0)}%</span>
      </div>
      <div class="gauge">
        <span class="gauge__label">Occupancy</span>
        <div class="bar bar--thick"><i class="bar__fill bar__fill--dram" style="width: {Math.min(100, (counters.occupancy || 0) * 100)}%"></i></div>
        <span class="gauge__val">{((counters.occupancy || 0) * 100).toFixed(0)}%</span>
      </div>
    </div>

    <div class="counts">
      <span><b>{(counters.threadsLaunched || 0).toLocaleString()}</b> threads</span>
      <span><b>{(counters.blocksLaunched || 0).toLocaleString()}</b> blocks</span>
      <span><b>{(counters.globalTransactions || 0).toLocaleString()}</b> DRAM transactions</span>
      <span><b>{(counters.syncthreadsCount || 0).toLocaleString()}</b> barrier arrivals</span>
      <span><b>{(counters.divergentBranches || 0).toLocaleString()}</b> divergent branches</span>
      <span><b>{(counters.atomicOps || 0).toLocaleString()}</b> atomics</span>
      {#if devices > 1}<span><b>{(counters.ncclOps || 0).toLocaleString()}</b> NCCL calls</span>{/if}
    </div>

    <section>
      <h4>Findings</h4>
      <ul class="findings">
        {#each report.findings as f}
          <li class="finding finding--{f.level}"><i></i><span>{f.text}</span></li>
        {/each}
      </ul>
    </section>
  </div>
{/if}

<style>
  .empty { padding: 3rem 1.5rem; text-align: center; color: var(--text-muted); }
  .empty__glyph { font-size: 2rem; color: var(--gold-primary); margin-bottom: .75rem; opacity: .7; }
  .empty p { max-width: 38ch; margin: 0 auto; font-size: .9rem; line-height: 1.6; }

  .report { display: flex; flex-direction: column; gap: 1.25rem; padding: 1.1rem; }
  .headline { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .75rem; }
  .stat { display: flex; flex-direction: column; gap: .15rem; padding: .85rem 1rem; border: 1px solid var(--border-gold); border-radius: var(--radius-md);
    background: linear-gradient(160deg, rgba(212,175,55,.09), rgba(212,175,55,.02)); }
  .stat__label { font-size: .68rem; letter-spacing: .09em; text-transform: uppercase; color: var(--text-muted); }
  .stat__val { font-family: var(--font-mono); font-size: 1.35rem; font-weight: 600; color: var(--gold-light); line-height: 1.2; }
  .stat__val small { font-size: .7rem; margin-left: .3rem; color: var(--text-muted); font-weight: 500; }
  .stat__val--memory-bound { color: #7dd3fc; font-size: 1.05rem; text-transform: capitalize; }
  .stat__val--compute-bound { color: #f0abfc; font-size: 1.05rem; text-transform: capitalize; }
  .stat__val--idle { color: var(--text-muted); font-size: 1.05rem; }
  .stat__sub { font-size: .72rem; color: var(--text-muted); }

  .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; }
  h4 { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--gold-primary); margin-bottom: .65rem; font-weight: 600; }
  .row { display: grid; grid-template-columns: 11.5rem 1fr 5.5rem; gap: .65rem; align-items: center; margin-bottom: .45rem; font-size: .8rem; }
  .row__label { color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .row__val { font-family: var(--font-mono); color: var(--text-primary); text-align: right; font-size: .76rem; }
  .bar { height: 7px; border-radius: 99px; background: rgba(255,255,255,.06); overflow: hidden; }
  .bar--thick { height: 9px; }
  .bar__fill { display: block; height: 100%; border-radius: 99px; transition: width .5s cubic-bezier(.2,.8,.2,1); }
  .bar__fill--host { background: linear-gradient(90deg, #64748b, #94a3b8); }
  .bar__fill--dram { background: linear-gradient(90deg, #997c22, #f5d77f); }
  .bar__fill--shared { background: linear-gradient(90deg, #0e7490, #67e8f9); }
  .bar__fill--link { background: linear-gradient(90deg, #7e22ce, #e879f9); }
  .bar__fill--time { background: linear-gradient(90deg, #b45309, #fbbf24); }
  .bar__fill--good { background: linear-gradient(90deg, #047857, #34d399); }
  .bar__fill--bad { background: linear-gradient(90deg, #b91c1c, #f87171); }
  .note { font-size: .7rem; color: var(--text-muted); margin-top: .5rem; }

  .gauges { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem; }
  .gauge { display: grid; grid-template-columns: 1fr auto; gap: .35rem .75rem; align-items: center; }
  .gauge__label { font-size: .76rem; color: var(--text-secondary); }
  .gauge__val { font-family: var(--font-mono); font-size: .8rem; color: var(--text-primary); }
  .gauge .bar { grid-column: 1 / -1; }

  .counts { display: flex; flex-wrap: wrap; gap: .4rem .9rem; font-size: .76rem; color: var(--text-muted); padding: .7rem .9rem; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); background: rgba(255,255,255,.015); }
  .counts b { color: var(--text-primary); font-family: var(--font-mono); font-weight: 600; }

  .findings { list-style: none; display: flex; flex-direction: column; gap: .5rem; }
  .finding { display: grid; grid-template-columns: 10px 1fr; gap: .7rem; align-items: start; font-size: .83rem; line-height: 1.5; color: var(--text-secondary);
    padding: .6rem .8rem; border-radius: var(--radius-md); background: rgba(255,255,255,.02); border: 1px solid var(--border-subtle); }
  .finding i { width: 8px; height: 8px; border-radius: 50%; margin-top: .45rem; background: var(--text-muted); }
  .finding--warn { border-color: var(--warning-border); background: var(--warning-bg); }
  .finding--warn i { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
  .finding--good { border-color: var(--success-border); background: var(--success-bg); }
  .finding--good i { background: var(--success); box-shadow: 0 0 8px var(--success); }
  .finding--info i { background: var(--gold-primary); }

  @media (max-width: 640px) { .row { grid-template-columns: 8.5rem 1fr 4.6rem; } }
</style>
