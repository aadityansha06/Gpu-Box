<script>
  import { onMount } from 'svelte';
  import { PRESETS } from '../lib/presets.js';
  import { runCUDA } from '../lib/engine.js';
  import { LINKS, PRICING } from '../lib/config.js';
  import MemoryBankGrid from '../lib/MemoryBankGrid.svelte';
  import MetricsPanel from '../lib/MetricsPanel.svelte';
  import CodeEditor from '../lib/CodeEditor.svelte';
  import parity from '../lib/parity.json';

  let selectedPresetId = PRESETS[0].id;
  let selectedProfile = 'RTX 4050 Laptop';
  let selectedDevices = 1;
  let code = PRESETS[0].code;

  let isRunning = false;
  let runStatus = 'IDLE';
  let runOutput = '';
  let runErrors = [];
  let rawCounters = null;
  let profileSpec = null;
  let activeTab = 'report';

  const profiles = [
    { id: 'RTX 4050 Laptop', label: 'RTX 4050 Laptop (hardware verified)' },
    { id: 'A100', label: 'A100 SXM4 80GB' },
    { id: 'H100', label: 'H100 SXM5 80GB' },
    { id: 'RTX 4090', label: 'RTX 4090' },
    { id: 'RTX 3090', label: 'RTX 3090' },
    { id: 'V100', label: 'Tesla V100' },
    { id: 'RTX 5060 Ti', label: 'RTX 5060 Ti (projected)' }
  ];
  const deviceCounts = [1, 2, 4, 8, 16];

  const groups = ['Basics', 'ML', 'Memory', 'Warp', 'Atomics', 'Multi-GPU', 'Debugging'];
  $: grouped = groups
    .map((g) => ({ name: g, items: PRESETS.filter((p) => p.tag === g) }))
    .filter((g) => g.items.length);
  $: current = PRESETS.find((p) => p.id === selectedPresetId);

  function choosePreset() {
    const p = PRESETS.find((x) => x.id === selectedPresetId);
    if (!p) return;
    code = p.code;
    selectedDevices = p.devices || 1;
    runStatus = 'IDLE';
    runOutput = '';
    runErrors = [];
    rawCounters = null;
    handleRun();
  }

  async function handleRun() {
    if (isRunning) return;
    isRunning = true;
    runStatus = 'RUNNING';
    try {
      const res = await runCUDA({ code, profile: selectedProfile, devices: selectedDevices });
      runOutput = res.output || (res.ok ? 'Completed with no console output.' : '');
      runErrors = res.errors || [];
      rawCounters = res.counters;
      profileSpec = res.profileSpec;
      runStatus = res.ok && runErrors.length === 0 ? 'PASS' : 'FAIL';
      if (runStatus === 'FAIL') activeTab = 'console';
      else if (activeTab === 'console') activeTab = 'report';
    } catch (err) {
      runStatus = 'ERROR';
      runOutput = '';
      runErrors = [err.message || String(err)];
      activeTab = 'console';
    } finally {
      isRunning = false;
    }
  }

  onMount(() => {
    const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 400));
    idle(() => handleRun());
  });

  const pillars = [
    { n: 'Data movement', d: 'Bytes at every level: host to device, DRAM reads and writes, shared memory and GPU to GPU links.' },
    { n: 'Compute', d: 'FP32 operations counted per thread, with arithmetic intensity and a memory- or compute-bound verdict.' },
    { n: 'Hazards', d: 'Out-of-bounds, shared-memory races, divergent barriers, bank conflicts and uncoalesced loads, with kernel, block and thread.' },
    { n: 'Clusters', d: '1 to 16 virtual GPUs with ring collectives, per-rank traffic, and a deadlock report when a rank skips a collective.' }
  ];
</script>

<svelte:head>
  <title>Gpubox | Pre-validate CUDA in the browser</title>
  <meta name="description" content="Run CUDA C++ and multi-GPU NCCL code in the browser and see bytes moved, FLOPs, bank conflicts, divergence and deadlocks before you spend GPU time. Verified against a physical RTX 4050." />
</svelte:head>

<div class="page">
  <section class="hero">
    <div class="wrap">
      <p class="eyebrow">CUDA pre-flight</p>
      <h1>Catch the CUDA bug<br />before it costs a GPU hour.</h1>
      <p class="lead">
        Run CUDA C++ and multi-GPU NCCL code in your browser. See the data moved, the compute executed and the hazards found, with nothing leaving your machine.
      </p>
      <div class="actions">
        <a href="#try" class="btn btn--primary">Try it now</a>
        <a href="/docs" class="btn btn--ghost">Documentation</a>
      </div>
      <dl class="facts">
        <div><dt>{parity.passed} / {parity.kernelsTested}</dt><dd>kernels match a physical RTX 4050</dd></div>
        <div><dt>205</dt><dd>automated checks passing</dd></div>
        <div><dt>1 to 16</dt><dd>virtual GPUs with NCCL</dd></div>
        <div><dt>0</dt><dd>bytes sent to a server</dd></div>
      </dl>
    </div>
  </section>

  <section class="block" id="try">
    <div class="wrap wrap--wide">
      <div class="ide">
        <div class="ide__bar">
          <select class="field field--grow" bind:value={selectedPresetId} on:change={choosePreset} aria-label="Example">
            {#each grouped as g}
              <optgroup label={g.name}>
                {#each g.items as p}
                  <option value={p.id}>{p.title}</option>
                {/each}
              </optgroup>
            {/each}
          </select>
          <select class="field" bind:value={selectedProfile} aria-label="GPU">
            {#each profiles as p}<option value={p.id}>{p.label}</option>{/each}
          </select>
          <select class="field" bind:value={selectedDevices} aria-label="Cluster size">
            {#each deviceCounts as n}<option value={n}>{n} GPU{n > 1 ? 's' : ''}</option>{/each}
          </select>
          <button class="btn btn--primary btn--sm" on:click={handleRun} disabled={isRunning}>
            {isRunning ? 'Running' : 'Run'}
          </button>
        </div>
        {#if current}<p class="ide__blurb">{current.blurb}</p>{/if}

        <div class="ide__body">
          <div class="ide__editor">
            <CodeEditor bind:code language="cuda" minHeight="480px" placeholder="// Paste CUDA C++ here" on:run={handleRun} />
          </div>
          <div class="ide__out">
            <div class="tabs">
              <button class:on={activeTab === 'report'} on:click={() => (activeTab = 'report')}>Report</button>
              <button class:on={activeTab === 'console'} on:click={() => (activeTab = 'console')}>
                Console
                {#if runStatus === 'PASS'}<i class="dot dot--ok"></i>{:else if runStatus === 'FAIL' || runStatus === 'ERROR'}<i class="dot dot--bad"></i>{/if}
              </button>
              <button class:on={activeTab === 'banks'} on:click={() => (activeTab = 'banks')}>
                Memory banks
                {#if rawCounters?.bankConflicts > 0}<i class="dot dot--warn"></i>{/if}
              </button>
              <a class="tabs__link" href="/playground">Full IDE</a>
            </div>
            <div class="ide__pane">
              {#if activeTab === 'report'}
                <MetricsPanel counters={rawCounters} spec={profileSpec} devices={selectedDevices} profileName={selectedProfile} />
              {:else if activeTab === 'console'}
                <div class="console">
                  {#if runErrors.length}
                    <div class="console__errors">
                      <b>Diagnostics</b>
                      {#each runErrors as e}<p>{e}</p>{/each}
                    </div>
                  {/if}
                  <pre>{runOutput || '// Run a kernel to see its output.'}</pre>
                </div>
              {:else}
                <div class="banks"><MemoryBankGrid counters={rawCounters} /></div>
              {/if}
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="block">
    <div class="wrap">
      <h2>Every run reads like a profiler report</h2>
      <p class="sub">Not just "it ran". What the GPU would have done, before you touch one.</p>
      <div class="pillars">
        {#each pillars as p, i}
          <div class="pillar">
            <span class="pillar__n">0{i + 1}</span>
            <h3>{p.n}</h3>
            <p>{p.d}</p>
          </div>
        {/each}
      </div>
    </div>
  </section>

  <section class="block" id="benchmarks">
    <div class="wrap">
      <h2>Checked against real silicon</h2>
      <p class="sub">
        The same CUDA source, compiled with NVRTC and run on a physical RTX 4050 Laptop GPU, compared element by element with the simulator. Reproduce it with <code>node scripts/gpu-parity.js</code>.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Kernel</th><th>Max rel. error</th><th>FLOPs</th><th>Global bytes</th><th>Bank conflicts</th><th></th></tr>
          </thead>
          <tbody>
            {#each parity.results as r}
              <tr>
                <td>{r.name}</td>
                <td class="num">{r.maxRelErr === 0 ? 'exact' : r.maxRelErr.toExponential(1)}</td>
                <td class="num">{r.simFlops.toLocaleString()}</td>
                <td class="num">{r.simGlobalBytes.toLocaleString()}</td>
                <td class="num">{r.simBankConflicts}</td>
                <td class="ok">{r.status === 'PASS' ? 'Match' : r.status}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="block">
    <div class="wrap">
      <h2>Know the limits</h2>
      <div class="limits">
        <div><h3>Not a replacement for hardware</h3><p>It filters logic, memory and collective bugs early. Final tuning, tensor cores and driver behaviour still need silicon.</p></div>
        <div><h3>Refused, never faked</h3><p>Inline PTX, WMMA, dynamic parallelism and CUDA Graphs fail at compile time with a clear error.</p></div>
        <div><h3>Numerics and timing</h3><p>Results match a GPU to about 1e-7. Integers in device memory are exact to 2<sup>24</sup>. Time is a roofline model, within 2 to 3x on small kernels.</p></div>
      </div>
    </div>
  </section>

  <section class="block" id="pricing">
    <div class="wrap">
      <h2>Licensing</h2>
      <p class="sub">Runs entirely client-side, so there is no server cost to pass on.</p>
      <div class="plans">
        <div class="plan">
          <h3>Education</h3>
          <p class="plan__price">${PRICING.education.price}<span>/{PRICING.education.period}</span></p>
          <p class="plan__for">Universities, bootcamps and courses.</p>
          <ul><li>Unlimited learner seats</li><li>Browser playground integration</li><li>Autograding runner</li><li>All GPU profiles</li></ul>
          <a href={LINKS.buyEducation} class="btn btn--ghost btn--full">Get Education</a>
        </div>
        <div class="plan plan--main">
          <h3>Commercial</h3>
          <p class="plan__price">${PRICING.commercial.price}<span>/{PRICING.commercial.period}</span></p>
          <p class="plan__for">Teams and platforms validating CUDA.</p>
          <ul><li>Commercial deployment rights</li><li>JS and WebAssembly bundle</li><li>Offline license key</li><li>1 to 16 GPU clusters</li><li>Race and deadlock detection</li><li>CI integration and updates</li></ul>
          <a href={LINKS.buyCommercial} class="btn btn--primary btn--full">Get Commercial</a>
        </div>
        <div class="plan">
          <h3>Enterprise</h3>
          <p class="plan__price">Custom</p>
          <p class="plan__for">Air-gapped and custom hardware.</p>
          <ul><li>On-premises deployment</li><li>Custom GPU profiles</li><li>Private registry</li><li>SLA and priority support</li></ul>
          <a href={LINKS.contact} class="btn btn--ghost btn--full">Contact us</a>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  .wrap { max-width: 1080px; margin: 0 auto; padding: 0 1.5rem; }
  .wrap--wide { max-width: 1320px; }

  .hero { padding: 7rem 0 4.5rem; }
  .eyebrow { font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; color: var(--gold-primary); margin-bottom: 1.5rem; }
  h1 { font-family: var(--font-sans); font-size: clamp(2.4rem, 5.2vw, 4.1rem); font-weight: 600; line-height: 1.08; letter-spacing: -0.035em; color: #fff; }
  .lead { max-width: 56ch; margin-top: 1.75rem; font-size: 1.12rem; line-height: 1.7; color: var(--text-secondary); }
  .actions { display: flex; gap: .75rem; margin-top: 2.25rem; }
  .facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-top: 4.5rem; padding-top: 2rem; border-top: 1px solid var(--line); }
  .facts dt { font-size: 1.7rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; }
  .facts dd { margin-top: .25rem; font-size: .84rem; color: var(--text-muted); line-height: 1.45; }

  .block { padding: 5rem 0; }
  h2 { font-size: clamp(1.6rem, 3vw, 2.15rem); font-weight: 600; letter-spacing: -0.03em; color: #fff; }
  .sub { margin-top: .85rem; max-width: 62ch; color: var(--text-secondary); line-height: 1.7; }
  .sub code { font-family: var(--font-mono); font-size: .85em; color: var(--gold-light); }

  .ide { border: 1px solid var(--line); border-radius: 14px; background: #050505; overflow: hidden; }
  .ide__bar { display: flex; gap: .6rem; align-items: center; padding: .75rem; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
  .ide__blurb { padding: .6rem 1rem; font-size: .82rem; color: var(--text-muted); border-bottom: 1px solid var(--line); }
  .field { background: #0c0c0c; color: var(--text-primary); border: 1px solid var(--line); border-radius: 8px; padding: .5rem .7rem; font-size: .83rem; }
  .field--grow { flex: 1; min-width: 200px; }
  .field:focus { outline: none; border-color: var(--gold-border); }
  .ide__body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .ide__editor { border-right: 1px solid var(--line); min-width: 0; background: #000; }
  .ide__out { min-width: 0; display: flex; flex-direction: column; background: #000; }
  .tabs { display: flex; align-items: center; gap: .25rem; padding: 0 .5rem; border-bottom: 1px solid var(--line); height: 2.9rem; }
  .tabs button { background: none; border: 0; color: var(--text-muted); font-size: .83rem; padding: .45rem .75rem; border-radius: 6px; cursor: pointer; display: inline-flex; align-items: center; gap: .4rem; }
  .tabs button:hover { color: #fff; }
  .tabs button.on { color: #fff; background: rgba(255,255,255,.07); }
  .tabs__link { margin-left: auto; font-size: .8rem; color: var(--text-muted); padding: .4rem .6rem; }
  .tabs__link:hover { color: var(--gold-light); }
  .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
  .dot--ok { background: var(--success); } .dot--bad { background: var(--danger); } .dot--warn { background: var(--warning); }
  .ide__pane { overflow: auto; max-height: 560px; flex: 1; }
  .console { padding: 1rem; font-family: var(--font-mono); font-size: .8rem; }
  .console pre { white-space: pre-wrap; color: var(--text-secondary); line-height: 1.6; }
  .console__errors { margin-bottom: 1rem; padding: .8rem 1rem; border: 1px solid var(--danger-border); background: var(--danger-bg); border-radius: 8px; color: #fca5a5; }
  .console__errors b { display: block; margin-bottom: .35rem; font-family: var(--font-sans); }
  .console__errors p { margin-top: .3rem; }
  .banks { padding: 1rem; }

  .pillars { display: grid; grid-template-columns: repeat(4, 1fr); margin-top: 3rem; border-top: 1px solid var(--line); }
  .pillar { padding: 1.75rem 1.5rem 0 0; }
  .pillar + .pillar { padding-left: 1.5rem; border-left: 1px solid var(--line); }
  .pillar__n { font-family: var(--font-mono); font-size: .75rem; color: var(--gold-primary); }
  .pillar h3 { margin: .8rem 0 .5rem; font-size: 1.05rem; font-weight: 600; color: #fff; }
  .pillar p { font-size: .9rem; line-height: 1.65; color: var(--text-muted); }

  .table-wrap { margin-top: 2.25rem; overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: .86rem; }
  th { text-align: left; font-weight: 500; font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); padding: .85rem 1rem; border-bottom: 1px solid var(--line); }
  td { padding: .8rem 1rem; border-bottom: 1px solid rgba(255,255,255,.05); color: var(--text-secondary); }
  tr:last-child td { border-bottom: 0; }
  td.num { font-family: var(--font-mono); font-size: .8rem; text-align: right; color: var(--text-primary); }
  th:nth-child(n+2):not(:last-child) { text-align: right; }
  td.ok { color: var(--success); font-size: .8rem; text-align: right; }

  .limits { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.5rem; margin-top: 2.5rem; }
  .limits h3 { font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: .55rem; }
  .limits p { font-size: .92rem; line-height: 1.7; color: var(--text-muted); }

  .plans { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2.5rem; align-items: stretch; }
  .plan { display: flex; flex-direction: column; padding: 1.75rem; border: 1px solid var(--line); border-radius: 14px; background: #060606; }
  .plan--main { border-color: var(--gold-border); background: linear-gradient(180deg, rgba(212,175,55,.06), #060606 55%); }
  .plan h3 { font-size: .95rem; font-weight: 500; color: var(--text-secondary); }
  .plan__price { font-size: 2.4rem; font-weight: 600; color: #fff; letter-spacing: -0.03em; margin: .6rem 0 .25rem; }
  .plan__price span { font-size: .9rem; font-weight: 400; color: var(--text-muted); margin-left: .15rem; }
  .plan__for { font-size: .88rem; color: var(--text-muted); }
  .plan ul { list-style: none; margin: 1.5rem 0; display: flex; flex-direction: column; gap: .6rem; flex: 1; }
  .plan li { font-size: .88rem; color: var(--text-secondary); padding-left: 1.1rem; position: relative; }
  .plan li::before { content: ''; position: absolute; left: 0; top: .55em; width: 5px; height: 5px; border-radius: 50%; background: var(--gold-primary); }

  @media (max-width: 920px) {
    .ide__body { grid-template-columns: 1fr; }
    .ide__editor { border-right: 0; border-bottom: 1px solid var(--line); }
    .facts, .pillars { grid-template-columns: repeat(2, 1fr); }
    .pillar:nth-child(3) { border-left: 0; padding-left: 0; }
    .limits, .plans { grid-template-columns: 1fr; }
    .hero { padding-top: 4.5rem; }
  }
</style>
