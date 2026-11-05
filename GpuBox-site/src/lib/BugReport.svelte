<script>
  import { CONTACT_EMAIL } from './config.js';

  export let open = false;
  export let code = '';
  export let profile = '';
  export let devices = 1;
  export let errors = [];
  export let output = '';

  let message = '';
  let email = '';
  let website = ''; // honeypot
  let includeCode = true;
  let images = [];
  let status = 'idle'; // idle | sending | sent | failed
  let mailHref = '';
  let problem = '';
  let dropping = false;

  function close() { open = false; if (status === 'sent') { message = ''; images = []; status = 'idle'; } }

  // Downscale to keep the email small: max 1400px, JPEG 0.82
  function shrink(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => {
          const k = Math.min(1, 1400 / Math.max(img.width, img.height));
          const c = document.createElement('canvas');
          c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
          c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
          resolve(c.toDataURL('image/jpeg', 0.82));
        };
        img.onerror = () => resolve(null);
        img.src = reader.result;
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  }

  async function addFiles(list) {
    for (const f of Array.from(list || [])) {
      if (!f.type.startsWith('image/') || images.length >= 3) continue;
      const d = await shrink(f);
      if (d) images = [...images, d];
    }
  }
  function onPaste(e) {
    const files = Array.from(e.clipboardData?.files || []);
    if (files.length) { e.preventDefault(); addFiles(files); }
  }
  function onDrop(e) { e.preventDefault(); dropping = false; addFiles(e.dataTransfer?.files); }

  async function send() {
    problem = '';
    if (message.trim().length < 5) { problem = 'Please describe what went wrong.'; return; }
    status = 'sending';
    const body = {
      message, email, website, images,
      code: includeCode ? code : '',
      context: { profile, devices, errors, output: includeCode ? output : '', page: location.href.split('#')[0], ua: navigator.userAgent }
    };
    try {
      const r = await fetch('/api/bug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await r.json().catch(() => ({}));
      if (r.ok && j.ok) { status = 'sent'; return; }
      throw new Error(j.error || 'failed');
    } catch (e) {
      // Never open the visitor's mail app on its own: keep their text and offer it as an explicit choice.
      mailHref = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent('[Gpubox bug] ' + message.split('
')[0].slice(0, 70))}&body=${encodeURIComponent([message, '', 'GPU: ' + profile + ', cluster: ' + devices, errors?.length ? 'Errors: ' + errors.join(' | ') : ''].join('
').slice(0, 1500))}`;
      status = 'failed';
    }
  }
</script>

<svelte:window on:keydown={(e) => open && e.key === 'Escape' && close()} />

{#if open}
  <div class="scrim" on:click|self={close} role="presentation">
    <div class="dlg" role="dialog" aria-modal="true" aria-labelledby="bug-title">
      <button class="x" on:click={close} aria-label="Close">×</button>

      {#if status === 'sent'}
        <div class="done">
          <h3 id="bug-title">Thank you</h3>
          <p>Your report reached me directly. Every bug you send makes the engine better.</p>
          <button class="btn btn--primary btn--sm" on:click={close}>Close</button>
        </div>
      {:else if status === 'failed'}
        <div class="done">
          <h3 id="bug-title">Could not send right now</h3>
          <p>The report service is not reachable. Your text is kept. You can try again, or send it by email instead.</p>
          <div class="row"><button class="btn btn--ghost btn--sm" on:click={() => (status = 'idle')}>Try again</button><a class="btn btn--primary btn--sm" href={mailHref}>Send by email</a></div>
        </div>
      {:else}
        <h3 id="bug-title">Report a bug</h3>
        <p class="lead">Tell me what you expected and what happened. A screenshot helps a lot. It goes straight to the founder.</p>

        <label for="bug-msg">What went wrong?</label>
        <textarea id="bug-msg" rows="5" bind:value={message} on:paste={onPaste} placeholder="e.g. My softmax kernel gives NaN when cols > 256"></textarea>

        <label>Screenshot <span class="opt">(optional, up to 3; paste, drop or choose)</span></label>
        <div class="drop {dropping ? 'drop--on' : ''}" on:dragover|preventDefault={() => (dropping = true)} on:dragleave={() => (dropping = false)} on:drop={onDrop}>
          {#each images as im, i}
            <div class="thumb"><img src={im} alt="Attached screenshot {i + 1}" /><button on:click={() => (images = images.filter((_, j) => j !== i))} aria-label="Remove screenshot {i + 1}">×</button></div>
          {/each}
          {#if images.length < 3}
            <label class="pick">Add image<input type="file" accept="image/*" multiple on:change={(e) => addFiles(e.target.files)} /></label>
          {/if}
        </div>

        <label for="bug-mail">Your email <span class="opt">(optional, so I can reply)</span></label>
        <input id="bug-mail" type="email" bind:value={email} placeholder="you@example.com" autocomplete="email" />
        <input class="hp" tabindex="-1" autocomplete="off" aria-hidden="true" bind:value={website} />

        <label class="chk"><input type="checkbox" bind:checked={includeCode} /> Include my code, GPU choice and run output</label>

        {#if problem}<p class="err">{problem}</p>{/if}
        <div class="row">
          <button class="btn btn--ghost btn--sm" on:click={close}>Cancel</button>
          <button class="btn btn--primary btn--sm" on:click={send} disabled={status === 'sending'}>{status === 'sending' ? 'Sending' : 'Send bug'}</button>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .scrim { position: fixed; inset: 0; z-index: 200; background: rgba(0,0,0,.7); backdrop-filter: blur(4px); display: grid; place-items: center; padding: 1rem; }
  .dlg { position: relative; width: min(520px, 100%); max-height: 92vh; overflow: auto; background: #0a0a0a; border: 1px solid var(--line, rgba(255,255,255,.1)); border-radius: 16px; padding: 1.6rem; box-shadow: 0 40px 100px rgba(0,0,0,.8); }
  h3 { font-size: 1.2rem; font-weight: 600; color: #fff; }
  .lead { margin: .4rem 0 1.1rem; font-size: .88rem; color: var(--text-muted); line-height: 1.6; }
  label { display: block; margin: .9rem 0 .35rem; font-size: .8rem; color: var(--text-secondary); }
  .opt { color: var(--text-muted); }
  textarea, input[type='email'] { width: 100%; background: #000; color: var(--text-primary); border: 1px solid var(--line, rgba(255,255,255,.1)); border-radius: 8px; padding: .65rem .75rem; font-size: .88rem; font-family: inherit; resize: vertical; }
  textarea:focus, input[type='email']:focus { outline: none; border-color: var(--gold-border); }
  .drop { display: flex; flex-wrap: wrap; gap: .5rem; padding: .6rem; border: 1px dashed var(--line, rgba(255,255,255,.18)); border-radius: 10px; min-height: 64px; align-items: center; }
  .drop--on { border-color: var(--gold-primary); background: rgba(212,175,55,.06); }
  .thumb { position: relative; width: 72px; height: 56px; }
  .thumb img { width: 100%; height: 100%; object-fit: cover; border-radius: 6px; border: 1px solid var(--line, rgba(255,255,255,.1)); }
  .thumb button { position: absolute; top: -6px; right: -6px; width: 20px; height: 20px; border-radius: 50%; border: 0; background: #fff; color: #000; cursor: pointer; line-height: 1; }
  .pick { margin: 0; padding: .45rem .8rem; border-radius: 8px; border: 1px solid var(--line, rgba(255,255,255,.14)); color: var(--text-secondary); font-size: .82rem; cursor: pointer; }
  .pick:hover { color: #fff; border-color: rgba(255,255,255,.3); }
  .pick input { display: none; }
  .chk { display: flex; align-items: center; gap: .5rem; margin-top: 1rem; }
  .hp { position: absolute; left: -9999px; width: 1px; height: 1px; opacity: 0; }
  .err { margin-top: .8rem; font-size: .82rem; color: #fca5a5; }
  .row { display: flex; justify-content: flex-end; gap: .6rem; margin-top: 1.3rem; }
  .x { position: absolute; top: .7rem; right: .9rem; width: 32px; height: 32px; background: none; border: 0; color: var(--text-muted); font-size: 1.5rem; cursor: pointer; }
  .x:hover { color: #fff; }
  .done { padding: .5rem 0; } .done p { margin: .6rem 0 1.2rem; color: var(--text-secondary); font-size: .92rem; line-height: 1.65; }
</style>
