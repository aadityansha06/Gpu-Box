<script>
  import '../app.css';
  import { page } from '$app/state';
  import GpuBoxLogo from '../lib/GpuBoxLogo.svelte';
  import { SITE_URL, CONTACT_EMAIL } from '../lib/config.js';

  const META = {
    '/': { d: 'Run CUDA C++ and multi-GPU NCCL code in your browser. See bytes moved, FLOPs, bank conflicts, divergence and deadlocks before you spend GPU time. Verified against a physical RTX 4050.' },
    '/playground': { d: 'Full browser IDE for CUDA C++: 17 runnable examples, 1 to 16 virtual GPUs, GPU report with data movement, compute and hazard findings.' },
    '/docs': { d: 'Gpubox documentation: quickstart, API reference, integration guide, recipes and known limitations for the browser CUDA simulator.' },
    '/benchmarks': { d: 'Gpubox benchmarks and hardware verification: simulator output compared with a physical NVIDIA RTX 4050 across 11 CUDA kernels.' }
  };
  $: path = page.url.pathname.replace(/\/$/, '') || '/';
  $: desc = (META[path] || META['/']).d;
  $: canonical = SITE_URL + (path === '/' ? '/' : path);
  const ld = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'Gpubox',
    applicationCategory: 'DeveloperApplication',
    operatingSystem: 'Web',
    description: 'Browser-based CUDA and multi-GPU simulation and pre-validation engine.',
    url: SITE_URL,
    offers: [
      { '@type': 'Offer', name: 'Education', price: '100', priceCurrency: 'USD' },
      { '@type': 'Offer', name: 'Commercial', price: '150', priceCurrency: 'USD' }
    ]
  });
</script>

<svelte:head>
  <meta name="description" content={desc} />
  <link rel="canonical" href={canonical} />
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Gpubox" />
  <meta property="og:title" content="Gpubox: pre-validate CUDA in the browser" />
  <meta property="og:description" content={desc} />
  <meta property="og:url" content={canonical} />
  <meta property="og:image" content={SITE_URL + '/og.svg'} />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="Gpubox: pre-validate CUDA in the browser" />
  <meta name="twitter:description" content={desc} />
  {@html '<script type="application/ld+json">' + ld + '</' + 'script>'}
</svelte:head>

<div class="site-layout">
  <header class="site-header">
    <div class="container site-header__inner">
      <a href="/" class="site-logo">
        <GpuBoxLogo size={30} />
        <span class="site-logo__text">GPUBOX</span>
      </a>

      <nav class="site-nav">
        <a href="/" class="site-nav__link">Overview</a>
        <a href="/playground" class="site-nav__link">Playground</a>
        <a href="/docs" class="site-nav__link">Docs</a>
        <a href="/benchmarks" class="site-nav__link">Benchmarks</a>
        <a href="/#pricing" class="site-nav__link">Pricing</a>
        <a href={"mailto:" + CONTACT_EMAIL} class="site-nav__link">Contact</a>
      </nav>

      <div class="site-header__actions">

        <a href="/playground" class="btn btn--primary btn--sm">Try it</a>
      </div>
    </div>
  </header>

  <main class="site-main">
    <slot />
  </main>

  <footer class="site-footer">
    <div class="container site-footer__inner">
      <div class="site-footer__brand">
        <div class="site-logo">
          <GpuBoxLogo size={28} />
          <span class="site-logo__text">GPUBOX</span>
        </div>
        <p class="site-footer__desc">
          Pre-validate CUDA and multi-GPU code in the browser, before it costs a GPU hour.</p>
        <p class="site-footer__copy">
          Copyright &copy; 2026 Aadityansha Verma. All rights reserved. Commercial License.
        </p>
      </div>

      <div class="site-footer__cols">
        <div class="site-footer__col">
          <h4>Documentation</h4>
          <ul>
            <li><a href="/docs">Overview & Quickstart</a></li>
            <li><a href="/docs#usage">Usage Guide</a></li>
            <li><a href="/docs#api-reference">API Reference</a></li>
            <li><a href="/docs#integration">Integration Guide</a></li>
            <li><a href="/docs#recipes">Recipes & Patterns</a></li>
            <li><a href="/docs#limitations">Known Limitations</a></li>
          </ul>
        </div>
        <div class="site-footer__col">
          <h4>Platform</h4>
          <ul>
            <li><a href="/playground">Interactive Playground</a></li>
            <li><a href="/benchmarks">Benchmarks & Verification</a></li>
            <li><a href="/#pricing">Licensing Plans</a></li>
            <li><a href={'mailto:' + CONTACT_EMAIL}>{CONTACT_EMAIL}</a></li>
          </ul>
        </div>
      </div>
    </div>

    <div class="container site-footer__disclaimer">
      <p>
        <strong>Trademark Disclaimer:</strong> Gpubox is not affiliated with, sponsored by, or endorsed by NVIDIA Corporation. CUDA, NVIDIA, NVLink, and NVSwitch are registered trademarks or trademarks of NVIDIA Corporation. All trademarks are used strictly for reference purposes to describe software compatibility and architectural profiles.
      </p>
    </div>
  </footer>
</div>

<style>
  .site-layout {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .site-header {
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(0, 0, 0, 0.94);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(212, 175, 55, 0.18);
  }

  .site-header__inner {
    height: 4.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .site-logo {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
  }

  .site-logo__text {
    font-size: 1.15rem;
    color: #ffffff;
    font-weight: 800;
    letter-spacing: 0.02em;
  }

  .site-logo__badge {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--gold-light);
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.25);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-family: var(--font-mono);
  }

  .site-nav {
    display: flex;
    align-items: center;
    gap: 1.75rem;
  }

  @media (max-width: 768px) {
    .site-nav {
      display: none;
    }
  }

  .site-nav__link {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .site-nav__link:hover {
    color: var(--gold-light);
  }

  .site-header__actions {
    display: flex;
    align-items: center;
    gap: 1.25rem;
  }

  .site-nav__icon-link {
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    transition: color 0.15s ease;
  }

  .site-nav__icon-link:hover {
    color: var(--gold-primary);
  }

  .site-main {
    flex: 1;
  }

  /* Footer */
  .site-footer {
    background: #000000;
    border-top: 1px solid rgba(212, 175, 55, 0.18);
    padding: 4.5rem 0 2rem;
  }

  .site-footer__inner {
    display: grid;
    grid-template-columns: 2fr 1.5fr;
    gap: 3rem;
    padding-bottom: 3rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  @media (max-width: 768px) {
    .site-footer__inner {
      grid-template-columns: 1fr;
    }
  }

  .site-footer__desc {
    margin-top: 1rem;
    color: var(--text-secondary);
    font-size: 0.88rem;
    max-width: 440px;
    line-height: 1.6;
  }

  .site-footer__copy {
    margin-top: 1.25rem;
    color: var(--text-muted);
    font-size: 0.78rem;
  }

  .site-footer__cols {
    display: flex;
    gap: 3.5rem;
  }

  @media (max-width: 600px) {
    .site-footer__cols {
      flex-direction: column;
      gap: 2rem;
    }
  }

  .site-footer__col h4 {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #ffffff;
    margin-bottom: 1rem;
  }

  .site-footer__col ul {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .site-footer__col a {
    font-size: 0.85rem;
    color: var(--text-secondary);
    transition: color 0.15s ease;
  }

  .site-footer__col a:hover {
    color: #ffffff;
  }

  .site-footer__disclaimer {
    padding-top: 2rem;
    font-size: 0.75rem;
    color: var(--text-muted);
    line-height: 1.6;
  }
</style>
