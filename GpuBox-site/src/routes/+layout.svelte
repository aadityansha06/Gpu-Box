<script>
  import '../app.css';
</script>

<div class="site-layout">
  <header class="site-header">
    <div class="container site-header__inner">
      <a href="/" class="site-logo">
        <span class="site-logo__mark"></span>
        <span class="site-logo__text">GPUBOX</span>
        <span class="site-logo__badge">v1.0.0</span>
      </a>

      <nav class="site-nav">
        <a href="/" class="site-nav__link">Overview</a>
        <a href="/playground" class="site-nav__link">Playground</a>
        <a href="/docs" class="site-nav__link">Docs</a>
        <a href="/benchmarks" class="site-nav__link">Benchmarks</a>
        <a href="/#pricing" class="site-nav__link">Pricing</a>
      </nav>

      <div class="site-header__actions">
        <a href="https://github.com/aadityansha06/Gpu-Box" target="_blank" rel="noreferrer" class="site-nav__icon-link" title="GitHub Repository">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
          </svg>
        </a>
        <a href="/playground" class="btn btn--primary btn--sm">Try Live</a>
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
          <span class="site-logo__mark"></span>
          <span class="site-logo__text">GPUBOX</span>
        </div>
        <p class="site-footer__desc">
          Browser-based CUDA and multi-GPU simulation and pre-validation library. Validate supported CUDA logic, 32-bank memory hazards, and 1 to 16 GPU cluster algorithms client-side before spending budget on hardware.
        </p>
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
            <li><a href="https://github.com/aadityansha06/Gpu-Box" target="_blank" rel="noreferrer">GitHub Repository</a></li>
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
    background: rgba(6, 8, 13, 0.88);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border-subtle);
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
    gap: 0.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
  }

  .site-logo__mark {
    width: 13px;
    height: 13px;
    background: #10b981;
    border-radius: 3px;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
  }

  .site-logo__text {
    font-size: 1.15rem;
    color: #ffffff;
    font-weight: 800;
  }

  .site-logo__badge {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted);
    background: rgba(255, 255, 255, 0.06);
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
    transition: color 0.15s ease;
  }

  .site-nav__link:hover {
    color: #ffffff;
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
    color: #ffffff;
  }

  .site-main {
    flex: 1;
  }

  /* Footer */
  .site-footer {
    background: #040508;
    border-top: 1px solid var(--border-subtle);
    padding: 4.5rem 0 2rem;
  }

  .site-footer__inner {
    display: grid;
    grid-template-columns: 2fr 1.5fr;
    gap: 3rem;
    padding-bottom: 3rem;
    border-bottom: 1px solid var(--border-subtle);
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
