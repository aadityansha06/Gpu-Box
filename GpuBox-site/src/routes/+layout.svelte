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
        <a href="#playground" class="site-nav__link">Playground</a>
        <a href="#features" class="site-nav__link">Features</a>
        <a href="#benchmarks" class="site-nav__link">Benchmarks</a>
        <a href="#pricing" class="site-nav__link">Pricing</a>
        <a href="https://github.com/aadityansha06/Gpu-Box/blob/main/docs/USAGE.md" target="_blank" rel="noreferrer" class="site-nav__link">Docs</a>
      </nav>

      <div class="site-header__actions">
        <a href="#playground" class="btn btn--primary btn--sm">Try Live</a>
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
          Browser-based CUDA and multi-GPU simulation and pre-validation library. Validate supported CUDA logic, memory hazards, and 1 to 16 GPU cluster algorithms before paying for hardware.
        </p>
        <p class="site-footer__copy">
          Copyright &copy; 2026 Aadityansha Verma. All rights reserved. Commercial License.
        </p>
      </div>

      <div class="site-footer__cols">
        <div class="site-footer__col">
          <h4>Documentation</h4>
          <ul>
            <li><a href="https://github.com/aadityansha06/Gpu-Box/blob/main/docs/USAGE.md" target="_blank" rel="noreferrer">User Guide</a></li>
            <li><a href="https://github.com/aadityansha06/Gpu-Box/blob/main/docs/API_REFERENCE.md" target="_blank" rel="noreferrer">API Reference</a></li>
            <li><a href="https://github.com/aadityansha06/Gpu-Box/blob/main/docs/INTEGRATION_GUIDE.md" target="_blank" rel="noreferrer">Integration Guide</a></li>
            <li><a href="https://github.com/aadityansha06/Gpu-Box/blob/main/docs/LIMITATIONS.md" target="_blank" rel="noreferrer">Limitations</a></li>
          </ul>
        </div>
        <div class="site-footer__col">
          <h4>Resources</h4>
          <ul>
            <li><a href="#playground">Interactive Playground</a></li>
            <li><a href="#benchmarks">Hardware Benchmarks</a></li>
            <li><a href="#pricing">Licensing Plans</a></li>
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
    background: rgba(7, 8, 12, 0.85);
    backdrop-filter: blur(12px);
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
    width: 14px;
    height: 14px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border-radius: 3px;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
  }

  .site-logo__text {
    font-size: 1.15rem;
    color: #ffffff;
    font-weight: 800;
  }

  .site-logo__badge {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    background: rgba(255, 255, 255, 0.06);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
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
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
    transition: color 0.15s ease;
  }

  .site-nav__link:hover {
    color: #ffffff;
  }

  .site-main {
    flex: 1;
  }

  /* Footer */
  .site-footer {
    background: #050608;
    border-top: 1px solid var(--border-subtle);
    padding: 4.5rem 0 2rem;
  }

  .site-footer__inner {
    display: grid;
    grid-template-columns: 2fr 1fr;
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
    font-size: 0.9rem;
    max-width: 440px;
    line-height: 1.6;
  }

  .site-footer__copy {
    margin-top: 1.25rem;
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  .site-footer__cols {
    display: flex;
    gap: 3.5rem;
  }

  .site-footer__col h4 {
    font-size: 0.85rem;
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
    font-size: 0.88rem;
    color: var(--text-secondary);
    transition: color 0.15s ease;
  }

  .site-footer__col a:hover {
    color: #ffffff;
  }

  .site-footer__disclaimer {
    padding-top: 2rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.6;
  }
</style>
