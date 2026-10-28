<script>
  export let size = 32;
  export let showText = false;
</script>

<div class="gpubox-logo-wrap">
  <svg
    class="gpubox-logo-svg"
    width={size}
    height={size}
    viewBox="0 0 48 48"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <!-- Luxury Gold Gradients -->
      <linearGradient id="gpuboxGold" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#fff2be" />
        <stop offset="45%" stop-color="#d4af37" />
        <stop offset="100%" stop-color="#91721d" />
      </linearGradient>

      <linearGradient id="gpuboxGoldLight" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#fdf0cd" />
        <stop offset="100%" stop-color="#d4af37" />
      </linearGradient>

      <linearGradient id="dieGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#14141c" />
        <stop offset="100%" stop-color="#050508" />
      </linearGradient>

      <!-- Glow Filter -->
      <filter id="goldGlow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="1.5" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>

    <!-- Outer 3D Isometric "Box" Frame -->
    <!-- Top Face of Box -->
    <polygon
      points="24,4 43,14 24,23 5,14"
      fill="rgba(212, 175, 55, 0.06)"
      stroke="url(#gpuboxGold)"
      stroke-width="1.4"
      stroke-linejoin="round"
    />

    <!-- Left Face of Box -->
    <polygon
      points="5,14 24,23 24,43 5,34"
      fill="rgba(212, 175, 55, 0.03)"
      stroke="url(#gpuboxGold)"
      stroke-width="1.4"
      stroke-linejoin="round"
    />

    <!-- Right Face of Box -->
    <polygon
      points="43,14 24,23 24,43 43,34"
      fill="rgba(212, 175, 55, 0.08)"
      stroke="url(#gpuboxGold)"
      stroke-width="1.4"
      stroke-linejoin="round"
    />

    <!-- Internal Box Isometric Depth Lines (Air-gapped Chamber) -->
    <line x1="24" y1="4" x2="24" y2="15" stroke="rgba(212, 175, 55, 0.25)" stroke-width="0.8" stroke-dasharray="2 2" />

    <!-- FLOATING GPU DIE INSIDE THE BOX -->
    <!-- GPU Substrate Package (Gold & Obsidian) -->
    <polygon
      points="24,15 36,21 24,27 12,21"
      fill="url(#dieGrad)"
      stroke="url(#gpuboxGoldLight)"
      stroke-width="1.2"
      filter="url(#goldGlow)"
    />

    <!-- GPU Substrate Depth Layer -->
    <polygon
      points="12,21 24,27 24,29.5 12,23.5"
      fill="#1c170a"
      stroke="url(#gpuboxGold)"
      stroke-width="0.8"
    />
    <polygon
      points="36,21 24,27 24,29.5 36,23.5"
      fill="#29220d"
      stroke="url(#gpuboxGold)"
      stroke-width="0.8"
    />

    <!-- GPU Silicon Die (Active SM Core Region) -->
    <polygon
      points="24,17.5 32,21.5 24,25.5 16,21.5"
      fill="#0c0c12"
      stroke="#f5d77f"
      stroke-width="0.9"
    />

    <!-- Streaming Multiprocessor (SM) Grid on Silicon Die -->
    <!-- SM Quadrant 1 -->
    <polygon points="24,18.8 27.2,20.4 24,21.8 20.8,20.4" fill="rgba(212, 175, 55, 0.4)" stroke="#fce59f" stroke-width="0.5" />
    
    <!-- Micro Interconnect Traces (Golden Bus lines extending from GPU die to Box) -->
    <line x1="12" y1="21" x2="6.5" y2="18.5" stroke="#f5d77f" stroke-width="0.75" opacity="0.8" />
    <line x1="36" y1="21" x2="41.5" y2="18.5" stroke="#f5d77f" stroke-width="0.75" opacity="0.8" />
    <line x1="24" y1="29.5" x2="24" y2="39" stroke="#f5d77f" stroke-width="0.75" opacity="0.8" />

    <!-- "GPU" Silicon Engraving Mark in Center -->
    <text
      x="24"
      y="22.2"
      text-anchor="middle"
      font-size="3.6"
      font-weight="900"
      font-family="system-ui, -apple-system, sans-serif"
      fill="#ffffff"
      letter-spacing="0.5"
    >GPU</text>

    <!-- Core Center Energy Sparkle -->
    <circle cx="24" cy="27" r="1" fill="#fff5cc" filter="url(#goldGlow)" />
  </svg>

  {#if showText}
    <span class="gpubox-logo-text">GPUBOX</span>
  {/if}
</div>

<style>
  .gpubox-logo-wrap {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    vertical-align: middle;
  }

  .gpubox-logo-svg {
    display: block;
    filter: drop-shadow(0 0 10px rgba(212, 175, 55, 0.4));
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), filter 0.25s ease;
  }

  .gpubox-logo-wrap:hover .gpubox-logo-svg {
    transform: scale(1.08) translateY(-1px);
    filter: drop-shadow(0 0 16px rgba(212, 175, 55, 0.7));
  }

  .gpubox-logo-text {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.04em;
    font-family: var(--font-hero, var(--font-sans));
  }
</style>
