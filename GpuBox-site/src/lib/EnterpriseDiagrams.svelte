<!-- src/lib/EnterpriseDiagrams.svelte -->
<!-- Handcrafted Gold & Black SVGs for Enterprise B2B Architectural Visualizations -->
<script>
  export let diagram = 'architecture'; // 'architecture' | 'cluster' | 'security'
</script>

{#if diagram === 'architecture'}
  <!-- Silicon Micro-Architecture & SIMT Pipeline SVG -->
  <div class="svg-wrapper">
    <svg viewBox="0 0 800 320" fill="none" xmlns="http://www.w3.org/2000/svg" class="enterprise-svg">
      <defs>
        <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#fce59f" />
          <stop offset="50%" stop-color="#d4af37" />
          <stop offset="100%" stop-color="#9e7d23" />
        </linearGradient>
        <linearGradient id="boxBg" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#14141d" />
          <stop offset="100%" stop-color="#09090e" />
        </linearGradient>
        <filter id="goldGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      <!-- Outer Die Shroud -->
      <rect x="20" y="20" width="760" height="280" rx="12" fill="url(#boxBg)" stroke="rgba(212, 175, 55, 0.25)" stroke-width="1.5" />
      <text x="45" y="52" fill="#d4af37" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" letter-spacing="0.1em">GPUBOX VIRTUAL ARCHITECTURE DIE &bull; SIMT EXECUTION PIPELINE</text>

      <!-- Stage 1: CUDA C++ Input / AST Parser -->
      <g transform="translate(45, 80)">
        <rect width="180" height="180" rx="8" fill="#0d0d15" stroke="rgba(212, 175, 55, 0.3)" stroke-width="1.2" />
        <rect x="12" y="16" width="32" height="32" rx="6" fill="rgba(212, 175, 55, 0.12)" stroke="#d4af37" stroke-width="1" />
        <path d="M22 36L28 26L34 36" stroke="#fce59f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        <text x="54" y="32" fill="#faf8f5" font-family="'Inter', sans-serif" font-size="13" font-weight="700">CUDA Transpiler</text>
        <text x="54" y="46" fill="#807a6f" font-family="'Inter', sans-serif" font-size="10">Pure Client-Side AST</text>

        <!-- Inner items -->
        <g transform="translate(14, 65)">
          <rect width="152" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; Dynamic Kernel AST</text>
        </g>
        <g transform="translate(14, 95)">
          <rect width="152" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; Zero-Copy Transpile</text>
        </g>
        <g transform="translate(14, 125)">
          <rect width="152" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; Sanitizer Injection</text>
        </g>
      </g>

      <!-- Connecting Arrow 1 -->
      <path d="M225 170 H275" stroke="#d4af37" stroke-width="2" stroke-dasharray="4 4" />
      <polygon points="275,166 283,170 275,174" fill="#d4af37" />

      <!-- Stage 2: SIMT Lockstep Warp Scheduler -->
      <g transform="translate(285, 80)">
        <rect width="210" height="180" rx="8" fill="#0d0d15" stroke="rgba(212, 175, 55, 0.4)" stroke-width="1.5" />
        <rect x="14" y="16" width="32" height="32" rx="6" fill="rgba(212, 175, 55, 0.15)" stroke="#d4af37" stroke-width="1" />
        <!-- Warp Grid Icon -->
        <line x1="22" y1="26" x2="38" y2="26" stroke="#fce59f" stroke-width="1.5"/>
        <line x1="22" y1="32" x2="38" y2="32" stroke="#fce59f" stroke-width="1.5"/>
        <line x1="22" y1="38" x2="38" y2="38" stroke="#fce59f" stroke-width="1.5"/>
        <text x="56" y="32" fill="#faf8f5" font-family="'Inter', sans-serif" font-size="13" font-weight="700">Warp SIMT Core</text>
        <text x="56" y="46" fill="#d4af37" font-family="'Inter', sans-serif" font-size="10">Lockstep SIMD Scheduler</text>

        <!-- Simulated Warps -->
        <g transform="translate(14, 65)">
          <rect width="182" height="24" rx="4" fill="rgba(212, 175, 55, 0.06)" stroke="rgba(212, 175, 55, 0.2)" />
          <text x="10" y="16" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10">Warp 0: 32 Threads Active</text>
        </g>
        <g transform="translate(14, 95)">
          <rect width="182" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; Branch Divergence Trap</text>
        </g>
        <g transform="translate(14, 125)">
          <rect width="182" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; __syncthreads() Barrier</text>
        </g>
      </g>

      <!-- Connecting Arrow 2 -->
      <path d="M495 170 H535" stroke="#d4af37" stroke-width="2" stroke-dasharray="4 4" />
      <polygon points="535,166 543,170 535,174" fill="#d4af37" />

      <!-- Stage 3: 32-Bank Memory & NCCL Fabric -->
      <g transform="translate(545, 80)">
        <rect width="205" height="180" rx="8" fill="#0d0d15" stroke="rgba(212, 175, 55, 0.3)" stroke-width="1.2" />
        <rect x="14" y="16" width="32" height="32" rx="6" fill="rgba(212, 175, 55, 0.12)" stroke="#d4af37" stroke-width="1" />
        <rect x="23" y="24" width="6" height="6" fill="#d4af37"/>
        <rect x="31" y="24" width="6" height="6" fill="#fce59f"/>
        <rect x="23" y="32" width="6" height="6" fill="#fce59f"/>
        <rect x="31" y="32" width="6" height="6" fill="#d4af37"/>
        <text x="56" y="32" fill="#faf8f5" font-family="'Inter', sans-serif" font-size="13" font-weight="700">32-Bank SRAM</text>
        <text x="56" y="46" fill="#807a6f" font-family="'Inter', sans-serif" font-size="10">Shared & Global Coalescing</text>

        <!-- Bank Matrix visual -->
        <g transform="translate(14, 65)">
          <rect width="177" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; 4-Byte Interleaved Banks</text>
        </g>
        <g transform="translate(14, 95)">
          <rect width="177" height="24" rx="4" fill="rgba(255, 255, 255, 0.03)" />
          <text x="10" y="16" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="10">&bull; Bank Conflict Detector</text>
        </g>
        <g transform="translate(14, 125)">
          <rect width="177" height="24" rx="4" fill="rgba(212, 175, 55, 0.06)" stroke="rgba(212, 175, 55, 0.2)" />
          <text x="10" y="16" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10">&bull; 1-16 GPU NCCL Ring Fabric</text>
        </g>
      </g>
    </svg>
  </div>

{:else if diagram === 'cluster'}
  <!-- Multi-GPU Ring AllReduce SVG Topology -->
  <div class="svg-wrapper">
    <svg viewBox="0 0 600 340" fill="none" xmlns="http://www.w3.org/2000/svg" class="enterprise-svg">
      <defs>
        <linearGradient id="ringGold" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#fce59f" />
          <stop offset="100%" stop-color="#9e7d23" />
        </linearGradient>
      </defs>

      <!-- Center Title -->
      <circle cx="300" cy="170" r="115" stroke="rgba(212, 175, 55, 0.25)" stroke-width="2" stroke-dasharray="6 6" fill="none" />
      <circle cx="300" cy="170" r="50" fill="#0d0d15" stroke="rgba(212, 175, 55, 0.4)" stroke-width="1.5" />
      <text x="300" y="165" text-anchor="middle" fill="#d4af37" font-family="'Inter', sans-serif" font-size="11" font-weight="700">NVLink Ring</text>
      <text x="300" y="180" text-anchor="middle" fill="#c5bfb4" font-family="'JetBrains Mono', monospace" font-size="9">600 GB/s</text>

      <!-- 8 GPU Node Nodes around ring -->
      <!-- Node 0: Top -->
      <g transform="translate(300, 55)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 0</text>
      </g>
      <!-- Node 1: Top-Right -->
      <g transform="translate(380, 88)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 1</text>
      </g>
      <!-- Node 2: Right -->
      <g transform="translate(415, 170)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 2</text>
      </g>
      <!-- Node 3: Bottom-Right -->
      <g transform="translate(380, 252)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 3</text>
      </g>
      <!-- Node 4: Bottom -->
      <g transform="translate(300, 285)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 4</text>
      </g>
      <!-- Node 5: Bottom-Left -->
      <g transform="translate(220, 252)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 5</text>
      </g>
      <!-- Node 6: Left -->
      <g transform="translate(185, 170)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 6</text>
      </g>
      <!-- Node 7: Top-Left -->
      <g transform="translate(220, 88)">
        <circle cx="0" cy="0" r="24" fill="#0e0e16" stroke="#d4af37" stroke-width="1.5" />
        <text x="0" y="4" text-anchor="middle" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">GPU 7</text>
      </g>

      <!-- Label -->
      <text x="300" y="325" text-anchor="middle" fill="#807a6f" font-family="'JetBrains Mono', monospace" font-size="10" letter-spacing="0.05em">
        NCCL ALLREDUCE &bull; 8 / 16 RANK SYNCHRONOUS RING PIPELINE
      </text>
    </svg>
  </div>

{:else if diagram === 'security'}
  <!-- Zero-Data-Egress Security Shield SVG -->
  <div class="svg-wrapper">
    <svg viewBox="0 0 480 260" fill="none" xmlns="http://www.w3.org/2000/svg" class="enterprise-svg">
      <defs>
        <linearGradient id="shieldGold" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#fce59f" />
          <stop offset="50%" stop-color="#d4af37" />
          <stop offset="100%" stop-color="#9e7d23" />
        </linearGradient>
      </defs>

      <!-- Shield Background -->
      <rect x="20" y="20" width="440" height="220" rx="10" fill="#0a0a10" stroke="rgba(212, 175, 55, 0.2)" stroke-width="1.5" />

      <!-- Center Shield Icon -->
      <g transform="translate(80, 75)">
        <path d="M40 10L10 25V65C10 95 40 115 40 115C40 115 70 95 70 65V25L40 10Z" fill="rgba(212, 175, 55, 0.1)" stroke="#d4af37" stroke-width="2" />
        <!-- Lock -->
        <rect x="30" y="55" width="20" height="18" rx="3" fill="#d4af37" />
        <path d="M34 55V47C34 43.7 36.7 41 40 41C43.3 41 46 43.7 46 47V55" stroke="#d4af37" stroke-width="2.2" stroke-linecap="round" />
      </g>

      <text x="180" y="85" fill="#faf8f5" font-family="'Inter', sans-serif" font-size="16" font-weight="700">100% Zero-Data-Egress Sandbox</text>
      <text x="180" y="110" fill="#d4af37" font-family="'JetBrains Mono', monospace" font-size="11">Proprietary Kernel & IP Protection</text>
      <text x="180" y="140" fill="#c5bfb4" font-family="'Inter', sans-serif" font-size="11" width="260">
        All CUDA parsing, transpilation, and kernel simulation run locally in WebAssembly and client JavaScript. Zero code egresses to external servers.
      </text>
      <g transform="translate(180, 180)">
        <rect width="130" height="22" rx="4" fill="rgba(212, 175, 55, 0.1)" stroke="rgba(212, 175, 55, 0.3)" />
        <text x="10" y="15" fill="#fce59f" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="700">SOC2 & HIPAA Compliant</text>
      </g>
    </svg>
  </div>
{/if}

<style>
  .svg-wrapper {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .enterprise-svg {
    width: 100%;
    height: auto;
    max-height: 340px;
    filter: drop-shadow(0 15px 35px rgba(0, 0, 0, 0.7));
  }
</style>
