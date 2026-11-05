<script>
  export let size = 32;
</script>

<!-- Mark: a GPU die (rounded square with pins) holding a 3x3 grid of threads; the lit diagonal reads as a warp. -->
<svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gpubox">
  <g stroke="#d4af37" stroke-width="1.4" stroke-linecap="round" opacity=".55">
    <path d="M10 2.5v3M16 2.5v3M22 2.5v3M10 26.5v3M16 26.5v3M22 26.5v3M2.5 10h3M2.5 16h3M2.5 22h3M26.5 10h3M26.5 16h3M26.5 22h3" />
  </g>
  <rect x="5.5" y="5.5" width="21" height="21" rx="5" fill="#0b0b0b" stroke="#d4af37" stroke-width="1.5" />
  <g fill="#d4af37">
    <circle cx="11.5" cy="11.5" r="1.7" />
    <circle cx="16" cy="16" r="1.7" />
    <circle cx="20.5" cy="20.5" r="1.7" />
  </g>
  <g fill="#d4af37" opacity=".28">
    <circle cx="16" cy="11.5" r="1.7" />
    <circle cx="20.5" cy="11.5" r="1.7" />
    <circle cx="11.5" cy="16" r="1.7" />
    <circle cx="20.5" cy="16" r="1.7" />
    <circle cx="11.5" cy="20.5" r="1.7" />
    <circle cx="16" cy="20.5" r="1.7" />
  </g>
</svg>
