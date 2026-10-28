<script>
  import { createEventDispatcher } from 'svelte';
  import { highlightCUDA, highlightJS } from './cuda-highlighter.js';

  export let code = '';
  export let language = 'cuda'; // 'cuda' | 'javascript'
  export let placeholder = '// Write or paste code here...';
  export let readOnly = false;
  export let minHeight = '360px';

  const dispatch = createEventDispatcher();

  let textareaRef;
  let preRef;
  let lines = [1];

  $: {
    const count = (code || '').split('\n').length;
    lines = Array.from({ length: Math.max(1, count) }, (_, i) => i + 1);
  }

  $: highlightedHTML = language === 'javascript'
    ? highlightJS(code)
    : highlightCUDA(code);

  function syncScroll() {
    if (textareaRef && preRef) {
      preRef.scrollTop = textareaRef.scrollTop;
      preRef.scrollLeft = textareaRef.scrollLeft;
    }
  }

  function handleKeydown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      dispatch('run');
      return;
    }

    if (e.key === 'Tab' && !readOnly) {
      e.preventDefault();
      const target = textareaRef;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      code = code.substring(0, start) + '    ' + code.substring(end);
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 4;
        syncScroll();
      }, 0);
    }
  }
</script>

<div class="editor-shell" style="min-height: {minHeight};">
  <div class="editor-gutter">
    {#each lines as num}
      <span class="gutter-num">{num}</span>
    {/each}
  </div>

  <div class="editor-core">
    <!-- Highlighted syntax layer (underneath) -->
    <pre
      bind:this={preRef}
      class="editor-highlight"
      aria-hidden="true"
    ><code class="token-code">{@html highlightedHTML + '\n'}</code></pre>

    <!-- Interactive transparent input textarea (on top) -->
    <textarea
      bind:this={textareaRef}
      bind:value={code}
      class="editor-textarea"
      spellcheck="false"
      autocapitalize="off"
      autocomplete="off"
      readonly={readOnly}
      {placeholder}
      on:scroll={syncScroll}
      on:keydown={handleKeydown}
      on:input={syncScroll}
    ></textarea>
  </div>
</div>

<style>
  .editor-shell {
    position: relative;
    display: flex;
    width: 100%;
    height: 100%;
    background: #000000;
    overflow: hidden;
    font-family: var(--font-mono, 'JetBrains Mono', 'Fira Code', monospace);
    font-size: 0.86rem;
    line-height: 1.6;
  }

  .editor-gutter {
    width: 48px;
    background: #000000;
    border-right: 1px solid rgba(212, 175, 55, 0.15);
    padding: 1rem 0;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    user-select: none;
    flex-shrink: 0;
    overflow: hidden;
  }

  .gutter-num {
    padding-right: 0.75rem;
    color: #444444;
    font-size: 0.75rem;
    height: 1.6em;
    line-height: 1.6;
  }

  .editor-core {
    position: relative;
    flex: 1;
    height: 100%;
    overflow: hidden;
    background: #000000;
  }

  .editor-highlight,
  .editor-textarea {
    position: absolute;
    inset: 0;
    margin: 0;
    padding: 1rem;
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
    tab-size: 4;
    white-space: pre;
    word-break: normal;
    overflow-wrap: normal;
    overflow: auto;
    border: none;
    outline: none;
  }

  .editor-highlight {
    background: transparent;
    pointer-events: none;
    z-index: 1;
    color: #cbd5e1;
  }

  .token-code {
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
  }

  .editor-textarea {
    background: transparent;
    color: transparent;
    caret-color: #d4af37;
    resize: none;
    z-index: 2;
  }

  .editor-textarea::selection {
    background: rgba(212, 175, 55, 0.35);
  }

  /* Prism Dark Theme Token Styles */
  :global(.token.comment),
  :global(.token.prolog),
  :global(.token.doctype),
  :global(.token.cdata) {
    color: #64748b;
    font-style: italic;
  }

  :global(.token.punctuation) {
    color: #94a3b8;
  }

  :global(.token.property),
  :global(.token.tag),
  :global(.token.boolean),
  :global(.token.number),
  :global(.token.constant),
  :global(.token.symbol),
  :global(.token.deleted) {
    color: #f59e0b;
  }

  :global(.token.selector),
  :global(.token.attr-name),
  :global(.token.string),
  :global(.token.char),
  :global(.token.builtin),
  :global(.token.inserted) {
    color: #10b981;
  }

  :global(.token.operator),
  :global(.token.entity),
  :global(.token.url),
  :global(.language-css .token.string),
  :global(.style .token.string) {
    color: #38bdf8;
  }

  :global(.token.atrule),
  :global(.token.attr-value),
  :global(.token.keyword),
  :global(.token.cuda-qualifier) {
    color: #c084fc;
    font-weight: 600;
  }

  :global(.token.function),
  :global(.token.class-name),
  :global(.token.cuda-function) {
    color: #60a5fa;
    font-weight: 600;
  }

  :global(.token.cuda-builtin),
  :global(.token.variable) {
    color: #f43f5e;
    font-weight: 600;
  }

  :global(.token.cuda-type) {
    color: #38bdf8;
    font-weight: 600;
  }

  :global(.token.regex),
  :global(.token.important) {
    color: #fb923c;
  }
</style>
