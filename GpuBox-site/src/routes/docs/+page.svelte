<script>
  import { onMount } from 'svelte';
  import { DOCS, CATEGORIES, getDoc } from '../../lib/docs-data.js';

  let selectedSlug = 'overview';
  let searchQuery = '';

  $: currentDoc = getDoc(selectedSlug);

  $: filteredDocs = searchQuery.trim() === ''
    ? DOCS
    : DOCS.filter(d => 
        d.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        d.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
        d.category.toLowerCase().includes(searchQuery.toLowerCase())
      );

  onMount(() => {
    // Check URL hash on load
    const hash = window.location.hash.replace('#', '');
    if (hash && DOCS.some(d => d.slug === hash)) {
      selectedSlug = hash;
    }

    const handleHash = () => {
      const h = window.location.hash.replace('#', '');
      if (h && DOCS.some(d => d.slug === h)) {
        selectedSlug = h;
      }
    };
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  });

  function selectDoc(slug) {
    selectedSlug = slug;
    window.location.hash = slug;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function getNextDoc() {
    const idx = DOCS.findIndex(d => d.slug === selectedSlug);
    return idx < DOCS.length - 1 ? DOCS[idx + 1] : null;
  }

  function getPrevDoc() {
    const idx = DOCS.findIndex(d => d.slug === selectedSlug);
    return idx > 0 ? DOCS[idx - 1] : null;
  }
</script>

<svelte:head>
  <title>{currentDoc.title} &bull; GPUBOX Documentation</title>
</svelte:head>

<div class="docs-container container-wide">
  <!-- Left Navigation Sidebar -->
  <aside class="docs-sidebar">
    <div class="docs-search-wrap">
      <input
        type="text"
        class="docs-search-input"
        placeholder="Search documentation..."
        bind:value={searchQuery}
      />
    </div>

    <nav class="docs-nav">
      {#each CATEGORIES as cat}
        {@const catDocs = filteredDocs.filter(d => d.category === cat)}
        {#if catDocs.length > 0}
          <div class="nav-group">
            <div class="nav-group__title">{cat}</div>
            <ul class="nav-group__list">
              {#each catDocs as d}
                <li>
                  <button
                    class="nav-link {selectedSlug === d.slug ? 'nav-link--active' : ''}"
                    on:click={() => selectDoc(d.slug)}
                  >
                    {d.title}
                  </button>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      {/each}
    </nav>
  </aside>

  <!-- Main Documentation Content View -->
  <main class="docs-content">
    <div class="doc-header">
      <div class="doc-meta">
        <span class="badge badge--neutral">{currentDoc.category}</span>
        <span class="doc-slug">docs/{currentDoc.slug}.md</span>
      </div>
      <h1 class="doc-title">{currentDoc.title}</h1>
      <p class="doc-summary">{currentDoc.summary}</p>
    </div>

    <article class="docs-prose">
      {@html currentDoc.html}
    </article>

    <!-- Bottom Pagination -->
    <div class="docs-pagination">
      {#if getPrevDoc()}
        {@const prev = getPrevDoc()}
        <button class="page-nav page-nav--prev" on:click={() => selectDoc(prev.slug)}>
          <span class="page-nav__direction">&larr; Previous</span>
          <span class="page-nav__title">{prev.title}</span>
        </button>
      {:else}
        <div></div>
      {/if}

      {#if getNextDoc()}
        {@const next = getNextDoc()}
        <button class="page-nav page-nav--next" on:click={() => selectDoc(next.slug)}>
          <span class="page-nav__direction">Next &rarr;</span>
          <span class="page-nav__title">{next.title}</span>
        </button>
      {/if}
    </div>
  </main>
</div>

<style>
  .docs-container {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 3rem;
    padding-top: 2rem;
    padding-bottom: 5rem;
    min-height: calc(100vh - 4.25rem);
  }

  @media (max-width: 900px) {
    .docs-container {
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }
  }

  /* Sidebar */
  .docs-sidebar {
    position: sticky;
    top: 5.5rem;
    height: calc(100vh - 7rem);
    overflow-y: auto;
    padding-right: 1rem;
    border-right: 1px solid var(--border-subtle);
  }

  @media (max-width: 900px) {
    .docs-sidebar {
      position: static;
      height: auto;
      border-right: none;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 1.5rem;
    }
  }

  .docs-search-wrap {
    margin-bottom: 1.5rem;
  }

  .docs-search-input {
    width: 100%;
    background: #08080d;
    border: 1px solid var(--gold-border-subtle);
    border-radius: var(--radius-md);
    padding: 0.55rem 0.85rem;
    font-size: 0.82rem;
    color: var(--text-primary);
    outline: none;
    transition: border-color 0.15s ease;
  }

  .docs-search-input:focus {
    border-color: var(--gold-primary);
  }

  .docs-nav {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .nav-group__title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
  }

  .nav-group__list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .nav-link {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    padding: 0.4rem 0.65rem;
    font-size: 0.84rem;
    color: var(--text-secondary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .nav-link:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.04);
  }

  .nav-link--active {
    color: var(--gold-light);
    font-weight: 600;
    background: rgba(212, 175, 55, 0.1);
    border-left-color: var(--gold-primary);
  }

  /* Main Content */
  .docs-content {
    max-width: 860px;
    width: 100%;
  }

  .doc-header {
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }

  .doc-meta {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }

  .doc-slug {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .doc-title {
    font-size: 2.25rem;
    margin-bottom: 0.6rem;
    color: #ffffff;
  }

  .doc-summary {
    font-size: 1.05rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  /* Bottom Page Navigation */
  .docs-pagination {
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    gap: 1.5rem;
  }

  .page-nav {
    display: flex;
    flex-direction: column;
    background: #08080d;
    border: 1px solid var(--gold-border-subtle);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.25rem;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s ease;
    min-width: 200px;
  }

  .page-nav:hover {
    border-color: var(--gold-primary);
    background: rgba(212, 175, 55, 0.06);
  }

  .page-nav--next {
    text-align: right;
    margin-left: auto;
  }

  .page-nav__direction {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .page-nav__title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #ffffff;
    margin-top: 0.2rem;
  }
</style>
