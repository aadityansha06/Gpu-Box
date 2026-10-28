# Gpubox Versioning & Stability Policy

Gpubox follows Semantic Versioning (SemVer 2.0.0).

## Versioning Rules

- **Major (X.0.0):** Breaking changes to the public TypeScript API (`index.ts`), modification of the `RunResult` JSON schema version, or removal of supported CUDA constructs.
- **Minor (1.X.0):** Addition of new GPU hardware profiles, new cluster topologies, additional NCCL collectives, or new validation harness modes without breaking existing callers.
- **Patch (1.0.X):** Bug fixes, diagnostic accuracy improvements, performance optimizations, and documentation updates.

## Stability Tiers

- **Stable:**
  - Public headless SDK (`createSimulator`, `compile`, `analyze`, `run`)
  - Multi-GPU collective runtimes (1 to 16 virtual GPUs)
  - `gpu-profiles.js` hardware registry
  - Validation harness (`validateOutput`)
- **Experimental:**
  - LLM multi-node analytical inference presets (labeled `[EXPERIMENTAL - analytical model, not validated]`)
  - Web Worker streaming transports

## Result Schema Versioning

All `RunResult` JSON objects carry `schemaVersion: "1.0.0"` and `engineVersion: "1.0.0"` ensuring automated downstream CI dashboards and storage systems can safely parse and migrate run reports.
