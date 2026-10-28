/**
 * api.cpp
 * ───────
 * The main entry point for the WASM build.
 * Exports functions callable from JavaScript via Emscripten Embind.
 *
 * Exported API:
 *   compile(source: string): CompileResult
 *   run(source: string, problemId: string): RunResult
 *   runMultiGPU(source: string, numGpus: number): RunResult
 *   getAllReduceModel(nGpus: number, countFloats: number): string (JSON)
 *   reset(): void
 *
 * Flow (for a single run):
 *   1. rewrite_cuda_source(source) → rewritten C++ string
 *   2. Student code is compiled into the same WASM binary at build time
 *      (see build system). At runtime, the student's kernel functions are
 *      registered via __sim_register_kernel().
 *   3. __sim_launch_impl() runs each block via the fiber scheduler.
 *   4. Results aggregated from counters + detectors → JSON.
 *
 * Note: In the WASM build, "compilation" of student code happens server-side
 * OR via a pre-compiled kernel registry. For pure client-side MVP, we use
 * a JavaScript-side compile step that calls clang-wasm (TinyCC or similar)
 * and links against our runtime. See js/src/loader.ts for details.
 */

#include "api.h"
#include "runtime.h"
#include "counters.h"
#include "detectors.h"
#include "multi_gpu.h"
#include "rewriter.h"
#include "../include/cuda_sim.h"
#include <string>
#include <sstream>
#include <cstdio>
#include <functional>
#include <unordered_map>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
using namespace emscripten;
#endif

// ─── Kernel launch bridge ─────────────────────────────────────────────────────
// The rewriter transforms <<<>>> into __sim_launch_kernel(name, grid, block, shmem, lambda).
// This template captures the lambda and converts it to the void(*)(void*) form.

template<typename Fn>
static void launch_trampoline(void* ctx) {
  (*reinterpret_cast<Fn*>(ctx))();
}

template<typename Fn>
void __sim_launch_kernel(const char* name, dim3 grid, dim3 block,
                          size_t shmem, Fn fn) {
  __sim_launch_impl(name, grid, block, shmem,
    &launch_trampoline<Fn>,
    reinterpret_cast<void*>(&fn));
}

// ─── Run result struct ────────────────────────────────────────────────────────
struct SimRunResult {
  bool        passed;
  bool        compile_error;
  std::string error_message;
  std::string counters_json;
  std::string diagnostics_json;
  std::string output;         // captured stdout from device printf
};

// ─── Global output capture ────────────────────────────────────────────────────
static std::string g_captured_output;

// ─── Main sim API ─────────────────────────────────────────────────────────────

void sim_reset() {
  counters_reset();
  detectors_reset();
  sim_runtime_reset();
  g_captured_output.clear();
}

std::string sim_get_counters_json() {
  return counters_to_json();
}

std::string sim_get_diagnostics_json() {
  return detectors_to_json();
}

std::string sim_get_allreduce_model(int n_gpus, int count_floats) {
  return multi_gpu_allreduce_model_json(n_gpus, (size_t)count_floats);
}

bool sim_has_errors() {
  return detectors_has_fatal();
}

// ─── Full result as JSON ──────────────────────────────────────────────────────
std::string sim_result_to_json(bool passed) {
  std::ostringstream o;
  o << "{\n";
  o << "  \"passed\": "       << (passed ? "true" : "false") << ",\n";
  o << "  \"hasErrors\": "    << (sim_has_errors() ? "true" : "false") << ",\n";
  o << "  \"counters\": "     << counters_to_json()    << ",\n";
  o << "  \"diagnostics\": "  << detectors_to_json()   << ",\n";
  o << "  \"output\": \""     << g_captured_output     << "\"\n";
  o << "}";
  return o.str();
}

// ─── Rewrite API (exposed to JS for client-side tooling) ─────────────────────
std::string sim_rewrite_source(const std::string& source) {
  try {
    return rewrite_cuda_source(source);
  } catch (const std::exception& e) {
    std::ostringstream o;
    o << "/* REWRITE_ERROR: " << e.what() << " */\n" << source;
    return o.str();
  }
}

// ─── Emscripten bindings ──────────────────────────────────────────────────────
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(cuda_sim) {
  function("simReset",            &sim_reset);
  function("simGetCounters",      &sim_get_counters_json);
  function("simGetDiagnostics",   &sim_get_diagnostics_json);
  function("simRewriteSource",    &sim_rewrite_source);
  function("simHasErrors",        &sim_has_errors);
  function("simResultToJson",     &sim_result_to_json);
  function("simGetAllReduceModel",&sim_get_allreduce_model);
  // simMultiGpuInit and per-GPU launch exposed in multi_gpu bindings
}
#endif
