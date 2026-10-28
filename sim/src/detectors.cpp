/**
 * detectors.cpp
 * ─────────────
 * Runtime bug detectors:
 *   1. Out-of-bounds (global + shared)
 *   2. Race conditions (randomized fiber schedule, repeat runs)
 *   3. Barrier divergence (threads exiting before __syncthreads)
 *   4. Deadlock (some fibers stuck at barrier, none runnable)
 *   5. Uninitialized reads (poison-byte NaN check)
 *
 * All findings are accumulated as DiagnosticMessage structs and serialized
 * to JSON by api.cpp.
 */

#include "detectors.h"
#include "runtime.h"
#include "../include/cuda_sim.h"
#include <vector>
#include <mutex>
#include <sstream>
#include <cstdio>
#include <cstring>

// ─── Diagnostic message ───────────────────────────────────────────────────────
struct DiagnosticMsg {
  std::string type;     // "oob", "race", "barrier_divergence", "deadlock", "uninit"
  std::string message;  // human-readable
  int block_x, block_y, block_z;
  int thread_x, thread_y, thread_z;
};

static std::vector<DiagnosticMsg> g_diagnostics;
static std::mutex                 g_diag_mutex;
static bool                       g_has_fatal = false;

static void add_diag(DiagnosticMsg&& d) {
  std::lock_guard<std::mutex> lk(g_diag_mutex);
  if (g_diagnostics.size() < 100) { // cap at 100 messages
    g_diagnostics.push_back(std::move(d));
  }
}

// ─── Out-of-bounds ────────────────────────────────────────────────────────────
void detector_check_bounds(const void* addr, int bytes, bool is_store,
                            dim3 block_idx, dim3 thread_idx) {
  auto& tl = __sim_tls::threadIdx_v; // use current TLS
  uintptr_t p = (uintptr_t)addr;
  // We rely on runtime.cpp's sim_query_address
  // (linked together; forward-declared)
  extern AllocationInfo sim_query_address(const void*);
  AllocationInfo info = sim_query_address(addr);

  if (!info.found) {
    DiagnosticMsg d;
    d.type    = "oob";
    d.block_x = (int)block_idx.x; d.block_y = (int)block_idx.y; d.block_z = (int)block_idx.z;
    d.thread_x= (int)thread_idx.x;d.thread_y= (int)thread_idx.y;d.thread_z= (int)thread_idx.z;
    char buf[512];
    snprintf(buf, sizeof(buf),
      "%s of %d bytes at address 0x%zx is outside any allocated device buffer "
      "(block(%d,%d,%d) thread(%d,%d,%d))",
      is_store ? "Write" : "Read", bytes, (size_t)p,
      d.block_x, d.block_y, d.block_z,
      d.thread_x, d.thread_y, d.thread_z);
    d.message = buf;
    add_diag(std::move(d));
    g_has_fatal = true;
    return;
  }

  if (info.in_guard) {
    DiagnosticMsg d;
    d.type    = "oob";
    d.block_x = (int)block_idx.x; d.block_y = (int)block_idx.y; d.block_z = (int)block_idx.z;
    d.thread_x= (int)thread_idx.x;d.thread_y= (int)thread_idx.y;d.thread_z= (int)thread_idx.z;
    char buf[512];
    snprintf(buf, sizeof(buf),
      "%s hit guard region near allocation at 0x%zx (size %zu) — likely off-by-one "
      "(block(%d,%d,%d) thread(%d,%d,%d))",
      is_store ? "Write" : "Read",
      (size_t)info.base_addr, info.alloc_size,
      d.block_x, d.block_y, d.block_z,
      d.thread_x, d.thread_y, d.thread_z);
    d.message = buf;
    add_diag(std::move(d));
    g_has_fatal = true;
  }
}

// ─── Barrier divergence ───────────────────────────────────────────────────────
void detector_report_barrier_divergence(dim3 block_idx,
                                         const std::vector<int>& missing_tids) {
  DiagnosticMsg d;
  d.type    = "barrier_divergence";
  d.block_x = (int)block_idx.x; d.block_y = (int)block_idx.y; d.block_z = (int)block_idx.z;
  d.thread_x = d.thread_y = d.thread_z = -1;

  std::ostringstream oss;
  oss << "__syncthreads() not reached by all threads in block("
      << block_idx.x << "," << block_idx.y << "," << block_idx.z
      << "): threads [";
  for (size_t i = 0; i < missing_tids.size() && i < 8; i++) {
    oss << missing_tids[i];
    if (i + 1 < missing_tids.size()) oss << ", ";
  }
  if (missing_tids.size() > 8) oss << ", ...";
  oss << "] exited early (common cause: early return before __syncthreads)";
  d.message = oss.str();
  add_diag(std::move(d));
  g_has_fatal = true;
}

// ─── Deadlock ─────────────────────────────────────────────────────────────────
void detector_report_deadlock(dim3 block_idx, int arrived, int n_alive) {
  DiagnosticMsg d;
  d.type    = "deadlock";
  d.block_x = (int)block_idx.x; d.block_y = (int)block_idx.y; d.block_z = (int)block_idx.z;
  d.thread_x = d.thread_y = d.thread_z = -1;

  char buf[512];
  snprintf(buf, sizeof(buf),
    "Deadlock in block(%d,%d,%d): %d/%d threads reached __syncthreads, "
    "but no threads are runnable. Check for conditional __syncthreads or "
    "data-dependent loop trip counts.",
    (int)block_idx.x, (int)block_idx.y, (int)block_idx.z,
    arrived, n_alive);
  d.message = buf;
  add_diag(std::move(d));
  g_has_fatal = true;
}

// ─── Uninitialized read ───────────────────────────────────────────────────────
void detector_check_uninit(const void* addr, int bytes, dim3 block_idx, dim3 thread_idx) {
  static const uint8_t kPoison = 0xBE;
  const uint8_t* p = (const uint8_t*)addr;
  bool poisoned = true;
  for (int i = 0; i < bytes && poisoned; i++) {
    if (p[i] != kPoison) poisoned = false;
  }
  if (poisoned) {
    DiagnosticMsg d;
    d.type    = "uninit";
    d.block_x = (int)block_idx.x; d.block_y = (int)block_idx.y; d.block_z = (int)block_idx.z;
    d.thread_x= (int)thread_idx.x;d.thread_y= (int)thread_idx.y;d.thread_z= (int)thread_idx.z;
    char buf[512];
    snprintf(buf, sizeof(buf),
      "Read of %d uninitialized bytes at 0x%zx (block(%d,%d,%d) thread(%d,%d,%d)) — "
      "did you forget cudaMemcpy or an initialization kernel?",
      bytes, (size_t)(uintptr_t)addr,
      d.block_x, d.block_y, d.block_z,
      d.thread_x, d.thread_y, d.thread_z);
    d.message = buf;
    add_diag(std::move(d));
  }
}

// ─── Race detection (randomized schedule helper) ──────────────────────────────
// Phase 2: capture outputs from 3 runs with different fiber orderings.
// Report race if outputs differ. For MVP this is a stub.
void detector_check_race_outputs(const void* /*out1*/, const void* /*out2*/,
                                  size_t /*bytes*/) {
  // TODO Phase 2: compare snapshots from 3 randomized fiber orderings
}

// ─── Reset ────────────────────────────────────────────────────────────────────
void detectors_reset() {
  std::lock_guard<std::mutex> lk(g_diag_mutex);
  g_diagnostics.clear();
  g_has_fatal = false;
}

bool detectors_has_fatal() { return g_has_fatal; }

// ─── Serialize to JSON ────────────────────────────────────────────────────────
std::string detectors_to_json() {
  std::lock_guard<std::mutex> lk(g_diag_mutex);
  std::ostringstream o;
  o << "[\n";
  for (size_t i = 0; i < g_diagnostics.size(); i++) {
    const auto& d = g_diagnostics[i];
    o << "  {\n";
    o << "    \"type\": \"" << d.type << "\",\n";
    // Escape message for JSON
    std::string msg = d.message;
    for (size_t j = 0; j < msg.size(); j++) {
      if (msg[j] == '"') { msg.insert(j, "\\"); j++; }
    }
    o << "    \"message\": \"" << msg << "\",\n";
    o << "    \"block\": [" << d.block_x << "," << d.block_y << "," << d.block_z << "],\n";
    o << "    \"thread\": [" << d.thread_x << "," << d.thread_y << "," << d.thread_z << "]\n";
    o << "  }";
    if (i + 1 < g_diagnostics.size()) o << ",";
    o << "\n";
  }
  o << "]";
  return o.str();
}
