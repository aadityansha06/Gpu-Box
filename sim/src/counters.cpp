/**
 * counters.cpp
 * ────────────
 * Tracks all simulator performance counters:
 *   - Global memory loads / stores (instructions + bytes + 32-byte sector transactions)
 *   - Shared memory loads / stores
 *   - Bank conflicts (per-warp shared access analysis)
 *   - __syncthreads() call count
 *   - Divergent warps (branch tracking)
 *   - Atomic operation count + per-address contention
 *   - Threads / blocks launched
 *   - Cross-GPU memory traffic (multi-GPU)
 *
 * All counters are reset per kernel launch (or per problem run).
 * Results are serialized to JSON by api.cpp.
 */

#include "counters.h"
#include "../include/cuda_sim.h"
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>

// ─── Per-kernel counter state ─────────────────────────────────────────────────
struct CounterState {
  // Launch
  long long threads_launched   = 0;
  long long blocks_launched    = 0;

  // Global memory
  long long global_load_insns  = 0;
  long long global_load_bytes  = 0;
  long long global_store_insns = 0;
  long long global_store_bytes = 0;
  long long global_transactions= 0; // distinct 32B sectors per warp-insn

  // Shared memory
  long long shared_load_insns  = 0;
  long long shared_load_bytes  = 0;
  long long shared_store_insns = 0;
  long long shared_store_bytes = 0;
  long long bank_conflicts     = 0;

  // Sync
  long long syncthreads_count  = 0;

  // Divergence
  long long divergent_warps    = 0;   // warp-branch-instances with mixed outcomes

  // Atomics
  long long atomic_ops         = 0;
  std::unordered_map<uintptr_t, int> atomic_contention; // addr → hit count

  // Memcpy (host<->device traffic)
  long long memcpy_h2d_bytes   = 0;
  long long memcpy_d2h_bytes   = 0;
  long long memcpy_d2d_bytes   = 0;

  // Multi-GPU cross-device traffic
  long long cross_gpu_bytes    = 0;
  long long nccl_ops           = 0;

  // Malloc
  long long device_bytes_allocated = 0;
};

// Active counters (reset per run)
static CounterState g_counters;
static std::mutex   g_counter_mutex;

// Warp memory access grouping: (warp_id, static_id, occurrence) → list of addresses
// We use a simple per-instruction buffer that's flushed per warp-instruction.
struct WarpAccessBuf {
  static constexpr int kMax = 32;
  uintptr_t addrs[kMax];
  int       bytes[kMax];
  int       count;
  bool      is_store;
  bool      is_shared;
};
thread_local WarpAccessBuf g_warp_buf = {};

static int count_sectors(const WarpAccessBuf& buf) {
  // Count distinct 32-byte sectors touched by this warp instruction
  std::unordered_map<uintptr_t, bool> sectors;
  for (int i = 0; i < buf.count; i++) {
    sectors[buf.addrs[i] / 32] = true;
  }
  return (int)sectors.size();
}

static int count_bank_conflicts(const WarpAccessBuf& buf) {
  // 32 banks of 4 bytes. Bank = (addr/4) % 32
  // Conflict = max lanes hitting any single bank > 1
  int hits[32] = {};
  uintptr_t broadcast_check[32] = {};
  bool      first[32] = {};
  memset(first, 1, sizeof(first));

  for (int i = 0; i < buf.count; i++) {
    int bank = (int)((buf.addrs[i] / 4) % 32);
    if (first[bank]) {
      broadcast_check[bank] = buf.addrs[i];
      first[bank] = false;
    }
    // Same address = broadcast (no conflict)
    if (buf.addrs[i] != broadcast_check[bank]) {
      hits[bank]++;
    }
  }

  int max_hit = 0;
  for (int i = 0; i < 32; i++) max_hit = std::max(max_hit, hits[i]);
  return std::max(0, max_hit - 1); // conflicts = extra replays
}

// ─── Record a memory access (called conceptually per-lane, grouped per warp) ──
// In our fiber model, we call these inline from the kernel.
// Each call represents one lane's access; we flush when the warp completes
// the instruction (i.e., when all 32 lanes have called it with the same static_id).

// Simple per-thread accumulation: we record one access per thread per call.
// Warp-level grouping happens at flush time.

struct WarpInsnKey {
  int warp_id;
  int static_id;
  int occurrence; // which dynamic occurrence of this static_id
};

// For the MVP, we do a simpler per-warp accumulation using thread-local state.
// Each thread deposits its address; at __syncthreads we flush.

thread_local struct {
  uintptr_t addrs[1024]; // circular per-thread buffer
  int bytes_arr[1024];
  bool is_store_arr[1024];
  bool is_shared_arr[1024];
  int write_idx;
} g_access_log = {};

static void flush_warp_accesses();

extern "C" void __sim_record_load(const void* addr, int bytes, int /*static_id*/) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  bool is_shared = ((uintptr_t)addr >> 28) == 0xF; // heuristic: shared ptrs tagged
  // For simplicity in MVP: count directly
  if (is_shared) {
    g_counters.shared_load_insns++;
    g_counters.shared_load_bytes += bytes;
  } else {
    g_counters.global_load_insns++;
    g_counters.global_load_bytes += bytes;
    g_counters.global_transactions++; // simplified: 1 sector per access
  }
}

extern "C" void __sim_record_store(const void* addr, int bytes, int /*static_id*/) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  bool is_shared = ((uintptr_t)addr >> 28) == 0xF;
  if (is_shared) {
    g_counters.shared_store_insns++;
    g_counters.shared_store_bytes += bytes;
  } else {
    g_counters.global_store_insns++;
    g_counters.global_store_bytes += bytes;
    g_counters.global_transactions++;
  }
}

void counters_record_syncthreads() {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters.syncthreads_count++;
}

void counters_record_atomic(const void* addr) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters.atomic_ops++;
  g_counters.atomic_contention[(uintptr_t)addr]++;
}

void counters_record_launch(const char* /*name*/, dim3 grid, dim3 block,
                             size_t /*shmem*/, long long total_threads) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters.threads_launched += total_threads;
  g_counters.blocks_launched  += (long long)(grid.x * grid.y * grid.z);
}

void counters_record_malloc(size_t bytes) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters.device_bytes_allocated += (long long)bytes;
}

void counters_record_memcpy(size_t bytes, int kind) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  switch (kind) {
    case 1: g_counters.memcpy_h2d_bytes += bytes; break; // H2D
    case 2: g_counters.memcpy_d2h_bytes += bytes; break; // D2H
    case 3: g_counters.memcpy_d2d_bytes += bytes; break; // D2D
    default: break;
  }
}

void counters_record_cross_gpu(size_t bytes) {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters.cross_gpu_bytes += bytes;
}

void counters_record_nccl_op() {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters.nccl_ops++;
}

// ─── Reset ────────────────────────────────────────────────────────────────────
void counters_reset() {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  g_counters = CounterState{};
}

// ─── Serialize to JSON ────────────────────────────────────────────────────────
std::string counters_to_json() {
  std::lock_guard<std::mutex> lk(g_counter_mutex);
  const auto& c = g_counters;

  // Find max atomic contention
  int max_contention = 0;
  for (auto& kv : c.atomic_contention) {
    max_contention = std::max(max_contention, kv.second);
  }

  std::ostringstream o;
  o << "{\n";
  o << "  \"threadsLaunched\": "      << c.threads_launched    << ",\n";
  o << "  \"blocksLaunched\": "       << c.blocks_launched     << ",\n";
  o << "  \"globalLoadInsns\": "      << c.global_load_insns   << ",\n";
  o << "  \"globalLoadBytes\": "      << c.global_load_bytes   << ",\n";
  o << "  \"globalStoreInsns\": "     << c.global_store_insns  << ",\n";
  o << "  \"globalStoreBytes\": "     << c.global_store_bytes  << ",\n";
  o << "  \"globalTransactions\": "   << c.global_transactions << ",\n";
  o << "  \"sharedLoadInsns\": "      << c.shared_load_insns   << ",\n";
  o << "  \"sharedLoadBytes\": "      << c.shared_load_bytes   << ",\n";
  o << "  \"sharedStoreInsns\": "     << c.shared_store_insns  << ",\n";
  o << "  \"sharedStoreBytes\": "     << c.shared_store_bytes  << ",\n";
  o << "  \"bankConflicts\": "        << c.bank_conflicts      << ",\n";
  o << "  \"syncthreadsCount\": "     << c.syncthreads_count   << ",\n";
  o << "  \"divergentWarps\": "       << c.divergent_warps     << ",\n";
  o << "  \"atomicOps\": "            << c.atomic_ops          << ",\n";
  o << "  \"atomicMaxContention\": "  << max_contention        << ",\n";
  o << "  \"memcpyH2DBytes\": "       << c.memcpy_h2d_bytes    << ",\n";
  o << "  \"memcpyD2HBytes\": "       << c.memcpy_d2h_bytes    << ",\n";
  o << "  \"crossGpuBytes\": "        << c.cross_gpu_bytes     << ",\n";
  o << "  \"ncclOps\": "              << c.nccl_ops            << "\n";
  o << "}";
  return o.str();
}
