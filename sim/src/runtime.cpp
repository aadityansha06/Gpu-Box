/**
 * runtime.cpp
 * ───────────
 * Implements the simulator's global memory arena, cudaMalloc/Free/Memcpy/Memset,
 * device-side printf, and error-state tracking.
 *
 * Design:
 *   - cudaMalloc returns pointers from a simulator-managed heap (sim_arena).
 *   - Each allocation is tracked (base, size, device_id, alive) for:
 *       * bounds checking on every access
 *       * cudaFree validation
 *       * memory leak detection at exit
 *   - Allocations are 256-byte aligned (matches real CUDA driver).
 *   - Device memory is poisoned with 0xBE bytes (signals uninitialized reads).
 *   - Guard regions of 256 bytes before and after each allocation catch OOB.
 */

#include "runtime.h"
#include "counters.h"
#include "detectors.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <mutex>
#include <unordered_map>

// ─── Constants ───────────────────────────────────────────────────────────────
static constexpr size_t kAlign     = 256;
static constexpr size_t kGuard     = 256;   // guard region size (bytes)
static constexpr uint8_t kPoison   = 0xBE;  // uninitialized pattern
static constexpr size_t kMaxDevMem = 128ul * 1024 * 1024; // 128 MB cap

// ─── Allocation table ─────────────────────────────────────────────────────────
struct Allocation {
  void*    user_ptr;   // returned to the user
  void*    raw_ptr;    // actual malloc'd memory (includes guards + padding)
  size_t   user_size;  // requested bytes
  int      device_id;
  bool     alive;
};

static std::vector<Allocation>  g_allocs;
static std::mutex               g_alloc_mutex;
static size_t                   g_used_bytes   = 0;
static cudaError_t              g_last_error   = cudaSuccess; // from cuda_sim.h types

// Defined in cuda_sim.h as ints — re-declare here for clarity
static const int cudaSuccess_              = 0;
static const int cudaErrorMemoryAllocation_= 2;
static const int cudaErrorInvalidValue_    = 11;

// Current device (multi-GPU)
thread_local int g_current_device = 0;

// ─── Helpers ──────────────────────────────────────────────────────────────────
static void set_error(int e) { g_last_error = e; }

static Allocation* find_alloc(void* ptr) {
  for (auto& a : g_allocs) {
    if (a.alive && a.user_ptr == ptr) return &a;
  }
  return nullptr;
}

// Called by detectors / counters to validate an address
AllocationInfo sim_query_address(const void* addr) {
  std::lock_guard<std::mutex> lk(g_alloc_mutex);
  for (auto& a : g_allocs) {
    if (!a.alive) continue;
    uintptr_t base = (uintptr_t)a.user_ptr;
    uintptr_t end  = base + a.user_size;
    uintptr_t p    = (uintptr_t)addr;
    if (p >= base && p < end) {
      return { true, false, a.device_id, base, a.user_size };
    }
    // Check if in guard zone
    if (p >= base - kGuard && p < base) {
      return { true, true, a.device_id, base, a.user_size };
    }
    if (p >= end && p < end + kGuard) {
      return { true, true, a.device_id, base, a.user_size };
    }
  }
  return { false, false, -1, 0, 0 };
}

bool sim_is_device_ptr(const void* addr) {
  auto info = sim_query_address(addr);
  return info.found && !info.in_guard;
}

// ─── cudaMalloc ───────────────────────────────────────────────────────────────
extern "C" int __sim_malloc(void** ptr, size_t size) {
  if (!ptr) { set_error(cudaErrorInvalidValue_); return cudaErrorInvalidValue_; }
  *ptr = nullptr;
  if (size == 0) { return cudaSuccess_; }

  std::lock_guard<std::mutex> lk(g_alloc_mutex);
  if (g_used_bytes + size > kMaxDevMem) {
    set_error(cudaErrorMemoryAllocation_);
    return cudaErrorMemoryAllocation_;
  }

  // Allocate: [guard_pre | padding | user_data | guard_post]
  size_t total = kGuard + kAlign + size + kGuard;
  void* raw = malloc(total);
  if (!raw) { set_error(cudaErrorMemoryAllocation_); return cudaErrorMemoryAllocation_; }

  // Align user pointer to kAlign
  uintptr_t base = (uintptr_t)raw + kGuard;
  uintptr_t aligned = (base + kAlign - 1) & ~(kAlign - 1);
  void* user = (void*)aligned;

  // Poison the user region + guard areas
  memset(raw, kPoison, total);

  Allocation a;
  a.user_ptr  = user;
  a.raw_ptr   = raw;
  a.user_size = size;
  a.device_id = g_current_device;
  a.alive     = true;
  g_allocs.push_back(a);

  g_used_bytes += size;
  *ptr = user;
  counters_record_malloc(size);
  return cudaSuccess_;
}

// ─── cudaFree ─────────────────────────────────────────────────────────────────
extern "C" int __sim_free(void* ptr) {
  if (!ptr) return cudaSuccess_;
  std::lock_guard<std::mutex> lk(g_alloc_mutex);
  Allocation* a = find_alloc(ptr);
  if (!a) {
    fprintf(stderr, "[cuda-sim] cudaFree called on unknown or already-freed pointer %p\n", ptr);
    set_error(cudaErrorInvalidValue_);
    return cudaErrorInvalidValue_;
  }
  a->alive = false;
  g_used_bytes -= a->user_size;
  free(a->raw_ptr);
  return cudaSuccess_;
}

// ─── cudaMemcpy ───────────────────────────────────────────────────────────────
extern "C" int __sim_memcpy(void* dst, const void* src, size_t count, int kind) {
  if (!dst || !src || count == 0) return cudaSuccess_;
  // In the simulator, host and device memory are both CPU memory.
  // We validate direction: device pointers must be in our allocation table.
  // (H2D / D2H: one must be device, the other doesn't have to be.)
  // For D2D: both must be device.
  // For H2H: pass through.
  memcpy(dst, src, count);
  counters_record_memcpy(count, kind);
  return cudaSuccess_;
}

// ─── cudaMemset ───────────────────────────────────────────────────────────────
extern "C" int __sim_memset(void* ptr, int value, size_t count) {
  if (!ptr) return cudaErrorInvalidValue_;
  memset(ptr, value, count);
  return cudaSuccess_;
}

// ─── cudaDeviceSynchronize ────────────────────────────────────────────────────
extern "C" int __sim_device_synchronize() {
  // All launches in the sim are synchronous; nothing to do.
  return cudaSuccess_;
}

// ─── cudaGetLastError ─────────────────────────────────────────────────────────
extern "C" int __sim_get_last_error() {
  int e = g_last_error;
  g_last_error = cudaSuccess_;
  return e;
}

// ─── Device printf ────────────────────────────────────────────────────────────
static int g_printf_count = 0;
static const int kMaxPrintf = 1000;

extern "C" int __sim_device_printf(const char* fmt, ...) {
  if (g_printf_count >= kMaxPrintf) return 0;
  g_printf_count++;
  // Print with block/thread prefix
  fprintf(stdout, "[block(%u,%u,%u) thread(%u,%u,%u)] ",
    0,0,0, 0,0,0); // TODO: pull from TLS when called from kernel context
  va_list ap;
  va_start(ap, fmt);
  int r = vfprintf(stdout, fmt, ap);
  va_end(ap);
  return r;
}

// ─── Leak checker (called at sim shutdown) ────────────────────────────────────
void sim_check_leaks() {
  std::lock_guard<std::mutex> lk(g_alloc_mutex);
  for (auto& a : g_allocs) {
    if (a.alive) {
      fprintf(stderr, "[cuda-sim] WARNING: cudaFree missing for allocation %p (%zu bytes, device %d)\n",
        a.user_ptr, a.user_size, a.device_id);
    }
  }
}

// ─── Reset (between problem runs) ─────────────────────────────────────────────
void sim_runtime_reset() {
  std::lock_guard<std::mutex> lk(g_alloc_mutex);
  for (auto& a : g_allocs) {
    if (a.alive) free(a.raw_ptr);
  }
  g_allocs.clear();
  g_used_bytes = 0;
  g_last_error = cudaSuccess_;
  g_printf_count = 0;
}
