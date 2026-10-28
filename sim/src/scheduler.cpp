/**
 * scheduler.cpp
 * ─────────────
 * Fiber-based CUDA thread scheduler.
 *
 * Model:
 *   - Each CUDA thread → one ucontext_t fiber with a private stack.
 *   - A block's fibers all share one OS thread (sequential within a block).
 *   - Blocks are dispatched sequentially (single-GPU, no true SM parallelism).
 *   - Warps = groups of 32 consecutive threads (by linear index).
 *   - __syncthreads() → yield to block scheduler; resume when all arrive.
 *
 * Stack sizes:
 *   - Default 64 KB per fiber.
 *   - A 1024-thread block uses ~64 MB of stack space (virtual, not physical).
 *   - Stacks are reused across blocks (allocated once per scheduler instance).
 *
 * Multi-GPU: each simulated GPU gets its own SimScheduler instance.
 */

#include "scheduler.h"
#include "counters.h"
#include "detectors.h"
#include <ucontext.h>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <functional>
#include <cstdio>

#include "../include/cuda_sim.h"

// ─── Constants ────────────────────────────────────────────────────────────────
static constexpr size_t kFiberStack   = 64 * 1024;  // 64 KB per fiber
static constexpr int    kMaxThreads   = 1024;
static constexpr int    kWarpSize     = 32;
static constexpr uint8_t kSharedPoison = 0xFF;       // NaN-byte pattern

// ─── Fiber state ─────────────────────────────────────────────────────────────
enum class FiberStatus { READY, RUNNING, BLOCKED_AT_SYNC, DONE };

struct Fiber {
  ucontext_t      ctx;
  std::vector<uint8_t> stack;
  FiberStatus     status;
  int             linear_tid;      // x + y*bx + z*bx*by
  int             lane_id;
  int             warp_id;
  std::function<void()> fn;        // kernel body for this thread
};

// ─── Block scheduler ─────────────────────────────────────────────────────────
struct BlockScheduler {
  std::vector<Fiber> fibers;
  ucontext_t         host_ctx;    // returns here after each yield
  int                n_threads;
  int                sync_epoch;  // incremented at each __syncthreads
  int                arrived;     // fibers that have called __syncthreads
  int                done_count;

  // Shared memory arena for this block
  std::vector<uint8_t> shared_arena;
  size_t               shared_offset;
  size_t               shared_limit;

  // The fiber currently running (for TLS updates)
  Fiber* current_fiber;
};

// Thread-local pointer to the active block scheduler (set during launch)
thread_local BlockScheduler* g_block_sched = nullptr;

// ─── ucontext trampoline ─────────────────────────────────────────────────────
static void fiber_entry() {
  BlockScheduler* bs = g_block_sched;
  Fiber* f = bs->current_fiber;
  f->status = FiberStatus::RUNNING;
  f->fn();   // run the kernel body for this thread
  f->status = FiberStatus::DONE;
  bs->done_count++;
  // Return to block scheduler
  swapcontext(&f->ctx, &bs->host_ctx);
}

// ─── __syncthreads implementation ────────────────────────────────────────────
extern "C" void __sim_syncthreads() {
  BlockScheduler* bs = g_block_sched;
  assert(bs && "syncthreads called outside kernel");

  Fiber* f = bs->current_fiber;
  assert(f->status == FiberStatus::RUNNING);

  f->status = FiberStatus::BLOCKED_AT_SYNC;
  bs->arrived++;

  counters_record_syncthreads();

  // Yield to host scheduler
  swapcontext(&f->ctx, &bs->host_ctx);
  // Resumed when all threads have arrived
}

extern "C" void __sim_syncwarp(unsigned /*mask*/) {
  // Within a block all warps are in the same fiber pool.
  // For MVP: treat as a full syncthreads within the warp group.
  // TODO: restrict to mask lanes only
  __sim_syncthreads();
}

// ─── Shared memory arena ──────────────────────────────────────────────────────
extern "C" void* __sim_shared_alloc(int bytes, int align, int /*static_id*/) {
  BlockScheduler* bs = g_block_sched;
  assert(bs && "shared_alloc called outside kernel");

  // Align offset
  size_t off = bs->shared_offset;
  off = (off + (size_t)align - 1) & ~((size_t)align - 1);

  if (off + (size_t)bytes > bs->shared_limit) {
    fprintf(stderr, "[cuda-sim] Shared memory overflow: requested %d bytes at offset %zu, limit %zu\n",
            bytes, off, bs->shared_limit);
    return nullptr;
  }
  bs->shared_offset = off + (size_t)bytes;
  return bs->shared_arena.data() + off;
}

// ─── Shuffle intrinsics ───────────────────────────────────────────────────────
// Implemented as direct value exchange since all lanes are in the same process.
// We store lane values in a per-warp buffer during shuffle, mimicking lockstep.

struct WarpShuffleBuf {
  float vals[kWarpSize];
  int   arrived;
};
// One buffer per warp-id per block
static thread_local WarpShuffleBuf g_shfl_buf[kMaxThreads / kWarpSize] = {};

extern "C" float __sim_shfl_down_sync(unsigned /*mask*/, float var, unsigned delta, int width) {
  BlockScheduler* bs = g_block_sched;
  Fiber* f = bs->current_fiber;
  int lane = f->lane_id;
  int warp = f->warp_id;
  WarpShuffleBuf& buf = g_shfl_buf[warp];

  buf.vals[lane] = var;
  buf.arrived++;
  // Wait for all lanes in warp to deposit
  if (buf.arrived < kWarpSize) {
    f->status = FiberStatus::BLOCKED_AT_SYNC;
    swapcontext(&f->ctx, &bs->host_ctx);
  } else {
    buf.arrived = 0;
    // Resume all waiting fibers in this warp
    // (handled by the main scheduler loop below)
  }

  // Read from source lane
  int src = lane + (int)delta;
  if ((src & (width - 1)) >= width || src >= kWarpSize) src = lane;
  return buf.vals[src];
}

extern "C" float __sim_shfl_xor_sync(unsigned mask, float var, int laneMask, int width) {
  BlockScheduler* bs = g_block_sched;
  Fiber* f = bs->current_fiber;
  int lane = f->lane_id;
  int warp = f->warp_id;
  WarpShuffleBuf& buf = g_shfl_buf[warp];

  buf.vals[lane] = var;
  buf.arrived++;
  if (buf.arrived < kWarpSize) {
    f->status = FiberStatus::BLOCKED_AT_SYNC;
    swapcontext(&f->ctx, &bs->host_ctx);
  } else {
    buf.arrived = 0;
  }
  int src = lane ^ laneMask;
  if (src < 0 || src >= kWarpSize) src = lane;
  return buf.vals[src];
}

extern "C" float __sim_shfl_sync(unsigned mask, float var, int srcLane, int width) {
  BlockScheduler* bs = g_block_sched;
  Fiber* f = bs->current_fiber;
  int lane = f->lane_id;
  int warp = f->warp_id;
  WarpShuffleBuf& buf = g_shfl_buf[warp];

  buf.vals[lane] = var;
  buf.arrived++;
  if (buf.arrived < kWarpSize) {
    f->status = FiberStatus::BLOCKED_AT_SYNC;
    swapcontext(&f->ctx, &bs->host_ctx);
  } else {
    buf.arrived = 0;
  }
  return buf.vals[srcLane < kWarpSize ? srcLane : lane];
}

extern "C" int __sim_shfl_down_sync_i(unsigned mask, int var, unsigned delta, int width) {
  return (int)__sim_shfl_down_sync(mask, (float)var, delta, width);
}

// ─── Atomics ──────────────────────────────────────────────────────────────────
// Within one OS thread (sequential fibers), plain ops are atomic.
// We record contention for the counter report.

extern "C" void __sim_atomic_add_f(float* addr, float val) {
  counters_record_atomic(addr);
  *addr += val;
}
extern "C" void __sim_atomic_add_i(int* addr, int val) {
  counters_record_atomic(addr);
  *addr += val;
}
extern "C" float __sim_atomic_max_f(float* addr, float val) {
  counters_record_atomic(addr);
  float old = *addr;
  if (val > *addr) *addr = val;
  return old;
}
extern "C" int __sim_atomic_max_i(int* addr, int val) {
  counters_record_atomic(addr);
  int old = *addr;
  if (val > *addr) *addr = val;
  return old;
}
extern "C" int __sim_atomic_cas(int* addr, int compare, int val) {
  counters_record_atomic(addr);
  int old = *addr;
  if (old == compare) *addr = val;
  return old;
}

// ─── Main block runner ────────────────────────────────────────────────────────
/**
 * Runs one CUDA block. Spawns N fibers (one per thread), then drives the
 * cooperative scheduler until all fibers finish.
 *
 * On each barrier:
 *   1. Resume all fibers round-robin until they block or finish.
 *   2. When arrived == n_alive (all living fibers are at the barrier), increment
 *      epoch and wake all blocked fibers.
 *   3. Detect deadlock: no runnable fibers but some still blocked.
 */
static void run_block(BlockScheduler& bs,
                      dim3 block_idx, dim3 block_dim, dim3 grid_dim,
                      size_t shmem_bytes,
                      std::function<void(int /*linear_tid*/)> make_fiber_fn) {
  int nx = (int)block_dim.x, ny = (int)block_dim.y, nz = (int)block_dim.z;
  int n = nx * ny * nz;
  bs.n_threads = n;
  bs.sync_epoch = 0;
  bs.arrived = 0;
  bs.done_count = 0;

  // Shared memory
  size_t total_shmem = shmem_bytes + 48 * 1024; // static + dynamic
  bs.shared_arena.assign(total_shmem, kSharedPoison);
  bs.shared_offset = 0;
  bs.shared_limit  = total_shmem;

  // Allocate / init fibers
  bs.fibers.resize(n);
  for (int tid = 0; tid < n; tid++) {
    Fiber& f = bs.fibers[tid];
    f.stack.resize(kFiberStack);
    f.linear_tid = tid;
    f.lane_id    = tid % kWarpSize;
    f.warp_id    = tid / kWarpSize;
    f.status     = FiberStatus::READY;
    f.fn         = [&bs, &make_fiber_fn, tid]() { make_fiber_fn(tid); };

    getcontext(&f.ctx);
    f.ctx.uc_stack.ss_sp   = f.stack.data();
    f.ctx.uc_stack.ss_size = kFiberStack;
    f.ctx.uc_link          = nullptr; // we manage control explicitly
    makecontext(&f.ctx, fiber_entry, 0);
  }

  // Run the block
  g_block_sched = &bs;
  int epoch = 0;

  while (bs.done_count < n) {
    // Run all READY fibers until they block or finish
    bool ran_any = false;
    for (int tid = 0; tid < n; tid++) {
      Fiber& f = bs.fibers[tid];
      if (f.status == FiberStatus::READY || f.status == FiberStatus::RUNNING) {
        bs.current_fiber = &f;

        // Set TLS for this thread
        int z = tid / (nx * ny);
        int rem = tid % (nx * ny);
        int y = rem / nx;
        int x = rem % nx;
        __sim_tls::threadIdx_v = {(unsigned)x, (unsigned)y, (unsigned)z};
        __sim_tls::blockIdx_v  = block_idx;
        __sim_tls::blockDim_v  = block_dim;
        __sim_tls::gridDim_v   = grid_dim;
        __sim_tls::laneId_v    = f.lane_id;
        __sim_tls::warpId_v    = f.warp_id;

        swapcontext(&bs.host_ctx, &f.ctx);
        ran_any = true;
      }
    }

    // Check if we should advance past a barrier
    int n_alive = n - bs.done_count;
    if (bs.arrived == n_alive && n_alive > 0) {
      // All living threads are at the barrier — wake them all
      bs.sync_epoch++;
      bs.arrived = 0;
      for (auto& f : bs.fibers) {
        if (f.status == FiberStatus::BLOCKED_AT_SYNC) {
          f.status = FiberStatus::READY;
        }
      }
    } else if (!ran_any && bs.arrived < n_alive) {
      // Deadlock: no runnable fibers, but not all are at the barrier
      detector_report_deadlock(block_idx, bs.arrived, n_alive);
      break;
    }
  }

  g_block_sched = nullptr;
}

// ─── Kernel launch ────────────────────────────────────────────────────────────
/**
 * Called by the rewritten <<<>>> syntax:
 *   __sim_launch(name, grid, block, shmem, kernel_thunk, ctx)
 *
 * Iterates over all blocks in the grid and runs each one sequentially.
 */
extern "C" void __sim_launch_impl(const char* name, dim3 grid, dim3 block,
                                   size_t shmem, void (*thunk)(void*), void* ctx) {
  // Validate launch config
  int threads_per_block = (int)(block.x * block.y * block.z);
  if (threads_per_block > kMaxThreads) {
    fprintf(stderr, "[cuda-sim] Launch error: %d threads per block exceeds limit %d\n",
            threads_per_block, kMaxThreads);
    return;
  }

  int total_blocks = (int)(grid.x * grid.y * grid.z);
  long long total_threads = (long long)total_blocks * threads_per_block;
  static const long long kMaxTotalThreads = 1024LL * 1024; // 1M threads
  if (total_threads > kMaxTotalThreads) {
    fprintf(stderr, "[cuda-sim] Launch error: %lld total threads exceeds cap %lld\n",
            total_threads, kMaxTotalThreads);
    return;
  }

  counters_record_launch(name, grid, block, shmem, total_threads);

  // Run each block
  BlockScheduler bs;
  for (unsigned bz = 0; bz < grid.z; bz++) {
    for (unsigned by = 0; by < grid.y; by++) {
      for (unsigned bx = 0; bx < grid.x; bx++) {
        dim3 block_idx = {bx, by, bz};

        // Reset shared offsets for each new block
        memset(g_shfl_buf, 0, sizeof(g_shfl_buf));

        run_block(bs, block_idx, block, grid, shmem,
          [&](int linear_tid) {
            // Set TLS here too (belt + suspenders; also set in run_block per-swap)
            int nx = (int)block.x, ny = (int)block.y;
            int z2 = linear_tid / (nx * ny);
            int rem = linear_tid % (nx * ny);
            int y2 = rem / nx, x2 = rem % nx;
            __sim_tls::threadIdx_v = {(unsigned)x2, (unsigned)y2, (unsigned)z2};
            __sim_tls::blockIdx_v  = block_idx;
            __sim_tls::blockDim_v  = block;
            __sim_tls::gridDim_v   = grid;
            thunk(ctx);  // call the kernel lambda
          });
      }
    }
  }
}
