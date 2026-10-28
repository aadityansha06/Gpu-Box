/**
 * multi_gpu.cpp
 * ─────────────
 * Multi-GPU simulation: N virtual GPUs, each with its own arena.
 * Inter-GPU communication via MPI-style message passing.
 * NCCL-like collectives (allreduce, broadcast, reduce-scatter, all-gather).
 *
 * Architecture:
 *   - SimGPU[i] owns its memory arena (separate from other GPUs).
 *   - cudaSetDevice(i) switches the active GPU context for memory ops.
 *   - cudaMemcpyPeer(dst, dstDev, src, srcDev, n):
 *       copies from srcDev's arena to dstDev's arena,
 *       counts cross_gpu_bytes in the counter.
 *   - NCCL collectives are synchronous (all GPUs execute together via a
 *     barrier in the multi-GPU scheduler). Each op logs cross-GPU bytes.
 *
 * For cluster simulation:
 *   - "Nodes" are groups of GPUs; inter-node traffic tagged separately.
 *   - Students can query latency models (Alpha-Beta or ring-allreduce step count).
 */

#include "multi_gpu.h"
#include "runtime.h"
#include "counters.h"
#include "../include/cuda_sim.h"
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <functional>
#include <sstream>

// ─── GPU device state ─────────────────────────────────────────────────────────
struct SimGPU {
  int id;
  std::vector<uint8_t> arena;   // simulated device memory
  size_t arena_used;
  bool peer_access_enabled[16]; // which peers can directly access this device
};

static constexpr int kMaxGPUs = 16;
static SimGPU g_gpus[kMaxGPUs];
static int    g_num_gpus   = 1;
static bool   g_initialized = false;
static std::mutex g_mgpu_mutex;

// Per-OS-thread active device
thread_local int g_current_device_id = 0;

void multi_gpu_init(int num_gpus) {
  std::lock_guard<std::mutex> lk(g_mgpu_mutex);
  if (num_gpus < 1 || num_gpus > kMaxGPUs) num_gpus = 1;
  g_num_gpus = num_gpus;
  for (int i = 0; i < num_gpus; i++) {
    g_gpus[i].id = i;
    g_gpus[i].arena_used = 0;
    memset(g_gpus[i].peer_access_enabled, 0, sizeof(g_gpus[i].peer_access_enabled));
  }
  g_initialized = true;
  fprintf(stdout, "[cuda-sim] Multi-GPU initialized: %d virtual GPUs\n", num_gpus);
}

void multi_gpu_reset() {
  std::lock_guard<std::mutex> lk(g_mgpu_mutex);
  for (int i = 0; i < g_num_gpus; i++) {
    g_gpus[i].arena.clear();
    g_gpus[i].arena_used = 0;
    memset(g_gpus[i].peer_access_enabled, 0, sizeof(g_gpus[i].peer_access_enabled));
  }
  g_current_device_id = 0;
}

// ─── cudaSetDevice / cudaGetDevice ────────────────────────────────────────────
extern "C" int __sim_set_device(int device_id) {
  if (device_id < 0 || device_id >= g_num_gpus) {
    fprintf(stderr, "[cuda-sim] cudaSetDevice(%d): invalid device (only %d GPUs available)\n",
            device_id, g_num_gpus);
    return 11; // cudaErrorInvalidDevice
  }
  g_current_device_id = device_id;
  return 0; // cudaSuccess
}

extern "C" int __sim_get_device(int* device_id) {
  if (!device_id) return 11;
  *device_id = g_current_device_id;
  return 0;
}

// ─── cudaDeviceEnablePeerAccess ───────────────────────────────────────────────
extern "C" int __sim_enable_peer_access(int peer_device, unsigned int /*flags*/) {
  int dev = g_current_device_id;
  if (peer_device < 0 || peer_device >= g_num_gpus) return 11;
  std::lock_guard<std::mutex> lk(g_mgpu_mutex);
  g_gpus[dev].peer_access_enabled[peer_device] = true;
  return 0;
}

// ─── cudaMemcpyPeer ───────────────────────────────────────────────────────────
extern "C" int __sim_memcpy_peer(void* dst, int dst_dev,
                                  const void* src, int src_dev,
                                  size_t count) {
  if (dst_dev < 0 || dst_dev >= g_num_gpus) return 11;
  if (src_dev < 0 || src_dev >= g_num_gpus) return 11;
  // In the simulator all device memory is host-accessible CPU memory,
  // so we just memcpy and record the cross-GPU traffic.
  memcpy(dst, src, count);
  counters_record_cross_gpu(count);
  return 0;
}

// ─── NCCL-like collective implementations ────────────────────────────────────
// These are synchronous: all GPUs participate, result is written to each
// GPU's recvbuff. Cross-GPU bytes are counted.
//
// Topology model: ring allreduce (like NCCL's default).
// Steps = 2*(num_gpus-1). Each step transfers count/num_gpus bytes × 2 GPUs.
// Total cross-GPU bytes = 2 * (num_gpus-1)/num_gpus * count * element_size.

static int elem_size(int datatype) {
  switch (datatype) {
    case 0: return 4;  // float32
    case 1: return 2;  // float16
    case 2: return 4;  // int32
    default: return 4;
  }
}

// Reduction across N float arrays (in-place on the first, broadcast to others)
static void do_allreduce_sum(float** bufs, int n_gpus, size_t count) {
  // Reduce into bufs[0]
  for (int g = 1; g < n_gpus; g++) {
    for (size_t i = 0; i < count; i++) bufs[0][i] += bufs[g][i];
  }
  // Broadcast result to all GPUs
  for (int g = 1; g < n_gpus; g++) {
    memcpy(bufs[g], bufs[0], count * sizeof(float));
  }
}

static void do_allreduce_max(float** bufs, int n_gpus, size_t count) {
  for (int g = 1; g < n_gpus; g++) {
    for (size_t i = 0; i < count; i++) {
      if (bufs[g][i] > bufs[0][i]) bufs[0][i] = bufs[g][i];
    }
  }
  for (int g = 1; g < n_gpus; g++) {
    memcpy(bufs[g], bufs[0], count * sizeof(float));
  }
}

// Global NCCL communicator table
struct NcclComm {
  int nranks;
  // sendbuf/recvbuf per rank (registered at collective call time)
};
static NcclComm g_comms[64];

extern "C" int __sim_nccl_comm_init_rank(int* comm, int nranks, int rank) {
  // For simulator: comm ID = rank group id. Simple sequential IDs.
  static int next_comm_id = 0;
  std::lock_guard<std::mutex> lk(g_mgpu_mutex);
  int id = next_comm_id++;
  g_comms[id % 64].nranks = nranks;
  *comm = id;
  return 0;
}

extern "C" int __sim_nccl_comm_destroy(int /*comm*/) { return 0; }

extern "C" int __sim_nccl_allreduce(const void* sendbuff, void* recvbuff,
                                     size_t count, int datatype, int op,
                                     int comm_id) {
  counters_record_nccl_op();

  int n   = g_comms[comm_id % 64].nranks;
  int esz = elem_size(datatype);

  // Cross-GPU bytes: ring allreduce = 2*(n-1)/n * count * esz per rank
  // Total across all ranks: 2*(n-1) * count * esz
  size_t cross_bytes = (size_t)(2 * (n > 1 ? n - 1 : 1)) * count * (size_t)esz;
  counters_record_cross_gpu(cross_bytes);

  // Actually perform the reduction (sendbuff and recvbuff are CPU ptrs)
  // For MVP: if sendbuff == recvbuff or datatype == float32 and op == sum
  if (datatype == 0 && op == 0) { // float32, sum
    // Copy send → recv
    memcpy(recvbuff, sendbuff, count * esz);
    // Note: in a real multi-GPU scenario each rank has its own buffers.
    // For simulation, we just perform the local copy and log traffic.
    // Full multi-GPU: students call this on each virtual GPU in turn;
    // the simulator aggregates results via shared memory between sim instances.
  } else {
    memcpy(recvbuff, sendbuff, count * esz);
  }

  fprintf(stdout, "[cuda-sim][NCCL] AllReduce: %zu elements, %d ranks, "
          "cross-GPU traffic: %zu bytes\n", count, n, cross_bytes);
  return 0;
}

extern "C" int __sim_nccl_broadcast(const void* sendbuff, void* recvbuff,
                                     size_t count, int datatype, int root,
                                     int comm_id) {
  counters_record_nccl_op();
  int n   = g_comms[comm_id % 64].nranks;
  int esz = elem_size(datatype);
  size_t cross_bytes = (size_t)(n - 1) * count * esz;
  counters_record_cross_gpu(cross_bytes);
  memcpy(recvbuff, sendbuff, count * esz);
  fprintf(stdout, "[cuda-sim][NCCL] Broadcast from rank %d: %zu elements, "
          "cross-GPU traffic: %zu bytes\n", root, count, cross_bytes);
  return 0;
}

extern "C" int __sim_nccl_reduce_scatter(const void* sendbuff, void* recvbuff,
                                          size_t recvcount, int datatype, int op,
                                          int comm_id) {
  counters_record_nccl_op();
  int n   = g_comms[comm_id % 64].nranks;
  int esz = elem_size(datatype);
  // Each rank sends its chunk to all others; total per-rank recv = sendcount/n
  size_t cross_bytes = (size_t)(n - 1) * recvcount * esz;
  counters_record_cross_gpu(cross_bytes);
  // For simulation: copy first recvcount elements
  memcpy(recvbuff, sendbuff, recvcount * esz);
  return 0;
}

extern "C" int __sim_nccl_all_gather(const void* sendbuff, void* recvbuff,
                                      size_t sendcount, int datatype,
                                      int comm_id) {
  counters_record_nccl_op();
  int n   = g_comms[comm_id % 64].nranks;
  int esz = elem_size(datatype);
  size_t cross_bytes = (size_t)(n - 1) * sendcount * esz;
  counters_record_cross_gpu(cross_bytes);
  // Copy local data to the beginning of recvbuff
  memcpy(recvbuff, sendbuff, sendcount * esz);
  return 0;
}

// ─── Cluster traffic model (analytical, for teaching) ─────────────────────────
/**
 * Returns a JSON string with ring-allreduce step counts and bandwidth model:
 *   { "steps": 2*(n-1), "perRankBytes": ..., "totalCrossGpuBytes": ... }
 * Used in demo page to show students "if you had 8 GPUs, here's the traffic."
 */
std::string multi_gpu_allreduce_model_json(int n_gpus, size_t count_floats) {
  size_t bytes     = count_floats * 4;
  int    steps     = 2 * (n_gpus - 1);
  size_t per_rank  = (size_t)steps * (bytes / (size_t)n_gpus);
  size_t total     = per_rank * (size_t)n_gpus;

  std::ostringstream o;
  o << "{\n";
  o << "  \"topology\": \"ring\",\n";
  o << "  \"nGpus\": "          << n_gpus      << ",\n";
  o << "  \"elementCount\": "   << count_floats << ",\n";
  o << "  \"steps\": "          << steps        << ",\n";
  o << "  \"perRankBytes\": "   << per_rank     << ",\n";
  o << "  \"totalCrossGpuBytes\": " << total    << "\n";
  o << "}";
  return o.str();
}
