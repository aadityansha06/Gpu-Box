#pragma once
/**
 * cuda_sim.h
 * ──────────
 * Drop-in shim that makes CUDA C++ compile and run on a CPU (host) as plain
 * C++17, driven by the cuda-sim fiber scheduler.
 *
 * Include this instead of <cuda_runtime.h> when compiling for the simulator.
 * All real GPU headers are replaced by this single file.
 */

#ifndef CUDA_SIM_H
#define CUDA_SIM_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cassert>
#include <algorithm>

// ─── Forward declarations from the runtime ───────────────────────────────────
extern "C" {
  void __sim_syncthreads();
  void __sim_syncwarp(unsigned mask);
  void __sim_record_load(const void* addr, int bytes, int static_id);
  void __sim_record_store(const void* addr, int bytes, int static_id);
  float __sim_shfl_down_sync(unsigned mask, float var, unsigned delta, int width);
  float __sim_shfl_xor_sync(unsigned mask, float var, int laneMask, int width);
  float __sim_shfl_sync(unsigned mask, float var, int srcLane, int width);
  int   __sim_shfl_down_sync_i(unsigned mask, int var, unsigned delta, int width);
  void* __sim_shared_alloc(int bytes, int align, int static_id);
  void  __sim_atomic_add_f(float* addr, float val);
  void  __sim_atomic_add_i(int*   addr, int   val);
  float __sim_atomic_max_f(float* addr, float val);
  int   __sim_atomic_max_i(int*   addr, int   val);
  int   __sim_atomic_cas(int* addr, int compare, int val);
}

// ─── Thread / block index structs ────────────────────────────────────────────
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned _x = 1, unsigned _y = 1, unsigned _z = 1)
    : x(_x), y(_y), z(_z) {}
  // allow implicit construction from int
  dim3(int _x) : x((unsigned)_x), y(1), z(1) {}
};

// Per-fiber (thread) state — set by the scheduler before invoking the kernel
namespace __sim_tls {
  inline thread_local dim3 threadIdx_v{0,0,0};
  inline thread_local dim3 blockIdx_v{0,0,0};
  inline thread_local dim3 blockDim_v{1,1,1};
  inline thread_local dim3 gridDim_v{1,1,1};
  inline thread_local int  laneId_v{0};
  inline thread_local int  warpId_v{0};
}

#define threadIdx  (__sim_tls::threadIdx_v)
#define blockIdx   (__sim_tls::blockIdx_v)
#define blockDim   (__sim_tls::blockDim_v)
#define gridDim    (__sim_tls::gridDim_v)

// ─── CUDA qualifiers → no-ops / attributes ───────────────────────────────────
#define __global__      __attribute__((noinline))
#define __device__      __attribute__((noinline))
#define __host__
#define __forceinline__ inline
#define __restrict__    __restrict
#define __launch_bounds__(...)

// __shared__ is handled by the per-block arena.
// We redirect static __shared__ arrays to the arena via a macro trick.
// Each unique callsite gets a unique static_id from __LINE__.
// The returned pointer is correctly aligned.
#define __shared__  /* see __sim_shared_alloc usage below */

// Helper macro for declaring shared arrays (used in starter code):
//   SIM_SHARED(float, tile, 32*32)
// expands to a local reference into the per-block arena.
#define SIM_SHARED(type, name, count)  \
  type* name = (type*)__sim_shared_alloc(sizeof(type)*(count), alignof(type), __LINE__)

// ─── Synchronization ─────────────────────────────────────────────────────────
#define __syncthreads()      __sim_syncthreads()
#define __syncwarp(mask)     __sim_syncwarp(mask)
#define __threadfence()      do {} while(0)  // memory ordering not modeled
#define __threadfence_block() do {} while(0)

// ─── Warp shuffle intrinsics ─────────────────────────────────────────────────
#define __shfl_down_sync(mask, var, delta, width) \
  __sim_shfl_down_sync(mask, (float)(var), delta, width)
#define __shfl_xor_sync(mask, var, laneMask, width) \
  __sim_shfl_xor_sync(mask, (float)(var), laneMask, width)
#define __shfl_sync(mask, var, srcLane, width) \
  __sim_shfl_sync(mask, (float)(var), srcLane, width)

// ─── Atomic operations ───────────────────────────────────────────────────────
inline float atomicAdd(float* addr, float val) {
  float old = *addr;
  __sim_atomic_add_f(addr, val);
  return old;
}
inline int atomicAdd(int* addr, int val) {
  int old = *addr;
  __sim_atomic_add_i(addr, val);
  return old;
}
inline float atomicMax(float* addr, float val) {
  return __sim_atomic_max_f(addr, val);
}
inline int atomicMax(int* addr, int val) {
  return __sim_atomic_max_i(addr, val);
}
inline int atomicCAS(int* addr, int compare, int val) {
  return __sim_atomic_cas(addr, compare, val);
}

// ─── Math functions (map to libm) ────────────────────────────────────────────
using std::expf;
using std::exp;
using std::logf;
using std::log;
using std::sqrtf;
using std::sqrt;
using std::fabsf;
using std::fabs;
using std::fmaxf;
using std::fminf;
using std::tanhf;
using std::tanh;
using std::sinf;
using std::cosf;
using std::powf;

// Fast-math approximations — map to libm for correctness
#define __expf(x)         expf(x)
#define __logf(x)         logf(x)
#define __fdividef(x,y)   ((x)/(y))
#define __fmaf_rn(a,b,c)  fmaf((a),(b),(c))
#define rsqrtf(x)         (1.0f/sqrtf(x))
#define __ldg(ptr)        (*(ptr))   // non-temporal load → plain load

// Special float values used in ML kernels
#ifndef CUDART_INF_F
#  define CUDART_INF_F __builtin_inff()
#endif
#ifndef CUDART_NAN_F
#  define CUDART_NAN_F __builtin_nanf("")
#endif

// ─── CUDA error codes / runtime API ──────────────────────────────────────────
typedef int cudaError_t;
static const cudaError_t cudaSuccess              = 0;
static const cudaError_t cudaErrorMemoryAllocation = 2;
static const cudaError_t cudaErrorInvalidValue     = 11;

inline const char* cudaGetErrorString(cudaError_t e) {
  switch(e) {
    case cudaSuccess:               return "cudaSuccess";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInvalidValue:     return "cudaErrorInvalidValue";
    default:                        return "cudaUnknownError";
  }
}

// These are implemented in runtime.cpp
extern "C" {
  cudaError_t __sim_malloc(void** ptr, size_t size);
  cudaError_t __sim_free(void* ptr);
  cudaError_t __sim_memcpy(void* dst, const void* src, size_t count, int kind);
  cudaError_t __sim_memset(void* ptr, int value, size_t count);
  cudaError_t __sim_device_synchronize();
  cudaError_t __sim_get_last_error();
}

inline cudaError_t cudaMalloc(void** ptr, size_t size) { return __sim_malloc(ptr, size); }
inline cudaError_t cudaFree(void* ptr)                 { return __sim_free(ptr); }
inline cudaError_t cudaMemset(void* ptr, int val, size_t n) { return __sim_memset(ptr, val, n); }
inline cudaError_t cudaDeviceSynchronize()             { return __sim_device_synchronize(); }
inline cudaError_t cudaGetLastError()                  { return __sim_get_last_error(); }

// cudaMemcpyKind
enum cudaMemcpyKind {
  cudaMemcpyHostToHost   = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
};
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return __sim_memcpy(dst, src, count, (int)kind);
}

// Multi-GPU peer APIs (implemented in multi_gpu.cpp)
extern "C" {
  cudaError_t __sim_set_device(int device_id);
  cudaError_t __sim_get_device(int* device_id);
  cudaError_t __sim_memcpy_peer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
  cudaError_t __sim_enable_peer_access(int peerDevice, unsigned int flags);
}

inline cudaError_t cudaSetDevice(int dev)   { return __sim_set_device(dev); }
inline cudaError_t cudaGetDevice(int* dev)  { return __sim_get_device(dev); }
inline cudaError_t cudaMemcpyPeer(void* dst, int dstDev, const void* src, int srcDev, size_t n) {
  return __sim_memcpy_peer(dst, dstDev, src, srcDev, n);
}
inline cudaError_t cudaDeviceEnablePeerAccess(int peer, unsigned flags) {
  return __sim_enable_peer_access(peer, flags);
}

// ─── Kernel launch (replaces <<<>>> after rewriting) ─────────────────────────
// The rewriter transforms:
//   my_kernel<<<grid, block, shmem, stream>>>(args...)
// into:
//   __sim_launch("my_kernel", my_kernel, grid, block, shmem, [&](){ my_kernel(args...); })
//
// Student code never calls this directly.
typedef void (*__sim_kernel_fn_t)();
extern "C" void __sim_launch_impl(const char* name, dim3 grid, dim3 block,
                                   size_t shmem, void (*thunk)(void*), void* ctx);

// printf from device code — limited, attributed
#define printf(...) __sim_device_printf(__VA_ARGS__)
extern "C" int __sim_device_printf(const char* fmt, ...);

// ─── Warp lane / warp id helpers ─────────────────────────────────────────────
inline int __lane_id()  { return __sim_tls::laneId_v; }
inline int __warp_id()  { return __sim_tls::warpId_v; }
#define __ballot_sync(mask, pred)   ((unsigned)((pred) ? (1u << __lane_id()) : 0u))
#define __all_sync(mask, pred)      ((pred) != 0)
#define __any_sync(mask, pred)      ((pred) != 0)

// ─── float4 / int4 vector types ──────────────────────────────────────────────
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int2   { int x, y; };
struct int4   { int x, y, z, w; };

inline float4 make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
inline float2 make_float2(float x, float y)                   { return {x,y}; }
inline int4   make_int4(int x, int y, int z, int w)           { return {x,y,z,w}; }

// ─── NCCL-like collective API (multi-GPU, implemented in multi_gpu.cpp) ───────
typedef int ncclComm_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;

static const ncclDataType_t ncclFloat32 = 0;
static const ncclDataType_t ncclFloat16 = 1;
static const ncclDataType_t ncclInt32   = 2;
static const ncclRedOp_t    ncclSum     = 0;
static const ncclRedOp_t    ncclMax     = 1;
static const ncclRedOp_t    ncclProd    = 2;

typedef int ncclResult_t;
static const ncclResult_t ncclSuccess = 0;

extern "C" {
  ncclResult_t __sim_nccl_allreduce(const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm);
  ncclResult_t __sim_nccl_broadcast(const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, int root, ncclComm_t comm);
  ncclResult_t __sim_nccl_reduce_scatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm);
  ncclResult_t __sim_nccl_all_gather(const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype, ncclComm_t comm);
  ncclResult_t __sim_nccl_comm_init_rank(ncclComm_t* comm, int nranks, int rank);
  ncclResult_t __sim_nccl_comm_destroy(ncclComm_t comm);
}

inline ncclResult_t ncclAllReduce(const void* s, void* r, size_t c,
  ncclDataType_t t, ncclRedOp_t op, ncclComm_t comm, void* /*stream*/) {
  return __sim_nccl_allreduce(s, r, c, t, op, comm);
}
inline ncclResult_t ncclBroadcast(const void* s, void* r, size_t c,
  ncclDataType_t t, int root, ncclComm_t comm, void* /*stream*/) {
  return __sim_nccl_broadcast(s, r, c, t, root, comm);
}
inline ncclResult_t ncclReduceScatter(const void* s, void* r, size_t rc,
  ncclDataType_t t, ncclRedOp_t op, ncclComm_t comm, void* /*stream*/) {
  return __sim_nccl_reduce_scatter(s, r, rc, t, op, comm);
}
inline ncclResult_t ncclAllGather(const void* s, void* r, size_t sc,
  ncclDataType_t t, ncclComm_t comm, void* /*stream*/) {
  return __sim_nccl_all_gather(s, r, sc, t, comm);
}
inline ncclResult_t ncclCommInitRank(ncclComm_t* c, int n, int r) {
  return __sim_nccl_comm_init_rank(c, n, r);
}
inline ncclResult_t ncclCommDestroy(ncclComm_t c) {
  return __sim_nccl_comm_destroy(c);
}

// cudaStream_t — ignored (everything runs synchronously in sim)
typedef void* cudaStream_t;
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

#endif // CUDA_SIM_H
