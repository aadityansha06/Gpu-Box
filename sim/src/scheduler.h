#pragma once
#include <cstddef>
#include "../include/cuda_sim.h"

extern "C" {
void __sim_launch_impl(const char* name, dim3 grid, dim3 block,
                       size_t shmem, void (*thunk)(void*), void* ctx);
void __sim_syncthreads();
}
