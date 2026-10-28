#pragma once
#include <string>
#include "../include/cuda_sim.h"

// Called by counters.cpp
void counters_reset();
void counters_record_launch(const char* name, dim3 grid, dim3 block, size_t shmem, long long total_threads);
void counters_record_syncthreads();
void counters_record_atomic(const void* addr);
void counters_record_malloc(size_t bytes);
void counters_record_memcpy(size_t bytes, int kind);
void counters_record_cross_gpu(size_t bytes);
void counters_record_nccl_op();
std::string counters_to_json();
