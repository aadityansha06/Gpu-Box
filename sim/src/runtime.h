#pragma once
#include <string>
#include <vector>
#include "../include/cuda_sim.h"

struct AllocationInfo {
  bool    found;
  bool    in_guard;
  int     device_id;
  uintptr_t base_addr;
  size_t    alloc_size;
};

AllocationInfo sim_query_address(const void* addr);
bool sim_is_device_ptr(const void* addr);
void sim_check_leaks();
void sim_runtime_reset();
