#pragma once
#include <string>
#include <vector>
#include "../include/cuda_sim.h"

void detector_check_bounds(const void* addr, int bytes, bool is_store, dim3 block_idx, dim3 thread_idx);
void detector_report_barrier_divergence(dim3 block_idx, const std::vector<int>& missing_tids);
void detector_report_deadlock(dim3 block_idx, int arrived, int n_alive);
void detector_check_uninit(const void* addr, int bytes, dim3 block_idx, dim3 thread_idx);
void detector_check_race_outputs(const void* out1, const void* out2, size_t bytes);
void detectors_reset();
bool detectors_has_fatal();
std::string detectors_to_json();
