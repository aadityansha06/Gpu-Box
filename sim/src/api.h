#pragma once
#include <string>

void sim_reset();
std::string sim_get_counters_json();
std::string sim_get_diagnostics_json();
std::string sim_rewrite_source(const std::string& source);
std::string sim_result_to_json(bool passed);
std::string sim_get_allreduce_model(int n_gpus, int count_floats);
bool sim_has_errors();
