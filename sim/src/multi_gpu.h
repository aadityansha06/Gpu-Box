#pragma once
#include <string>

void multi_gpu_init(int num_gpus);
void multi_gpu_reset();
std::string multi_gpu_allreduce_model_json(int n_gpus, size_t count_floats);
